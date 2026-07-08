import base64
import json
import pathlib
import re
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import zarr
from mcap.reader import make_reader

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.umi_dataset import UmiDataset
from umi.common.cv_util import get_image_transform

_ROBOT_EEF_RE = re.compile(r'^robot(\d+)/eef/state$')
_CAMERA_RE = re.compile(r'^/cameras/camera(\d+)/image$')


def _nearest_indices(sorted_ref: np.ndarray, query: np.ndarray) -> np.ndarray:
    """For each value in `query`, find the index of the nearest value in `sorted_ref`."""
    idx = np.clip(np.searchsorted(sorted_ref, query), 1, len(sorted_ref) - 1)
    left, right = sorted_ref[idx - 1], sorted_ref[idx]
    return idx - ((query - left) <= (right - query)).astype(np.int64)


class UmiMcapDataset(UmiDataset):
    """UmiDataset variant that reads trumi's per-episode .mcap files directly,
    instead of a pre-built dataset.zarr.zip. Everything downstream of replay
    buffer construction (SequenceSampler, get_normalizer, __getitem__) is
    inherited from UmiDataset unchanged.
    """

    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        out_res: Tuple[int, int] = (224, 224),
        cache_dir: Optional[str] = None,
        pose_repr: dict = {},
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_duration: Optional[float] = None
    ):
        self.out_res = tuple(out_res)
        super().__init__(
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            cache_dir=cache_dir,
            pose_repr=pose_repr,
            action_padding=action_padding,
            temporally_independent_normalization=temporally_independent_normalization,
            repeat_frame_prob=repeat_frame_prob,
            seed=seed,
            val_ratio=val_ratio,
            max_duration=max_duration
        )

    def _load_replay_buffer(self, dataset_path: str, cache_dir: Optional[str]) -> ReplayBuffer:
        if cache_dir is not None:
            warnings.warn(
                'UmiMcapDataset ignores cache_dir; mcap episodes are always '
                'read into an in-memory MemoryStore.')

        episode_paths = sorted(pathlib.Path(dataset_path).expanduser().glob('episode_*.mcap'))
        if not episode_paths:
            raise FileNotFoundError(f'No episode_*.mcap files found in {dataset_path}')

        replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
        ref_grippers, ref_cameras = None, None
        for ep_path in episode_paths:
            episode_data, gripper_ids, camera_ids = self._read_episode(ep_path)
            if ref_grippers is None:
                ref_grippers, ref_cameras = gripper_ids, camera_ids
            elif gripper_ids != ref_grippers or camera_ids != ref_cameras:
                raise ValueError(
                    f'{ep_path.name}: grippers={gripper_ids}, cameras={camera_ids} '
                    f'!= first episode grippers={ref_grippers}, cameras={ref_cameras}. '
                    'All episodes in a dataset must have identical gripper/camera indices.')
            replay_buffer.add_episode(data=episode_data, compressors=None)
        return replay_buffer

    def _read_episode(self, mcap_path: pathlib.Path) -> Tuple[Dict[str, np.ndarray], List[int], List[int]]:
        with open(mcap_path, 'rb') as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = {ch.topic for ch in summary.channels.values()}
            gripper_ids = sorted(
                int(m.group(1)) for t in topics if (m := _ROBOT_EEF_RE.match(t)))
            camera_ids = sorted(
                int(m.group(1)) for t in topics if (m := _CAMERA_RE.match(t)))
            if not gripper_ids:
                raise ValueError(f'{mcap_path.name}: no robot{{N}}/eef/state topics found')

            episode_data = {}
            t_grid = None
            for g in gripper_ids:
                pos, rot, log_times = self._read_pose_topic(reader, f'robot{g}/eef/state')
                widths = self._read_gripper_topic(reader, f'robot{g}/gripper/state')
                assert len(widths) == len(pos), (
                    f'{mcap_path.name}: robot{g} eef/state ({len(pos)}) and '
                    f'gripper/state ({len(widths)}) message counts differ')
                start_pos, start_rot = self._read_single_pose(reader, f'robot{g}/eef_demo_start/state')
                end_pos, end_rot = self._read_single_pose(reader, f'robot{g}/eef_demo_end/state')

                T = len(pos)
                if t_grid is None:
                    t_grid = log_times
                elif T != len(t_grid):
                    raise ValueError(
                        f'{mcap_path.name}: robot{g} has T={T} timesteps, expected '
                        f'T={len(t_grid)} (all grippers in an episode must share the '
                        'same timestep count)')

                episode_data[f'robot{g}_eef_pos'] = pos.astype(np.float32)
                episode_data[f'robot{g}_eef_rot_axis_angle'] = rot.astype(np.float32)
                episode_data[f'robot{g}_gripper_width'] = widths.astype(np.float32)[:, None]
                start6 = np.concatenate([start_pos, start_rot]).astype(np.float32)
                end6 = np.concatenate([end_pos, end_rot]).astype(np.float32)
                episode_data[f'robot{g}_demo_start_pose'] = np.tile(start6, (T, 1))
                episode_data[f'robot{g}_demo_end_pose'] = np.tile(end6, (T, 1))

            for c in camera_ids:
                episode_data[f'camera{c}_rgb'] = self._read_camera_topic(
                    reader, summary, f'/cameras/camera{c}/image', t_grid, mcap_path, c)

            return episode_data, gripper_ids, camera_ids

    @staticmethod
    def _read_pose_topic(reader, topic: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_times, positions, rotations = [], [], []
        for _, _, message in reader.iter_messages(topics=[topic]):
            payload = json.loads(message.data)
            log_times.append(message.log_time)
            positions.append(payload['pos'])
            rotations.append(payload['rot_axis_angle'])
        if not log_times:
            raise ValueError(f'No messages found on topic {topic}')
        return np.array(positions, dtype=np.float64), np.array(rotations, dtype=np.float64), \
            np.array(log_times, dtype=np.int64)

    @staticmethod
    def _read_gripper_topic(reader, topic: str) -> np.ndarray:
        widths = []
        for _, _, message in reader.iter_messages(topics=[topic]):
            payload = json.loads(message.data)
            widths.append(payload['width'])
        if not widths:
            raise ValueError(f'No messages found on topic {topic}')
        return np.array(widths, dtype=np.float64)

    @staticmethod
    def _read_single_pose(reader, topic: str) -> Tuple[np.ndarray, np.ndarray]:
        messages = list(reader.iter_messages(topics=[topic]))
        if not messages:
            raise ValueError(f'No messages found on topic {topic}')
        payload = json.loads(messages[0][2].data)
        return np.array(payload['pos'], dtype=np.float64), \
            np.array(payload['rot_axis_angle'], dtype=np.float64)

    def _read_camera_topic(self, reader, summary, topic: str,
            t_grid: np.ndarray, mcap_path: pathlib.Path, cam_id: int) -> np.ndarray:
        channel = next(ch for ch in summary.channels.values() if ch.topic == topic)
        schema_name = summary.schemas[channel.schema_id].name
        if schema_name != 'foxglove.CompressedImage':
            raise NotImplementedError(
                f'UmiMcapDataset only supports JPEG-encoded camera topics '
                f'(schema foxglove.CompressedImage). Got schema "{schema_name}" for '
                f'camera{cam_id} in {mcap_path.name} (likely recorded with '
                '--video_codec h264). Re-export the mcap dataset with --video_codec jpeg.')

        # pass 1: collect timestamps only (cheap, no JSON/base64 decode)
        img_log_times = np.array(
            [message.log_time for _, _, message in reader.iter_messages(topics=[topic])],
            dtype=np.int64)
        if len(img_log_times) == 0:
            raise ValueError(f'No messages found on topic {topic}')

        nearest = _nearest_indices(img_log_times, t_grid)
        needed_ordinals = set(nearest.tolist())

        # pass 2: decode only the frames actually needed
        decoded_cache = {}
        transform = None
        for ordinal, (_, _, message) in enumerate(reader.iter_messages(topics=[topic])):
            if ordinal not in needed_ordinals:
                continue
            payload = json.loads(message.data)
            if payload.get('format') != 'jpeg':
                raise NotImplementedError(
                    f'Expected jpeg-formatted frames on {topic}, got '
                    f'format="{payload.get("format")}" in {mcap_path.name}')
            jpeg_bytes = base64.b64decode(payload['data'])
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if transform is None:
                ih, iw = img.shape[:2]
                transform = get_image_transform(in_res=(iw, ih), out_res=self.out_res, bgr_to_rgb=True)
            decoded_cache[ordinal] = transform(img)

        out = np.empty((len(t_grid),) + self.out_res[::-1] + (3,), dtype=np.uint8)
        for i, ordinal in enumerate(nearest):
            out[i] = decoded_cache[int(ordinal)]
        return out
