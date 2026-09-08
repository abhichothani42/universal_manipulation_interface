# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import json
import yaml
import av
import numpy as np
import cv2
from tqdm import tqdm

from umi.common.cv_util import (
    parse_aruco_config,
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    draw_predefined_mask,
    get_gripper_width
)

# our 3D-printed finger tags: gripper0 -> ids 0/1, gripper1 -> ids 6/7
GRIPPER_TAG_LABELS = {
    0: ('gripper0', 'left'),
    1: ('gripper0', 'right'),
    6: ('gripper1', 'left'),
    7: ('gripper1', 'right'),
}
GRIPPER_LEFT_RIGHT_IDS = {
    'gripper0': (0, 1),
    'gripper1': (6, 7),
}
GRIPPER_COLORS = {
    'gripper0': (0, 255, 255),   # cyan
    'gripper1': (255, 0, 255),   # magenta
}

# %%
@click.command()
@click.option('-i', '--input', required=True, help='Input GoPro video file.')
@click.option('-o', '--output', default=None, help='Optional annotated output video path.')
@click.option('-ij', '--intrinsics_json', required=True)
@click.option('-ay', '--aruco_yaml', default='example/calibration/aruco_config.yaml')
@click.option('--show', is_flag=True, default=False, help='Preview annotated frames in a window while processing.')
@click.option('-n', '--num_workers', type=int, default=4)
def main(input, output, intrinsics_json, aruco_yaml, show, num_workers):
    cv2.setNumThreads(num_workers)

    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']

    raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r')))

    tag_ids = sorted(GRIPPER_TAG_LABELS.keys())
    detect_counts = {tag_id: 0 for tag_id in tag_ids}
    width_samples = {name: [] for name in GRIPPER_LEFT_RIGHT_IDS}
    n_frames = 0

    out_container = None
    out_stream = None
    need_vis = show or (output is not None)

    with av.open(os.path.expanduser(input)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        in_res = np.array([in_stream.height, in_stream.width])[::-1]
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)

        if output is not None:
            out_container = av.open(os.path.expanduser(output), mode='w')
            out_stream = out_container.add_stream('h264', rate=in_stream.rate)
            out_stream.thread_type = 'AUTO'
            out_stream.thread_count = num_workers
            out_stream.width = in_stream.width
            out_stream.height = in_stream.height
            out_stream.codec_context.options = {'crf': '20', 'profile': 'high'}

        for i, frame in tqdm(enumerate(in_container.decode(in_stream)), total=in_stream.frames):
            img = frame.to_ndarray(format='rgb24')
            # mask mirrors only -- the finger tags we care about live outside that region
            det_img = draw_predefined_mask(
                img.copy(), color=(0, 0, 0), mirror=True, gripper=False, finger=False)

            tag_dict = detect_localize_aruco_tags(
                img=det_img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=fisheye_intr,
                refine_subpix=True
            )

            n_frames += 1
            for tag_id in tag_ids:
                if tag_id in tag_dict:
                    detect_counts[tag_id] += 1

            for gripper_name, (left_id, right_id) in GRIPPER_LEFT_RIGHT_IDS.items():
                width = get_gripper_width(tag_dict, left_id, right_id)
                if width is not None:
                    width_samples[gripper_name].append(width)

            if need_vis:
                vis_img = img.copy()
                for tag_id, (gripper_name, side) in GRIPPER_TAG_LABELS.items():
                    color = GRIPPER_COLORS[gripper_name]
                    if tag_id in tag_dict:
                        corners = tag_dict[tag_id]['corners'].astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(vis_img, [corners], True, color, 2)
                        center = tag_dict[tag_id]['corners'].mean(axis=0).astype(int)
                        cv2.putText(vis_img, f'{gripper_name}:{side}({tag_id})',
                                    tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(vis_img, f'frame {i}', (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if show:
                    disp = cv2.resize(vis_img, (960, 540))
                    cv2.imshow('aruco_detection', cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if out_stream is not None:
                    out_frame = av.VideoFrame.from_ndarray(vis_img, format='rgb24')
                    for packet in out_stream.encode(out_frame):
                        out_container.mux(packet)

        if out_stream is not None:
            for packet in out_stream.encode():
                out_container.mux(packet)

    if out_container is not None:
        out_container.close()
    if show:
        cv2.destroyAllWindows()

    print(f'\nProcessed {n_frames} frames from {input}')
    print('Per-tag detection rate:')
    for tag_id in tag_ids:
        gripper_name, side = GRIPPER_TAG_LABELS[tag_id]
        rate = detect_counts[tag_id] / n_frames if n_frames else 0.0
        print(f'  id={tag_id} ({gripper_name} {side:>5}): {detect_counts[tag_id]:>6d}/{n_frames} = {rate*100:5.1f}%')

    print('Gripper width sanity (both tags visible, nominal_z=0.072m +/- 0.008m):')
    for gripper_name, samples in width_samples.items():
        if samples:
            arr = np.array(samples)
            print(f'  {gripper_name}: n={len(arr)}, mean={arr.mean()*1000:.1f}mm, '
                  f'std={arr.std()*1000:.2f}mm, min={arr.min()*1000:.1f}mm, max={arr.max()*1000:.1f}mm')
        else:
            print(f'  {gripper_name}: no frames with both tags visible')

# %%
if __name__ == "__main__":
    main()
