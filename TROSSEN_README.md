# Trossen Fork Notes

This fork of UMI replaces the original UR5 + SpaceMouse real-world eval
hardware with **Trossen leader/follower arms**, moves the project to
`uv`/`pyproject.toml`, and adds a couple of dependency-driven bug fixes and a
training-data path that reads `trumi`'s MCAP recordings directly. This doc
covers what changed and why; the data collection / generation pipeline for
Trossen-specific UMI lives in a separate repo,
[TrossenRobotics/trumi](https://github.com/TrossenRobotics/trumi/tree/main) —
follow that repo's docs for data collection. The original UMI SLAM
pipeline in the main `README.md` still applies as-is and is unaffected by
this fork; follow it too.

## Install (uv)

The project moved from the conda environment to [uv](https://docs.astral.sh/uv/),
pinned to Python 3.13 and PyTorch 2.11+ (CUDA 12.8):

```console
$ uv sync
```

`pyproject.toml` pulls `torch`/`torchvision`/`torchaudio` from PyTorch's
`cu128` wheel index (see `[[tool.uv.index]]`) — swap that index if you're on a
different CUDA version. `uv.lock` is checked in; commit it whenever
`pyproject.toml` changes so installs stay reproducible.

Everything below runs as `uv run python <script>` instead of inside an
activated conda env.

## Trossen hardware eval

### Architecture

`eval_real.py` → `BimanualUmiEnv`, same multiprocess structure as the
original UR5 path (camera workers + robot controller(s) as `mp.Process`,
communicating over `SharedMemoryRingBuffer`/`SharedMemoryQueue`), with two
new processes for Trossen:

- **`LeaderArm`** (`umi/real_world/leader_arm_shared_memory.py`) — replaces
  the SpaceMouse. Puts the leader arm in gravity-compensated
  `external_effort` mode so it can be moved freely by hand, and publishes its
  EEF pose + gripper width to a ring buffer at 100 Hz. The follower mirrors
  the leader's *absolute* pose (not incremental deltas, unlike the
  SpaceMouse).
- **`TrossenArmController`** (`umi/real_world/trossen_arm_controller.py`) —
  drop-in replacement for `RTDEInterpolationController`: same
  `SCHEDULE_WAYPOINT`/queue/ring-buffer/`PoseTrajectoryInterpolator` shape, so
  `BimanualUmiEnv` and `eval_real.py`'s control loop barely changed. Adds a
  `SCHEDULE_GRIPPER` command since the Trossen gripper is the arm's 7th
  joint, not a separate WSG controller — `TrossenGripperController` is a
  thin adapter that forwards gripper commands into the same process/queue so
  `BimanualUmiEnv` can still treat it as a generic gripper.
- UR5/Franka controller imports in `bimanual_umi_env.py` are now lazy (only
  imported if that `robot_type` is actually configured), so their SDKs
  aren't required when only Trossen hardware is present.

Config lives in `example/eval_trossen_config.yaml` — robot/leader IPs,
per-arm latencies, `gripper_max_width`, collision settings. Uncomment the
second `robots`/`grippers` block for bimanual.

```console
$ uv run python eval_real.py --robot_config=example/eval_trossen_config.yaml \
    -i <checkpoint.ckpt> -o data/eval_output
```

### Pose conventions

Two different coordinate/tool conventions meet at the robot boundary, and
getting them backwards silently produces plausible-but-wrong motion:

- **Dataset / policy convention**: gripper pose in the ArUco *tag* frame
  (Z-up world), orientation in the **GoPro camera** convention (+X right, +Y
  down, +Z forward/approach). This is what the policy was trained on.
- **Trossen TCP convention**: +X forward/approach, +Y left, +Z up — a
  different axis labeling of the *same physical gripper*, plus the fingertip
  origin is ~85mm forward of Trossen's stock TCP (the UMI fingers are longer
  than stock).

`umi/common/pose_util.py` defines the fixed tool-frame relabel:

```python
X_TOOL_CAM_TO_TROSSEN   # camera convention -> Trossen TCP convention
X_TOOL_TROSSEN_TO_CAM   # inverse
```

`umi/real_world/real_inference_util.py` applies `X_TOOL_TROSSEN_TO_CAM` to
every pose read from the env (so the policy always sees camera-convention
poses, matching training) and `X_TOOL_CAM_TO_TROSSEN` to the policy's output
action before sending it to the robot. This is the only place eval needs to
know about the conversion — `TrossenArmController` itself just moves to
whatever pose it's given, in Trossen TCP convention.

`scripts_real/replay_real_bimanual_umi.py` needs an *additional* fix beyond
the tool-frame relabel: the original UR5 script anchored dataset episodes to
the robot's base frame with a hand-measured `tx_tag_robot` extrinsic
calibration, which we don't have for Trossen. Instead,
`episode_pose_to_robot()` anchors each episode's first frame onto the
follower's *current actual pose* at replay time — the robot starts wherever
it already is and moves through the episode from there.

Gripper width is carried as **total opening** (the full gap between both
fingers) everywhere in the ring buffer / zarr / dataset convention. The
Trossen driver's `get_gripper_position()`/`set_gripper_position()` use
**one-side stroke**, so `TrossenArmController` and `LeaderArm` multiply by 2
reading from hardware and divide by 2 writing to it — the only place this
conversion happens.

### Calibration workflow

```console
# measure robot_action_latency for the follower (drives it from the leader,
# cross-correlates commanded vs. actual EEF trajectory)
$ uv run python scripts/calibrate_trossen_robot_latency.py \
    --follower_ip 192.168.1.4 --leader_ip 192.168.1.2 --duration 60

# measure camera_obs_latency (same as upstream UMI, robustness fixes below)
$ uv run python scripts/calibrate_uvc_camera_latency.py --camera_idx 1 --fps 60 --n_frames 120

# sanity-check the finger ArUco tags (detection rate + measured gripper
# width) against a recorded GoPro clip, before trusting a dataset's
# robotN_gripper_width column
$ uv run python scripts/test_gripper_aruco_detection.py \
    -i <video.mp4> -ij <gopro_intrinsics.json>
```

Plug the measured `robot_action_latency` / `camera_obs_latency` into
`example/eval_trossen_config.yaml`.

### Replaying a recorded episode on the Trossen

```console
$ uv run python scripts_real/replay_real_bimanual_umi.py \
    --input data/cup_in_the_wild.zarr.zip \
    --output data/replay_output \
    --robot_ip 192.168.1.4 \
    --leader_ip 192.168.1.2 \
    --replay_episode 0 \
    --camera_reorder 1
```

## OpenCV / Qt / X11 threadlock

`cv2`'s Qt (xcb) backend has to open its X11 connection *before* `av`
(PyAV) and `torch` load their own ffmpeg/threading libraries, and before any
`UvcCamera` subprocess touches X11 — otherwise the first `cv2` GUI call
(`imshow`/`namedWindow`) deadlocks inside Qt's shared-memory probe. Every
entry point that opens a camera GUI window imports `cv2` first and calls
`cv2.setNumThreads(1)` before anything else:

- `eval_real.py`
- `scripts/calibrate_uvc_camera_latency.py` (also pre-opens its windows in
  the main process before spawning the `UvcCamera` subprocess, sets
  `QT_X11_NO_MITSHM=1` to avoid `xcb_shm_create_segment()` crashes, and
  waits for Elgato capture cards to re-enumerate after a firmware reset)
- `scripts_real/replay_real_bimanual_umi.py`

If you add a new script that opens a `cv2` window and spawns `UvcCamera`
processes, follow the same import order.

## MCAP training data

`diffusion_policy/dataset/umi_mcap_dataset.py` (`UmiMcapDataset`) reads
`trumi`'s per-episode `episode_*.mcap` recordings directly — JPEG-encoded
camera topics + JSON pose/gripper topics — into an in-memory replay buffer,
skipping the `run_slam_pipeline.py` → `dataset.zarr.zip` export step
entirely. Everything past replay-buffer construction (sampler, normalizer,
`__getitem__`) is inherited from `UmiDataset` unchanged; its
`_load_replay_buffer()` was factored out to make that override possible.

Not yet supported: ArUco-tag inpainting/masking, and H.264-encoded camera
topics (mcap datasets must be recorded with `--video_codec jpeg`).

```console
$ uv run python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
    task=umi_mcap task.dataset_path=<path to trumi episode_*.mcap directory>
```

## Other dependency-driven fixes

Pinning newer package versions (`opencv-python`, `diffusers`) than upstream
UMI's conda environment broke a few call sites:

- **`umi/common/cv_util.py`**: `cv2.aruco.detectMarkers()` /
  `estimatePoseSingleMarkers()` were removed from newer OpenCV. Replaced with
  `cv2.aruco.ArucoDetector.detectMarkers()` and `cv2.solvePnP()` against
  explicit marker corner object points.
- **`diffusion_policy/model/common/lr_scheduler.py`**: `diffusers>=0.38`'s
  `diffusers.optimization` no longer re-exports `Union`/`Optional` (upstream
  UMI pins an older `diffusers` where it still does — note upstream actually
  tried and reverted this same fix, since their pin doesn't need it). Import
  `Union`/`Optional` from `typing` and `Optimizer` from `torch.optim`
  directly instead.
- **`scripts/calibrate_slam_tag.py`**: replaced the `skfda`-based
  `geometric_median` with a small local Weiszfeld-algorithm implementation,
  dropping the `scikit-fda` dependency for one function.
