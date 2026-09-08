"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import zarr
# cv2 and numpy must be imported before av and torch so OpenCV's Qt/X11
# backend initialises before ffmpeg/torch load their threading libraries.
import cv2
import numpy as np
cv2.setNumThreads(1)
import av
import click
import dill
import hydra
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from umi.common.interpolation_util import get_interp1d, PoseInterpolator

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.real_world.leader_arm_shared_memory import LeaderArm
from umi.common.pose_util import (pose_to_mat, mat_to_pose,
    X_TOOL_CAM_TO_TROSSEN, X_TOOL_TROSSEN_TO_CAM)
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

OmegaConf.register_new_resolver("eval", eval, replace=True)

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    if delta > 0:
        print(delta)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

@click.command()
@click.option('--input', '-i', required=True, help='Path to dataset')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--replay_episode', '-re', type=int, default=0)
# @click.option('--robot_ip', '-ri', default='172.24.95.9')
# @click.option('--gripper_ip', '-gi', default='172.24.95.17')
# @click.option('--robot_ip', default='172.24.95.8')
# @click.option('--gripper_ip', default='172.24.95.18')
@click.option('--robot_ip', default='192.168.1.4', help='Follower arm IP')
@click.option('--leader_ip', default='192.168.1.2', help='Leader arm IP (replaces SpaceMouse)')
@click.option('--gripper_ip', default='172.24.95.27', help='Unused for Trossen; kept for compatibility')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='120')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
def main(input, output, replay_episode, robot_ip, leader_ip, gripper_ip,
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    no_mirror):
    # total gripper opening in meters (full gap between both fingers)
    max_gripper_width = 0.088
    gripper_speed = 0.2

    # ================================================================
    # Replaying a dataset trajectory on the Trossen requires TWO fixes.
    #
    # The dataset stores each pose as tx_tag_cam: the gripper pose in the ArUco
    # TAG frame (Z-up), with orientation in the GoPro CAMERA convention
    # (+X right, +Y down, +Z forward/approach). We want tx_base_tcp: the pose in
    # the Trossen BASE frame (X fwd, Y left, Z up), in the Trossen TCP convention.
    #
    #     tx_base_tcp = tx_base_tag @ tx_tag_cam @ X_tool
    #                   \_________/              \______/
    #                   (1) world offset         (2) tool-frame relabel
    #
    # (1) tx_base_tag - WORLD offset (tag -> base). UR5 hard-coded a measured
    #     matrix (tx_tag_right/left); we don't have that calibration, so we BUILD
    #     one per episode by anchoring the episode's FIRST dataset frame to the
    #     robot's CURRENT actual pose:
    #         tx_base_tag = tx_base_tcp_now @ inv(X_tool) @ inv(tx_tag_cam[0])
    #     => frame 0 maps exactly onto where the robot already is, so it starts in
    #        place and then moves along the demo from there.
    #
    # (2) X_tool - fixed TOOL-frame relabel (camera convention -> Trossen TCP
    #     convention). UR5 needed none (its TCP is already Z=approach like the
    #     camera). Trossen's TCP is X=approach, so we right-multiply by this
    #     constant rotation. Columns = Trossen axes written in camera coords:
    #       Trossen X (approach) = camera +Z
    #       Trossen Y (left)     = camera -X
    #       Trossen Z (up)       = camera -Y
    #     It rotates orientation only; the gripper-tip position is unchanged.
    #     Defined once in umi/common/pose_util.py and shared with eval.
    X_tool = X_TOOL_CAM_TO_TROSSEN
    X_tool_inv = X_TOOL_TROSSEN_TO_CAM


    # load replay buffer
    # with zarr.ZipStore(input, mode='r') as zip_store:
    #     replay_buffer = ReplayBuffer.copy_from_store(
    #         src_store=zip_store, 
    #         store=None,
    #         keys=[
    #             'robot0_eef_pos',
    #             'robot0_eef_rot_axis_angle',
    #             'robot0_gripper_width',
    #             'robot1_eef_pos',
    #             'robot1_eef_rot_axis_angle',
    #             'robot1_gripper_width',
    #         ])
    zip_store = zarr.ZipStore(input, mode='a')
    root = zarr.group(zip_store)
    replay_buffer = ReplayBuffer.create_from_group(root)

    episode_data = None

    # replay_episode = replay_buffer.n_episodes - 1

    def episode_pose_to_robot(episode_idx, robot_anchor_poses):
        """Convert one episode's dataset poses to Trossen base-frame TCP poses:
            tx_base_tcp = tx_base_tag @ tx_tag_cam @ X_tool
        tx_base_tag is anchored so the episode's FIRST frame maps onto the robot's
        current actual pose, i.e. the robot starts where it is and moves from there.

        robot_anchor_poses: list of (6,) actual TCP poses [x,y,z,rx,ry,rz]
            (base frame), one per robot - the pose the robot is in right now.
        Returns (pose_data, episode_slice); pose_data[f'robot{i}_tcp_pose'] is an
        (N, 6) array already in the Trossen base frame.
        """
        s = replay_buffer.get_episode_slice(episode_idx)
        out = dict()
        for robot_idx in range(len(robot_anchor_poses)):
            pos = replay_buffer.data[f'robot{robot_idx}_eef_pos'][s]
            rot = replay_buffer.data[f'robot{robot_idx}_eef_rot_axis_angle'][s]
            tx_tag_cam = pose_to_mat(np.concatenate([pos, rot], axis=-1))  # (N,4,4)

            # anchor frame 0 onto the robot's current pose:
            #   tx_base_tag = tx_base_tcp_now @ inv(X_tool) @ inv(tx_tag_cam[0])
            tx_base_tcp_now = pose_to_mat(np.asarray(robot_anchor_poses[robot_idx]))
            tx_base_tag = tx_base_tcp_now @ X_tool_inv @ np.linalg.inv(tx_tag_cam[0])

            tx_base_tcp = tx_base_tag @ tx_tag_cam @ X_tool  # (N,4,4)
            out[f'robot{robot_idx}_tcp_pose'] = mat_to_pose(tx_base_tcp)
        return out, s


    # setup experiment
    dt = 1/frequency

    obs_res = (224, 224)
    # load fisheye converter
    fisheye_converter = None

    robots_config = [
        {
            'robot_type': 'trossen',
            'robot_ip': robot_ip,
            'leader_ip': leader_ip,
            'frequency': 125,
            'robot_obs_latency': 0.0001, 'robot_action_latency': 0.02,
            'gripper_max_width': max_gripper_width,
            'init_joints_pos': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'height_threshold': -1000000.0,   # set to table z to enable collision avoidance
            'sphere_radius': 0.0, 'sphere_center': [0, 0, 0],
        }
    ]
    # For Trossen the gripper is the 7th joint of the arm — no separate gripper controller.
    # gripper_ip/port are unused; obs/action latencies match robot_action_latency.
    grippers_config = [
        {
            'gripper_ip': '', 'gripper_port': 0,
            'gripper_obs_latency': 0.0001, 'gripper_action_latency': 0.02,
        }
    ]

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with LeaderArm(
                shm_manager=shm_manager,
                leader_ip=leader_ip,
                frequency=100,
                init_joints_pos=(np.array(robots_config[0]['init_joints_pos'])
                    if robots_config[0].get('init_joints_pos') is not None else None),
            ) as leader, \
            KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.1,
                # obs
                camera_obs_horizon=2,
                robot_obs_horizon=2,
                gripper_obs_horizon=2,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                # action
                max_pos_speed=0.25/4,
                max_rot_speed=0.6/4,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            # leader arm used for teleoperation; one per follower robot
            leaders = [leader]
            print("Waiting for camera")
            time.sleep(1.0)

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states])

                gripper_states = env.get_gripper_state()
                gripper_target_pos = np.asarray([gs['gripper_position'] for gs in gripper_states])
                
                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]

                    text = f'Episode: {replay_episode}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            replay_episode = min(replay_episode + 1, replay_buffer.n_episodes-1)
                            print('e')
                        elif key_stroke == KeyCode(char='w'):
                            # Prev episode
                            replay_episode = max(replay_episode - 1, 0)
                            print('w')
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 5.0
                            # ep = replay_buffer.get_episode(replay_episode)
                            anchors = [rs['ActualTCPPose'] for rs in env.get_robot_state()]
                            pose_data, s = episode_pose_to_robot(replay_episode, anchors)

                            for robot_idx in range(len(robots_config)):
                                pose = pose_data[f'robot{robot_idx}_tcp_pose'][0]
                                grip = float(np.squeeze(replay_buffer.data[f'robot{robot_idx}_gripper_width'][s.start]))
                                env.robots[robot_idx].schedule_waypoint(pose, target_time=time.time() + duration)
                                env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                target_pose[robot_idx] = pose
                                gripper_target_pos[robot_idx] = grip

                            start_t = time.time()
                            episode_data = replay_buffer.get_episode(replay_episode)
                            time.sleep(max(duration - (time.time() - start_t), 0))

                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]

                    if start_policy:
                        break

                    precise_wait(t_sample)
                    # get teleop command from leader arm(s):
                    # follower mirrors the leader's absolute 6-DOF pose and gripper width.
                    # LeaderGripperPos is total-width (driver one-side × 2), matching zarr convention.
                    for robot_idx in range(target_pose.shape[0]):
                        leader_state = leaders[robot_idx].get_state()
                        target_pose[robot_idx] = np.array(leader_state['LeaderTCPPose'])
                        gripper_target_pos[robot_idx] = np.clip(
                            float(leader_state['LeaderGripperPos']), 0, max_gripper_width)

                    # solve collision with table
                    for robot_idx in control_robot_idx_list:
                        solve_table_collision(
                            ee_pose=target_pose[robot_idx],
                            gripper_width=gripper_target_pos[robot_idx],
                            height_threshold=robots_config[robot_idx]['height_threshold'])
                    
                    # solve collison between two robots
                    solve_sphere_collision(
                        ee_poses=target_pose,
                        robots_config=robots_config
                    )

                    action = np.zeros((7 * target_pose.shape[0],))

                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]


                    # execute teleop command
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    episode_data = replay_buffer.get_episode(replay_episode)
                    anchors = [rs['ActualTCPPose'] for rs in env.get_robot_state()]
                    pose_data, s = episode_pose_to_robot(replay_episode, anchors)

                    # pre-compute interpolation
                    data_frequency = 60.0
                    slowdown = 10.0
                    n_data_samples = len(pose_data['robot0_tcp_pose'])
                    data_timestamps = np.arange(n_data_samples).astype(np.float32) / data_frequency
                    exec_timestamps = np.arange(int(np.floor(data_timestamps[-1] * frequency * slowdown))) / frequency / slowdown
                    exec_data_idxs = np.round(np.clip(exec_timestamps, 0, data_timestamps[-1]) * data_frequency).astype(np.int32)

                    actions = np.zeros((len(exec_timestamps), 7 * len(robots_config)))
                    for robot_idx in range(len(robots_config)):
                        data_pose = pose_data[f'robot{robot_idx}_tcp_pose']
                        data_pose_interpolator = PoseInterpolator(data_timestamps, data_pose)
                        data_gripper_interpolator = get_interp1d(data_timestamps, episode_data[f'robot{robot_idx}_gripper_width'])
                        exec_pose = data_pose_interpolator(exec_timestamps)
                        exec_grip = data_gripper_interpolator(exec_timestamps)

                        for i in range(len(exec_pose)):
                            solve_table_collision(
                                ee_pose=exec_pose[i],
                                gripper_width=exec_grip[i,0],
                                height_threshold=robots_config[robot_idx]['height_threshold'])

                        actions[:,robot_idx*7:robot_idx*7+6] = exec_pose
                        actions[:,robot_idx*7+6:robot_idx*7+7] = exec_grip

                    # start episode
                    start_delay = 1.0 
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    # ts = eval_t_start + np.arange(len(actions)) / frequency
                    # env.exec_actions(
                    #     actions=actions, 
                    #     timestamps=ts)

                    n_cameras = sum(1 for k in episode_data if k.startswith('camera') and k.endswith('_rgb'))
                    for iter_idx, _ in enumerate(exec_timestamps):
                        t = iter_idx / frequency
                        t_cycle_start = t_start + t
                        t_cycle_end = t_cycle_start + 1/frequency

                        # pump obs
                        obs = env.get_obs()

                        action = actions[iter_idx]

                        env.exec_actions(
                            actions=[action], 
                            timestamps=[t_cycle_end-time.monotonic()+time.time()])
                        
                        # plot image overlay
                        data_idx = exec_data_idxs[iter_idx]
                        vis_imgs = list()
                        for camera_idx in range(n_cameras):
                            img = episode_data[f'camera{camera_idx}_rgb'][data_idx]
                            vis_img = obs[f'camera{camera_idx}_rgb'][-1]
                            match_img = img.astype(np.float32) / 255
                            avg_img = (vis_img + match_img) / 2
                            vis_img = np.concatenate([vis_img, avg_img, match_img], axis=1)
                            vis_imgs.append(vis_img[...,::-1])
                        vis_img = np.concatenate(vis_imgs, axis=0)
                        cv2.imshow('default', vis_img)
                        key_stroke = cv2.pollKey()

                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break
                        
                        precise_wait(t_cycle_end)
                    env.end_episode()

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()