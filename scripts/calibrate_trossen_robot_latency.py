"""
Measures robot_action_latency for a Trossen follower arm.

Usage:
    uv run python scripts/calibrate_trossen_robot_latency.py \
        --follower_ip 192.168.1.4 \
        --leader_ip   192.168.1.2

Move the leader arm freely for the full recording duration.
The script cross-correlates the commanded EEF trajectory against the
follower's actual EEF trajectory and prints the measured latency.
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager

from umi.real_world.trossen_arm_controller import TrossenArmController
from umi.real_world.leader_arm_shared_memory import LeaderArm
from umi.common.precise_sleep import precise_wait
from umi.common.latency_util import get_latency
from matplotlib import pyplot as plt


# %%
@click.command()
@click.option('--follower_ip', '-fi', default='192.168.1.4', show_default=True)
@click.option('--leader_ip',   '-li', default='192.168.1.2', show_default=True)
@click.option('--frequency',   '-f',  default=30,  type=float, show_default=True,
              help='Command loop rate (Hz). Keep ≤ follower controller rate.')
@click.option('--duration',    '-d',  default=20.0, type=float, show_default=True,
              help='Recording duration in seconds. Move the leader during this window.')
def main(follower_ip, leader_ip, frequency, duration):
    max_pos_speed = 0.5   # m/s  – speed limit passed to PoseTrajectoryInterpolator
    max_rot_speed = 1.2   # rad/s
    dt = 1.0 / frequency
    command_latency = dt / 2   # sample halfway through each cycle

    with SharedMemoryManager() as shm_manager:
        with TrossenArmController(
            shm_manager=shm_manager,
            follower_ip=follower_ip,
            frequency=125,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            receive_latency=0.0,   # measure raw, no pre-correction
            verbose=False
        ) as controller, LeaderArm(
            shm_manager=shm_manager,
            leader_ip=leader_ip,
            frequency=100,
        ) as leader:
            print('Both arms ready.')
            print(f'Recording for {duration:.0f} s — move the leader arm now.')

            # Seed target from follower's current pose so first command is safe.
            state = controller.get_state()
            target_pose = np.array(state['ActualTCPPose'])

            t_target = []
            x_target = []

            t_start = time.monotonic()
            iter_idx = 0
            while True:
                elapsed = time.monotonic() - t_start
                if elapsed >= duration:
                    break

                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample     = t_cycle_end - command_latency
                # target_time one dt ahead keeps the interpolator from receiving
                # a deadline that is already in the past.
                t_command_target = t_cycle_end + dt

                precise_wait(t_sample, time_func=time.monotonic)

                # Read leader pose (absolute EEF, same convention as follower ActualTCPPose)
                leader_state = leader.get_state()
                target_pose = np.array(leader_state['LeaderTCPPose'])

                # Wall-clock target time for schedule_waypoint
                t_command_wall = time.monotonic() - time.time() + time.time() + (dt)
                # Simpler: just use time.time() + dt directly
                t_command_wall = time.time() + dt

                controller.schedule_waypoint(target_pose, target_time=t_command_wall)

                t_target.append(t_command_wall)
                x_target.append(target_pose.copy())

                remaining = duration - elapsed
                if iter_idx % (frequency * 5) == 0:
                    print(f'  {remaining:.0f} s remaining…')

                precise_wait(t_cycle_end, time_func=time.monotonic)
                iter_idx += 1

            print('Recording done. Reading follower state history…')
            states = controller.get_all_state()

    # ── post-process ──────────────────────────────────────────────────────────
    t_target = np.array(t_target)
    x_target = np.array(x_target)          # (N, 6)
    t_actual = states['robot_receive_timestamp']
    x_actual = states['ActualTCPPose']     # (M, 6)

    dim_names = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    n_dims = 6
    latencies = []

    fig, axes = plt.subplots(n_dims, 3, figsize=(15, 15))

    for i in range(n_dims):
        latency, info = get_latency(
            x_target[..., i], t_target,
            x_actual[..., i], t_actual
        )
        latencies.append(latency)

        row = axes[i]
        row[0].plot(info['lags'], info['correlation'])
        row[0].set_title(f'{dim_names[i]} cross-correlation')
        row[0].set_xlabel('lag (s)')

        row[1].plot(t_target - t_target[0], x_target[..., i], label='target')
        row[1].plot(t_actual - t_actual[0], x_actual[..., i], label='actual')
        row[1].legend()
        row[1].set_title(f'{dim_names[i]} raw')

        t_s = info['t_samples'] - info['t_samples'][0]
        row[2].plot(t_s, info['x_target'], label='target')
        row[2].plot(t_s - latency, info['x_actual'], label='actual (shifted)')
        row[2].legend()
        row[2].set_title(f'{dim_names[i]} aligned  latency={latency:.4f}s')

    fig.tight_layout()

    latencies = np.array(latencies)
    print('\n── Results ─────────────────────────────────────────────')
    for name, lat in zip(dim_names, latencies):
        print(f'  {name:4s}: {lat:.4f} s ({lat*1000:.1f} ms)')
    print(f'  MEAN: {latencies.mean():.4f} s ({latencies.mean()*1000:.1f} ms)')
    print(f'  Set robot_action_latency = {latencies.mean():.4f}  in eval_trossen_config.yaml')
    print('────────────────────────────────────────────────────────')

    plt.show()


# %%
if __name__ == '__main__':
    main()
