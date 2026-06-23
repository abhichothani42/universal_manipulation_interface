import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import trossen_arm
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait

class Command(enum.Enum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1  # Cartesian pose target, sent by env.exec_actions() at eval time
    SCHEDULE_GRIPPER = 2   # gripper width target (metres), sent by env.exec_actions()
class TrossenArmController(mp.Process):
    def __init__(self, 
                 shm_manager: SharedMemoryManager, 
                 follower_ip, 
                 frequency=125,
                 max_pos_speed=0.25,
                 max_rot_speed=0.6,
                 launch_timeout=3,
                 init_joints_pos=None,
                 soft_real_time=False,
                 verbose=False,
                 receive_keys=None,
                 receive_latency=0.0,
                 gripper_max_width=0.04,
                 get_max_k=None):
        # verify
        assert 0 < frequency <= 500
        if init_joints_pos is not None:
            init_joints_pos = np.array(init_joints_pos)
            assert init_joints_pos.shape == (7,)

        super().__init__(name="TrossenArmController")
        self.follower_ip = follower_ip
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.init_joints_pos=init_joints_pos
        self.soft_real_time = soft_real_time
        # one-way robot->PC latency used to back-date obs timestamps
        self.receive_latency = receive_latency
        # physical gripper stroke (meters); policy/leader widths are clamped to this
        self.gripper_max_width = gripper_max_width
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        # Carries commands from main process → controller process.
        # STOP: shut down cleanly.
        # SCHEDULE_WAYPOINT: a Cartesian target pose + wall-clock deadline, sent by
        #   env.exec_actions() at eval time (same as RTDE's schedule_waypoint).
        #   During data collection this queue is never used — the follower mirrors
        #   the interpolator which is fed by leader poses read inside run().
        example = {
            'cmd':            Command.STOP.value,
            'target_pose':    np.zeros((6,), dtype=np.float64),
            'target_gripper': 0.0,
            'target_time':    0.0,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        # Publishes follower + leader state for the main process (real_env.get_obs()).
        # Key names match RTDE convention so real_env.py's DEFAULT_OBS_KEY_MAP works unchanged:
        #   ActualTCPPose  → robot_eef_pose   (follower actual EEF)
        #   ActualQ        → robot_joint       (follower joint positions)
        #   ActualQd       → robot_joint_vel   (follower joint velocities)
        # receive_keys is kept for future use but not needed for building the example.

        example = {
            # follower state — observations
            'ActualTCPPose':  np.zeros((6,), dtype=np.float64),  # follower EEF [x,y,z,rx,ry,rz]
            'ActualTCPSpeed': np.zeros((6,), dtype=np.float64),  # follower EEF velocity
            'ActualQ':        np.zeros((7,), dtype=np.float64),  # follower joint positions
            'ActualQd':       np.zeros((7,), dtype=np.float64),  # follower joint velocities
            'TargetTCPPose':  np.zeros((6,), dtype=np.float64),  # last interpolated pose command
            # gripper state — observations (width in meters)
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time(),
            # raw + latency-compensated timestamps used by the env to align robot obs
            'robot_receive_timestamp': time.time(),
            'robot_timestamp': time.time(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[TrossenArmController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pose, target_time):
        """Called by env.exec_actions() at eval time to send policy-predicted targets."""
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def schedule_gripper(self, pos, target_time):
        """Schedule a gripper width target (meters). Driven by env.exec_actions() via
        the TrossenGripperController adapter and by leader teleop."""
        message = {
            'cmd': Command.SCHEDULE_GRIPPER.value,
            'target_gripper': float(pos),
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        # a call to os setting priority to 20 and RR scheduling
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        # Configure the drivers
        follower_driver = trossen_arm.TrossenArmDriver()
        follower_driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_follower,
            self.follower_ip,
            True
        )

        try:
            if self.verbose:
                print(f"[TrossenArmController] Connect to follower: {self.follower_ip}")
            
            # move to init position
            if self.init_joints_pos is not None:
                print(f"[TrossenArmController] Moving to Init position")
                follower_driver.set_all_modes(trossen_arm.Mode.position)
                follower_driver.set_all_positions(self.init_joints_pos, 2.0, True)

            follower_driver.set_all_modes(trossen_arm.Mode.position)

            # main loop
            dt = 1. / self.frequency
            curr_pose = follower_driver.get_cartesian_positions()
            curr_gripper = follower_driver.get_gripper_position()
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            last_gripper_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            # gripper width (meters) is interpolated as the first element of a 6d pose,
            gripper_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_gripper, 0, 0, 0, 0, 0]]
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                # t_now: monotonic clock, used by PoseTrajectoryInterpolator (must match curr_t)
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                pose_command = pose_interp(t_now)
                follower_driver.set_cartesian_positions(pose_command, 
                                                        trossen_arm.InterpolationSpace.cartesian,
                                                        dt,
                                                        False)

                # command gripper to the interpolated width
                gripper_command = float(np.clip(
                    gripper_interp(t_now)[0], 0.0, self.gripper_max_width))
                follower_driver.set_gripper_position(
                    gripper_command, dt, False)

                # update robot state
                # Read follower actual state + leader state and publish to ring buffer.
                # The main process reads this via get_all_state() inside real_env.get_obs().
                t_recv = time.time()
                state = {
                    'ActualTCPPose':  np.array(follower_driver.get_cartesian_positions()),
                    'ActualTCPSpeed': np.array(follower_driver.get_cartesian_velocities()),
                    'ActualQ':        np.array(follower_driver.get_all_positions()),
                    'ActualQd':       np.array(follower_driver.get_all_velocities()),
                    'TargetTCPPose':  np.array(pose_command),  # last pose sent to follower
                    'gripper_position': follower_driver.get_gripper_position(),
                    'gripper_velocity': follower_driver.get_gripper_velocity(),
                    'gripper_receive_timestamp': t_recv,
                    'gripper_timestamp': t_recv - self.receive_latency,
                    'robot_receive_timestamp': t_recv,
                    'robot_timestamp': t_recv - self.receive_latency,
                }
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.SCHEDULE_GRIPPER.value:
                        target_gripper = float(command['target_gripper'])
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        gripper_interp = gripper_interp.schedule_waypoint(
                            pose=[target_gripper, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_gripper_waypoint_time
                        )
                        last_gripper_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[TrossenArmController] Actual frequency {1/(time.monotonic() - t_now):.1f} Hz")
        finally:
            try:
                follower_driver.set_all_modes(trossen_arm.Mode.position)
                follower_driver.set_all_positions(
                    np.zeros(follower_driver.get_num_joints()),
                    2.0,
                    True
                )
                follower_driver.set_all_modes(trossen_arm.Mode.idle)
            except Exception:
                pass
            follower_driver.cleanup()

            self.ready_event.set()

            if self.verbose:
                print(f"[TrossenArmController] Disconnected from follower: {self.follower_ip}")


class TrossenGripperController:
    """
    Lightweight adapter that exposes the WSGController-style gripper interface
    expected by BimanualUmiEnv, backed by a TrossenArmController.

    The Trossen end-effector is the 7th joint of the same arm, so a single arm
    process owns both the arm and the gripper. This adapter therefore does not
    spawn its own process: it forwards gripper commands to the arm's input queue
    and reads gripper state from the arm's ring buffer. Lifecycle methods are
    no-ops because the arm process is started/stopped by the env.
    """

    def __init__(self, arm: TrossenArmController):
        self.arm = arm

    # ========= launch method (owned by the arm process) ===========
    def start(self, wait=True):
        pass

    def stop(self, wait=True):
        pass

    def start_wait(self):
        pass

    def stop_wait(self):
        pass

    @property
    def is_ready(self):
        return self.arm.is_ready

    # ========= context manager ===========
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        self.arm.schedule_gripper(pos, target_time)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        return self.arm.get_state(k=k, out=out)

    def get_all_state(self):
        return self.arm.get_all_state()