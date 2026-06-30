import time
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import trossen_arm
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class LeaderArm(mp.Process):
    """
    Continuously reads the leader arm's Cartesian pose (and gripper width) and
    publishes it to a shared memory ring buffer.

    The leader arm is set to external_effort mode with zero efforts (gravity
    compensation) so it can be moved freely.
    """

    def __init__(self,
            shm_manager: SharedMemoryManager,
            leader_ip: str,
            frequency: int = 100,
            get_max_k: int = 30,
            init_joints_pos=None,
            ):
        """
        leader_ip:  IP address of the leader arm controller
        frequency:  polling rate in Hz — how often leader state is read and published
        get_max_k:  maximum number of past readings the main loop can request at once
        """
        super().__init__(name="LeaderArm")

        self.leader_ip = leader_ip
        self.frequency = frequency
        if init_joints_pos is not None:
            init_joints_pos = np.array(init_joints_pos)
            assert init_joints_pos.shape == (7,)
        self.init_joints_pos = init_joints_pos

        example = {
            # Cartesian EEF pose of the leader: [x, y, z, rx, ry, rz]
            # in meters and radians (angle-axis), same convention as UR5 ActualTCPPose
            'LeaderTCPPose': np.zeros((6,), dtype=np.float64),
            # leader gripper width in meters (same convention as the follower / WSG)
            'LeaderGripperPos': 0.0,
            'receive_timestamp': time.time(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ======= get state APIs ==========

    def get_state(self):
        """
        Return the latest leader arm snapshot (single most recent ring buffer entry).

        Returns dict with keys: LeaderTCPPose (6,), LeaderGripperPos, receive_timestamp.
        """
        return self.ring_buffer.get()

    # ========== start / stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========

    def run(self):
        driver = trossen_arm.TrossenArmDriver()
        driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            self.leader_ip,
            False
        )

        try:
            # move to init position
            if self.init_joints_pos is not None:
                print(f"[LeaderArm] Moving to Init position")
                driver.set_all_modes(trossen_arm.Mode.position)
                driver.set_all_positions(self.init_joints_pos, 2.0, True)

            # gravity compensation
            driver.set_all_modes(trossen_arm.Mode.external_effort)
            driver.set_all_external_efforts(
                np.zeros(driver.get_num_joints()),
                0.0,
                False
            )

            # publish one reading immediately so the main loop can start reading
            # without waiting for the first sleep cycle (mirrors Spacemouse.run())
            self.ring_buffer.put({
                'LeaderTCPPose':     np.array(driver.get_cartesian_positions()),
                # Multiply by 2: driver returns one-side stroke; zarr/ring buffer
                # convention is total-width.
                'LeaderGripperPos':  driver.get_gripper_position() * 2.0,
                'receive_timestamp': time.time(),
            })
            self.ready_event.set()

            dt = 1.0 / self.frequency
            while not self.stop_event.is_set():
                t_start = time.perf_counter()

                # overwrite with latest hardware reading (not accumulate — same as SpaceMouse)
                self.ring_buffer.put({
                    'LeaderTCPPose':     np.array(driver.get_cartesian_positions()),
                    # Multiply by 2: driver returns one-side stroke; zarr/ring buffer
                    # convention is total-width.
                    'LeaderGripperPos':  driver.get_gripper_position() * 2.0,
                    'receive_timestamp': time.time(),
                })

                elapsed = time.perf_counter() - t_start
                time.sleep(max(0, dt - elapsed))

        finally:
            try:
                driver.set_all_modes(trossen_arm.Mode.position)
                driver.set_all_positions(
                    np.zeros(driver.get_num_joints()),
                    2.0,
                    True
                )
                driver.set_all_modes(trossen_arm.Mode.idle)
            except Exception:
                pass
            driver.cleanup()

            self.ready_event.set()
