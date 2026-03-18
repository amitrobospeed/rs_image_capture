from __future__ import annotations

import time

from dorna2 import Dorna

from zipper_script import (
    CMDS_PATH,
    DORNA_HOST,
    DORNA_PORT,
    FINISH_TIMEOUT_S,
    MOTOR_SETTLE_S,
    START_TIMEOUT_S,
    _wait_until_idle,
)

TARGET_CYCLES = 20


def run_cycle_script(robot: Dorna, *, cycles: int) -> None:
    if cycles <= 0:
        print("No cycles requested. Nothing to do.")
        return
    if not CMDS_PATH.exists():
        raise FileNotFoundError(f"Command script not found: {CMDS_PATH}")

    print("[Cycle] Enabling motors")
    robot.play(-1, {"cmd": "motor", "motor": 1})
    time.sleep(MOTOR_SETTLE_S)

    print(f"[Cycle] Waiting for idle before first play_script: {CMDS_PATH}")
    if not _wait_until_idle(robot, timeout_s=START_TIMEOUT_S):
        raise TimeoutError("Robot did not become idle before cycle start")

    for cycle_index in range(1, cycles + 1):
        t0 = time.perf_counter()
        robot.play_script(str(CMDS_PATH))
        submit_dt = time.perf_counter() - t0
        print(f"[Cycle] Submitted cycle {cycle_index}/{cycles} in {submit_dt:.4f}s")

        if not _wait_until_idle(robot, timeout_s=FINISH_TIMEOUT_S):
            raise TimeoutError(f"Robot did not finish cycle {cycle_index} before timeout")

        print(f"[Cycle] Completed cycle {cycle_index}/{cycles}")

    print(f"[Cycle] Done. Completed {cycles} requested cycles using {CMDS_PATH.name}")


def main(robot: Dorna) -> None:
    run_cycle_script(robot, cycles=TARGET_CYCLES)


if __name__ == "__main__":
    robot = Dorna()
    try:
        robot.connect(host=DORNA_HOST, port=DORNA_PORT)
        time.sleep(1.0)
        main(robot)
    finally:
        robot.close()
