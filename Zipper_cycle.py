from __future__ import annotations

from pathlib import Path
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
DWELL_S = 0.5
REVERSE_CMDS_PATH = Path(__file__).with_name("cmds_reverse.txt")


def _run_named_script(robot: Dorna, script_path: Path, *, cycle_index: int, total_cycles: int, label: str) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Command script not found: {script_path}")

    t0 = time.perf_counter()
    robot.play_script(str(script_path))
    submit_dt = time.perf_counter() - t0
    print(f"[Cycle] Submitted {label} for cycle {cycle_index}/{total_cycles} in {submit_dt:.4f}s")

    if not _wait_until_idle(robot, timeout_s=FINISH_TIMEOUT_S):
        raise TimeoutError(f"Robot did not finish {label} for cycle {cycle_index} before timeout")

    print(f"[Cycle] Completed {label} for cycle {cycle_index}/{total_cycles}")


def run_cycle_script(robot: Dorna, *, cycles: int) -> None:
    if cycles <= 0:
        print("No cycles requested. Nothing to do.")
        return
    if not CMDS_PATH.exists():
        raise FileNotFoundError(f"Forward command script not found: {CMDS_PATH}")
    if not REVERSE_CMDS_PATH.exists():
        raise FileNotFoundError(f"Reverse command script not found: {REVERSE_CMDS_PATH}")

    print("[Cycle] Enabling motors")
    robot.play(-1, {"cmd": "motor", "motor": 1})
    time.sleep(MOTOR_SETTLE_S)

    print(f"[Cycle] Waiting for idle before first play_script: {CMDS_PATH}")
    if not _wait_until_idle(robot, timeout_s=START_TIMEOUT_S):
        raise TimeoutError("Robot did not become idle before cycle start")

    for cycle_index in range(1, cycles + 1):
        _run_named_script(robot, CMDS_PATH, cycle_index=cycle_index, total_cycles=cycles, label="forward path")
        print(f"[Cycle] Dwelling at D for {DWELL_S:.1f}s")
        time.sleep(DWELL_S)

        _run_named_script(robot, REVERSE_CMDS_PATH, cycle_index=cycle_index, total_cycles=cycles, label="reverse path")
        print(f"[Cycle] Dwelling at A for {DWELL_S:.1f}s")
        time.sleep(DWELL_S)

        print(f"[Cycle] Completed full cycle {cycle_index}/{cycles}")

    print(f"[Cycle] Done. Completed {cycles} requested cycles using {CMDS_PATH.name} + {REVERSE_CMDS_PATH.name}")


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
