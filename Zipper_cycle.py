from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from dorna2 import Dorna

DORNA_HOST = "192.168.1.24"
DORNA_PORT = 443
TARGET_CYCLES = 20
STATUS_POLL_S = 0.02
START_TIMEOUT_S = 20.0
CYCLE_TIMEOUT_S = 60.0
MOTOR_SETTLE_S = 1.0
CMDS_PATH = Path(__file__).with_name("cmds.txt")


def _extract_stat(resp: Any) -> int | None:
    if isinstance(resp, dict):
        if "stat" in resp:
            try:
                return int(resp["stat"])
            except (TypeError, ValueError):
                return None
        union = resp.get("union")
        if isinstance(union, dict) and "stat" in union:
            try:
                return int(union["stat"])
            except (TypeError, ValueError):
                return None
        msgs = resp.get("msgs")
        if isinstance(msgs, list):
            for msg in msgs:
                if isinstance(msg, dict) and "stat" in msg:
                    try:
                        return int(msg["stat"])
                    except (TypeError, ValueError):
                        return None
    return None


def _robot_idle(robot: Dorna) -> bool:
    try:
        return _extract_stat(robot.play(-1, {"cmd": "stat"})) == -1
    except Exception:
        return False


def _wait_until_idle(robot: Dorna, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _robot_idle(robot):
            return True
        time.sleep(STATUS_POLL_S)
    return False

def _build_cycle_buffer(cycles: int) -> list[dict[str, Any]]:
    if cycles <= 0:
        return []

def run_cycle_script(robot: Dorna, *, cycles: int, cmds_path: Path = CMDS_PATH) -> None:
    if cycles <= 0:
        print("No cycles requested. Nothing to do.")
        return
    if not cmds_path.exists():
        raise FileNotFoundError(f"Command script not found: {cmds_path}")

    print("[Cycle] Enabling motors")
    robot.play(-1, {"cmd": "motor", "motor": 1})
    time.sleep(MOTOR_SETTLE_S)

    print(f"[Cycle] Waiting for idle before first play_script: {cmds_path}")
    if not _wait_until_idle(robot, START_TIMEOUT_S):
        raise TimeoutError("Robot did not report idle before cycle start")

    for cycle_index in range(1, cycles + 1):
        t0 = time.perf_counter()
        robot.play_script(str(cmds_path))
        submit_dt = time.perf_counter() - t0
        print(f"[Cycle] Submitted cycle {cycle_index}/{cycles} in {submit_dt:.4f}s")

        if not _wait_until_idle(robot, CYCLE_TIMEOUT_S):
            raise TimeoutError(f"Timed out waiting for cycle {cycle_index} to finish")

        print(f"[Cycle] Completed cycle {cycle_index}/{cycles}")

    print(f"[Cycle] Done. Completed {cycles} requested cycles using {cmds_path.name}")


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
