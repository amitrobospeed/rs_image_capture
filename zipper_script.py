from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from dorna2 import Dorna

DORNA_HOST = "192.168.1.24"
DORNA_PORT = 443
MOTOR_SETTLE_S = 1.0
STATUS_POLL_S = 0.02
START_TIMEOUT_S = 10.0
FINISH_TIMEOUT_S = 60.0
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


def run_script_test(robot: Dorna) -> None:
    if not CMDS_PATH.exists():
        raise FileNotFoundError(f"Command script not found: {CMDS_PATH}")

    print("[Script] Enabling motors")
    robot.play(-1, {"cmd": "motor", "motor": 1})
    time.sleep(MOTOR_SETTLE_S)

    print(f"[Script] Waiting for idle before play_script: {CMDS_PATH}")
    if not _wait_until_idle(robot, timeout_s=START_TIMEOUT_S):
        raise TimeoutError("Robot did not become idle before play_script test")

    t0 = time.perf_counter()
    robot.play_script(str(CMDS_PATH))
    dt = time.perf_counter() - t0
    print(f"[Script] play_script submitted in {dt:.4f}s")

    print("[Script] Waiting for final idle after cmds.txt")
    if not _wait_until_idle(robot, timeout_s=FINISH_TIMEOUT_S):
        raise TimeoutError("Robot did not finish cmds.txt before timeout")

    print("[Script] Test complete")


def main() -> None:
    robot = Dorna()
    try:
        print(f"[Script] Connecting to {DORNA_HOST}:{DORNA_PORT}")
        robot.connect(host=DORNA_HOST, port=DORNA_PORT)
        run_script_test(robot)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
