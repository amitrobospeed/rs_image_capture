from __future__ import annotations

import time
from typing import Any

from dorna2 import Dorna

DORNA_HOST = "192.168.1.24"
DORNA_PORT = 443
MOTOR_SETTLE_S = 1.0
STATUS_POLL_S = 0.02
FINISH_TIMEOUT_S = 60.0

MOVE_DEFAULTS = {
    "cmd": "lmove",
    "rel": 0,
    "vel": 100,
    "acc": 800,
    "jerk": 1000,
    "cont": 1,
    "corner": 100,
}

WAYPOINTS = {
    "A": {"x": 386.777475, "y": -104.275356, "z": 169.734268, "a": 176.101107, "b": -33.250912, "c": 6.131734},
    "B": {"x": 266.843852, "y": -104.206306, "z": 169.708871, "a": 176.111387, "b": -33.251148, "c": 6.142103},
    "C": {"x": 266.711805, "y": 175.773619, "z": 169.707272, "a": 176.107832, "b": -33.244479, "c": 6.112397},
    "D": {"x": 386.783937, "y": 175.752657, "z": 169.835723, "a": 176.116134, "b": -33.260301, "c": 6.145552},
}

TEST_PATH = ["A", "B", "C", "D"]


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


def _build_move(name: str, *, terminal: bool = False) -> dict[str, Any]:
    move = dict(MOVE_DEFAULTS)
    move.update(WAYPOINTS[name])
    if terminal:
        move["cont"] = 0
        move["corner"] = 0
    return move


def run_blending_test(robot: Dorna) -> None:
    print("[Simple] Enabling motors")
    robot.play(-1, {"cmd": "motor", "motor": 1})
    time.sleep(MOTOR_SETTLE_S)

    print("[Simple] Waiting for idle before sending test path")
    if not _wait_until_idle(robot, timeout_s=10.0):
        raise TimeoutError("Robot did not become idle before test start")

    print("[Simple] Sending minimal contour A -> B -> C -> D")
    for index, name in enumerate(TEST_PATH):
        terminal = index == len(TEST_PATH) - 1
        move = _build_move(name, terminal=terminal)
        t0 = time.perf_counter()
        robot.play(-1, move)
        dt = time.perf_counter() - t0
        print(f"[Simple] Submitted {name} in {dt:.4f}s terminal={terminal} move={move}")

    print("[Simple] All moves submitted; waiting for final idle")
    if not _wait_until_idle(robot, timeout_s=FINISH_TIMEOUT_S):
        raise TimeoutError("Robot did not finish simple blending test before timeout")

    print("[Simple] Test complete")


def main() -> None:
    robot = Dorna()
    try:
        print(f"[Simple] Connecting to {DORNA_HOST}:{DORNA_PORT}")
        robot.connect(host=DORNA_HOST, port=DORNA_PORT)
        run_blending_test(robot)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
