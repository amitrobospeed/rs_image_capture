from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from dorna2 import Dorna

DORNA_HOST = "192.168.1.24"
DORNA_PORT = 443

TARGET_CYCLES = 20
STATUS_POLL_S = 0.02
START_TIMEOUT_S = 20.0
FINISH_TIMEOUT_S = 180.0
MOTOR_SETTLE_S = 1.0
GENERATED_CMDS_PATH = Path(__file__).with_name("zipper_cycle_cmds.txt")

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

# One logical zipper cycle. The last B->A transition is created automatically when the
# next cycle starts or when we append the explicit terminal A for the final stop.
CYCLE_PATTERN = ["A", "B", "C", "D", "C", "B"]


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


def _make_lmove(name: str, *, terminal: bool = False) -> dict[str, Any]:
    move = dict(MOVE_DEFAULTS)
    move.update(WAYPOINTS[name])
    if terminal:
        move["cont"] = 0
        move["corner"] = 0
    return move


def _build_cycle_buffer(cycles: int) -> list[dict[str, Any]]:
    if cycles <= 0:
        return []

    moves: list[dict[str, Any]] = []
    for _cycle_idx in range(cycles):
        for waypoint_name in CYCLE_PATTERN:
            moves.append(_make_lmove(waypoint_name))

    # Explicitly return to A so the final cycle closes the contour cleanly before stop.
    moves.append(_make_lmove("A", terminal=True))
    return moves


def _write_cmds_file(moves: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(move, separators=(",", ":")) for move in moves) + "\n",
        encoding="utf-8",
    )


def run_cycle_script(robot: Dorna, *, cycles: int, cmds_path: Path = GENERATED_CMDS_PATH) -> None:
    moves = _build_cycle_buffer(cycles)
    if not moves:
        print("No cycles requested. Nothing to do.")
        return

    _write_cmds_file(moves, cmds_path)
    print(f"[Cycle] Wrote {len(moves)} moves to {cmds_path}")

    print("[Cycle] Enabling motors")
    robot.play(-1, {"cmd": "motor", "motor": 1})
    time.sleep(MOTOR_SETTLE_S)

    print("[Cycle] Waiting for idle before play_script")
    if not _wait_until_idle(robot, START_TIMEOUT_S):
        raise TimeoutError("Robot did not report idle before play_script start")

    t0 = time.perf_counter()
    robot.play_script(str(cmds_path))
    dt = time.perf_counter() - t0
    print(f"[Cycle] play_script submitted in {dt:.4f}s for {cycles} cycles")

    print("[Cycle] Waiting for robot to finish generated command script")
    if not _wait_until_idle(robot, FINISH_TIMEOUT_S):
        raise TimeoutError("Timed out waiting for zipper cycle script to finish")

    print(f"[Cycle] Done. Completed {cycles} requested cycles")


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
