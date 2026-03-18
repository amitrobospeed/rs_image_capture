from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from dorna2 import Dorna

DORNA_HOST = "192.168.1.24"
DORNA_PORT = 443

TARGET_CYCLES = 20
QUEUE_LOOKAHEAD = 8
QUEUE_POLL_S = 0.01
START_TIMEOUT_S = 20.0
FINISH_TIMEOUT_S = 120.0
MOTOR_SETTLE_S = 1.0

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


@dataclass
class CycleState:
    running: bool = False
    paused: bool = False
    stopped: bool = True
    cycle_count: int = 0
    segment_index: int = 0
    dispatched_segments: int = 0
    aligned_to_start: bool = False


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
        stat = _extract_stat(robot.play(-1, {"cmd": "stat"}))
    except Exception:
        return False
    return stat == -1


def _wait_until_idle(robot: Dorna, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _robot_idle(robot):
            return True
        time.sleep(QUEUE_POLL_S)
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


class ZipperCycleRunner:
    def __init__(self, robot: Dorna, cycles: int):
        self.robot = robot
        self.cycles = max(0, int(cycles))
        self.state = CycleState()
        self.sequence = _build_cycle_buffer(self.cycles)

    def _send_motor_on(self) -> None:
        self.robot.play(-1, {"cmd": "motor", "motor": 1})
        time.sleep(MOTOR_SETTLE_S)

    def _prime_queue(self) -> None:
        while self.state.segment_index < len(self.sequence) and self.state.dispatched_segments < QUEUE_LOOKAHEAD:
            self._dispatch_next()

    def _dispatch_next(self) -> None:
        move = self.sequence[self.state.segment_index]
        self.robot.play(-1, move)
        self.state.segment_index += 1
        self.state.dispatched_segments += 1

        if self.state.segment_index % len(CYCLE_PATTERN) == 0 and self.state.segment_index <= self.cycles * len(CYCLE_PATTERN):
            self.state.cycle_count += 1
            print(f"[Cycle] Dispatched cycle {self.state.cycle_count}/{self.cycles}")

    def _feed_remaining(self) -> None:
        # Keep feeding moves while the robot is still active. By staying ahead of the arm,
        # the controller can retain lookahead for cont/corner blending instead of stopping
        # at each waypoint.
        while self.state.segment_index < len(self.sequence):
            self._dispatch_next()
            time.sleep(QUEUE_POLL_S)

    def run(self) -> None:
        if not self.sequence:
            print("No cycles requested. Nothing to do.")
            return

        print("[Run] Enabling motors")
        self._send_motor_on()

        print("[Run] Waiting for idle before start")
        if not _wait_until_idle(self.robot, START_TIMEOUT_S):
            raise TimeoutError("Robot never reported idle before start")

        self.state.running = True
        self.state.paused = False
        self.state.stopped = False
        self.state.aligned_to_start = True

        print(f"[Run] Priming queue with up to {QUEUE_LOOKAHEAD} blended moves")
        self._prime_queue()

        print("[Run] Feeding remaining zipper moves")
        self._feed_remaining()

        print("[Run] Waiting for robot to finish final move")
        if not _wait_until_idle(self.robot, FINISH_TIMEOUT_S):
            raise TimeoutError("Timed out waiting for zipper cycle to finish")

        self.state.running = False
        self.state.stopped = True
        print(f"[Run] Done. Completed {self.state.cycle_count}/{self.cycles} cycles")


def main(robot: Dorna) -> None:
    runner = ZipperCycleRunner(robot=robot, cycles=TARGET_CYCLES)
    runner.run()


if __name__ == "__main__":
    robot = Dorna()
    try:
        robot.connect(host=DORNA_HOST, port=DORNA_PORT)
        time.sleep(1.0)
        main(robot)
    finally:
        robot.close()
