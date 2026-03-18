from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from dorna2 import Dorna

from zipper_script import (
    CMDS_PATH,
    DORNA_HOST,
    DORNA_PORT,
    FINISH_TIMEOUT_S,
    MOTOR_SETTLE_S,
    START_TIMEOUT_S,
    STATUS_POLL_S,
    _extract_stat,
)

TARGET_CYCLES = 20
DWELL_S = 0.5
REVERSE_CMDS_PATH = Path(__file__).with_name("cmds_reverse.txt")
LOOP_FORWARD_CMDS_PATH = Path(__file__).with_name("cmds_cycle_forward.txt")
PROBLEM_KEYWORDS = ("alarm", "error", "err", "fault", "protect", "emergency")
MESSAGE_KEYS = ("msg", "message", "detail", "reason")


def _value_is_problem(value: Any) -> bool:
    if value in (None, False, 0, 0.0, "", "0"):
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered not in {"0", "false", "none", "null", "ok", "idle", "ready", "no_alarm", "no alarm"}
    return bool(value)


def _find_problem(resp: Any, path: str = "root") -> str | None:
    if isinstance(resp, dict):
        for key, value in resp.items():
            key_lower = str(key).lower()
            child_path = f"{path}.{key}"
            if any(token in key_lower for token in PROBLEM_KEYWORDS) and _value_is_problem(value):
                return f"{child_path}={value!r}"
            if key_lower in MESSAGE_KEYS and isinstance(value, str):
                lowered = value.lower()
                if any(token in lowered for token in PROBLEM_KEYWORDS):
                    return f"{child_path}={value!r}"
            nested_problem = _find_problem(value, child_path)
            if nested_problem:
                return nested_problem
    elif isinstance(resp, list):
        for idx, item in enumerate(resp):
            nested_problem = _find_problem(item, f"{path}[{idx}]")
            if nested_problem:
                return nested_problem
    return None


def _query_robot_status(robot: Dorna) -> tuple[int | None, str | None, Any]:
    resp = robot.play(-1, {"cmd": "stat"})
    stat = _extract_stat(resp)
    problem = _find_problem(resp)
    return stat, problem, resp


def _wait_until_healthy_idle(robot: Dorna, timeout_s: float, *, label: str) -> None:
    deadline = time.time() + timeout_s
    last_stat = None
    while time.time() < deadline:
        try:
            stat, problem, resp = _query_robot_status(robot)
        except Exception as exc:
            raise RuntimeError(f"{label}: failed to query robot status: {exc}") from exc

        if problem:
            raise RuntimeError(f"{label}: robot reported problem {problem}; raw={resp!r}")
        if stat == -1:
            return

        last_stat = stat
        time.sleep(STATUS_POLL_S)

    raise TimeoutError(f"{label}: timed out waiting for healthy idle (last_stat={last_stat})")


def _run_named_script(robot: Dorna, script_path: Path, *, cycle_index: int, total_cycles: int, label: str) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Command script not found: {script_path}")

    _wait_until_healthy_idle(robot, START_TIMEOUT_S, label=f"Pre-check before {label} cycle {cycle_index}")

    t0 = time.perf_counter()
    robot.play_script(str(script_path))
    submit_dt = time.perf_counter() - t0
    print(f"[Cycle] Submitted {label} for cycle {cycle_index}/{total_cycles} in {submit_dt:.4f}s")

    _wait_until_healthy_idle(robot, FINISH_TIMEOUT_S, label=f"Waiting for {label} cycle {cycle_index}")
    print(f"[Cycle] Completed {label} for cycle {cycle_index}/{total_cycles}")


def _forward_script_for_cycle(cycle_index: int) -> tuple[Path, str]:
    if cycle_index == 1:
        return CMDS_PATH, "forward path (startup A→B→C→D)"
    return LOOP_FORWARD_CMDS_PATH, "forward path (loop B→C→D)"


def run_cycle_script(robot: Dorna, *, cycles: int) -> None:
    if cycles <= 0:
        print("No cycles requested. Nothing to do.")
        return
    if not CMDS_PATH.exists():
        raise FileNotFoundError(f"Forward command script not found: {CMDS_PATH}")
    if not LOOP_FORWARD_CMDS_PATH.exists():
        raise FileNotFoundError(f"Loop forward command script not found: {LOOP_FORWARD_CMDS_PATH}")
    if not REVERSE_CMDS_PATH.exists():
        raise FileNotFoundError(f"Reverse command script not found: {REVERSE_CMDS_PATH}")

    print("[Cycle] Enabling motors")
    robot.play(-1, {"cmd": "motor", "motor": 1})
    time.sleep(MOTOR_SETTLE_S)

    _wait_until_healthy_idle(robot, START_TIMEOUT_S, label="Pre-start readiness")
    print(
        f"[Cycle] Robot ready. Starting cycles with first-pass {CMDS_PATH.name}, "
        f"loop-pass {LOOP_FORWARD_CMDS_PATH.name}, reverse {REVERSE_CMDS_PATH.name}"
    )

    for cycle_index in range(1, cycles + 1):
        forward_script, forward_label = _forward_script_for_cycle(cycle_index)
        _run_named_script(robot, forward_script, cycle_index=cycle_index, total_cycles=cycles, label=forward_label)
        print(f"[Cycle] Dwelling at D for {DWELL_S:.1f}s")
        time.sleep(DWELL_S)
        _wait_until_healthy_idle(robot, START_TIMEOUT_S, label=f"Post-D dwell check cycle {cycle_index}")

        _run_named_script(robot, REVERSE_CMDS_PATH, cycle_index=cycle_index, total_cycles=cycles, label="reverse path (C→B→A)")
        print(f"[Cycle] Dwelling at A for {DWELL_S:.1f}s")
        time.sleep(DWELL_S)
        _wait_until_healthy_idle(robot, START_TIMEOUT_S, label=f"Post-A dwell check cycle {cycle_index}")

        print(f"[Cycle] Completed full cycle {cycle_index}/{cycles}")

    print(
        f"[Cycle] Done. Completed {cycles} requested cycles using "
        f"{CMDS_PATH.name}/{LOOP_FORWARD_CMDS_PATH.name} + {REVERSE_CMDS_PATH.name}"
    )


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
