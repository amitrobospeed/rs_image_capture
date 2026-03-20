"""
robospeed_controller.py  –  Hardware / Backend Controller
RoboSpeed Durability Intelligence Platform  v2.8

This module is the bridge between the physical robot hardware (or a hardware
abstraction layer) and the PyQt6 GUI defined in robospeed_gui_main.py.

─────────────────────────────────────────────────────────────────────
ARCHITECTURE
─────────────────────────────────────────────────────────────────────
  robospeed_controller.py              robospeed_gui_main.py
  ┌──────────────────────┐             ┌────────────────────────────┐
  │  RoboSpeedController │◄──────────► │  MainWindow (GUI)          │
  │                      │  Qt signals │                            │
  │  MotionDriver        │             │  LeftPanel                 │
  │  ForceDAQ            │             │  RightPanel                │
  │  VisionDriver        │             │   ├─ DefectModule (cosmetic)│
  │  DataLogger          │             │   ├─ DefectModule (led)     │
  └──────────────────────┘             │   ├─ DefectModule (geometry)│
                                       │   ├─ Visual Controls        │
                                       │   └─ AI Analyst             │
                                       │  ForceGraph  (rolling win.) │
                                       │  VisionPanel               │
                                       └────────────────────────────┘

HOW TO RUN
─────────────────────────────────────────────────────────────────────
  # Simulation / demo (no hardware):
  python robospeed_controller.py --sim

  # With real hardware:
  python robospeed_controller.py --no-sim --motion-port COM3 --daq-port COM4

  The controller stops the GUI's built-in MockDataThread and replaces it
  with real hardware data from MotionDriver + ForceDAQ + VisionDriver.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys
import os
import time
import logging
import argparse
import threading
from dataclasses import dataclass
from typing import Optional

# Qt
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

# GUI module (must be in the same directory or on PYTHONPATH)
import robospeed_gui_main as gui

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RSController")

BUTTON_ORDER = ("A", "B", "C", "D")
PHASES = ("above", "press", "retract")
PEAK_START_THRESHOLD = 0.5
DEVIATION_THRESHOLD = 0.50

BUTTON_POSES = {
    "A": {
        "above":   {"x": 318.06, "y": -38.16, "z": 127.44, "a": -173.0,  "b": 41.62, "c": -3.53},
        "press":   {"x": 318.06, "y": -38.16, "z": 120.40, "a": -173.0,  "b": 41.62, "c": -3.53},
        "retract": {"x": 318.06, "y": -38.16, "z": 127.44, "a": -173.0,  "b": 41.62, "c": -3.53},
    },
    "B": {
        "above":   {"x": 327.02, "y": -21.69, "z": 127.44, "a": -173.8,  "b": 32.25, "c": -4.81},
        "press":   {"x": 327.02, "y": -21.69, "z": 118.70, "a": -173.8,  "b": 32.25, "c": -4.81},
        "retract": {"x": 327.02, "y": -21.69, "z": 127.44, "a": -173.8,  "b": 32.25, "c": -4.81},
    },
    "C": {
        "above":   {"x": 342.30, "y": -29.64, "z": 127.44, "a": -174.83, "b": 34.08, "c": -6.71},
        "press":   {"x": 342.30, "y": -29.64, "z": 120.38, "a": -174.83, "b": 34.08, "c": -6.71},
        "retract": {"x": 342.30, "y": -29.64, "z": 127.44, "a": -174.83, "b": 34.08, "c": -6.71},
    },
    "D": {
        "above":   {"x": 335.08, "y": -43.88, "z": 125.13, "a": -173.84, "b": 43.25, "c": -4.86},
        "press":   {"x": 335.08, "y": -43.88, "z": 121.70, "a": -173.84, "b": 43.25, "c": -4.86},
        "retract": {"x": 335.08, "y": -43.88, "z": 127.44, "a": -173.84, "b": 43.25, "c": -4.86},
    },
}


# ═══════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════
@dataclass
class MotionParams:
    velocity       : int   = 300
    acceleration   : int   = 300
    jerk           : int   = 1000
    target_cycles  : int   = 100
    baseline_cycles: int   = 30
    force_min_lbs  : float = 0.5
    force_max_lbs  : float = 1.8


@dataclass
class CycleResult:
    cycle_num      : int
    timestamp      : float
    peak_force_lbs : float
    force_ok       : bool
    retract_ok     : bool
    button         : str = ""
    anomaly_type   : str = "normal"


@dataclass
class PeakEvent:
    cycle_num      : int
    button         : str
    phase          : str
    timestamp      : float
    cycle_x        : int
    peak_force_lbs : float
    missed         : bool
    anomaly_type   : str
    force_ok       : bool
    retract_ok     : bool
    baseline_count : int
    baseline_ready : bool
    baseline_mean  : Optional[float]
    message        : str


@dataclass
class CycleSummary:
    cycle_num      : int
    timestamp      : float
    baseline_count : int
    baseline_ready : bool
    baseline_means : dict[str, float]
    status_message : str


@dataclass
class InspectionCapture:
    cycle_num  : int
    camera     : str
    frame_type : str
    data       : Optional[bytes] = None


@dataclass
class ControllerConfig:
    motion_port     : str  = "COM3"
    daq_port        : str  = "COM4"
    camera_c1_id    : int  = 0
    camera_c2_id    : int  = 1
    sim_mode        : bool = True


# ═══════════════════════════════════════════════════════════════════
# HARDWARE DRIVERS
# ═══════════════════════════════════════════════════════════════════
class MotionDriver:
    def __init__(self, config: ControllerConfig):
        self._cfg = config
        self._connected = False
        self._params = MotionParams()
        self._sim_phase_seconds = {"above": 0.12, "press": 0.18, "retract": 0.14}

    def connect(self) -> bool:
        if self._cfg.sim_mode:
            log.info("MotionDriver: simulation mode — no serial connection")
            self._connected = True
            return True
        try:
            import serial
            self._ser = serial.Serial(self._cfg.motion_port, 115200, timeout=1)
            self._connected = True
            log.info(f"MotionDriver: connected on {self._cfg.motion_port}")
            return True
        except Exception as e:
            log.error(f"MotionDriver.connect failed: {e}")
            return False

    def disconnect(self):
        if not self._cfg.sim_mode and hasattr(self, "_ser"):
            self._ser.close()
        self._connected = False

    def home(self) -> bool:
        log.info("MotionDriver: HOME command")
        return True

    def set_params(self, params: MotionParams):
        self._params = params
        log.info(
            "MotionDriver: set_params vel=%s acc=%s jerk=%s",
            params.velocity,
            params.acceleration,
            params.jerk,
        )

    def stop(self):
        log.info("MotionDriver: STOP command")

    def record_trajectory(self):
        log.info("MotionDriver: RECORD TRAJECTORY command")

    def phase_duration(self, phase: str) -> float:
        return self._sim_phase_seconds.get(phase, 0.12)

    def move_to_pose(self, button: str, phase: str) -> bool:
        pose = BUTTON_POSES.get(button, {}).get(phase)
        if pose is None:
            log.error(f"MotionDriver: unknown pose {button}-{phase}")
            return False
        if self._cfg.sim_mode:
            time.sleep(self.phase_duration(phase))
            return True
        try:
            cmd = dict(
                cmd="jmove",
                rel=0,
                vel=self._params.velocity,
                acc=self._params.acceleration,
                jerk=self._params.jerk,
                **pose,
            )
            log.info(f"MotionDriver: move {button}-{phase} -> {cmd}")
            # TODO: replace log-only path with real controller protocol and completion poll.
            return True
        except Exception as e:
            log.error(f"MotionDriver.move_to_pose failed for {button}-{phase}: {e}")
            return False

    def safe_align(self) -> bool:
        log.info("MotionDriver: SAFE ALIGN A-above")
        return self.move_to_pose("A", "above")


class ForceDAQ:
    def __init__(self, config: ControllerConfig):
        self._cfg = config
        self._connected = False
        self._t0 = time.time()

    def connect(self) -> bool:
        if self._cfg.sim_mode:
            self._connected = True
            return True
        try:
            import serial
            self._ser = serial.Serial(self._cfg.daq_port, 115200, timeout=0.1)
            self._connected = True
            return True
        except Exception as e:
            log.error(f"ForceDAQ.connect failed: {e}")
            return False

    def disconnect(self):
        if not self._cfg.sim_mode and hasattr(self, "_ser"):
            self._ser.close()

    def read_lbs(self) -> float:
        if self._cfg.sim_mode:
            import math
            import random
            t = time.time() - self._t0
            return max(0.0, 1.1 * math.sin(2 * math.pi * 0.2 * t) + random.gauss(0, 0.015))
        return 0.0


class VisionDriver:
    def __init__(self, config: ControllerConfig):
        self._cfg = config
        self._caps: dict = {}

    def connect(self) -> bool:
        if self._cfg.sim_mode:
            return True
        try:
            import cv2
            for name, idx in (("C1", self._cfg.camera_c1_id), ("C2", self._cfg.camera_c2_id)):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    self._caps[name] = cap
                    log.info(f"VisionDriver: {name} opened (index {idx})")
                else:
                    log.warning(f"VisionDriver: {name} not found (index {idx})")
            return True
        except ImportError:
            log.warning("VisionDriver: OpenCV not installed — vision disabled")
            return False

    def capture(self, camera: str = "C1") -> Optional[bytes]:
        if self._cfg.sim_mode:
            return None
        import cv2
        cap = self._caps.get(camera)
        if cap is None:
            return None
        ret, frame = cap.read()
        if not ret:
            return None
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()

    def disconnect(self):
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass


class DataLogger:
    def __init__(self, project: str, profile: str):
        self._project = project
        self._profile = profile
        self._results: list[CycleResult] = []
        self._log_dir = os.path.join("logs", f"{project}_{profile}".replace(" ", "_"))
        os.makedirs(self._log_dir, exist_ok=True)
        log.info(f"DataLogger: writing to {self._log_dir}")

    def record_cycle(self, result: CycleResult):
        self._results.append(result)
        if not result.force_ok:
            log.warning(
                "Cycle %s (%s): %s %.3f lbs",
                result.cycle_num,
                result.button or "?",
                result.anomaly_type,
                result.peak_force_lbs,
            )

    def record_capture(self, capture: InspectionCapture):
        if capture.data:
            fname = f"{capture.frame_type}_cycle{capture.cycle_num:05d}_{capture.camera}.jpg"
            with open(os.path.join(self._log_dir, fname), "wb") as f:
                f.write(capture.data)

    def generate_report(self) -> str:
        import csv
        path = os.path.join(self._log_dir, "report.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cycle", "button", "timestamp", "peak_force_lbs", "force_ok", "retract_ok", "anomaly_type"])
            for r in self._results:
                w.writerow([
                    r.cycle_num,
                    r.button,
                    f"{r.timestamp:.3f}",
                    f"{r.peak_force_lbs:.4f}",
                    r.force_ok,
                    r.retract_ok,
                    r.anomaly_type,
                ])
        log.info(f"DataLogger: report saved → {path}")
        return path


# ═══════════════════════════════════════════════════════════════════
# TEST LOOP THREAD
# ═══════════════════════════════════════════════════════════════════
class TestLoopThread(QThread):
    sig_cycle_done = pyqtSignal(object)
    sig_force_live = pyqtSignal(float, float, int)
    sig_robot_event = pyqtSignal(object)
    sig_error = pyqtSignal(str)
    sig_finished = pyqtSignal()

    def __init__(
        self,
        motion: MotionDriver,
        daq: ForceDAQ,
        vision: VisionDriver,
        logger: DataLogger,
        params: MotionParams,
        vision_cfg: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._motion = motion
        self._daq = daq
        self._vision = vision
        self._logger = logger
        self._params = params
        self._vis_cfg = vision_cfg
        self._running = False
        self._paused = False
        self._stop_req = False
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._baseline_peaks = {btn: [] for btn in BUTTON_ORDER}
        self._baseline_mean: dict[str, float] = {}
        self._sim_base_force = {"A": 1.08, "B": 1.16, "C": 1.24, "D": 1.33}

    def request_stop(self):
        with self._lock:
            self._stop_req = True
            self._motion.stop()

    def request_pause(self):
        with self._lock:
            self._paused = True

    def request_resume(self):
        with self._lock:
            self._paused = False

    def _wait_if_paused(self) -> bool:
        while True:
            with self._lock:
                if self._stop_req:
                    return False
                if not self._paused:
                    return True
            time.sleep(0.05)

    def _next_position(self, button: str, phase: str) -> tuple[str, str]:
        btn_idx = BUTTON_ORDER.index(button)
        phase_idx = PHASES.index(phase)
        if phase_idx < len(PHASES) - 1:
            return button, PHASES[phase_idx + 1]
        if btn_idx < len(BUTTON_ORDER) - 1:
            return BUTTON_ORDER[btn_idx + 1], PHASES[0]
        return BUTTON_ORDER[0], PHASES[0]

    def _emit_phase_event(self, *, cycle: int, button: str, phase: str):
        next_button, next_phase = self._next_position(button, phase)
        self.sig_robot_event.emit(
            {
                "event_type": "phase_changed",
                "cycle_num": cycle,
                "button": button,
                "phase": phase,
                "next_button": next_button,
                "next_phase": next_phase,
                "message": f"{button}-{phase} in progress",
            }
        )

    def _plan_sim_anomaly(self, cycle: int) -> dict[str, str]:
        import random

        plan: dict[str, str] = {btn: "normal" for btn in BUTTON_ORDER}
        for btn in BUTTON_ORDER:
            roll = random.random()
            if roll < 0.03:
                plan[btn] = "missed_peak"
            elif cycle > self._params.baseline_cycles and roll < 0.08:
                plan[btn] = "baseline_deviation"
            elif roll < 0.12:
                plan[btn] = "force_out_of_range"
        return plan

    def _simulate_force_sample(self, *, button: str, phase: str, elapsed: float, duration: float, anomaly_hint: str) -> float:
        import random

        progress = min(1.0, elapsed / max(duration, 1e-6))
        peak = self._sim_base_force[button] + random.gauss(0, 0.015)
        if anomaly_hint == "force_out_of_range":
            peak += 0.9
        elif anomaly_hint == "baseline_deviation":
            peak += 0.65
        elif anomaly_hint == "missed_peak":
            peak = 0.18 + random.gauss(0, 0.01)

        if phase == "above":
            return max(0.0, random.gauss(0, 0.006))
        if phase == "press":
            return max(0.0, peak * (progress ** 2) + random.gauss(0, 0.01))
        if phase == "retract":
            return max(0.0, peak * max(0.0, 1.0 - progress ** 2) + random.gauss(0, 0.01))
        return 0.0

    def _read_force_sample(self, *, button: str, phase: str, elapsed: float, duration: float, anomaly_hint: str) -> float:
        if self._daq._cfg.sim_mode:
            return self._simulate_force_sample(
                button=button,
                phase=phase,
                elapsed=elapsed,
                duration=duration,
                anomaly_hint=anomaly_hint,
            )
        return self._daq.read_lbs()

    def _evaluate_button_peak(self, *, cycle: int, button: str, peak: float, peak_time: float) -> PeakEvent:
        missed = peak < PEAK_START_THRESHOLD
        retract_ok = not missed
        anomaly_type = "missed_peak" if missed else "normal"
        force_ok = self._params.force_min_lbs <= peak <= self._params.force_max_lbs

        if not missed and cycle <= self._params.baseline_cycles:
            self._baseline_peaks[button].append(float(peak))

        baseline_count = min(len(v) for v in self._baseline_peaks.values())
        baseline_ready = baseline_count >= self._params.baseline_cycles
        if baseline_ready and not self._baseline_mean:
            self._baseline_mean = {
                btn: float(sum(vals) / len(vals))
                for btn, vals in self._baseline_peaks.items()
                if vals
            }

        baseline_mean = self._baseline_mean.get(button)
        if not missed and baseline_ready and baseline_mean and baseline_mean > 0:
            deviation = abs(peak - baseline_mean) / baseline_mean
            if deviation > DEVIATION_THRESHOLD:
                anomaly_type = "baseline_deviation"

        if not force_ok and anomaly_type == "normal":
            anomaly_type = "force_out_of_range"

        if anomaly_type == "baseline_deviation":
            force_ok = False
            message = f"Baseline dev {button}: {peak:.2f} vs {baseline_mean:.2f}"
        elif anomaly_type == "force_out_of_range":
            force_ok = False
            message = f"Force out of range {button}: {peak:.2f} lbs"
        elif anomaly_type == "missed_peak":
            message = f"Missed peak on {button}"
        else:
            message = f"{button} peak {peak:.2f} lbs"

        return PeakEvent(
            cycle_num=cycle,
            button=button,
            phase="retract",
            timestamp=peak_time,
            cycle_x=cycle,
            peak_force_lbs=float(peak),
            missed=missed,
            anomaly_type=anomaly_type,
            force_ok=force_ok,
            retract_ok=retract_ok,
            baseline_count=baseline_count,
            baseline_ready=baseline_ready,
            baseline_mean=baseline_mean,
            message=message,
        )

    def run(self):
        self._running = True
        self._stop_req = False
        self._t0 = time.time()
        self._motion.set_params(self._params)
        force_sample_interval = 0.02
        cycle = 0

        if not self._motion.safe_align():
            self.sig_error.emit("Robot failed to align at A-above")
            self._running = False
            self.sig_finished.emit()
            return

        for cycle in range(1, self._params.target_cycles + 1):
            if not self._wait_if_paused():
                break

            anomaly_plan = self._plan_sim_anomaly(cycle)
            for button in BUTTON_ORDER:
                window_peak = 0.0
                window_peak_time = time.time() - self._t0

                for phase in PHASES:
                    if not self._wait_if_paused():
                        break
                    self._emit_phase_event(cycle=cycle, button=button, phase=phase)
                    if not self._motion.move_to_pose(button, phase):
                        self.sig_error.emit(f"Motion error at cycle {cycle} ({button}-{phase})")
                        self._running = False
                        self.sig_finished.emit()
                        return

                    duration = self._motion.phase_duration(phase)
                    phase_start = time.time()
                    while time.time() - phase_start < duration:
                        elapsed = time.time() - phase_start
                        force = self._read_force_sample(
                            button=button,
                            phase=phase,
                            elapsed=elapsed,
                            duration=duration,
                            anomaly_hint=anomaly_plan.get(button, "normal"),
                        )
                        t = time.time() - self._t0
                        self.sig_force_live.emit(t, force, cycle)
                        if phase in ("press", "retract") and force >= window_peak:
                            window_peak = force
                            window_peak_time = t
                        time.sleep(force_sample_interval)

                    if phase == "retract":
                        peak_event = self._evaluate_button_peak(
                            cycle=cycle,
                            button=button,
                            peak=window_peak,
                            peak_time=window_peak_time,
                        )
                        self._logger.record_cycle(
                            CycleResult(
                                cycle_num=cycle,
                                timestamp=peak_event.timestamp,
                                peak_force_lbs=peak_event.peak_force_lbs,
                                force_ok=peak_event.force_ok,
                                retract_ok=peak_event.retract_ok,
                                button=button,
                                anomaly_type=peak_event.anomaly_type,
                            )
                        )
                        self.sig_robot_event.emit(peak_event)
                else:
                    continue
                break

            baseline_count = min(len(v) for v in self._baseline_peaks.values())
            baseline_ready = baseline_count >= self._params.baseline_cycles
            self.sig_cycle_done.emit(
                CycleSummary(
                    cycle_num=cycle,
                    timestamp=time.time() - self._t0,
                    baseline_count=baseline_count,
                    baseline_ready=baseline_ready,
                    baseline_means=dict(self._baseline_mean),
                    status_message=f"Cycle {cycle}/{self._params.target_cycles} complete",
                )
            )

            surf_every = self._vis_cfg.get("surface_capture_every", 25)
            led_every = self._vis_cfg.get("led_capture_every", 25)
            cloud_every = self._vis_cfg.get("point_cloud_capture_every", 50)
            for every, ftype, cam in [
                (surf_every, "surface", "C1"),
                (led_every, "led", "C1"),
                (cloud_every, "point_cloud", "C2"),
            ]:
                if every > 0 and cycle % every == 0:
                    data = self._vision.capture(cam)
                    self._logger.record_capture(InspectionCapture(cycle, cam, ftype, data))

        self._running = False
        self.sig_finished.emit()
        log.info(f"TestLoopThread: finished after {cycle} cycles")


# ═══════════════════════════════════════════════════════════════════
# MAIN CONTROLLER
# ═══════════════════════════════════════════════════════════════════
class RoboSpeedController(QObject):
    def __init__(self, config: ControllerConfig):
        super().__init__()
        self._cfg = config
        self._motion = MotionDriver(config)
        self._daq = ForceDAQ(config)
        self._vision = VisionDriver(config)
        self._logger: Optional[DataLogger] = None
        self._loop: Optional[TestLoopThread] = None
        self._params = MotionParams()
        self._vis_cfg: dict = {}
        self._win: Optional[gui.MainWindow] = None

        if not self._motion.connect():
            log.warning("Motion controller not connected — running in degraded mode")
        if not self._daq.connect():
            log.warning("Force DAQ not connected — force readings will be zero")
        self._vision.connect()

    def attach(self, win: gui.MainWindow):
        self._win = win
        self._win._peak_events.clear()
        win.left.sig_start.connect(self._on_gui_start)
        win.left.sig_pause.connect(self._on_gui_pause)
        win.left.sig_stop.connect(self._on_gui_stop)
        win.left.sig_home.connect(self._on_gui_home)
        win.left.sig_reset.connect(self._on_gui_reset)
        win.left.sig_report.connect(self._on_gui_report)
        win.left.sig_exit.connect(self._on_gui_exit)
        win.left.sig_fields.connect(self._on_gui_fields)
        win.left.sig_record.connect(self._on_gui_record)
        win.right.sig_camera_changed.connect(self._on_camera_changed)
        win.right.sig_freq_updated.connect(self._on_freq_updated)
        self._reset_runtime_state()
        log.info("Controller attached to GUI")

    def _reset_runtime_state(self):
        with self._win._lock:
            self._win._state.update(
                force_out_of_range=dict(A=0, B=0, C=0, D=0),
                button_did_not_retract=dict(A=0, B=0, C=0, D=0),
                cycle_count=0,
                baseline_ready=False,
                baseline_count=0,
                baseline_mean={},
                baseline_peaks={btn: [] for btn in BUTTON_ORDER},
                current_button="—",
                current_phase="idle",
                next_button="A",
                next_phase="above",
                status_detail="Ready",
            )

    def _set_status(self, color: str, alert_msg: str, detail: str, duration: float):
        self._win._alert(color, alert_msg, duration)
        with self._win._lock:
            self._win._state["status_detail"] = detail
        self._win.statusBar().showMessage(detail, int(duration * 1000))

    def _on_gui_start(self):
        if self._loop and self._loop.isRunning():
            self._loop.request_resume()
            with self._win._lock:
                self._win._state.update(running=True, paused=False, stopped=False)
            self._set_status(gui.C["GREEN"], "Resumed", "Cycle test resumed", 2.0)
            log.info("Test resumed")
            return

        project = self._win.txtProject.text().strip() or "Project"
        profile = self._win.txtTestProfile.text().strip() or "Profile"
        self._logger = DataLogger(project, profile)
        self._vis_cfg = self._win.right.get_vision_settings()
        self._win._peak_events.clear()
        self._reset_runtime_state()
        self._win.right.mod_cosmetic.set_state("PROCESSING…")
        self._win.right.mod_led.set_state("PROCESSING…")

        self._loop = TestLoopThread(
            self._motion,
            self._daq,
            self._vision,
            self._logger,
            self._params,
            self._vis_cfg,
        )
        self._loop.sig_force_live.connect(self._win.force_graph.push)
        self._loop.sig_cycle_done.connect(self._on_cycle_done)
        self._loop.sig_robot_event.connect(self._on_robot_event)
        self._loop.sig_error.connect(self._on_loop_error)
        self._loop.sig_finished.connect(self._on_loop_finished)
        self._loop.start()

        with self._win._lock:
            self._win._state.update(
                running=True,
                paused=False,
                stopped=False,
                current_button="A",
                current_phase="above",
                next_button="A",
                next_phase="press",
                status_detail="Robot + force monitoring active",
            )
        self._set_status(gui.C["GREEN"], "Test started", "Robot + force monitoring active", 2.0)
        log.info("Test started")

    def _on_gui_pause(self):
        if self._loop:
            self._loop.request_pause()
        with self._win._lock:
            self._win._state.update(running=False, paused=True, status_detail="Paused at cycle boundary")
        self._set_status(gui.C["AMBER"], "Paused", "Paused at cycle boundary", 2.0)
        log.info("Test paused")

    def _on_gui_stop(self):
        if self._loop:
            self._loop.request_stop()
            self._loop.wait(3000)
            self._loop = None
        with self._win._lock:
            self._win._state.update(running=False, paused=False, stopped=True, status_detail="Stopped")
        self._set_status(gui.C["DOT_STOP"], "Stopped", "Robot stopped; ready", 2.0)
        log.info("Test stopped")

    def _on_gui_home(self):
        self._motion.home()
        self._set_status(gui.C["ACCENT"], "Homing robot…", "Robot returning home", 2.0)
        log.info("Homing")

    def _on_gui_reset(self):
        self._reset_runtime_state()
        self._win._peak_events.clear()
        self._set_status(gui.C["GREEN"], "Counters reset", "Baseline + per-button counters cleared", 2.0)
        log.info("Counters reset")

    def _on_gui_report(self):
        if self._logger:
            path = self._logger.generate_report()
            self._win.statusBar().showMessage(f"Report saved: {path}", 5000)
        else:
            self._win.statusBar().showMessage("No data to report yet", 3000)

    def _on_gui_exit(self):
        self._on_gui_stop()
        self._motion.disconnect()
        self._daq.disconnect()
        self._vision.disconnect()

    def _on_gui_fields(self, d: dict):
        self._params = MotionParams(
            velocity=d.get("vel", self._params.velocity),
            acceleration=d.get("acc", self._params.acceleration),
            jerk=d.get("jerk", self._params.jerk),
            target_cycles=d.get("target_cycles", self._params.target_cycles),
            baseline_cycles=d.get("baseline_cycles", self._params.baseline_cycles),
            force_min_lbs=d.get("force_min", self._params.force_min_lbs),
            force_max_lbs=d.get("force_max", self._params.force_max_lbs),
        )
        with self._win._lock:
            self._win._state.update(
                vel=self._params.velocity,
                acc=self._params.acceleration,
                jerk=self._params.jerk,
                target_cycles=self._params.target_cycles,
                baseline_cycles=self._params.baseline_cycles,
                force_min=self._params.force_min_lbs,
                force_max=self._params.force_max_lbs,
            )
        log.debug(f"Params updated: {self._params}")

    def _on_gui_record(self):
        self._motion.record_trajectory()

    def _on_camera_changed(self, label: str):
        log.info(f"Camera selection: {label}")

    def _on_freq_updated(self, msg: str):
        log.info(f"Inspection frequency changed: {msg}")

    def _on_cycle_done(self, summary_obj):
        if not isinstance(summary_obj, CycleSummary):
            return
        with self._win._lock:
            st = self._win._state
            st["cycle_count"] = summary_obj.cycle_num
            st["baseline_count"] = summary_obj.baseline_count
            st["baseline_ready"] = summary_obj.baseline_ready
            st["baseline_mean"] = dict(summary_obj.baseline_means)
            st["status_detail"] = summary_obj.status_message
        if summary_obj.baseline_ready:
            self._set_status(gui.C["GREEN"], "Baseline ready", summary_obj.status_message, 1.5)
        else:
            self._win.statusBar().showMessage(summary_obj.status_message, 2500)

    def _on_robot_event(self, event):
        if isinstance(event, PeakEvent):
            self._record_peak_event(event)
            return
        if not isinstance(event, dict) or event.get("event_type") != "phase_changed":
            return
        with self._win._lock:
            st = self._win._state
            st["current_button"] = event["button"]
            st["current_phase"] = event["phase"]
            st["next_button"] = event["next_button"]
            st["next_phase"] = event["next_phase"]
            st["status_detail"] = event["message"]
        self._win.statusBar().showMessage(event["message"], 1200)

    def _record_peak_event(self, event: PeakEvent):
        with self._win._lock:
            st = self._win._state
            st["baseline_count"] = event.baseline_count
            st["baseline_ready"] = event.baseline_ready
            if event.baseline_mean is not None:
                st["baseline_mean"][event.button] = event.baseline_mean
            st["status_detail"] = event.message
            if event.anomaly_type in ("force_out_of_range", "baseline_deviation"):
                st["force_out_of_range"][event.button] += 1
            if not event.retract_ok:
                st["button_did_not_retract"][event.button] += 1

        self._win._peak_events.append(
            {
                "t": event.timestamp,
                "cycle": event.cycle_x,
                "y": event.peak_force_lbs,
                "button": event.button,
                "missed": event.missed,
                "anomaly_type": event.anomaly_type,
            }
        )

        if event.anomaly_type == "missed_peak":
            self._set_status(gui.C["RED"], event.message, f"Cycle {event.cycle_num} · {event.button}", 3.0)
        elif event.anomaly_type == "baseline_deviation":
            self._set_status(gui.C["AMBER"], event.message, f"Cycle {event.cycle_num} · baseline deviation", 3.0)
        elif event.anomaly_type == "force_out_of_range":
            self._set_status(gui.C["AMBER"], event.message, f"Cycle {event.cycle_num} · out of range", 3.0)
        else:
            self._win.statusBar().showMessage(event.message, 1200)

    def _on_loop_error(self, msg: str):
        log.error(f"Test loop error: {msg}")
        self._set_status(gui.C["RED"], f"Error: {msg}", "Controller loop error", 5.0)
        self._win.right.mod_cosmetic.set_state("ERROR")
        self._win.right.mod_led.set_state("ERROR")

    def _on_loop_finished(self):
        with self._win._lock:
            self._win._state.update(
                running=False,
                paused=False,
                stopped=True,
                current_button="—",
                current_phase="idle",
                next_button="A",
                next_phase="above",
                status_detail="Test complete",
            )
        self._win.right.mod_cosmetic.set_state("COMPLETE")
        self._win.right.mod_led.set_state("COMPLETE")
        self._set_status(gui.C["GREEN"], "Test complete", "Robot + force sequence complete", 3.0)
        self._loop = None
        log.info("Test loop finished")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def _parse_args():
    p = argparse.ArgumentParser(description="RoboSpeed Controller")
    p.add_argument("--sim", action="store_true", default=True, help="Simulation mode (no real hardware)")
    p.add_argument("--no-sim", action="store_false", dest="sim", help="Connect to real hardware")
    p.add_argument("--motion-port", default="COM3", help="Serial port for motion controller (default COM3)")
    p.add_argument("--daq-port", default="COM4", help="Serial port for force DAQ (default COM4)")
    p.add_argument("--cam-c1", type=int, default=0, help="OpenCV index for camera C1")
    p.add_argument("--cam-c2", type=int, default=1, help="OpenCV index for camera C2")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = ControllerConfig(
        motion_port=args.motion_port,
        daq_port=args.daq_port,
        camera_c1_id=args.cam_c1,
        camera_c2_id=args.cam_c2,
        sim_mode=args.sim,
    )

    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication(sys.argv)
    app.setApplicationName("RoboSpeed DIP")
    app.setOrganizationName("RoboSpeed")
    app.setFont(gui.mkfont(10))

    _logo, _icon = gui._find_logo()
    if _icon and os.path.exists(_icon):
        from PyQt6.QtGui import QIcon
        app.setWindowIcon(QIcon(_icon))
    elif _logo and os.path.exists(_logo):
        from PyQt6.QtGui import QIcon
        app.setWindowIcon(QIcon(_logo))

    win = gui.MainWindow()
    win._thread.stop()

    ctrl = RoboSpeedController(cfg)
    ctrl.attach(win)

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
