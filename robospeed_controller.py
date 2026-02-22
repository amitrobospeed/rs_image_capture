"""
robospeed_controller.py  –  Hardware / Backend Controller
RoboSpeed Durability Intelligence Platform  v2.4

This module is the bridge between the physical robot hardware (or a hardware
abstraction layer) and the PyQt6 GUI defined in robospeed_gui_main.py.

─────────────────────────────────────────────────────────────────────
ARCHITECTURE
─────────────────────────────────────────────────────────────────────
  robospeed_controller.py          robospeed_gui_main.py
  ┌──────────────────────┐         ┌──────────────────────┐
  │  RoboSpeedController │◄──────► │    MainWindow (GUI)  │
  │                      │  Qt     │                      │
  │  MotionDriver        │ signals │  LeftPanel           │
  │  ForceDAQ            │         │  RightPanel          │
  │  VisionDriver        │         │  ForceGraph          │
  │  DataLogger          │         │  VisionPanel         │
  └──────────────────────┘         └──────────────────────┘

HOW TO RUN
─────────────────────────────────────────────────────────────────────
  # Simulation / demo (no hardware):
  python robospeed_controller.py --sim

  # With real hardware:
  python robospeed_controller.py --motion-port COM3 --daq-port COM4

  The controller creates the QApplication + MainWindow and then injects
  itself via MainWindow.set_controller(controller).
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys
import os
import time
import logging
import argparse
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

# Qt
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer
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
    force_ok       : bool   # True if within [force_min, force_max]
    retract_ok     : bool   # True if button retracted correctly


@dataclass
class InspectionCapture:
    cycle_num  : int
    camera     : str          # "C1" | "C2"
    frame_type : str          # "surface" | "led" | "point_cloud"
    data       : Optional[bytes] = None   # raw image / point-cloud bytes


@dataclass
class ControllerConfig:
    motion_port     : str  = "COM3"     # serial port for robot motion controller
    motion_host     : str  = "192.168.1.24"  # ethernet host for Dorna motion controller
    motion_tcp_port : int  = 443         # ethernet port for Dorna motion controller
    daq_port        : str  = "COM4"     # serial port for force DAQ
    phidget_serial  : int  = 781028     # Phidget bridge serial for force sensor
    phidget_channel : int  = 0          # Phidget channel index
    force_calibration_factor: float = 68000.0  # voltage ratio -> lbs
    force_data_interval_ms : int   = 8          # Phidget sampling interval
    camera_c1_id    : int  = 0          # OpenCV camera index
    camera_c2_id    : int  = 1
    sim_mode        : bool = True       # True = no real hardware


# ═══════════════════════════════════════════════════════════════════
# HARDWARE DRIVERS  (stubs — replace with real implementation)
# ═══════════════════════════════════════════════════════════════════
class MotionDriver:
    """
    Wraps the motion controller serial / Ethernet interface.
    Replace the stub methods with real protocol calls.
    """
    def __init__(self, config: ControllerConfig):
        self._cfg = config
        self._connected = False
        self._transport = "sim"
        self._robot = None
        self._ser = None
        self._last_params = MotionParams()

    @staticmethod
    def _extract_stat(resp) -> Optional[int]:
        """Extract Dorna 'stat' from possible response envelope shapes."""
        if isinstance(resp, dict):
            if "stat" in resp:
                return int(resp["stat"])
            union = resp.get("union")
            if isinstance(union, dict) and "stat" in union:
                return int(union["stat"])
            msgs = resp.get("msgs")
            if isinstance(msgs, list) and msgs:
                m0 = msgs[0]
                if isinstance(m0, dict) and "stat" in m0:
                    return int(m0["stat"])
        return None

    def _wait_dorna_idle(self, timeout_s: float = 10.0) -> bool:
        if not self._robot:
            return False
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                s = self._extract_stat(self._robot.play(-1, {"cmd": "stat"}))
                if s == -1:  # idle in your firmware
                    return True
            except Exception as e:
                log.debug(f"MotionDriver: stat poll failed: {e}")
            time.sleep(0.05)
        return False

    def connect(self) -> bool:
        if self._cfg.sim_mode:
            log.info("MotionDriver: simulation mode — no motion connection")
            self._connected = True
            self._transport = "sim"
            return True

        # Preferred path: Dorna over Ethernet
        try:
            from dorna2 import Dorna
            self._robot = Dorna()
            self._robot.connect(host=self._cfg.motion_host,
                                port=self._cfg.motion_tcp_port)
            self._connected = True
            self._transport = "dorna_ethernet"
            log.info(
                f"MotionDriver: connected to Dorna @ "
                f"{self._cfg.motion_host}:{self._cfg.motion_tcp_port}"
            )
            return True
        except Exception as e:
            log.warning(f"MotionDriver: Dorna ethernet connect failed: {e}")

        # Fallback: generic serial motion controller
        try:
            import serial
            self._ser = serial.Serial(self._cfg.motion_port, 115200, timeout=1)
            self._connected = True
            self._transport = "serial"
            log.info(f"MotionDriver: connected on serial {self._cfg.motion_port}")
            return True
        except Exception as e:
            log.error(f"MotionDriver.connect failed: {e}")
            return False

    def disconnect(self):
        if self._transport == "serial" and self._ser is not None:
            self._ser.close()
        if self._transport == "dorna_ethernet" and self._robot is not None:
            try:
                self._robot.close()
            except Exception:
                pass
        self._connected = False

    def home(self) -> bool:
        log.info("MotionDriver: HOME command")
        if self._cfg.sim_mode:
            return True
        if self._transport == "dorna_ethernet" and self._robot is not None:
            try:
                self._robot.play(0, {
                    "cmd": "jmove",
                    "rel": 0,
                    "vel": self._last_params.velocity,
                    "accel": self._last_params.acceleration,
                    "jerk": self._last_params.jerk,
                    "x": 318.06,
                    "y": -38.16,
                    "z": 200.0,
                    "a": -173.0,
                    "b": 41.62,
                    "c": -3.53,
                })
                return self._wait_dorna_idle(timeout_s=15.0)
            except Exception as e:
                log.error(f"MotionDriver.home failed (ethernet): {e}")
                return False

        if self._transport == "serial" and self._ser is not None:
            try:
                self._ser.write(b"HOME\r\n")
                return self._ser.readline().strip() == b"OK"
            except Exception as e:
                log.error(f"MotionDriver.home failed (serial): {e}")
                return False
        return False

    def set_params(self, params: MotionParams):
        log.info(f"MotionDriver: set_params vel={params.velocity} "
                 f"acc={params.acceleration} jerk={params.jerk}")
        self._last_params = params
        if self._cfg.sim_mode:
            return
        if self._transport == "serial" and self._ser is not None:
            try:
                cmd = (
                    f"SET VEL {params.velocity};ACC {params.acceleration};"
                    f"JERK {params.jerk}\r\n"
                )
                self._ser.write(cmd.encode("ascii"))
                self._ser.readline()
            except Exception as e:
                log.warning(f"MotionDriver.set_params serial write failed: {e}")

    def run_cycle(self) -> bool:
        """Execute one press-release cycle. Returns True when complete."""
        if self._cfg.sim_mode:
            time.sleep(0.05)   # sim: fake 50 ms cycle time
            return True

        if self._transport == "dorna_ethernet" and self._robot is not None:
            try:
                # One cycle = visit A/B/C/D, each with above -> press -> retract.
                # Matches Stage D trajectory semantics.
                cycle_poses = [
                    # A
                    {"x": 318.06, "y": -38.16, "z": 127.44, "a": -173.0,  "b": 41.62, "c": -3.53},
                    {"x": 318.06, "y": -38.16, "z": 120.40, "a": -173.0,  "b": 41.62, "c": -3.53},
                    {"x": 318.06, "y": -38.16, "z": 127.44, "a": -173.0,  "b": 41.62, "c": -3.53},
                    # B
                    {"x": 327.02, "y": -21.69, "z": 127.44, "a": -173.8,  "b": 32.25, "c": -4.81},
                    {"x": 327.02, "y": -21.69, "z": 118.70, "a": -173.8,  "b": 32.25, "c": -4.81},
                    {"x": 327.02, "y": -21.69, "z": 127.44, "a": -173.8,  "b": 32.25, "c": -4.81},
                    # C
                    {"x": 342.30, "y": -29.64, "z": 127.44, "a": -174.83, "b": 34.08, "c": -6.71},
                    {"x": 342.30, "y": -29.64, "z": 120.38, "a": -174.83, "b": 34.08, "c": -6.71},
                    {"x": 342.30, "y": -29.64, "z": 127.44, "a": -174.83, "b": 34.08, "c": -6.71},
                    # D
                    {"x": 335.08, "y": -43.88, "z": 125.13, "a": -173.84, "b": 43.25, "c": -4.86},
                    {"x": 335.08, "y": -43.88, "z": 121.70, "a": -173.84, "b": 43.25, "c": -4.86},
                    {"x": 335.08, "y": -43.88, "z": 127.44, "a": -173.84, "b": 43.25, "c": -4.86},
                ]

                for pose in cycle_poses:
                    self._robot.play(0, {
                        "cmd": "jmove",
                        "rel": 0,
                        "vel": self._last_params.velocity,
                        "accel": self._last_params.acceleration,
                        "jerk": self._last_params.jerk,
                        **pose,
                    })

                return self._wait_dorna_idle(timeout_s=30.0)
                self._robot.play(0, {
                    "cmd": "jmove",
                    "rel": 0,
                    "vel": self._last_params.velocity,
                    "accel": self._last_params.acceleration,
                    "jerk": self._last_params.jerk,
                    "x": 318.06,
                    "y": -38.16,
                    "z": 127.44,
                    "a": -173.0,
                    "b": 41.62,
                    "c": -3.53,
                })
                self._robot.play(0, {
                    "cmd": "jmove",
                    "rel": 0,
                    "vel": self._last_params.velocity,
                    "accel": self._last_params.acceleration,
                    "jerk": self._last_params.jerk,
                    "x": 318.06,
                    "y": -38.16,
                    "z": 120.40,
                    "a": -173.0,
                    "b": 41.62,
                    "c": -3.53,
                })
                self._robot.play(0, {
                    "cmd": "jmove",
                    "rel": 0,
                    "vel": self._last_params.velocity,
                    "accel": self._last_params.acceleration,
                    "jerk": self._last_params.jerk,
                    "x": 318.06,
                    "y": -38.16,
                    "z": 127.44,
                    "a": -173.0,
                    "b": 41.62,
                    "c": -3.53,
                })
                return self._wait_dorna_idle(timeout_s=10.0)
            except Exception as e:
                log.error(f"MotionDriver.run_cycle failed (ethernet): {e}")
                return False

        if self._transport == "serial" and self._ser is not None:
            try:
                self._ser.write(b"CYCLE\r\n")
                return self._ser.readline().strip() == b"OK"
            except Exception as e:
                log.error(f"MotionDriver.run_cycle failed (serial): {e}")
                return False
        return False

    def stop(self):
        log.info("MotionDriver: STOP command")
        if self._cfg.sim_mode:
            return
        if self._transport == "dorna_ethernet" and self._robot is not None:
            try:
                self._robot.play(0, {"cmd": "halt"})
            except Exception as e:
                log.error(f"MotionDriver.stop failed (ethernet): {e}")
            return
        if self._transport == "serial" and self._ser is not None:
            try:
                self._ser.write(b"STOP\r\n")
            except Exception as e:
                log.error(f"MotionDriver.stop failed (serial): {e}")

    def record_trajectory(self):
        log.info("MotionDriver: RECORD TRAJECTORY command")
        # TODO: put controller in teach mode


class ForceDAQ:
    """
    Reads force sensor via Phidget bridge or serial DAQ.
    In sim_mode generates synthetic force data.
    """
    def __init__(self, config: ControllerConfig):
        self._cfg = config
        self._connected = False
        self._t0 = time.time()
        self._zero_offset = 0.0
        self._transport = "sim"
        self._bridge = None
        self._ser = None

    def connect(self) -> bool:
        if self._cfg.sim_mode:
            self._connected = True
            self._transport = "sim"
            return True

        # Preferred path: Phidget22 bridge (matches Stage D v24)
        try:
            from Phidget22.Devices.VoltageRatioInput import VoltageRatioInput
            self._bridge = VoltageRatioInput()
            self._bridge.setDeviceSerialNumber(self._cfg.phidget_serial)
            self._bridge.setChannel(self._cfg.phidget_channel)
            self._bridge.openWaitForAttachment(5000)
            self._bridge.setDataInterval(self._cfg.force_data_interval_ms)

            # Tare / zero-offset at connect time
            samples = []
            for _ in range(200):
                samples.append(self._bridge.getVoltageRatio())
                time.sleep(0.001)
            self._zero_offset = sum(samples) / len(samples)

            self._connected = True
            self._transport = "phidget"
            log.info(
                f"ForceDAQ: connected via Phidget serial={self._cfg.phidget_serial} "
                f"channel={self._cfg.phidget_channel}, zero={self._zero_offset:.8f}"
            )
            return True
        except Exception as e:
            log.warning(f"ForceDAQ: Phidget connect failed: {e}")

        # Fallback path: generic serial DAQ
        try:
            import serial
            self._ser = serial.Serial(self._cfg.daq_port, 115200, timeout=0.1)
            self._connected = True
            self._transport = "serial"
            log.info(f"ForceDAQ: connected on serial {self._cfg.daq_port}")
            return True
        except Exception as e:
            log.error(f"ForceDAQ.connect failed: {e}")
            return False

    def disconnect(self):
        if self._transport == "phidget" and self._bridge is not None:
            try:
                self._bridge.close()
            except Exception:
                pass
        if self._transport == "serial" and self._ser is not None:
            self._ser.close()

    def read_lbs(self) -> float:
        """Return the instantaneous force reading in lbs."""
        if self._cfg.sim_mode:
            import math, random
            t = time.time() - self._t0
            return max(0.0, 1.1 * math.sin(2 * math.pi * 0.2 * t)
                       + random.gauss(0, 0.015))

        if self._transport == "phidget" and self._bridge is not None:
            try:
                raw = self._bridge.getVoltageRatio()
                force = (raw - self._zero_offset) * self._cfg.force_calibration_factor
                return float(max(0.0, force))
            except Exception as e:
                log.debug(f"ForceDAQ.read_lbs phidget read failed: {e}")
                return 0.0

        if self._transport == "serial" and self._ser is not None:
            try:
                self._ser.write(b"READ\r\n")
                raw = self._ser.readline().strip()
                return float(raw)
            except Exception as e:
                log.debug(f"ForceDAQ.read_lbs serial read failed: {e}")
                return 0.0

        return 0.0


class VisionDriver:
    """
    Manages camera(s) and captures frames / point clouds.
    """
    def __init__(self, config: ControllerConfig):
        self._cfg = config
        self._caps: dict = {}

    def connect(self) -> bool:
        if self._cfg.sim_mode:
            return True
        try:
            import cv2
            for name, idx in (("C1", self._cfg.camera_c1_id),
                               ("C2", self._cfg.camera_c2_id)):
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
        """Return a JPEG-encoded frame as bytes, or None on failure."""
        if self._cfg.sim_mode:
            return None   # GUI uses its own noise image
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
            try: cap.release()
            except: pass


class DataLogger:
    """
    Persists cycle results and inspection captures to disk.
    Extend to write to a database, InfluxDB, CSV, etc.
    """
    def __init__(self, project: str, profile: str):
        self._project = project
        self._profile = profile
        self._results: list[CycleResult] = []
        self._log_dir = os.path.join("logs", f"{project}_{profile}".replace(" ", "_"))
        os.makedirs(self._log_dir, exist_ok=True)
        log.info(f"DataLogger: writing to {self._log_dir}")

    def record_cycle(self, result: CycleResult):
        self._results.append(result)
        # TODO: append to CSV / database
        if not result.force_ok:
            log.warning(f"Cycle {result.cycle_num}: force OUT OF RANGE "
                        f"({result.peak_force_lbs:.3f} lbs)")

    def record_capture(self, capture: InspectionCapture):
        if capture.data:
            fname = (f"{capture.frame_type}_cycle{capture.cycle_num:05d}"
                     f"_{capture.camera}.jpg")
            with open(os.path.join(self._log_dir, fname), "wb") as f:
                f.write(capture.data)

    def generate_report(self) -> str:
        """Return a path to a generated report file."""
        import csv
        path = os.path.join(self._log_dir, "report.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cycle", "timestamp", "peak_force_lbs",
                        "force_ok", "retract_ok"])
            for r in self._results:
                w.writerow([r.cycle_num, f"{r.timestamp:.3f}",
                             f"{r.peak_force_lbs:.4f}",
                             r.force_ok, r.retract_ok])
        log.info(f"DataLogger: report saved → {path}")
        return path


# ═══════════════════════════════════════════════════════════════════
# TEST LOOP THREAD
# ═══════════════════════════════════════════════════════════════════
class TestLoopThread(QThread):
    """
    Runs the main durability-test loop in a background thread.
    Emits Qt signals that the GUI can safely consume from the main thread.
    """
    sig_cycle_done  = pyqtSignal(int, float, bool, bool)  # cycle, peak_force, force_ok, retract_ok
    sig_force_live  = pyqtSignal(float, float, int)        # time, force, cycle
    sig_error       = pyqtSignal(str)
    sig_finished    = pyqtSignal()

    def __init__(self,
                 motion: MotionDriver,
                 daq: ForceDAQ,
                 vision: VisionDriver,
                 logger: DataLogger,
                 params: MotionParams,
                 vision_cfg: dict,
                 parent=None):
        super().__init__(parent)
        self._motion    = motion
        self._daq       = daq
        self._vision    = vision
        self._logger    = logger
        self._params    = params
        self._vis_cfg   = vision_cfg
        self._running   = False
        self._paused    = False
        self._stop_req  = False
        self._lock      = threading.Lock()
        self._t0        = time.time()

    # ── control ──────────────────────────────────────────────────
    def request_stop(self):
        with self._lock: self._stop_req = True; self._motion.stop()

    def request_pause(self):
        with self._lock: self._paused = True

    def request_resume(self):
        with self._lock: self._paused = False

    # ── main loop ─────────────────────────────────────────────────
    def run(self):
        self._running  = True
        self._stop_req = False
        self._t0       = time.time()

        self._motion.set_params(self._params)
        cycle = 0
        peaks_this_cycle: list[float] = []
        force_sample_interval = 0.02   # 50 Hz

        for cycle in range(1, self._params.target_cycles + 1):
            with self._lock:
                if self._stop_req: break

            # Wait while paused
            while True:
                with self._lock:
                    if not self._paused or self._stop_req: break
                time.sleep(0.05)

            # ── Execute one cycle ──────────────────────────────────
            peaks_this_cycle.clear()
            t_cycle_start = time.time()

            ok = self._motion.run_cycle()
            if not ok:
                self.sig_error.emit(f"Motion error at cycle {cycle}")
                break

            # ── Sample force during cycle ──────────────────────────
            while time.time() - t_cycle_start < 0.3:   # 300 ms window
                f = self._daq.read_lbs()
                t = time.time() - self._t0
                self.sig_force_live.emit(t, f, cycle)
                peaks_this_cycle.append(f)
                time.sleep(force_sample_interval)

            peak = max(peaks_this_cycle) if peaks_this_cycle else 0.0
            force_ok   = self._params.force_min_lbs <= peak <= self._params.force_max_lbs
            retract_ok = True   # TODO: read retract sensor

            result = CycleResult(
                cycle_num=cycle, timestamp=time.time() - self._t0,
                peak_force_lbs=peak, force_ok=force_ok, retract_ok=retract_ok,
            )
            self._logger.record_cycle(result)
            self.sig_cycle_done.emit(cycle, peak, force_ok, retract_ok)

            # ── Vision captures ────────────────────────────────────
            surf_every  = self._vis_cfg.get("surface_capture_every",  25)
            led_every   = self._vis_cfg.get("led_capture_every",       25)
            cloud_every = self._vis_cfg.get("point_cloud_capture_every",50)

            for every, ftype, cam in [
                (surf_every,  "surface",     "C1"),
                (led_every,   "led",         "C1"),
                (cloud_every, "point_cloud", "C2"),
            ]:
                if every > 0 and cycle % every == 0:
                    data = self._vision.capture(cam)
                    self._logger.record_capture(
                        InspectionCapture(cycle, cam, ftype, data))

        self._running = False
        self.sig_finished.emit()
        log.info(f"TestLoopThread: finished after {cycle} cycles")


# ═══════════════════════════════════════════════════════════════════
# MAIN CONTROLLER
# ═══════════════════════════════════════════════════════════════════
class RoboSpeedController(QObject):
    """
    Top-level controller.  Creates drivers, owns the test loop thread,
    and wires everything to the GUI MainWindow signals.
    """

    def __init__(self, config: ControllerConfig):
        super().__init__()
        self._cfg    = config
        self._motion = MotionDriver(config)
        self._daq    = ForceDAQ(config)
        self._vision = VisionDriver(config)
        self._logger: Optional[DataLogger] = None
        self._loop:   Optional[TestLoopThread] = None
        self._params = MotionParams()
        self._vis_cfg: dict = {}
        self._win:    Optional[gui.MainWindow] = None

        # Connect hardware
        if not self._motion.connect():
            log.warning("Motion controller not connected — running in degraded mode")
        if not self._daq.connect():
            log.warning("Force DAQ not connected — force readings will be zero")
        self._vision.connect()

    # ── Attach to GUI window ──────────────────────────────────────
    def attach(self, win: gui.MainWindow):
        """Call this after MainWindow is created to wire all signals."""
        self._win = win

        # LeftPanel → controller
        win.left.sig_start.connect(self._on_gui_start)
        win.left.sig_pause.connect(self._on_gui_pause)
        win.left.sig_stop.connect(self._on_gui_stop)
        win.left.sig_home.connect(self._on_gui_home)
        win.left.sig_reset.connect(self._on_gui_reset)
        win.left.sig_report.connect(self._on_gui_report)
        win.left.sig_exit.connect(self._on_gui_exit)
        win.left.sig_fields.connect(self._on_gui_fields)
        win.left.sig_record.connect(self._on_gui_record)

        # RightPanel → controller
        win.right.sig_camera_changed.connect(self._on_camera_changed)
        win.right.sig_freq_updated.connect(self._on_freq_updated)

        log.info("Controller attached to GUI")

    # ── GUI signal handlers ───────────────────────────────────────
    def _on_gui_start(self):
        project = self._win.txtProject.text().strip() or "Project"
        profile = self._win.txtTestProfile.text().strip() or "Profile"
        self._logger = DataLogger(project, profile)
        vis_cfg = self._win.right.get_vision_settings()
        self._vis_cfg = vis_cfg

        self._loop = TestLoopThread(
            self._motion, self._daq, self._vision,
            self._logger, self._params, self._vis_cfg,
        )
        self._loop.sig_force_live.connect(self._win.force_graph.push)
        self._loop.sig_cycle_done.connect(self._on_cycle_done)
        self._loop.sig_error.connect(self._on_loop_error)
        self._loop.sig_finished.connect(self._on_loop_finished)
        self._loop.start()

        # Update GUI state
        with self._win._lock:
            self._win._state.update(running=True, paused=False, stopped=False)
        log.info("Test started")

    def _on_gui_pause(self):
        if self._loop:
            self._loop.request_pause()
        with self._win._lock:
            self._win._state.update(running=False, paused=True)
        log.info("Test paused")

    def _on_gui_stop(self):
        if self._loop:
            self._loop.request_stop()
            self._loop.wait(3000)
        with self._win._lock:
            self._win._state.update(running=False, paused=False, stopped=True)
        log.info("Test stopped")

    def _on_gui_home(self):
        self._motion.home()
        log.info("Homing")

    def _on_gui_reset(self):
        with self._win._lock:
            self._win._state.update(
                force_out_of_range=dict(A=0,B=0,C=0,D=0),
                button_did_not_retract=dict(A=0,B=0,C=0,D=0),
                cycle_count=0, baseline_ready=False, baseline_count=0,
            )
        self._win._peak_events.clear()
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
            velocity        = d.get("vel",             self._params.velocity),
            acceleration    = d.get("acc",             self._params.acceleration),
            jerk            = d.get("jerk",            self._params.jerk),
            target_cycles   = d.get("target_cycles",   self._params.target_cycles),
            baseline_cycles = d.get("baseline_cycles", self._params.baseline_cycles),
            force_min_lbs   = d.get("force_min",       self._params.force_min_lbs),
            force_max_lbs   = d.get("force_max",       self._params.force_max_lbs),
        )
        log.debug(f"Params updated: {self._params}")

    def _on_gui_record(self):
        self._motion.record_trajectory()

    def _on_camera_changed(self, label: str):
        log.info(f"Camera selection: {label}")
        # TODO: switch live feed stream

    def _on_freq_updated(self, msg: str):
        log.info(f"Inspection frequency changed: {msg}")

    # ── Test loop callbacks ───────────────────────────────────────
    def _on_cycle_done(self, cycle: int, peak: float, force_ok: bool, retract_ok: bool):
        with self._win._lock:
            st = self._win._state
            st["cycle_count"] = cycle
            if not force_ok:
                st["force_out_of_range"]["A"] += 1   # TODO: map to actual button A/B/C/D
            if not retract_ok:
                st["button_did_not_retract"]["A"] += 1

    def _on_loop_error(self, msg: str):
        log.error(f"Test loop error: {msg}")
        self._win._alert(gui.C["RED"], f"Error: {msg}", 5.0)

    def _on_loop_finished(self):
        with self._win._lock:
            self._win._state.update(running=False, stopped=True)
        self._win.statusBar().showMessage("Test complete", 5000)
        log.info("Test loop finished")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def _parse_args():
    p = argparse.ArgumentParser(description="RoboSpeed Controller")
    p.add_argument("--sim",          action="store_true", default=True,
                   help="Simulation mode (no real hardware)")
    p.add_argument("--no-sim",       action="store_false", dest="sim",
                   help="Connect to real hardware")
    p.add_argument("--motion-port",  default="COM3",
                   help="Serial port for motion controller (default COM3)")
    p.add_argument("--motion-host",  default="192.168.1.24",
                   help="Ethernet host for Dorna motion controller")
    p.add_argument("--motion-tcp-port",  type=int, default=443,
                   help="Ethernet TCP port for Dorna motion controller")
    p.add_argument("--daq-port",     default="COM4",
                   help="Serial port for force DAQ (default COM4)")
    p.add_argument("--phidget-serial", type=int, default=781028,
                   help="Phidget bridge serial for force sensor")
    p.add_argument("--phidget-channel", type=int, default=0,
                   help="Phidget channel index for force sensor")
    p.add_argument("--force-calibration-factor", type=float, default=68000.0,
                   help="Force conversion factor from voltage ratio to lbs")
    p.add_argument("--force-data-interval-ms", type=int, default=8,
                   help="Phidget data interval in milliseconds")
    p.add_argument("--cam-c1",       type=int, default=0,
                   help="OpenCV index for camera C1")
    p.add_argument("--cam-c2",       type=int, default=1,
                   help="OpenCV index for camera C2")
    return p.parse_args()


def main():
    args  = _parse_args()
    cfg   = ControllerConfig(
        motion_port  = args.motion_port,
        motion_host  = args.motion_host,
        motion_tcp_port = args.motion_tcp_port,
        daq_port     = args.daq_port,
        phidget_serial = args.phidget_serial,
        phidget_channel = args.phidget_channel,
        force_calibration_factor = args.force_calibration_factor,
        force_data_interval_ms = args.force_data_interval_ms,
        camera_c1_id = args.cam_c1,
        camera_c2_id = args.cam_c2,
        sim_mode     = args.sim,
    )

    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication(sys.argv)
    app.setApplicationName("RoboSpeed DIP")
    app.setOrganizationName("RoboSpeed")
    app.setFont(gui.mkfont(10))

    # Set app icon
    _logo, _icon = gui._find_logo()
    if _icon and os.path.exists(_icon):
        from PyQt6.QtGui import QIcon
        app.setWindowIcon(QIcon(_icon))
    elif _logo and os.path.exists(_logo):
        from PyQt6.QtGui import QIcon
        app.setWindowIcon(QIcon(_logo))

    # Build GUI
    win = gui.MainWindow()

    # Stop the built-in mock data thread — controller provides real data
    win._thread.stop()

    # Build & attach controller
    ctrl = RoboSpeedController(cfg)
    ctrl.attach(win)

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
