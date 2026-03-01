import time
import threading
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, TextBox
from matplotlib.backends.backend_pdf import PdfPages
import os
import csv
import sys
import subprocess
from datetime import datetime

import cv2
import pyrealsense2 as rs

from Phidget22.Devices.VoltageRatioInput import VoltageRatioInput
from dorna2 import Dorna


# ================================
# FORCE SETTINGS
# ================================
calibration_factor = 68000
window_seconds = 10
update_interval = 0.2
data_interval_ms = 8

peak_start_threshold = 0.5

force_min_valid = 0.5
force_max_valid = 1.8

baseline_cycles = 5
deviation_threshold = 0.50

ALERT_FLASH_S = 1.0
TARE_WARMUP_S = 3.0
TARE_SAMPLES = 200

# ================================
# UI SETTINGS
# ================================
LOGO_CANDIDATE_PATHS = [
    "assets/robospeed_logo.png",
    "robospeed_logo.png",
]

COLOR_BG = "#0f172a"
COLOR_PANEL = "#111827"
COLOR_PANEL_BORDER = "#334155"
COLOR_TEXT = "#e5e7eb"

CAMERA_WINDOW_NAME = "IC Camera"
CAMERA_OUTPUT_DIR = "inspection_output"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 15
TARGET_LUMA_MEAN = 95
MAX_SAT_PCT = 3.0
TUNE_MAX_ITERS = 10
EXPOSURE_MIN = 2000
EXPOSURE_MAX = 15000
GAIN_MIN = 0
GAIN_MAX = 32

DEFAULT_AUTO_CAPTURE_RETRIES = 2
AUTO_FAIL_POLICY_OPTIONS = ("safe_stop", "continue")
IC_CAPTURE_SETTLE_S = 2.0
CAPTURE_AVG_MIN_FRAMES = 10
CAPTURE_AVG_TIMEOUT_S = 2.5
INSPECTION_DIFF_THRESHOLD = 25
INSPECTION_MIN_DEFECT_AREA = 15
INSPECTION_MIN_DEFECT_W = 3
INSPECTION_MIN_DEFECT_H = 3
INSPECTION_EDGE_IGNORE_PX = 2
BUTTON_WHITE_RATIO_DROP_PCT = 0.20
BUTTON_MIN_COLOR_PIXELS = 50

BUTTON_COLOR_RULES = {
    "A": {"name": "green",  "hsv": [((35, 40, 40), (90, 255, 255))]},
    "B": {"name": "blue",   "hsv": [((90, 40, 40), (140, 255, 255))]},
    "C": {"name": "red",    "hsv": [((0, 40, 40), (10, 255, 255)), ((160, 40, 40), (179, 255, 255))]},
    "D": {"name": "yellow", "hsv": [((15, 40, 40), (40, 255, 255))]},
}
WHITE_HSV_RANGE = ((0, 0, 160), (179, 60, 255))


# ================================
# DORNA SETTINGS
# ================================
DORNA_HOST = "192.168.1.24"
DORNA_PORT = 443

# ✅ Updated ranges per your request
VEL_MIN, VEL_MAX = 0, 1000
ACC_MIN, ACC_MAX = 0, 2000
JERK_MIN, JERK_MAX = 0, 10000

SAFE_START_VEL = 100
SAFE_START_ACC = 200
SAFE_START_JERK = 2000
IC_CLEAR_J0_REL = -70
IC_CLEAR_J0_JERK = 1000


# ================================
# HOME POSITION
# ================================
HOME_POSE = {
    "x": 318.06,
    "y": -38.16,
    "z": 200.0,
    "a": -173.0,
    "b": 41.62,
    "c": -3.53
}


# ================================
# BUTTON POSES (hardcoded)
# ================================
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

PHASES = ["above", "press", "retract"]
BUTTON_ORDER = ["A", "B", "C", "D"]

TRAJECTORY = []
INDEX_TO_META = []
for btn in BUTTON_ORDER:
    for ph in PHASES:
        TRAJECTORY.append(BUTTON_POSES[btn][ph])
        INDEX_TO_META.append((btn, ph))


# ================================
# STAT PARSER (your firmware: idle == -1)
# ================================
def _extract_stat(resp):
    if isinstance(resp, dict):
        if "stat" in resp:
            return int(resp["stat"])
        if "union" in resp and isinstance(resp["union"], dict) and "stat" in resp["union"]:
            return int(resp["union"]["stat"])
        if "msgs" in resp and isinstance(resp["msgs"], list) and len(resp["msgs"]) > 0:
            m0 = resp["msgs"][0]
            if isinstance(m0, dict) and "stat" in m0:
                return int(m0["stat"])
    return None


def is_idle(robot):
    try:
        resp = robot.play(-1, {"cmd": "stat"})
        s = _extract_stat(resp)
        return (s == -1)
    except Exception:
        return False


# ================================
# STATE
# ================================
@dataclass
class SystemState:
    running: bool = False
    paused: bool = False
    stopped: bool = True

    vel: int = 300
    acc: int = 300
    jerk: int = 1000  # ✅ added
    target_cycles: int = 30
    baseline_cycles: int = baseline_cycles
    force_min: float = force_min_valid
    force_max: float = force_max_valid

    traj_index: int = 0
    cycle_count: int = 0
    aligned_to_A: bool = False

    force_out_of_range: dict = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0})
    button_did_not_retract: dict = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0})

    baseline_peaks: dict = field(default_factory=lambda: {"A": [], "B": [], "C": [], "D": []})
    baseline_mean: dict = field(default_factory=dict)
    baseline_ready: bool = False

    window_active: bool = False
    window_button: str | None = None
    window_peak_force: float | None = None
    window_peak_time: float | None = None

    alert_until_wall: float = 0.0
    alert_color: str = "gray"
    alert_msg: str = ""
    manual_intervention_requested: bool = False
    manual_mode_active: bool = False
    image_capture_count: int = 0
    tare_on_start: bool = True
    tare_in_progress: bool = False
    test_name: str = "test_report"
    exit_requested: bool = False

    capture_every_x_cycles: int = 5
    golden_ready: bool = False
    auto_capture_enabled: bool = False
    next_auto_capture_cycle: int = 0
    last_capture_result: str = "none"
    auto_capture_retries: int = DEFAULT_AUTO_CAPTURE_RETRIES
    auto_fail_policy: str = "safe_stop"
    detect_contour_enabled: bool = False
    detect_white_ratio_enabled: bool = True


def main():
    # --- Connect robot (keep alive) ---
    robot = Dorna()
    robot_connected = True
    robot_mode_text = "ROBOT"
    print("Connecting to Dorna...")
    try:
        robot.connect(host=DORNA_HOST, port=DORNA_PORT)
        print("Robot Connected!")
    except Exception as exc:
        robot_connected = False
        robot_mode_text = "VISUAL_INSPECTION_ONLY"
        print(f"[Robot] Connect failed -> visual inspection mode: {exc}")

    # --- Force sensor ---
    bridge = VoltageRatioInput()
    bridge.setDeviceSerialNumber(781028)
    bridge.setChannel(0)
    bridge.openWaitForAttachment(5000)
    bridge.setDataInterval(data_interval_ms)

    print("Taring...")
    time.sleep(TARE_WARMUP_S)
    zero_offset = float(np.mean([bridge.getVoltageRatio() for _ in range(TARE_SAMPLES)]))

    state = SystemState()
    state_lock = threading.RLock()

    # --- Camera preview (Phase 2A / Phase 3A controls) ---
    os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)
    camera_lock = threading.RLock()
    camera_hw_lock = threading.RLock()
    camera_latest_frame = None
    camera_status = "camera:not_started"
    camera_stop_evt = threading.Event()
    camera_thread = None
    camera_sensor = None

    # session-level camera settings lock
    camera_exposure = 4500
    camera_gain = 8
    camera_settings_locked = False
    camera_tuned_once = False

    # capture/inspection session artifacts
    golden_frame = None
    golden_path = None
    last_capture_frame = None
    last_capture_path = None
    inspection_records = []
    anomaly_stats = {
        "total_scored": 0,
        "pass_count": 0,
        "fail_count": 0,
        "warn_count": 0,
        "first_fail_cycle": "",
        "worst_score": -1.0,
        "worst_cycle": "",
    }
    locked_roi = None
    roi_locked = False
    button_rois = {}
    button_color_baselines = {}
    button_roi_locked = False

    manifest_path = os.path.join(CAMERA_OUTPUT_DIR, "manifest.csv")
    reports_dir = os.path.join(CAMERA_OUTPUT_DIR, "reports")
    masks_dir = os.path.join(CAMERA_OUTPUT_DIR, "masks")
    cycle_video_path = None
    cycle_video_writer = None
    cycle_video_started = False
    last_saved_report_path = None
    auto_report_written_cycle = -1

    for _d in [
        os.path.join(CAMERA_OUTPUT_DIR, "golden"),
        os.path.join(CAMERA_OUTPUT_DIR, "cyc"),
        os.path.join(CAMERA_OUTPUT_DIR, "anomaly"),
        masks_dir,
        os.path.join(CAMERA_OUTPUT_DIR, "video"),
        reports_dir,
    ]:
        os.makedirs(_d, exist_ok=True)

    def _set_camera_status(msg):
        nonlocal camera_status
        with camera_lock:
            camera_status = msg

    def compute_luma_stats(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        sat_pct = float(np.mean(gray >= 250) * 100.0)
        return mean_luma, sat_pct

    def start_camera_preview():
        nonlocal camera_thread, camera_latest_frame, camera_sensor
        if camera_thread is not None and camera_thread.is_alive():
            return

        camera_stop_evt.clear()

        def _camera_worker():
            nonlocal camera_latest_frame, camera_sensor
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)

            try:
                profile = pipeline.start(config)
            except Exception:
                _set_camera_status("camera:open_failed")
                return

            try:
                sensor = profile.get_device().first_color_sensor()
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, int(camera_exposure))
                sensor.set_option(rs.option.gain, int(camera_gain))
                with camera_hw_lock:
                    camera_sensor = sensor
            except Exception:
                _set_camera_status("camera:sensor_setup_failed")

            _set_camera_status("camera:live")

            # warmup frames
            for _ in range(8):
                try:
                    pipeline.wait_for_frames(1000)
                except Exception:
                    break

            try:
                while not camera_stop_evt.is_set():
                    try:
                        frames = pipeline.wait_for_frames(1000)
                    except Exception:
                        _set_camera_status("camera:frame_timeout")
                        time.sleep(0.05)
                        continue

                    color = frames.get_color_frame()
                    if not color:
                        _set_camera_status("camera:no_color_frame")
                        continue

                    frame = np.asanyarray(color.get_data())
                    with camera_lock:
                        camera_latest_frame = frame.copy()

                    # Frame is rendered inside the matplotlib window (same app window).
            finally:
                with camera_hw_lock:
                    camera_sensor = None
                try:
                    pipeline.stop()
                except Exception:
                    pass

        camera_thread = threading.Thread(target=_camera_worker, daemon=True)
        camera_thread.start()

    def stop_camera_preview():
        camera_stop_evt.set()
        if camera_thread is not None:
            camera_thread.join(timeout=1.5)

    def get_latest_camera_frame():
        with camera_lock:
            if camera_latest_frame is None:
                return None
            return camera_latest_frame.copy()

    def capture_average_frame(min_frames=CAPTURE_AVG_MIN_FRAMES, timeout_s=CAPTURE_AVG_TIMEOUT_S):
        deadline = time.time() + max(0.1, float(timeout_s))
        frames = []
        while time.time() < deadline:
            fr = get_latest_camera_frame()
            if fr is not None:
                frames.append(fr.astype(np.float32))
                if len(frames) >= int(min_frames):
                    break
            time.sleep(0.02)

        if len(frames) < int(min_frames):
            return None, len(frames)

        avg = np.mean(frames, axis=0)
        return np.clip(avg, 0, 255).astype(np.uint8), len(frames)

    def run_camera_auto_tune():
        nonlocal camera_exposure, camera_gain, camera_settings_locked, camera_tuned_once
        with camera_hw_lock:
            sensor = camera_sensor
        if sensor is None:
            set_alert("red", "Camera tune failed: sensor not ready")
            return False

        frame, frames_used = capture_average_frame(min_frames=6, timeout_s=1.5)
        if frame is None:
            set_alert("red", f"Camera tune failed: averaged frame unavailable ({frames_used}/6)")
            return False

        exp = int(camera_exposure)
        gain = int(camera_gain)

        # Stage 1: exposure-first alignment
        for _ in range(TUNE_MAX_ITERS):
            mean_l, sat = _compute_button_luma_stats(frame)

            if sat > MAX_SAT_PCT:
                exp = int(exp * 0.85)
            elif mean_l > TARGET_LUMA_MEAN + 5:
                exp = int(exp * 0.9)
            elif mean_l < TARGET_LUMA_MEAN - 5:
                exp = int(exp * 1.05)
            else:
                break

            exp = int(np.clip(exp, EXPOSURE_MIN, EXPOSURE_MAX))

            try:
                sensor.set_option(rs.option.exposure, exp)
                sensor.set_option(rs.option.gain, gain)
            except Exception:
                set_alert("red", "Camera tune failed: cannot apply exposure")
                return False

            time.sleep(0.12)
            frame, _ = capture_average_frame(min_frames=6, timeout_s=1.2)
            if frame is None:
                frame = get_latest_camera_frame()
                if frame is None:
                    break

        # Stage 2: gain-only fine tune
        for _ in range(max(2, TUNE_MAX_ITERS // 2)):
            mean_l, sat = _compute_button_luma_stats(frame)

            if sat > MAX_SAT_PCT:
                gain = int(gain * 0.9)
            elif mean_l > TARGET_LUMA_MEAN + 3:
                gain = int(gain * 0.9)
            elif mean_l < TARGET_LUMA_MEAN - 3:
                gain = int(gain * 1.1)
            else:
                break

            gain = int(np.clip(gain, GAIN_MIN, GAIN_MAX))

            try:
                sensor.set_option(rs.option.gain, gain)
            except Exception:
                set_alert("red", "Camera tune failed: cannot apply gain")
                return False

            time.sleep(0.10)
            frame, _ = capture_average_frame(min_frames=6, timeout_s=1.0)
            if frame is None:
                frame = get_latest_camera_frame()
                if frame is None:
                    break

        camera_exposure = exp
        camera_gain = gain
        camera_settings_locked = True
        camera_tuned_once = True
        set_alert("green", f"Camera tuned+locked exp={exp} gain={gain}")
        return True

    def _manifest_write(row):
        file_exists = os.path.exists(manifest_path)
        fields = [
            "run_id", "cycle", "capture_type", "timestamp", "camera_status", "result",
            "message", "reason_code", "file_path", "score", "threshold", "verdict",
            "golden_path", "video_path", "anomaly_path", "policy",
            "decision_logic", "failed_metric", "max_contour_area", "max_bbox", "score_role",
            "button_drop_pct", "white_ratio_button", "white_ratio_change_pct"
        ]
        with open(manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})

    def _record_anomaly_stats(cycle_num, verdict, score):
        if score in ("", None):
            return
        try:
            s = float(score)
        except Exception:
            return
        anomaly_stats["total_scored"] += 1
        if verdict == "PASS":
            anomaly_stats["pass_count"] += 1
        elif verdict == "FAIL":
            anomaly_stats["fail_count"] += 1
            if anomaly_stats["first_fail_cycle"] == "":
                anomaly_stats["first_fail_cycle"] = cycle_num
        else:
            anomaly_stats["warn_count"] += 1
        if s > anomaly_stats["worst_score"]:
            anomaly_stats["worst_score"] = s
            anomaly_stats["worst_cycle"] = cycle_num

    def _recover_camera_preview():
        set_alert("#f59e0b", "Camera recovery: restarting preview")
        stop_camera_preview()
        time.sleep(0.2)
        start_camera_preview()
        time.sleep(0.3)

    def _stamp(frame, text, color=(0, 255, 255)):
        out = frame.copy()
        cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return out

    def _roi_bounds(roi_like, shape):
        if isinstance(roi_like, dict):
            rshape = roi_like.get("shape", "rect")
            if rshape == "circle":
                cx, cy, r = int(roi_like.get("cx", 0)), int(roi_like.get("cy", 0)), int(roi_like.get("r", 0))
                return (cx - r, cy - r, 2 * r, 2 * r)
            if rshape == "poly":
                pts = np.array(roi_like.get("points", []), dtype=np.int32)
                if len(pts) < 3:
                    return (0, 0, 1, 1)
                x, y, w, h = cv2.boundingRect(pts)
                return (x, y, w, h)
            return (int(roi_like.get("x", 0)), int(roi_like.get("y", 0)), int(roi_like.get("w", 0)), int(roi_like.get("h", 0)))
        return tuple(map(int, roi_like))

    def _sanitize_roi(roi_like, shape):
        x, y, w, h = _roi_bounds(roi_like, shape)
        h_img, w_img = shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        if isinstance(roi_like, dict):
            rshape = roi_like.get("shape", "rect")
            if rshape == "circle":
                cx = int(np.clip(int(roi_like.get("cx", x + w // 2)), x, x + w - 1))
                cy = int(np.clip(int(roi_like.get("cy", y + h // 2)), y, y + h - 1))
                r_max = max(1, min(cx - x, x + w - 1 - cx, cy - y, y + h - 1 - cy))
                r = int(max(1, min(int(roi_like.get("r", 1)), r_max)))
                return {"shape": "circle", "cx": cx, "cy": cy, "r": r}
            if rshape == "poly":
                pts = []
                for px, py in roi_like.get("points", []):
                    pts.append((int(np.clip(int(px), 0, w_img - 1)), int(np.clip(int(py), 0, h_img - 1))))
                if len(pts) < 3:
                    pts = [(x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1)]
                return {"shape": "poly", "points": pts}
            return {"shape": "rect", "x": x, "y": y, "w": w, "h": h}
        return (x, y, w, h)

    def _draw_roi(frame, roi_like, color=(255, 0, 0), label=None):
        out = frame
        roi = _sanitize_roi(roi_like, frame.shape)
        if isinstance(roi, dict):
            if roi.get("shape") == "circle":
                cv2.circle(out, (roi["cx"], roi["cy"]), roi["r"], color, 1)
                lx, ly = roi["cx"] - roi["r"], roi["cy"] - roi["r"]
            elif roi.get("shape") == "poly":
                pts = np.array(roi.get("points", []), dtype=np.int32)
                cv2.polylines(out, [pts], True, color, 1)
                lx, ly = int(pts[:, 0].min()), int(pts[:, 1].min())
            else:
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
                lx, ly = x, y
        else:
            x, y, w, h = roi
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
            lx, ly = x, y
        if label:
            cv2.putText(out, label, (lx + 2, max(14, ly - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out

    def _roi_mask_from_spec(shape, roi_like):
        roi = _sanitize_roi(roi_like, shape)
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if isinstance(roi, dict):
            if roi.get("shape") == "circle":
                cv2.circle(mask, (roi["cx"], roi["cy"]), roi["r"], 255, -1)
            elif roi.get("shape") == "poly":
                pts = np.array(roi.get("points", []), dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            else:
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                mask[y:y + h, x:x + w] = 255
        else:
            x, y, w, h = roi
            mask[y:y + h, x:x + w] = 255
        return mask, _roi_bounds(roi, shape)

    def _select_roi_with_shape(window_name, frame):
        shape_mode = "rect"
        start_pt = None
        drag_pt = None
        poly_pts = []
        finalized = None
        actions = {}
        status_msg = "Draw ROI, then press Enter to confirm"

        def _is_valid_roi(roi):
            if not roi:
                return False
            if isinstance(roi, dict):
                rshape = roi.get("shape", "rect")
                if rshape == "rect":
                    return int(roi.get("w", 0)) >= 4 and int(roi.get("h", 0)) >= 4
                if rshape == "circle":
                    return int(roi.get("r", 0)) >= 3
                if rshape == "poly":
                    return len(roi.get("points", [])) >= 3
                return False
            return len(roi) == 4 and int(roi[2]) >= 4 and int(roi[3]) >= 4

        def _finalize_current():
            nonlocal finalized, status_msg
            if shape_mode == "poly":
                if len(poly_pts) >= 3:
                    finalized = {"shape": "poly", "points": poly_pts.copy()}
                    status_msg = "Polygon ready. Press Enter to confirm"
                    return True
                status_msg = "Polygon needs at least 3 points"
                return False
            if _is_valid_roi(finalized):
                status_msg = "ROI ready. Press Enter to confirm"
                return True
            status_msg = "Draw a larger ROI first"
            return False

        def _draw_buttons(canvas):
            nonlocal actions
            actions = {}
            items = [("rect", "Rectangle"), ("circle", "Circle"), ("poly", "Polygon"), ("lock", "Lock ROI")]
            x0 = 20
            for key, title in items:
                w, h = 150, 34
                x1, y1 = x0 + w, 50 + h
                fill = (46, 204, 113) if (key == shape_mode and key != "lock") else ((192, 57, 43) if key == "lock" else (44, 62, 80))
                cv2.rectangle(canvas, (x0, 50), (x1, y1), fill, -1)
                cv2.rectangle(canvas, (x0, 50), (x1, y1), (236, 240, 241), 1)
                cv2.putText(canvas, title, (x0 + 8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                actions[key] = (x0, 50, x1, y1)
                x0 += w + 10

        def _hit_button(x, y):
            for key, (x0, y0, x1, y1) in actions.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return key
            return None

        def _mouse_cb(evt, x, y, _flags, _param):
            nonlocal shape_mode, start_pt, drag_pt, poly_pts, finalized, status_msg
            if evt == cv2.EVENT_LBUTTONDOWN:
                k = _hit_button(x, y)
                if k in ("rect", "circle", "poly"):
                    shape_mode = k
                    start_pt = None
                    drag_pt = None
                    finalized = None
                    if k != "poly":
                        poly_pts = []
                    status_msg = f"{k.title()} mode selected"
                    return
                if k == "lock":
                    if _finalize_current():
                        status_msg = "ROI locked. Press Enter to continue"
                    return
                if shape_mode == "poly":
                    poly_pts.append((x, y))
                    finalized = None
                    status_msg = f"Polygon points: {len(poly_pts)}"
                    return
                start_pt = (x, y)
                drag_pt = (x, y)
            elif evt == cv2.EVENT_MOUSEMOVE:
                if shape_mode in ("rect", "circle") and start_pt is not None:
                    drag_pt = (x, y)
            elif evt == cv2.EVENT_LBUTTONUP:
                if shape_mode in ("rect", "circle") and start_pt is not None and drag_pt is not None:
                    sx, sy = start_pt
                    ex, ey = drag_pt
                    if shape_mode == "rect":
                        finalized = {"shape": "rect", "x": min(sx, ex), "y": min(sy, ey), "w": abs(ex - sx), "h": abs(ey - sy)}
                    else:
                        r = int(np.hypot(ex - sx, ey - sy))
                        finalized = {"shape": "circle", "cx": sx, "cy": sy, "r": r}
                    start_pt = None
                    status_msg = "ROI drawn. Press Enter to confirm"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, _mouse_cb)
        while True:
            canvas = frame.copy()
            cv2.putText(canvas, "Draw area to inspect and press enter to continue", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(canvas, "Choose ROI shape: Rectangle / Circle / Polygon", (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
            cv2.putText(canvas, "Polygon: click points, Enter/Lock ROI to finish", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(canvas, status_msg, (20, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
            _draw_buttons(canvas)

            if shape_mode == "poly" and poly_pts:
                pts = np.array(poly_pts, dtype=np.int32)
                cv2.polylines(canvas, [pts], False, (255, 0, 0), 1)
                for p in poly_pts:
                    cv2.circle(canvas, p, 3, (255, 0, 0), -1)
                if finalized and finalized.get("shape") == "poly":
                    _draw_roi(canvas, finalized, (255, 0, 0))
            elif start_pt is not None and drag_pt is not None and shape_mode in ("rect", "circle"):
                sx, sy = start_pt
                ex, ey = drag_pt
                preview = {"shape": "rect", "x": min(sx, ex), "y": min(sy, ey), "w": abs(ex - sx), "h": abs(ey - sy)} if shape_mode == "rect" else {"shape": "circle", "cx": sx, "cy": sy, "r": int(np.hypot(ex - sx, ey - sy))}
                _draw_roi(canvas, preview, (255, 0, 0))
            elif finalized is not None:
                _draw_roi(canvas, finalized, (255, 0, 0))

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                if _finalize_current() and _is_valid_roi(finalized):
                    return _sanitize_roi(finalized, frame.shape)
            elif key == 27:
                return None
            elif key in (8, 127):
                if shape_mode == "poly" and poly_pts:
                    poly_pts.pop()
                    finalized = None
                    status_msg = f"Polygon points: {len(poly_pts)}"

    def _roi_crop(frame, roi):
        x, y, w, h = _roi_bounds(_sanitize_roi(roi, frame.shape), frame.shape)
        return frame[y:y+h, x:x+w], (x, y, w, h)

    def _compute_button_color_stats(frame, roi, button_name):
        roi_norm = _sanitize_roi(roi, frame.shape)
        mask_full, (x, y, w, h) = _roi_mask_from_spec(frame.shape, roi_norm)
        crop = frame[y:y+h, x:x+w]
        local_mask = mask_full[y:y+h, x:x+w]
        if crop.size == 0 or local_mask.size == 0 or np.count_nonzero(local_mask) == 0:
            return {
                "roi": roi_norm,
                "white_px": 0,
                "color_px": 0,
                "white_to_color": 0.0,
                "luma_mean": 0.0,
                "sat_pct": 0.0,
            }
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white_lo, white_hi = WHITE_HSV_RANGE
        white_mask = cv2.inRange(hsv, np.array(white_lo, dtype=np.uint8), np.array(white_hi, dtype=np.uint8))

        color_mask = np.zeros(white_mask.shape, dtype=np.uint8)
        for lo, hi in BUTTON_COLOR_RULES.get(button_name, {}).get("hsv", []):
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))

        white_mask = cv2.bitwise_and(white_mask, local_mask)
        color_mask = cv2.bitwise_and(color_mask, local_mask)

        white_px = int(np.count_nonzero(white_mask))
        color_px = int(np.count_nonzero(color_mask))
        ratio = float(white_px / max(color_px, 1))
        luma_mean, sat_pct = compute_luma_stats(cv2.bitwise_and(crop, crop, mask=local_mask))
        return {
            "roi": roi_norm,
            "white_px": white_px,
            "color_px": color_px,
            "white_to_color": ratio,
            "luma_mean": luma_mean,
            "sat_pct": sat_pct,
        }

    def _compute_button_luma_stats(frame):
        if not button_roi_locked or not button_rois:
            return compute_luma_stats(frame)
        means = []
        sats = []
        for btn, roi in button_rois.items():
            st = _compute_button_color_stats(frame, roi, btn)
            means.append(st["luma_mean"])
            sats.append(st["sat_pct"])
        if not means:
            return compute_luma_stats(frame)
        return float(np.mean(means)), float(max(sats))

    def _select_button_rois_and_calibrate(frame):
        nonlocal button_rois, button_color_baselines, button_roi_locked, locked_roi, roi_locked
        selected = dict(button_rois) if (button_roi_locked and button_rois) else {}
        shape_mode = "rect"
        start_pt = None
        drag_pt = None
        poly_pts = []
        working_roi = None
        actions = {}
        current_idx = 0
        status_msg = "Draw ROI then Next; Draw all four then Lock All"

        def _is_valid_roi(roi):
            if not roi:
                return False
            if isinstance(roi, dict):
                rshape = roi.get("shape", "rect")
                if rshape == "rect":
                    return int(roi.get("w", 0)) >= 4 and int(roi.get("h", 0)) >= 4
                if rshape == "circle":
                    return int(roi.get("r", 0)) >= 3
                if rshape == "poly":
                    return len(roi.get("points", [])) >= 3
            return False

        def _target_btn():
            return BUTTON_ORDER[current_idx]

        def _draw_buttons(canvas):
            nonlocal actions
            actions = {}
            items = [
                ("rect", "Rectangle", 140),
                ("circle", "Circle", 120),
                ("poly", "Polygon", 130),
                ("next", "Next Button", 150),
                ("lock", "Lock All", 120),
            ]
            x0, y0 = 20, 50
            for key, title, w in items:
                h = 34
                x1, y1 = x0 + w, y0 + h
                active = key == shape_mode and key in ("rect", "circle", "poly")
                fill = (46, 204, 113) if active else ((192, 57, 43) if key == "lock" else (44, 62, 80))
                cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (236, 240, 241), 1)
                cv2.putText(canvas, title, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                actions[key] = (x0, y0, x1, y1)
                x0 += w + 8

        def _hit_button(x, y):
            for key, (x0, y0, x1, y1) in actions.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return key
            return None

        def _finalize_working_from_poly():
            nonlocal working_roi, status_msg
            if len(poly_pts) >= 3:
                working_roi = {"shape": "poly", "points": poly_pts.copy()}
                status_msg = f"{_target_btn()} polygon ready"
                return True
            status_msg = "Polygon needs >=3 points"
            return False

        def _commit_current_btn():
            nonlocal status_msg, working_roi
            btn = _target_btn()
            if shape_mode == "poly":
                _finalize_working_from_poly()
            if not _is_valid_roi(working_roi):
                status_msg = f"Draw valid ROI for button {btn}"
                return False
            selected[btn] = _sanitize_roi(working_roi, frame.shape)
            status_msg = f"Saved ROI for button {btn}"
            return True

        def _try_lock_all():
            nonlocal status_msg
            if _is_valid_roi(working_roi):
                _commit_current_btn()
            missing = [b for b in BUTTON_ORDER if b not in selected]
            if missing:
                status_msg = f"Missing ROI: {','.join(missing)}"
                return False
            return True

        def _mouse_cb(evt, x, y, _flags, _param):
            nonlocal shape_mode, start_pt, drag_pt, poly_pts, working_roi, current_idx, status_msg
            if evt == cv2.EVENT_LBUTTONDOWN:
                k = _hit_button(x, y)
                if k in ("rect", "circle", "poly"):
                    shape_mode = k
                    start_pt = None
                    drag_pt = None
                    working_roi = None
                    if k != "poly":
                        poly_pts = []
                    status_msg = f"{k.title()} mode for button {_target_btn()}"
                    return
                if k == "next":
                    if _commit_current_btn():
                        current_idx = (current_idx + 1) % len(BUTTON_ORDER)
                        start_pt = None
                        drag_pt = None
                        poly_pts = []
                        working_roi = selected.get(_target_btn())
                    return
                if k == "lock":
                    _try_lock_all()
                    return
                if shape_mode == "poly":
                    poly_pts.append((x, y))
                    working_roi = None
                    status_msg = f"{_target_btn()} polygon points: {len(poly_pts)}"
                    return
                start_pt = (x, y)
                drag_pt = (x, y)
            elif evt == cv2.EVENT_MOUSEMOVE:
                if shape_mode in ("rect", "circle") and start_pt is not None:
                    drag_pt = (x, y)
            elif evt == cv2.EVENT_LBUTTONUP:
                if shape_mode in ("rect", "circle") and start_pt is not None and drag_pt is not None:
                    sx, sy = start_pt
                    ex, ey = drag_pt
                    if shape_mode == "rect":
                        working_roi = {"shape": "rect", "x": min(sx, ex), "y": min(sy, ey), "w": abs(ex - sx), "h": abs(ey - sy)}
                    else:
                        working_roi = {"shape": "circle", "cx": sx, "cy": sy, "r": int(np.hypot(ex - sx, ey - sy))}
                    start_pt = None
                    status_msg = f"ROI drawn for {_target_btn()}"

        window_name = "ROI Selector All Buttons"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, _mouse_cb)
        while True:
            canvas = frame.copy()
            cv2.putText(canvas, "Draw all button ROIs in one window, then Lock All", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(canvas, f"Current button: {_target_btn()}  |  Enter=Lock All  Esc=Cancel  Backspace=Undo poly point", (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cv2.putText(canvas, status_msg, (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
            _draw_buttons(canvas)

            for b in BUTTON_ORDER:
                if b in selected:
                    _draw_roi(canvas, selected[b], color=(255, 0, 0), label=b)

            if shape_mode == "poly" and poly_pts:
                pts = np.array(poly_pts, dtype=np.int32)
                cv2.polylines(canvas, [pts], False, (0, 255, 255), 1)
                for p in poly_pts:
                    cv2.circle(canvas, p, 2, (0, 255, 255), -1)
            elif start_pt is not None and drag_pt is not None and shape_mode in ("rect", "circle"):
                sx, sy = start_pt
                ex, ey = drag_pt
                preview = {"shape": "rect", "x": min(sx, ex), "y": min(sy, ey), "w": abs(ex - sx), "h": abs(ey - sy)} if shape_mode == "rect" else {"shape": "circle", "cx": sx, "cy": sy, "r": int(np.hypot(ex - sx, ey - sy))}
                _draw_roi(canvas, preview, color=(0, 255, 255), label=_target_btn())
            elif working_roi is not None:
                _draw_roi(canvas, working_roi, color=(0, 255, 255), label=_target_btn())

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                if _try_lock_all():
                    break
            elif key == 27:
                return False
            elif key in (8, 127) and shape_mode == "poly" and poly_pts:
                poly_pts.pop()
                working_roi = None
                status_msg = f"{_target_btn()} polygon points: {len(poly_pts)}"

        button_rois = {b: _sanitize_roi(selected[b], frame.shape) for b in BUTTON_ORDER}
        button_color_baselines = {b: _compute_button_color_stats(frame, button_rois[b], b) for b in BUTTON_ORDER}
        button_roi_locked = True

        bounds = [_roi_bounds(r, frame.shape) for r in button_rois.values()]
        xs = [r[0] for r in bounds]
        ys = [r[1] for r in bounds]
        x2 = [r[0] + r[2] for r in bounds]
        y2 = [r[1] + r[3] for r in bounds]
        locked_roi = (min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys))
        roi_locked = True
        return True

    def _ensure_cycle_video(golden_img, run_id):
        nonlocal cycle_video_path, cycle_video_writer, cycle_video_started
        if cycle_video_writer is not None:
            return True
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cycle_video_path = os.path.join(CAMERA_OUTPUT_DIR, "video", f"cycle_inspection_{run_id}_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cycle_video_writer = cv2.VideoWriter(cycle_video_path, fourcc, max(1, CAMERA_FPS), (CAMERA_WIDTH, CAMERA_HEIGHT))
        if not cycle_video_writer.isOpened():
            cycle_video_writer = None
            cycle_video_path = None
            return False
        golden_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        intro_src = golden_img.copy()
        if button_roi_locked and button_rois:
            for _btn, _roi in button_rois.items():
                _draw_roi(intro_src, _roi, color=(255, 0, 0), label=_btn)
        elif roi_locked and locked_roi is not None:
            x, y, w, h = locked_roi
            cv2.rectangle(intro_src, (x, y), (x + w, y + h), (255, 0, 0), 2)
        intro = _stamp(intro_src, f"GOLDEN | run={run_id} | {golden_tag}")
        cycle_video_writer.write(intro)
        cycle_video_started = True
        return True

    def _append_cycle_video_frame(frame, cycle_num):
        if cycle_video_writer is None:
            return
        tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stamped = _stamp(frame, f"CYCLE {cycle_num} | {tag}", color=(0, 255, 0))
        cycle_video_writer.write(stamped)

    def save_capture_frame(frame, capture_type, run_id, cycle_num=0):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if capture_type == "golden":
            out_name = f"golden_{run_id}.png"
            out_path = os.path.join(CAMERA_OUTPUT_DIR, "golden", out_name)
            if os.path.exists(out_path):
                out_name = f"golden_{run_id}_{ts}.png"
                out_path = os.path.join(CAMERA_OUTPUT_DIR, "golden", out_name)
        elif capture_type == "cyc":
            out_name = f"cycle_{cycle_num}_{ts}.png"
            out_path = os.path.join(CAMERA_OUTPUT_DIR, "cyc", out_name)
        elif capture_type == "anomaly":
            out_name = f"frame_anamoly_{cycle_num}_{ts}.png"
            out_path = os.path.join(CAMERA_OUTPUT_DIR, "anomaly", out_name)
        else:
            out_name = f"{capture_type}_{cycle_num}_{ts}.png"
            out_path = os.path.join(CAMERA_OUTPUT_DIR, out_name)
        ok = cv2.imwrite(out_path, frame)
        return ok, out_name, out_path

    def run_basic_inspection(golden, cyc, run_id, cycle_num):
        nonlocal locked_roi, button_rois, button_color_baselines, button_roi_locked
        g_src = golden
        c_src = cyc
        if roi_locked and locked_roi is not None:
            x, y, w, h = locked_roi
            h_img, w_img = golden.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            locked_roi = (x, y, w, h)
            g_src = golden[y:y + h, x:x + w]
            c_src = cyc[y:y + h, x:x + w]

        g_gray = cv2.cvtColor(g_src, cv2.COLOR_BGR2GRAY)
        c_gray = cv2.cvtColor(c_src, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_norm = clahe.apply(g_gray)
        c_norm = clahe.apply(c_gray)

        g_blur = cv2.GaussianBlur(g_norm, (5, 5), 0)
        c_blur = cv2.GaussianBlur(c_norm, (5, 5), 0)

        diff = cv2.absdiff(g_blur, c_blur)
        score = float(np.mean(diff))
        _, mask = cv2.threshold(diff, INSPECTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        edge = max(0, int(INSPECTION_EDGE_IGNORE_PX))
        if edge > 0 and mask.shape[0] > 2 * edge and mask.shape[1] > 2 * edge:
            mask[:edge, :] = 0
            mask[-edge:, :] = 0
            mask[:, :edge] = 0
            mask[:, -edge:] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with state_lock:
            contour_enabled = state.detect_contour_enabled
            white_enabled = state.detect_white_ratio_enabled

        defect_found = False
        defect_rect = None
        max_area = 0.0
        max_bbox = (0, 0)
        ratio_fail_buttons = []
        ratio_detail = {}
        ratio_drop_by_button = {b: 0.0 for b in BUTTON_ORDER}

        contour_fail = False
        if contour_enabled:
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area <= INSPECTION_MIN_DEFECT_AREA:
                    continue
                rx, ry, rw, rh = cv2.boundingRect(contour)
                if rw < INSPECTION_MIN_DEFECT_W or rh < INSPECTION_MIN_DEFECT_H:
                    continue
                if area > max_area:
                    max_area = area
                    max_bbox = (rw, rh)
                    contour_fail = True
                    defect_rect = (rx, ry, rw, rh)

        white_fail = False
        worst_ratio_drop_pct = 0.0
        worst_ratio_btn = ""
        if white_enabled and button_roi_locked and button_rois and button_color_baselines:
            for btn, roi in button_rois.items():
                base = button_color_baselines.get(btn)
                if not base:
                    continue
                cur = _compute_button_color_stats(cyc, roi, btn)
                base_ratio = float(base.get("white_to_color", 0.0))
                cur_ratio = float(cur.get("white_to_color", 0.0))
                ratio_drop = (base_ratio - cur_ratio) / max(base_ratio, 1e-6) if base_ratio > 0 else 0.0
                ratio_drop_pct = max(0.0, ratio_drop * 100.0)
                ratio_drop_by_button[btn] = ratio_drop_pct
                if ratio_drop_pct >= worst_ratio_drop_pct:
                    worst_ratio_drop_pct = ratio_drop_pct
                    worst_ratio_btn = btn
                low_color = cur.get("color_px", 0) < BUTTON_MIN_COLOR_PIXELS
                ratio_detail[btn] = f"{base_ratio:.3f}->{cur_ratio:.3f} drop={ratio_drop_pct:.1f}%"
                if ratio_drop > BUTTON_WHITE_RATIO_DROP_PCT or low_color:
                    ratio_fail_buttons.append(btn)
            white_fail = len(ratio_fail_buttons) > 0

        defect_found = contour_fail or white_fail

        verdict = "FAIL" if defect_found else "PASS"
        decision_logic = (
            f"Enabled detectors -> contour={'ON' if contour_enabled else 'OFF'}, white_ratio={'ON' if white_enabled else 'OFF'}; "
            f"contour gate: diff>{INSPECTION_DIFF_THRESHOLD}, area>{INSPECTION_MIN_DEFECT_AREA}, "
            f"bbox>={INSPECTION_MIN_DEFECT_W}x{INSPECTION_MIN_DEFECT_H}; "
            f"white gate: drop>{int(BUTTON_WHITE_RATIO_DROP_PCT*100)}%"
        )
        if contour_fail and white_fail:
            failed_metric = f"contour+white_ratio:{','.join(ratio_fail_buttons)}"
        elif white_fail:
            failed_metric = f"white_ratio_drop:{','.join(ratio_fail_buttons)}"
        elif contour_fail:
            failed_metric = "contour_gate_triggered"
        else:
            failed_metric = "none"
        score_role = "informational_mean_pixel_diff" if contour_enabled else "white_ratio_change_pct"
        score_value = score if contour_enabled else worst_ratio_drop_pct
        decision_trace = {
            "decision_logic": decision_logic,
            "failed_metric": failed_metric,
            "max_contour_area": f"{max_area:.2f}",
            "max_bbox": f"{max_bbox[0]}x{max_bbox[1]}",
            "score_role": score_role,
            "button_ratio": "; ".join([f"{k}:{v}" for k, v in ratio_detail.items()]),
            "button_drop_pct": "|".join([f"{b}:{ratio_drop_by_button.get(b, 0.0):.2f}" for b in BUTTON_ORDER]),
            "white_ratio_change_pct": f"{worst_ratio_drop_pct:.2f}",
            "white_ratio_button": worst_ratio_btn,
            "display_score": f"{score_value:.2f}",
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mask_path = os.path.join(masks_dir, f"inspection_mask_{cycle_num}_{ts}.png")
        diff_path = os.path.join(masks_dir, f"inspection_diff_{cycle_num}_{ts}.png")
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(diff_path, diff)

        disp = cyc.copy()
        if contour_enabled:
            metric_txt = f"Mean pixel diff:{score:.2f}"
        else:
            suffix = f" ({worst_ratio_btn})" if worst_ratio_btn else ""
            fail_btn_txt = f" failBtn:{','.join(ratio_fail_buttons)}" if ratio_fail_buttons else ""
            metric_txt = f"White ratio change:{worst_ratio_drop_pct:.2f}%{suffix}{fail_btn_txt}"
        label = f"Cycle:{cycle_num} Verdict:{verdict} {metric_txt}"
        (_, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        tx = 20
        ty = max(th + 72, 72)
        cv2.putText(disp, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if verdict == "PASS" else (0, 0, 255), 2)

        if defect_rect is not None:
            rx, ry, rw, rh = defect_rect
            if roi_locked and locked_roi is not None:
                ox, oy, _, _ = locked_roi
                cv2.rectangle(disp, (ox + rx, oy + ry), (ox + rx + rw, oy + ry + rh), (0, 0, 255), 2)
            else:
                cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

        if button_roi_locked and button_rois:
            for _btn, _roi in button_rois.items():
                _draw_roi(disp, _roi, color=(255, 0, 0), label=_btn)
        elif roi_locked and locked_roi is not None:
            x, y, w, h = locked_roi
            cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ok_a, _, anomaly_path = save_capture_frame(disp, "anomaly", run_id, cycle_num)
        if not ok_a:
            anomaly_path = ""

        score_out = score if contour_enabled else worst_ratio_drop_pct
        return verdict, score_out, mask_path, anomaly_path, disp, decision_trace

    def _auto_capture_cycle(cycle_num):
        nonlocal last_capture_frame, last_capture_path, golden_frame, golden_path
        run_id = state.test_name.strip() or "test_report"
        with state_lock:
            retries = max(0, int(state.auto_capture_retries))
            fail_policy = state.auto_fail_policy if state.auto_fail_policy in AUTO_FAIL_POLICY_OPTIONS else "safe_stop"

        def _log_fail(message, reason_code):
            _manifest_write({
                "run_id": run_id,
                "cycle": cycle_num,
                "capture_type": "cyc",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "camera_status": camera_status,
                "result": "FAIL",
                "message": message,
                "reason_code": reason_code,
                "file_path": "",
                "score": "",
                "threshold": f"thr>{INSPECTION_DIFF_THRESHOLD}|area>{INSPECTION_MIN_DEFECT_AREA}|wh>={INSPECTION_MIN_DEFECT_W}x{INSPECTION_MIN_DEFECT_H}",
                "verdict": "FAIL",
                "golden_path": golden_path or "",
                "video_path": cycle_video_path or "",
                "anomaly_path": "",
                "policy": fail_policy,
            })

        ok_ckpt = False
        for _ in range(retries + 1):
            ok_ckpt = go_ic_home_checkpoint()
            if ok_ckpt:
                break
            time.sleep(0.1)
        if not ok_ckpt:
            with state_lock:
                state.last_capture_result = f"cycle {cycle_num}: checkpoint failed"
            _log_fail("checkpoint_failed", "checkpoint_failed")
            return False

        time.sleep(IC_CAPTURE_SETTLE_S)

        frame = None
        frames_used = 0
        for _ in range(retries + 1):
            frame, frames_used = capture_average_frame()
            if frame is not None:
                break
            _recover_camera_preview()
            time.sleep(0.1)
        if frame is None:
            with state_lock:
                state.last_capture_result = f"cycle {cycle_num}: avg no frame ({frames_used}/{CAPTURE_AVG_MIN_FRAMES})"
            _log_fail("no_camera_frame", "avg_no_camera_frame")
            return False

        capture_type = "cyc"

        ok, out_name, out_path = save_capture_frame(frame, capture_type, run_id, cycle_num)
        if not ok:
            with state_lock:
                state.last_capture_result = f"cycle {cycle_num}: save failed"
            _log_fail("save_failed", "save_failed")
            return False

        last_capture_frame = frame.copy()
        last_capture_path = out_path

        verdict = "WARN"
        score = ""
        anomaly_path = ""
        decision_trace = {"decision_logic": "", "failed_metric": "", "max_contour_area": "", "max_bbox": "", "score_role": "", "button_drop_pct": "", "white_ratio_button": "", "white_ratio_change_pct": ""}
        msg = "captured"

        if capture_type == "golden":
            golden_frame = frame.copy()
            golden_path = out_path
            with state_lock:
                state.golden_ready = True
            verdict = "GOLDEN"
            msg = "golden_ready"
            _ensure_cycle_video(golden_frame, run_id)
            reason_code = "golden_initialized"
        elif golden_frame is not None:
            verdict, score, _mask_path, anomaly_path, disp, decision_trace = run_basic_inspection(golden_frame, frame, run_id, cycle_num)
            _ensure_cycle_video(golden_frame, run_id)
            _append_cycle_video_frame(disp, cycle_num)
            msg = "inspection_done"
            reason_code = "inspection_scored"
        else:
            verdict = "WARN"
            msg = "golden_missing"
            reason_code = "golden_missing"

        rec = {
            "run_id": run_id,
            "cycle": cycle_num,
            "capture_type": capture_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "camera_status": camera_status,
            "result": "OK",
            "message": msg,
            "reason_code": reason_code,
            "file_path": out_path,
            "score": f"{score:.2f}" if score not in ("", None) else "",
            "threshold": f"thr>{INSPECTION_DIFF_THRESHOLD}|area>{INSPECTION_MIN_DEFECT_AREA}|wh>={INSPECTION_MIN_DEFECT_W}x{INSPECTION_MIN_DEFECT_H}",
            "verdict": verdict,
            "golden_path": golden_path or "",
            "video_path": cycle_video_path or "",
            "anomaly_path": anomaly_path,
            "policy": fail_policy,
            "decision_logic": decision_trace.get("decision_logic", ""),
            "failed_metric": decision_trace.get("failed_metric", ""),
            "max_contour_area": decision_trace.get("max_contour_area", ""),
            "max_bbox": decision_trace.get("max_bbox", ""),
            "score_role": decision_trace.get("score_role", ""),
            "button_drop_pct": decision_trace.get("button_drop_pct", ""),
            "white_ratio_button": decision_trace.get("white_ratio_button", ""),
            "white_ratio_change_pct": decision_trace.get("white_ratio_change_pct", ""),
        }
        _manifest_write(rec)
        inspection_records.append(dict(rec, anomaly_path=anomaly_path))
        _record_anomaly_stats(cycle_num, verdict, score)
        with state_lock:
            state.last_capture_result = f"cycle {cycle_num}: {capture_type}/{verdict}"
        set_alert("#2563eb", f"Auto capture cycle {cycle_num}: {capture_type}/{verdict}")

        go_home()
        if not wait_until_idle():
            return False
        with state_lock:
            state.running = True
            state.paused = False
            state.manual_mode_active = False
            state.manual_intervention_requested = False
            if state.capture_every_x_cycles > 0:
                state.next_auto_capture_cycle = state.cycle_count + state.capture_every_x_cycles
        return True

    start_camera_preview()

    # --- GUI layout (same as Stage D force window v5) ---
    fig = plt.figure(figsize=(14, 8), facecolor=COLOR_BG)

    # Determine app logo once and reuse it in GUI + PDF report
    app_logo_path = None
    for logo_path in LOGO_CANDIDATE_PATHS:
        if os.path.exists(logo_path):
            app_logo_path = logo_path
            break

    # left panel background sections (professional grouping)
    fig.patches.append(Rectangle((0.02, 0.0628), 0.2076, 0.9098, transform=fig.transFigure,
                                 facecolor=COLOR_PANEL, edgecolor=COLOR_PANEL_BORDER, linewidth=1.0, zorder=-1))

    # Plot area (force + camera in same matplotlib window)
    force_ax_x, force_ax_y, force_ax_w = 0.30, 0.25, 0.32
    force_ax_h = force_ax_w * 9.0 / 16.0
    ax = fig.add_axes([force_ax_x, force_ax_y, force_ax_w, force_ax_h])
    ax.set_ylim(-0.2, max(2.0, state.force_max + 1.5))
    ax.set_ylabel("Force (lbs)", fontsize=22, color="white")
    ax.set_xlabel("Time (s)", fontsize=16, color="white")
    ax.set_title("Live Force (Last 10s)", fontsize=26, color=COLOR_TEXT)
    ax.set_facecolor("#f8fafc")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.grid(True, alpha=0.35)
    force_band = ax.axhspan(state.force_min, state.force_max, alpha=0.18, color="#93c5fd")

    (line,) = ax.plot([], [], linewidth=2.5, color="#0ea5e9")

    # Camera pane (same window as force graph)
    ax_cam = fig.add_axes([0.65, 0.40, 0.32, 0.50])
    ax_cam.set_title("IC Camera", fontsize=20, color=COLOR_TEXT)
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    ax_cam.set_facecolor("#0b1220")
    cam_placeholder = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    camera_im = ax_cam.imshow(cam_placeholder)

    # Messages (no bbox), keep two-line spacing at ~1.5 lines
    status_line_y = 0.095
    line_spacing_y = ((1.5 * 13.0) / 72.0) / 8.0  # 1.5 lines at 13pt on 8in figure
    fail_line_y = status_line_y - line_spacing_y
    msg_text_x = 0.31
    status_line = fig.text(msg_text_x, status_line_y, "", fontsize=13, color=COLOR_TEXT)
    param_line  = fig.text(msg_text_x, fail_line_y, "", fontsize=13, color=COLOR_TEXT)
    fail_line_1 = fig.text(msg_text_x, fail_line_y, "", fontsize=11, color="#fbbf24")
    fail_line_2 = fig.text(msg_text_x, fail_line_y, "", fontsize=11, color="#fca5a5")

    # status indicator: dedicated square axes keeps a true circle and aligns with status text line
    status_diameter = 0.03  # 1.5x previous 0.02 diameter
    status_ax = fig.add_axes([0.282, status_line_y - 0.0063, status_diameter, status_diameter])
    status_ax.set_aspect('equal')
    status_ax.axis('off')
    status_dot = Circle((0.5, 0.5), 0.45, transform=status_ax.transAxes, facecolor="gray", edgecolor="black")
    status_ax.add_patch(status_dot)

    # Bottom-right logo (same anchor style as gui_test), enlarged 1.5x and nudged left/up
    if app_logo_path:
        try:
            force_ax_x, force_ax_y, force_ax_w = 0.30, 0.25, 0.67
            _force_ax_h = force_ax_w * 9.0 / 16.0
            logo_w, logo_h = 0.297, 0.0432  # 1.5x current size while preserving aspect ratio
            logo_x = (force_ax_x + force_ax_w) - logo_w + (1.0 / 14.0) - ((3.0 / 25.4) / 14.0)
            logo_y = 0.01 + ((1.0 / 25.4) / 8.0)
            ax_logo = fig.add_axes([logo_x, logo_y, logo_w, logo_h])
            ax_logo.imshow(mpimg.imread(app_logo_path))
            ax_logo.axis("off")
        except Exception:
            pass
    else:
        fig.text(0.035, 0.95, "ROBOSPEED", fontsize=16, color=COLOR_TEXT, weight="bold")

    # ---- LEFT PANEL ----
    y_top = 0.875
    dy = 0.062

    fig.text(0.0594, y_top + 0.063, "Run Controls", color="#e2e8f0", fontsize=12, weight="bold", zorder=5)
    fig.text(0.0594, y_top - 7.09*dy + 0.02, "Settings", color="#e2e8f0", fontsize=12, weight="bold", zorder=5)

    btn_start = Button(fig.add_axes([0.0594, y_top, 0.14, 0.05]), "Start", color="#22c55e", hovercolor="#16a34a")
    btn_pause = Button(fig.add_axes([0.0594, y_top - dy, 0.14, 0.05]), "Pause", color="#f59e0b", hovercolor="#d97706")
    btn_stop = Button(fig.add_axes([0.0594, y_top - 2*dy, 0.14, 0.05]), "Stop", color="#ef4444", hovercolor="#dc2626")
    btn_home = Button(fig.add_axes([0.0594, y_top - 3*dy, 0.14, 0.05]), "Home", color="#64748b", hovercolor="#475569")
    btn_reset = Button(fig.add_axes([0.0594, y_top - 4*dy, 0.14, 0.05]), "Reset", color="#94a3b8", hovercolor="#64748b")
    btn_report = Button(fig.add_axes([0.0594, y_top - 5*dy, 0.14, 0.05]), "View Report", color="#3b82f6", hovercolor="#2563eb")
    btn_exit = Button(fig.add_axes([0.0594, y_top - 6*dy, 0.14, 0.05]), "Exit", color="#b91c1c", hovercolor="#991b1b")

    tb_w = 0.06174
    tb_x = (0.0594 + 0.14) - tb_w
    tb_y_shift = 0.005
    tb_vel = TextBox(fig.add_axes([tb_x, y_top - 7.98*dy - tb_y_shift + 0.02, tb_w, 0.05]), "Vel (0-1000)", initial=str(state.vel))
    tb_acc = TextBox(fig.add_axes([tb_x, y_top - 8.98*dy - tb_y_shift + 0.02, tb_w, 0.05]), "Acc(0-2000)", initial=str(state.acc))
    tb_jerk = TextBox(fig.add_axes([tb_x, y_top - 9.98*dy - tb_y_shift + 0.02, tb_w, 0.05]), "Jerk(0-10,000)", initial=str(state.jerk))
    tb_cyc = TextBox(fig.add_axes([tb_x, y_top - 10.97*dy - tb_y_shift + 0.02, tb_w, 0.05]), "Cycles", initial=str(state.target_cycles))
    tb_base = TextBox(fig.add_axes([tb_x, y_top - 11.97*dy - tb_y_shift + 0.02, tb_w, 0.05]), "Baseline", initial=str(state.baseline_cycles))
    force_row_y = y_top - 12.97*dy - tb_y_shift + 0.02
    force_box_gap = 0.006
    force_box_w = (tb_w - force_box_gap) / 2
    fmin_x = tb_x
    fmax_x = tb_x + force_box_w + force_box_gap
    tb_fmin = TextBox(fig.add_axes([fmin_x, force_row_y, force_box_w, 0.05]), "Force", initial=str(state.force_min))
    tb_fmax = TextBox(fig.add_axes([fmax_x, force_row_y, force_box_w, 0.05]), "", initial=str(state.force_max))
    mm_to_fig_y = (1.0 / 25.4) / 8.0
    manual_btn_h = 0.033
    manual_btn_gap = 0.004

    # Two-row manual/camera controls above camera pane
    camera_ax_x, camera_ax_y, camera_ax_w, camera_ax_h = 0.65, 0.40, 0.32, 0.50
    # Keep control rows above force/camera top edge
    force_ax_top = force_ax_y + force_ax_h
    row2_y = force_ax_top + 0.004
    row1_y = row2_y + manual_btn_h + manual_btn_gap
    manual_btn_w = (camera_ax_w - (4 * manual_btn_gap)) / 5

    fig.text(camera_ax_x, row1_y + manual_btn_h + 0.006, "Manual IC / Camera", color="#e2e8f0", fontsize=11, weight="bold", zorder=5)

    x0 = camera_ax_x
    # Row 1: IC Home, Camera Tune, Golden Capture, Image Capture, Run Inspection
    btn_ic_home = Button(fig.add_axes([x0 + 0 * (manual_btn_w + manual_btn_gap), row1_y, manual_btn_w, manual_btn_h]), "IC Home", color="#0ea5e9", hovercolor="#0284c7")
    btn_camera_tune = Button(fig.add_axes([x0 + 1 * (manual_btn_w + manual_btn_gap), row1_y, manual_btn_w, manual_btn_h]), "Camera Tune", color="#16a34a", hovercolor="#15803d")
    btn_golden_capture = Button(fig.add_axes([x0 + 2 * (manual_btn_w + manual_btn_gap), row1_y, manual_btn_w, manual_btn_h]), "Golden Capture", color="#d97706", hovercolor="#b45309")
    btn_image_capture = Button(fig.add_axes([x0 + 3 * (manual_btn_w + manual_btn_gap), row1_y, manual_btn_w, manual_btn_h]), "Image Capture", color="#2563eb", hovercolor="#1d4ed8")
    btn_run_inspection = Button(fig.add_axes([x0 + 4 * (manual_btn_w + manual_btn_gap), row1_y, manual_btn_w, manual_btn_h]), "Run Inspection", color="#7c3aed", hovercolor="#6d28d9")

    # Row 2: Return to Test, Re-tare, Tare@Start ON/OFF
    btn_return_test = Button(fig.add_axes([x0 + 0 * (manual_btn_w + manual_btn_gap), row2_y, manual_btn_w, manual_btn_h]), "Return to Test", color="#0891b2", hovercolor="#0e7490")
    btn_re_tare = Button(fig.add_axes([x0 + 1 * (manual_btn_w + manual_btn_gap), row2_y, manual_btn_w, manual_btn_h]), "Re-tare", color="#475569", hovercolor="#334155")
    btn_tare_on_start = Button(fig.add_axes([x0 + 2 * (manual_btn_w + manual_btn_gap), row2_y, manual_btn_w, manual_btn_h]), "Tare@Start: ON", color="#0f766e", hovercolor="#115e59")
    btn_auto_cap = Button(fig.add_axes([x0 + 3 * (manual_btn_w + manual_btn_gap), row2_y, manual_btn_w, manual_btn_h]), "AutoCap: OFF", color="#1d4ed8", hovercolor="#1e40af")

    # Row 3: Anomaly detector toggles
    row3_y = row2_y - (manual_btn_h + manual_btn_gap)
    btn_detect_contour = Button(fig.add_axes([x0 + 0 * (manual_btn_w + manual_btn_gap), row3_y, manual_btn_w, manual_btn_h]), "Contour: ON", color="#166534", hovercolor="#15803d")
    btn_detect_white = Button(fig.add_axes([x0 + 1 * (manual_btn_w + manual_btn_gap), row3_y, manual_btn_w, manual_btn_h]), "WhiteRatio: ON", color="#166534", hovercolor="#15803d")
    btn_lock_roi = Button(fig.add_axes([x0 + 2 * (manual_btn_w + manual_btn_gap), row3_y, manual_btn_w, manual_btn_h]), "Lock ROI", color="#0f766e", hovercolor="#115e59")

    # Auto-capture settings/status panel
    px_x = 1.0 / (fig.get_figwidth() * fig.dpi)
    px_y = 1.0 / (fig.get_figheight() * fig.dpi)
    content_shift_x = 8.0 * px_x
    panel_pad_x = 3.0 * px_x
    panel_pad_y = 3.0 * px_y

    auto_panel_bottom = 0.25
    auto_panel_top = camera_ax_y - 0.01
    content_x = camera_ax_x + content_shift_x + panel_pad_x

    auto_msg_font = 8.8
    auto_row_h = ((1.5 * auto_msg_font) / 72.0) / fig.get_figheight()
    auto_tb_w = tb_w  # same width as Baseline box
    auto_tb_h = 0.03
    title_y = auto_panel_top - (2.2 * auto_row_h)
    cap_label_y = title_y - (2 * auto_row_h)
    cap_box_y = cap_label_y - auto_tb_h - 0.003
    retry_label_x = content_x + auto_tb_w + 0.012
    fig.text(content_x, title_y, "Auto Capture Settings / Camera Status", color="#e2e8f0", fontsize=10, weight="bold", zorder=5)
    fig.text(content_x, cap_label_y, "Cap every", color="#e2e8f0", fontsize=10, zorder=5)
    fig.text(retry_label_x, cap_label_y, "Retries", color="#e2e8f0", fontsize=10, zorder=5)
    tb_cap_every = TextBox(fig.add_axes([content_x, cap_box_y, auto_tb_w, auto_tb_h]), "", initial=str(state.capture_every_x_cycles))
    tb_cap_retry = TextBox(fig.add_axes([retry_label_x, cap_box_y, auto_tb_w, auto_tb_h]), "", initial=str(state.auto_capture_retries))
    btn_fail_policy = Button(fig.add_axes([retry_label_x + auto_tb_w + 0.012, cap_box_y, auto_tb_w + 0.02, auto_tb_h]), "Fail:STOP", color="#991b1b", hovercolor="#7f1d1d")

    msg_row_1 = cap_box_y - (1.3 * auto_row_h)
    auto_status_1 = fig.text(content_x, msg_row_1 - (0 * auto_row_h), "", fontsize=auto_msg_font, color="#e2e8f0")
    auto_status_2 = fig.text(content_x, msg_row_1 - (1 * auto_row_h), "", fontsize=auto_msg_font, color="#e2e8f0")
    auto_status_3 = fig.text(content_x, msg_row_1 - (2 * auto_row_h), "", fontsize=auto_msg_font, color="#e2e8f0")
    auto_status_4 = fig.text(content_x, msg_row_1 - (3 * auto_row_h), "", fontsize=auto_msg_font, color="#e2e8f0")
    auto_status_5 = fig.text(content_x, msg_row_1 - (4 * auto_row_h), "", fontsize=auto_msg_font, color="#e2e8f0")

    content_right = retry_label_x + auto_tb_w + 0.012 + (auto_tb_w + 0.02)
    content_top = title_y + auto_row_h
    content_bottom = msg_row_1 - (4 * auto_row_h)
    panel_x = content_x - panel_pad_x
    panel_y = content_bottom - panel_pad_y
    panel_w = (content_right - content_x) + (2 * panel_pad_x)
    panel_h = (content_top - content_bottom) + (2 * panel_pad_y)
    fig.patches.append(Rectangle((panel_x, panel_y), panel_w, panel_h,
                                 transform=fig.transFigure, facecolor="#0b1220", edgecolor=COLOR_PANEL_BORDER, linewidth=1.0, zorder=-0.5))

    for _btn in [btn_start, btn_pause, btn_stop, btn_home, btn_reset, btn_exit, btn_report, btn_tare_on_start, btn_auto_cap, btn_fail_policy,
                 btn_ic_home, btn_return_test, btn_camera_tune, btn_golden_capture, btn_image_capture, btn_run_inspection, btn_re_tare, btn_detect_contour, btn_detect_white, btn_lock_roi]:
        _btn.label.set_color("white")
        _btn.label.set_fontsize(8 if _btn in [btn_ic_home, btn_return_test, btn_camera_tune, btn_golden_capture, btn_image_capture, btn_run_inspection, btn_re_tare, btn_detect_contour, btn_detect_white, btn_lock_roi, btn_tare_on_start, btn_auto_cap, btn_fail_policy] else 12)

    for _tb in [tb_vel, tb_acc, tb_jerk, tb_cyc, tb_base]:
        _tb.label.set_color("white")
        _tb.label.set_horizontalalignment("left")
        _tb.label.set_position((-1.27, 0.5))
    for _tb in [tb_fmin, tb_fmax, tb_cap_every, tb_cap_retry]:
        _tb.label.set_color("white")
    tb_fmin.label.set_horizontalalignment("left")
    force_label_x = -1.27 * (tb_w / force_box_w)
    tb_fmin.label.set_position((force_label_x, 0.5))

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def set_alert(color, msg):
        with state_lock:
            state.alert_color = color
            state.alert_msg = msg
            state.alert_until_wall = time.time() + ALERT_FLASH_S

    def update_tare_toggle_button():
        with state_lock:
            enabled = state.tare_on_start
        if enabled:
            btn_tare_on_start.label.set_text("Tare@Start: ON")
            btn_tare_on_start.ax.set_facecolor("#0f766e")
        else:
            btn_tare_on_start.label.set_text("Tare@Start: OFF")
            btn_tare_on_start.ax.set_facecolor("#7f1d1d")

    def update_auto_cap_button():
        with state_lock:
            enabled = state.auto_capture_enabled and state.capture_every_x_cycles > 0
            every = state.capture_every_x_cycles
        if enabled:
            btn_auto_cap.label.set_text(f"AutoCap:{every}")
            btn_auto_cap.ax.set_facecolor("#166534")
        else:
            btn_auto_cap.label.set_text("AutoCap: OFF")
            btn_auto_cap.ax.set_facecolor("#1d4ed8")

    def update_fail_policy_button():
        with state_lock:
            pol = state.auto_fail_policy if state.auto_fail_policy in AUTO_FAIL_POLICY_OPTIONS else "safe_stop"
        if pol == "continue":
            btn_fail_policy.label.set_text("Fail:CONT")
            btn_fail_policy.ax.set_facecolor("#92400e")
        else:
            btn_fail_policy.label.set_text("Fail:STOP")
            btn_fail_policy.ax.set_facecolor("#991b1b")

    def update_detector_buttons():
        with state_lock:
            contour_on = state.detect_contour_enabled
            white_on = state.detect_white_ratio_enabled
        btn_detect_contour.label.set_text("Contour: ON" if contour_on else "Contour: OFF")
        btn_detect_contour.ax.set_facecolor("#166534" if contour_on else "#7f1d1d")
        btn_detect_white.label.set_text("WhiteRatio: ON" if white_on else "WhiteRatio: OFF")
        btn_detect_white.ax.set_facecolor("#166534" if white_on else "#7f1d1d")

    def go_a_above():
        print("[Robot] Going to A-above for tare")
        robot.play(-1, {
            "cmd": "jmove", "rel": 0,
            "vel": SAFE_START_VEL,
            "acc": SAFE_START_ACC,
            "jerk": SAFE_START_JERK,
            **BUTTON_POSES["A"]["above"]
        })

    def perform_tare(reason):
        nonlocal zero_offset
        with state_lock:
            if state.tare_in_progress:
                set_alert("#475569", "Tare already in progress")
                return False
            state.tare_in_progress = True
        try:
            set_alert("#475569", f"Tare warm-up ({TARE_WARMUP_S:.0f}s): {reason}")
            time.sleep(TARE_WARMUP_S)
            samples = [bridge.getVoltageRatio() for _ in range(TARE_SAMPLES)]
            zero_offset = float(np.mean(samples))
            set_alert("green", f"Tare complete ({reason})")
            print(f"[Force] Tare complete ({reason}) zero_offset={zero_offset:.8f}")
            return True
        except Exception as exc:
            set_alert("red", f"Tare failed: {exc}")
            print(f"[Force] Tare failed: {exc}")
            return False
        finally:
            with state_lock:
                state.tare_in_progress = False

    def go_home():
        if not robot_connected:
            return False
        print("[Robot] Going Home")
        robot.play(-1, {
            "cmd": "jmove", "rel": 0,
            "vel": SAFE_START_VEL,
            "acc": SAFE_START_ACC,
            "jerk": SAFE_START_JERK,
            **HOME_POSE
        })
        return True

    def wait_until_idle(timeout_s=20.0):
        if not robot_connected:
            return False
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if is_idle(robot):
                return True
            time.sleep(0.02)
        return False

    def go_ic_home_checkpoint():
        if not robot_connected:
            return False
        go_home()
        if not wait_until_idle():
            print("[Robot] Timeout waiting to reach Home before IC clear")
            return False

        print(f"[Robot] IC clear move: rel j0={IC_CLEAR_J0_REL}")
        robot.play(-1, {
            "cmd": "jmove", "rel": 1,
            "vel": SAFE_START_VEL,
            "acc": SAFE_START_ACC,
            "jerk": IC_CLEAR_J0_JERK,
            "j0": IC_CLEAR_J0_REL
        })
        if not wait_until_idle():
            print("[Robot] Timeout waiting at IC checkpoint")
            return False
        return True

    # ✅ KEY FIX: apply current TextBox values on Start (even if on_submit didn't fire)
    def apply_textbox_values():
        def _parse_int(text, fallback):
            try:
                return int(float(text))
            except Exception:
                return fallback

        def _parse_float(text, fallback):
            try:
                return float(text)
            except Exception:
                return fallback

        with state_lock:
            v = _parse_int(tb_vel.text, state.vel)
            a = _parse_int(tb_acc.text, state.acc)
            j = _parse_int(tb_jerk.text, state.jerk)
            c = _parse_int(tb_cyc.text, state.target_cycles)
            b = _parse_int(tb_base.text, state.baseline_cycles)
            fmin = _parse_float(tb_fmin.text, state.force_min)
            fmax = _parse_float(tb_fmax.text, state.force_max)
            cap_every = _parse_int(tb_cap_every.text, state.capture_every_x_cycles)
            cap_retry = _parse_int(tb_cap_retry.text, state.auto_capture_retries)

            state.vel = clamp(v, VEL_MIN, VEL_MAX)
            state.acc = clamp(a, ACC_MIN, ACC_MAX)
            state.jerk = clamp(j, JERK_MIN, JERK_MAX)
            if c > 0:
                state.target_cycles = c
            new_base = clamp(b, 1, 500)
            if new_base != state.baseline_cycles:
                state.baseline_cycles = new_base
                state.baseline_peaks = {"A": [], "B": [], "C": [], "D": []}
                state.baseline_mean = {}
                state.baseline_ready = False
                state.image_capture_count = 0
            else:
                state.baseline_cycles = new_base

            if fmax > fmin:
                state.force_min = fmin
                state.force_max = fmax

            state.capture_every_x_cycles = max(0, cap_every)
            state.auto_capture_enabled = state.capture_every_x_cycles > 0
            state.next_auto_capture_cycle = state.capture_every_x_cycles if state.auto_capture_enabled else 0
            state.auto_capture_retries = clamp(cap_retry, 0, 10)


    def on_start(_evt):
        apply_textbox_values()
        if not robot_connected:
            with state_lock:
                state.running = False
                state.paused = True
                state.stopped = True
                state.manual_mode_active = True
            set_alert("#d97706", "Robot disconnected: visual inspection mode only")
            print("[GUI] Start blocked: robot disconnected (visual inspection mode)")
            return
        nonlocal last_capture_frame, last_capture_path, golden_frame, golden_path, locked_roi, roi_locked
        nonlocal button_rois, button_color_baselines, button_roi_locked
        nonlocal auto_report_written_cycle
        with state_lock:
            state.running = False
            state.paused = True
            state.stopped = False
            state.traj_index = 0
            state.cycle_count = 0
            state.aligned_to_A = False
            state.window_active = False
            state.window_button = None
            state.window_peak_force = None
            state.window_peak_time = None
            state.manual_intervention_requested = False
            state.manual_mode_active = False
            state.image_capture_count = 0
            state.golden_ready = False
            state.last_capture_result = "none"
            state.next_auto_capture_cycle = state.capture_every_x_cycles if state.capture_every_x_cycles > 0 else 0

        last_capture_frame = None
        last_capture_path = None
        golden_frame = None
        golden_path = None
        locked_roi = None
        roi_locked = False
        button_rois = {}
        button_color_baselines = {}
        button_roi_locked = False
        auto_report_written_cycle = -1

        print("[GUI] Start pressed -> Going Home before cycle start")
        go_home()
        if not wait_until_idle():
            with state_lock:
                state.running = False
                state.paused = False
                state.stopped = True
            set_alert("red", "Start failed: robot did not reach Home")
            print("[GUI] Start aborted: timeout waiting at Home")
            return

        with state_lock:
            tare_on_start = state.tare_on_start
        if tare_on_start:
            go_a_above()
            if not wait_until_idle():
                with state_lock:
                    state.running = False
                    state.paused = False
                    state.stopped = True
                set_alert("red", "Start failed: robot did not reach A-above for tare")
                print("[GUI] Start aborted: timeout waiting at A-above for tare")
                return
            if not perform_tare("start"):
                with state_lock:
                    state.running = False
                    state.paused = False
                    state.stopped = True
                print("[GUI] Start aborted: tare failed")
                return

        with state_lock:
            auto_mode = state.auto_capture_enabled and state.capture_every_x_cycles > 0

        if auto_mode:
            print("[GUI] Auto mode start: preparing golden at IC checkpoint")
            ok_ckpt = go_ic_home_checkpoint()
            if not ok_ckpt:
                with state_lock:
                    state.running = False
                    state.paused = True
                    state.stopped = True
                set_alert("red", "Auto start failed: could not reach IC checkpoint")
                return

            time.sleep(IC_CAPTURE_SETTLE_S)
            if not run_camera_auto_tune():
                with state_lock:
                    state.running = False
                    state.paused = True
                    state.stopped = True
                set_alert("red", "Auto start failed: camera tune failed")
                return

            frame, frames_used = capture_average_frame()
            if frame is None:
                with state_lock:
                    state.running = False
                    state.paused = True
                    state.stopped = True
                set_alert("red", f"Auto start failed: averaged frame unavailable ({frames_used}/{CAPTURE_AVG_MIN_FRAMES})")
                return

            run_id = state.test_name.strip() or "test_report"
            ok, out_name, out_path = save_capture_frame(frame, "golden", run_id, 0)
            if not ok:
                with state_lock:
                    state.running = False
                    state.paused = True
                    state.stopped = True
                set_alert("red", "Auto start failed: could not save golden")
                return

            if not _select_button_rois_and_calibrate(frame):
                with state_lock:
                    state.running = False
                    state.paused = True
                    state.stopped = True
                set_alert("red", "Auto start aborted: all button ROIs not selected")
                return

            golden_frame = frame.copy()
            golden_path = out_path
            _ensure_cycle_video(golden_frame, run_id)
            with state_lock:
                state.golden_ready = True

            _manifest_write({
                "run_id": run_id,
                "cycle": 0,
                "capture_type": "golden",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "camera_status": camera_status,
                "result": "OK",
                "message": "auto_start_golden_ready",
                "reason_code": "golden_initialized",
                "file_path": out_path,
                "score": "",
                "threshold": f"thr>{INSPECTION_DIFF_THRESHOLD}|area>{INSPECTION_MIN_DEFECT_AREA}|wh>={INSPECTION_MIN_DEFECT_W}x{INSPECTION_MIN_DEFECT_H}",
                "verdict": "GOLDEN",
                "golden_path": out_path,
                "video_path": cycle_video_path or "",
                "anomaly_path": "",
                "policy": state.auto_fail_policy,
            })

            go_home()
            if not wait_until_idle():
                with state_lock:
                    state.running = False
                    state.paused = True
                    state.stopped = True
                set_alert("red", "Auto start failed: unable to return home")
                return

        with state_lock:
            state.running = True
            state.paused = False
            state.stopped = False
            if state.capture_every_x_cycles > 0:
                state.next_auto_capture_cycle = state.capture_every_x_cycles
        set_alert("green", "At Home. Starting cycle test")

    def on_pause(_evt):
        with state_lock:
            state.running = False
            state.paused = True
        print("[GUI] Pause pressed")

    def on_stop(_evt):
        nonlocal last_saved_report_path, auto_report_written_cycle
        with state_lock:
            stop_cycle = state.cycle_count
            state.running = False
            state.paused = False
            state.stopped = True
            state.traj_index = 0
            state.aligned_to_A = False
            state.window_active = False
            state.window_button = None
            state.window_peak_force = None
            state.window_peak_time = None
            state.manual_intervention_requested = False
            state.manual_mode_active = False
            state.image_capture_count = 0
            state.last_capture_result = f"stopped_at_cycle_{stop_cycle}"
        if stop_cycle > 0 and auto_report_written_cycle != stop_cycle:
            try:
                path = _auto_report_path()
                build_report_pdf(path)
                last_saved_report_path = path
                auto_report_written_cycle = stop_cycle
                print(f"[Report] Auto-saved on stop: {path}")
            except Exception as exc:
                print(f"[Report] Auto-save on stop failed: {exc}")
        print("[GUI] Stop pressed -> Going Home")
        go_home()

    def on_home(_evt):
        if not robot_connected:
            set_alert("#d97706", "Robot disconnected: Home unavailable in visual mode")
            return
        go_home()

    def on_re_tare(_evt):
        with state_lock:
            if state.running:
                set_alert("#475569", "Re-tare blocked while running")
                print("[GUI] Re-tare blocked: running")
                return
        if not wait_until_idle(timeout_s=5.0):
            set_alert("#475569", "Re-tare blocked: robot not idle")
            print("[GUI] Re-tare blocked: robot not idle")
            return
        perform_tare("manual")

    def on_toggle_tare_on_start(_evt):
        with state_lock:
            state.tare_on_start = not state.tare_on_start
            enabled = state.tare_on_start
        update_tare_toggle_button()
        set_alert("#0f766e" if enabled else "#7f1d1d", f"Tare-on-start {'enabled' if enabled else 'disabled'}")

    def on_toggle_auto_cap(_evt):
        with state_lock:
            if state.capture_every_x_cycles <= 0:
                state.auto_capture_enabled = False
                msg = "Auto capture needs CapEvery > 0"
            else:
                state.auto_capture_enabled = not state.auto_capture_enabled
                if state.auto_capture_enabled:
                    state.next_auto_capture_cycle = state.cycle_count + state.capture_every_x_cycles
                msg = "Auto capture enabled" if state.auto_capture_enabled else "Auto capture disabled"
        update_auto_cap_button()
        set_alert("#1d4ed8", msg)

    def on_toggle_fail_policy(_evt):
        with state_lock:
            state.auto_fail_policy = "continue" if state.auto_fail_policy == "safe_stop" else "safe_stop"
            pol = state.auto_fail_policy
        update_fail_policy_button()
        set_alert("#92400e" if pol == "continue" else "#991b1b", f"Auto capture fail policy: {pol}")

    def on_toggle_detect_contour(_evt):
        with state_lock:
            state.detect_contour_enabled = not state.detect_contour_enabled
            contour_on = state.detect_contour_enabled
            white_on = state.detect_white_ratio_enabled
        if not contour_on and not white_on:
            with state_lock:
                state.detect_white_ratio_enabled = True
        update_detector_buttons()
        set_alert("#166534" if contour_on else "#7f1d1d", f"Contour detector {'enabled' if contour_on else 'disabled'}")

    def on_toggle_detect_white(_evt):
        with state_lock:
            state.detect_white_ratio_enabled = not state.detect_white_ratio_enabled
            white_on = state.detect_white_ratio_enabled
            contour_on = state.detect_contour_enabled
        if not contour_on and not white_on:
            with state_lock:
                state.detect_contour_enabled = True
        update_detector_buttons()
        set_alert("#166534" if white_on else "#7f1d1d", f"White-ratio detector {'enabled' if white_on else 'disabled'}")

    def on_lock_roi(_evt):
        nonlocal golden_frame, button_rois, button_color_baselines, button_roi_locked
        frame, frames_used = capture_average_frame()
        if frame is None:
            set_alert("#7f1d1d", f"Lock ROI failed: no camera frame ({frames_used}/{CAPTURE_AVG_MIN_FRAMES})")
            return
        if _select_button_rois_and_calibrate(frame):
            golden_frame = frame.copy() if golden_frame is None else golden_frame
            set_alert("#0f766e", f"Button ROIs locked: {list(button_rois.keys())}")
            print(f"[GUI] Button ROIs locked -> {button_rois}")
        else:
            if button_roi_locked and button_rois:
                set_alert("#d97706", "ROI update canceled. Existing locked ROIs kept")
            else:
                set_alert("#d97706", "ROI selection canceled before lock")

    def on_ic_home(_evt):
        if not robot_connected:
            with state_lock:
                state.manual_mode_active = True
                state.paused = True
            set_alert("#d97706", "Robot disconnected: visual inspection mode active")
            return
        with state_lock:
            if state.manual_mode_active:
                set_alert("#0ea5e9", "Already at IC checkpoint")
                print("[GUI] IC Home ignored: already in manual checkpoint mode")
                return

            if state.running:
                state.manual_intervention_requested = True
                set_alert("#0ea5e9", "IC Home requested (soft interrupt at cycle boundary)")
                print("[GUI] IC Home pressed")
                return

            state.manual_intervention_requested = False
            state.manual_mode_active = True
            state.running = False
            state.paused = True
            state.stopped = False

        print("[GUI] IC Home pressed -> no active cycle, moving to IC checkpoint now")
        ok = go_ic_home_checkpoint()
        if ok:
            set_alert("#0ea5e9", "At IC checkpoint. Press Return to Test to resume")
        else:
            with state_lock:
                state.manual_mode_active = False
            set_alert("red", "IC checkpoint move failed. Check robot state")

    def on_image_capture(_evt):
        nonlocal last_capture_frame, last_capture_path
        with state_lock:
            state.image_capture_count += 1
            capture_num = state.image_capture_count

        frame, frames_used = capture_average_frame()
        if frame is None:
            set_alert("red", f"Image Capture failed: averaged frame unavailable ({frames_used}/{CAPTURE_AVG_MIN_FRAMES})")
            print("[GUI] Image Capture failed: not enough camera frames for averaging")
            _manifest_write({
                "run_id": state.test_name.strip() or "test_report",
                "cycle": max(1, state.cycle_count),
                "capture_type": "manual",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "camera_status": camera_status,
                "result": "FAIL",
                "message": "manual_capture_no_frame",
                "reason_code": "avg_no_camera_frame",
                "file_path": "",
                "score": "",
                "threshold": f"thr>{INSPECTION_DIFF_THRESHOLD}|area>{INSPECTION_MIN_DEFECT_AREA}|wh>={INSPECTION_MIN_DEFECT_W}x{INSPECTION_MIN_DEFECT_H}",
                "verdict": "FAIL",
                "golden_path": golden_path or "",
                "video_path": cycle_video_path or "",
                "anomaly_path": "",
                "policy": state.auto_fail_policy,
            })
            return

        run_id = state.test_name.strip() or "test_report"
        ok, out_name, out_path = save_capture_frame(frame, "cyc", run_id, capture_num)
        if not ok:
            set_alert("red", "Image Capture failed: save error")
            print(f"[GUI] Image Capture failed: could not save {out_path}")
            _manifest_write({
                "run_id": run_id,
                "cycle": capture_num,
                "capture_type": "manual",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "camera_status": camera_status,
                "result": "FAIL",
                "message": "manual_capture_save_failed",
                "reason_code": "save_failed",
                "file_path": "",
                "score": "",
                "threshold": f"thr>{INSPECTION_DIFF_THRESHOLD}|area>{INSPECTION_MIN_DEFECT_AREA}|wh>={INSPECTION_MIN_DEFECT_W}x{INSPECTION_MIN_DEFECT_H}",
                "verdict": "FAIL",
                "golden_path": golden_path or "",
                "video_path": cycle_video_path or "",
                "anomaly_path": "",
                "policy": state.auto_fail_policy,
            })
            return

        last_capture_frame = frame.copy()
        last_capture_path = out_path
        _manifest_write({
            "run_id": run_id,
            "cycle": capture_num,
            "capture_type": "manual",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "camera_status": camera_status,
            "result": "OK",
            "message": "manual_capture",
            "reason_code": "manual_capture",
            "file_path": out_path,
            "score": "",
            "threshold": f"thr>{INSPECTION_DIFF_THRESHOLD}|area>{INSPECTION_MIN_DEFECT_AREA}|wh>={INSPECTION_MIN_DEFECT_W}x{INSPECTION_MIN_DEFECT_H}",
            "verdict": "CAPTURED",
            "golden_path": golden_path or "",
            "video_path": cycle_video_path or "",
            "anomaly_path": "",
            "policy": state.auto_fail_policy,
        })
        set_alert("#2563eb", f"Image Capture saved: {out_name}")
        print(f"[GUI] Image Capture saved -> {out_path}")

    def on_camera_tune(_evt):
        ok = run_camera_auto_tune()
        if ok:
            print(f"[GUI] Camera tuned and locked exp={camera_exposure} gain={camera_gain}")

    def on_golden_capture(_evt):
        nonlocal golden_frame, golden_path, locked_roi, roi_locked, button_rois, button_color_baselines, button_roi_locked
        frame, frames_used = capture_average_frame()
        if frame is None:
            set_alert("red", f"Golden Capture failed: averaged frame unavailable ({frames_used}/{CAPTURE_AVG_MIN_FRAMES})")
            print("[GUI] Golden Capture failed: not enough camera frames for averaging")
            return

        run_id = state.test_name.strip() or "test_report"
        ok, out_name, out_path = save_capture_frame(frame, "golden", run_id, 0)
        if not ok:
            set_alert("red", "Golden Capture failed: save error")
            print(f"[GUI] Golden Capture failed: could not save {out_path}")
            return

        golden_frame = frame.copy()
        golden_path = out_path
        with state_lock:
            state.golden_ready = True
        if _select_button_rois_and_calibrate(golden_frame):
            set_alert("#d97706", f"Golden saved + button ROIs locked: {out_name}")
            print(f"[GUI] Button ROIs locked -> {button_rois}")
        else:
            locked_roi = None
            roi_locked = False
            button_rois = {}
            button_color_baselines = {}
            button_roi_locked = False
            set_alert("orange", "Golden saved but button ROIs not fully selected")
        print(f"[GUI] Golden saved -> {out_path}")

    def on_run_inspection(_evt):
        nonlocal last_capture_frame
        if golden_frame is None:
            set_alert("#7c3aed", "Run Inspection blocked: capture Golden first")
            print("[GUI] Run Inspection blocked: golden missing")
            return
        if last_capture_frame is None:
            set_alert("#7c3aed", "Run Inspection blocked: capture Image first")
            print("[GUI] Run Inspection blocked: latest capture missing")
            return

        run_id = state.test_name.strip() or "test_report"
        cyc_num = max(1, state.cycle_count)
        if (not button_roi_locked and (not roi_locked or locked_roi is None)):
            set_alert("#7c3aed", "Run Inspection blocked: capture Golden + select ROI first")
            print("[GUI] Run Inspection blocked: ROI missing")
            return

        verdict, score, mask_path, anomaly_path, disp, decision_trace = run_basic_inspection(golden_frame, last_capture_frame, run_id, cyc_num)
        _ensure_cycle_video(golden_frame, run_id)
        _append_cycle_video_frame(disp, cyc_num)
        _manifest_write({"run_id": run_id, "cycle": cyc_num, "capture_type": "manual", "timestamp": datetime.now().isoformat(timespec="seconds"), "camera_status": camera_status, "result": "OK", "message": "manual_inspection", "reason_code": "inspection_scored", "file_path": last_capture_path or "", "score": f"{score:.2f}", "threshold": f"thr>{INSPECTION_DIFF_THRESHOLD}|area>{INSPECTION_MIN_DEFECT_AREA}|wh>={INSPECTION_MIN_DEFECT_W}x{INSPECTION_MIN_DEFECT_H}", "verdict": verdict, "golden_path": golden_path or "", "video_path": cycle_video_path or "", "anomaly_path": anomaly_path, "policy": state.auto_fail_policy, "decision_logic": decision_trace.get("decision_logic", ""), "failed_metric": decision_trace.get("failed_metric", ""), "max_contour_area": decision_trace.get("max_contour_area", ""), "max_bbox": decision_trace.get("max_bbox", ""), "score_role": decision_trace.get("score_role", ""), "button_drop_pct": decision_trace.get("button_drop_pct", ""), "white_ratio_button": decision_trace.get("white_ratio_button", ""), "white_ratio_change_pct": decision_trace.get("white_ratio_change_pct", "")})
        inspection_records.append({"run_id": run_id, "cycle": cyc_num, "capture_type": "manual", "timestamp": datetime.now().isoformat(timespec="seconds"), "camera_status": camera_status, "result": "OK", "message": "manual_inspection", "reason_code": "inspection_scored", "file_path": last_capture_path or "", "score": f"{score:.2f}", "threshold": f"thr>{INSPECTION_DIFF_THRESHOLD}|area>{INSPECTION_MIN_DEFECT_AREA}|wh>={INSPECTION_MIN_DEFECT_W}x{INSPECTION_MIN_DEFECT_H}", "verdict": verdict, "golden_path": golden_path or "", "video_path": cycle_video_path or "", "anomaly_path": anomaly_path, "policy": state.auto_fail_policy, "decision_logic": decision_trace.get("decision_logic", ""), "failed_metric": decision_trace.get("failed_metric", ""), "max_contour_area": decision_trace.get("max_contour_area", ""), "max_bbox": decision_trace.get("max_bbox", ""), "score_role": decision_trace.get("score_role", ""), "button_drop_pct": decision_trace.get("button_drop_pct", ""), "white_ratio_button": decision_trace.get("white_ratio_button", ""), "white_ratio_change_pct": decision_trace.get("white_ratio_change_pct", "")})
        _record_anomaly_stats(cyc_num, verdict, score)
        with state_lock:
            state.last_capture_result = f"manual/{verdict}"
        metric_label = decision_trace.get("score_role", "inspection_metric")
        set_alert("green" if verdict == "PASS" else "orange", f"Inspection {verdict} {metric_label}={score:.2f}")
        print(f"[GUI] Run Inspection -> {verdict} {metric_label}={score:.2f} mask={mask_path} anomaly={anomaly_path}")

    def on_return_to_test(_evt):
        with state_lock:
            state.manual_intervention_requested = False
            was_manual = state.manual_mode_active
            if not was_manual:
                set_alert("#0891b2", "Return to Test ignored (not in manual mode)")
                print("[GUI] Return to Test ignored: robot was not in manual checkpoint mode")
                return

            state.manual_mode_active = False
            state.running = False
            state.paused = True
            state.stopped = False
            state.aligned_to_A = False
            state.traj_index = 0

        print("[GUI] Return to Test pressed -> Going Home before restart")
        go_home()
        if not wait_until_idle():
            with state_lock:
                state.running = False
                state.paused = False
                state.stopped = True
            set_alert("red", "Return to Test failed: robot did not reach Home")
            print("[GUI] Return to Test aborted: timeout waiting at Home")
            return

        with state_lock:
            state.running = True
            state.paused = False
            state.stopped = False
        set_alert("#0891b2", "At Home. Restarting cycle test")
        print("[GUI] Return to Test: automatic cycle resumed")

    def on_reset(_evt):
        nonlocal golden_frame, golden_path, last_capture_frame, last_capture_path, locked_roi, roi_locked
        nonlocal button_rois, button_color_baselines, button_roi_locked
        nonlocal cycle_video_writer, cycle_video_path, cycle_video_started, auto_report_written_cycle
        with state_lock:
            state.force_out_of_range = {"A": 0, "B": 0, "C": 0, "D": 0}
            state.button_did_not_retract = {"A": 0, "B": 0, "C": 0, "D": 0}
            state.baseline_peaks = {"A": [], "B": [], "C": [], "D": []}
            state.baseline_mean = {}
            state.baseline_ready = False
            state.image_capture_count = 0
        golden_frame = None
        golden_path = None
        last_capture_frame = None
        last_capture_path = None
        locked_roi = None
        roi_locked = False
        inspection_records.clear()
        anomaly_stats.update({
            "total_scored": 0,
            "pass_count": 0,
            "fail_count": 0,
            "warn_count": 0,
            "first_fail_cycle": "",
            "worst_score": -1.0,
            "worst_cycle": "",
        })
        if cycle_video_writer is not None:
            try:
                cycle_video_writer.release()
            except Exception:
                pass
        cycle_video_writer = None
        cycle_video_path = None
        cycle_video_started = False
        auto_report_written_cycle = -1
        _recover_camera_preview()
        with state_lock:
            state.golden_ready = False
            state.last_capture_result = "none"
            state.next_auto_capture_cycle = state.capture_every_x_cycles if state.capture_every_x_cycles > 0 else 0
        print("[GUI] Reset pressed -> counters + baseline cleared")

    def build_report_pdf(path):
        target_dir = os.path.dirname(path) or "."
        if not os.access(target_dir, os.W_OK):
            raise PermissionError(f"No write permission to directory: {target_dir}")

        with state_lock:
            report_name = state.test_name.strip() or "test_report"
            peak_copy = {b: list(report_peak_cycles[b]) for b in BUTTON_ORDER}
            force_copy = {b: list(report_peak_forces[b]) for b in BUTTON_ORDER}
            miss_copy = {b: list(report_missed_cycles[b]) for b in BUTTON_ORDER}
            oor_copy = {b: list(report_oor_cycles[b]) for b in BUTTON_ORDER}
            state_cycles = state.cycle_count
            baseline_used = state.baseline_cycles
            force_min_used = state.force_min
            force_max_used = state.force_max
            settings_used = {
                "Velocity": state.vel,
                "Acceleration": state.acc,
                "Jerk": state.jerk,
                "Target cycles": state.target_cycles,
                "Baseline cycles": state.baseline_cycles,
                "Force min": state.force_min,
                "Force max": state.force_max,
                "Auto cap retries": state.auto_capture_retries,
                "Auto fail policy": state.auto_fail_policy,
            }

        observed_cycles = []
        for b in BUTTON_ORDER:
            observed_cycles.extend(peak_copy[b])
            observed_cycles.extend(miss_copy[b])
            observed_cycles.extend(oor_copy[b])
        total_cycles = max(observed_cycles) if observed_cycles else state_cycles

        pdf = PdfPages(path)
        try:
            landscape_size = (11, 8.5)
            summary_fig = plt.figure(figsize=landscape_size, facecolor="white")
            summary_fig.suptitle(f"Test Report: {report_name}", fontsize=20, weight="bold", y=0.96)

            if app_logo_path:
                try:
                    ax_logo_rep = summary_fig.add_axes([0.66, 0.03, 0.30, 0.08])
                    ax_logo_rep.imshow(mpimg.imread(app_logo_path))
                    ax_logo_rep.axis("off")
                except Exception:
                    pass

            txt = (
                f"Total cycles completed: {total_cycles}\n"
                f"Baseline cycles setting: {baseline_used}\n"
                f"Force valid range: {force_min_used} to {force_max_used} lbs"
            )
            summary_fig.text(0.08, 0.78, txt, fontsize=12, va='top')
            settings_txt = "\n".join([f"{k}: {v}" for k, v in settings_used.items()])
            summary_fig.text(0.08, 0.53, "Settings used for this test:", fontsize=12, weight="bold", va='top')
            summary_fig.text(0.08, 0.49, settings_txt, fontsize=11, va='top')
            summary_fig.text(0.08, 0.60, "Buttons tracked: A, B, C, D", fontsize=11, color="#334155")
            pdf.savefig(summary_fig)
            plt.close(summary_fig)

            for b in BUTTON_ORDER:
                fig_btn, ax_btn = plt.subplots(figsize=landscape_size)
                ax_btn.set_title(f"Button {b}: Force vs Cycle")
                ax_btn.set_xlabel("Cycle Count")
                ax_btn.set_ylabel("Peak Force (lbs)")
                ax_btn.grid(True, axis='y', alpha=0.3)

                if peak_copy[b]:
                    ax_btn.plot(peak_copy[b], force_copy[b], linewidth=1.5, color='black', label='Peak force')

                for i, cyc in enumerate(miss_copy[b]):
                    ax_btn.axvline(cyc, color='red', linestyle='--', linewidth=1.5,
                                   label='Did not retract' if i == 0 else None)

                for i, cyc in enumerate(oor_copy[b]):
                    ax_btn.axvline(cyc, color='orange', linestyle='--', linewidth=1.5,
                                   label='Force out of range' if i == 0 else None)

                handles, labels = ax_btn.get_legend_handles_labels()
                if labels:
                    ax_btn.legend(loc='best')

                oor_n = len(oor_copy[b])
                miss_n = len(miss_copy[b])
                ax_btn.text(
                    0.01, -0.22,
                    f"Total force out of range failures: {oor_n}    Total button did not retract failures: {miss_n}",
                    transform=ax_btn.transAxes,
                    fontsize=10,
                    color='black',
                    va='top'
                )

                fig_btn.tight_layout(rect=[0, 0.08, 1, 1])
                pdf.savefig(fig_btn)
                plt.close(fig_btn)

            anomaly_fig = plt.figure(figsize=landscape_size, facecolor="white")
            anomaly_fig.suptitle("Anomaly Detection During Cycling", fontsize=18, weight="bold", y=0.96)
            y = 0.90
            anomaly_fig.text(0.06, y, "Original v23 report sections preserved. This section is appended.", fontsize=10, color="#334155")
            y -= 0.05
            worst_score_txt = f"{anomaly_stats['worst_score']:.2f}" if anomaly_stats["worst_score"] >= 0 else "n/a"
            stats_txt = (
                f"Scored: {anomaly_stats['total_scored']}   PASS: {anomaly_stats['pass_count']}   FAIL: {anomaly_stats['fail_count']}   WARN: {anomaly_stats['warn_count']}\n"
                f"First FAIL cycle: {anomaly_stats['first_fail_cycle'] or 'n/a'}   Worst inspection metric: {worst_score_txt} (cycle {anomaly_stats['worst_cycle'] or 'n/a'})"
            )
            anomaly_fig.text(0.06, y, stats_txt, fontsize=9)
            y -= 0.06
            if inspection_records:
                head = "Cycle | Type | Verdict | Metric | FailedBtn | A% | B% | C% | D% | Reason"
                anomaly_fig.text(0.06, y, head, fontsize=10.5, weight="bold")
                y -= 0.03
                for rec in inspection_records[-28:]:
                    try:
                        score_txt = f"{float(rec.get('score', '')):.2f}"
                    except Exception:
                        score_txt = str(rec.get('score', ''))
                    drops = {b: "0.00" for b in BUTTON_ORDER}
                    drop_raw = str(rec.get("button_drop_pct", "") or "")
                    for token in drop_raw.split("|"):
                        if ":" not in token:
                            continue
                        k, v = token.split(":", 1)
                        k = k.strip()
                        if k in drops:
                            try:
                                drops[k] = f"{float(v):.2f}"
                            except Exception:
                                drops[k] = v.strip()
                    failed_btn = str(rec.get("white_ratio_button", "") or "")
                    if not failed_btn:
                        fm = str(rec.get("failed_metric", "") or "")
                        if ":" in fm:
                            failed_btn = fm.split(":", 1)[1]
                    metric_name = rec.get("score_role", "inspection_metric")
                    line_txt = (
                        f"{rec.get('cycle','')} | {rec.get('capture_type','')} | {rec.get('verdict','')} | "
                        f"{metric_name}:{score_txt} | {failed_btn or '-'} | "
                        f"{drops['A']} | {drops['B']} | {drops['C']} | {drops['D']} | "
                        f"{rec.get('reason_code', rec.get('message',''))}"
                    )
                    anomaly_fig.text(0.06, y, line_txt, fontsize=8.3)
                    y -= 0.024
                    if y < 0.08:
                        break
                fail_rows = [r for r in inspection_records if str(r.get("verdict", "")).upper() == "FAIL"]
                if fail_rows:
                    y = max(0.30, y - 0.01)
                    anomaly_fig.text(0.06, y, "FAIL Explainability (per inspection decision)", fontsize=11, weight="bold")
                    y -= 0.03
                    anomaly_fig.text(0.06, y, "Legend: Red = metric that triggered FAIL", fontsize=9, color="red")
                    y -= 0.025
                    for rec in fail_rows[-8:]:
                        logic = rec.get("decision_logic", "") or "Decision trace unavailable"
                        area_val = rec.get("max_contour_area", "") or "n/a"
                        bbox_val = rec.get("max_bbox", "") or "n/a"
                        cyc = rec.get("cycle", "")
                        anomaly_fig.text(0.06, y, f"Cycle {cyc}: {logic}", fontsize=8.8)
                        y -= 0.021
                        try:
                            fail_score_txt = f"{float(rec.get('score', '')):.2f}"
                        except Exception:
                            fail_score_txt = str(rec.get('score', ''))
                        metric_label = rec.get("score_role", "inspection_metric")
                        anomaly_fig.text(0.08, y, f"{metric_label}: {fail_score_txt}  ", fontsize=8.6, color="#334155")
                        y -= 0.020
                        anomaly_fig.text(0.08, y, f"Triggered metric: {rec.get('failed_metric', 'n/a')} (area={area_val}, bbox={bbox_val})", fontsize=8.6, color="red")
                        y -= 0.024
                        if y < 0.10:
                            break
                recent_anomaly = inspection_records[-1].get("anomaly_path", "") if inspection_records else ""
                anomaly_fig.text(0.06, 0.07, f"Latest frame_anamoly: {recent_anomaly or 'not_created'}", fontsize=9)
                anomaly_fig.text(0.06, 0.05, f"Cycle inspection video: {cycle_video_path or 'not_created'}", fontsize=9)
            else:
                anomaly_fig.text(0.06, y, "No inspection records captured.", fontsize=11)
            pdf.savefig(anomaly_fig)
            plt.close(anomaly_fig)
        finally:
            pdf.close()

    def _auto_report_path():
        run_id = state.test_name.strip() or "test_report"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(reports_dir, f"report_{run_id}_{ts}.pdf")

    def _open_file(path):
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)

    def _latest_report_path():
        files = [os.path.join(reports_dir, f) for f in os.listdir(reports_dir) if f.lower().endswith(".pdf")]
        if not files:
            return None
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files[0]

    def on_download_report(_evt):
        nonlocal last_saved_report_path
        apply_textbox_values()
        try:
            path = last_saved_report_path if (last_saved_report_path and os.path.exists(last_saved_report_path)) else _latest_report_path()
            if path is None:
                path = _auto_report_path()
                build_report_pdf(path)
                last_saved_report_path = path
            _open_file(path)
            set_alert("green", f"Report opened: {path}")
            print(f"[Report] Opened: {path}")
        except PermissionError as exc:
            set_alert("red", f"Report open failed: {exc}")
            print(f"[Report] Permission error: {exc}")
        except Exception as exc:
            set_alert("red", f"Report open failed: {exc}")
            print(f"[Report] Open failed: {exc}")

    def on_exit(_evt):
        nonlocal last_saved_report_path, auto_report_written_cycle
        with state_lock:
            stop_cycle = state.cycle_count
            state.exit_requested = True
            state.running = False
            state.paused = False
            state.stopped = True
        if stop_cycle > 0 and auto_report_written_cycle != stop_cycle:
            try:
                path = _auto_report_path()
                build_report_pdf(path)
                last_saved_report_path = path
                auto_report_written_cycle = stop_cycle
                print(f"[Report] Auto-saved on exit: {path}")
            except Exception as exc:
                print(f"[Report] Auto-save on exit failed: {exc}")
        plt.close(fig)

    btn_start.on_clicked(on_start)
    btn_pause.on_clicked(on_pause)
    btn_stop.on_clicked(on_stop)
    btn_home.on_clicked(on_home)
    btn_re_tare.on_clicked(on_re_tare)
    btn_tare_on_start.on_clicked(on_toggle_tare_on_start)
    btn_auto_cap.on_clicked(on_toggle_auto_cap)
    btn_fail_policy.on_clicked(on_toggle_fail_policy)
    btn_detect_contour.on_clicked(on_toggle_detect_contour)
    btn_detect_white.on_clicked(on_toggle_detect_white)
    btn_lock_roi.on_clicked(on_lock_roi)
    btn_ic_home.on_clicked(on_ic_home)
    btn_return_test.on_clicked(on_return_to_test)
    btn_camera_tune.on_clicked(on_camera_tune)
    btn_golden_capture.on_clicked(on_golden_capture)
    btn_image_capture.on_clicked(on_image_capture)
    btn_run_inspection.on_clicked(on_run_inspection)
    btn_reset.on_clicked(on_reset)
    btn_report.on_clicked(on_download_report)
    btn_exit.on_clicked(on_exit)

    update_tare_toggle_button()
    update_auto_cap_button()
    update_fail_policy_button()
    update_detector_buttons()

    # TextBox callbacks (kept, but now also locked)
    def on_vel_submit(text):
        try:
            v = int(float(text))
            with state_lock:
                state.vel = clamp(v, VEL_MIN, VEL_MAX)
        except Exception:
            pass

    def on_acc_submit(text):
        try:
            a = int(float(text))
            with state_lock:
                state.acc = clamp(a, ACC_MIN, ACC_MAX)
        except Exception:
            pass

    def on_jerk_submit(text):
        try:
            j = int(float(text))
            with state_lock:
                state.jerk = clamp(j, JERK_MIN, JERK_MAX)
        except Exception:
            pass

    def on_cyc_submit(text):
        try:
            c = int(float(text))
            if c > 0:
                with state_lock:
                    state.target_cycles = c
        except Exception:
            pass

    def on_base_submit(text):
        try:
            b = int(float(text))
            with state_lock:
                new_base = clamp(b, 1, 500)
                if new_base != state.baseline_cycles:
                    state.baseline_cycles = new_base
                    state.baseline_peaks = {"A": [], "B": [], "C": [], "D": []}
                    state.baseline_mean = {}
                    state.baseline_ready = False
                else:
                    state.baseline_cycles = new_base
        except Exception:
            pass

    def on_fmin_submit(text):
        try:
            fmin = float(text)
            with state_lock:
                if state.force_max > fmin:
                    state.force_min = fmin
        except Exception:
            pass

    def on_fmax_submit(text):
        try:
            fmax = float(text)
            with state_lock:
                if fmax > state.force_min:
                    state.force_max = fmax
        except Exception:
            pass

    def on_cap_every_submit(text):
        try:
            v = int(float(text))
            with state_lock:
                state.capture_every_x_cycles = max(0, v)
                state.auto_capture_enabled = state.capture_every_x_cycles > 0
                state.next_auto_capture_cycle = state.capture_every_x_cycles if state.auto_capture_enabled else 0
            update_auto_cap_button()
        except Exception:
            pass

    def on_cap_retry_submit(text):
        try:
            v = int(float(text))
            with state_lock:
                state.auto_capture_retries = clamp(v, 0, 10)
        except Exception:
            pass

    tb_vel.on_submit(on_vel_submit)
    tb_acc.on_submit(on_acc_submit)
    tb_jerk.on_submit(on_jerk_submit)
    tb_cyc.on_submit(on_cyc_submit)
    tb_base.on_submit(on_base_submit)
    tb_fmin.on_submit(on_fmin_submit)
    tb_fmax.on_submit(on_fmax_submit)
    tb_cap_every.on_submit(on_cap_every_submit)
    tb_cap_retry.on_submit(on_cap_retry_submit)

    # ---- Force buffers ----
    times = deque()
    forces = deque()
    start_time = time.time()

    peak_events = deque()
    peak_scatter = None
    peak_texts = []

    report_peak_cycles = {b: [] for b in BUTTON_ORDER}
    report_peak_forces = {b: [] for b in BUTTON_ORDER}
    report_missed_cycles = {b: [] for b in BUTTON_ORDER}
    report_oor_cycles = {b: [] for b in BUTTON_ORDER}

    print("Streaming + GUI running (Stage D force window v5).")

    try:
        while True:
            with state_lock:
                if state.exit_requested:
                    break
            # FORCE sample
            raw = bridge.getVoltageRatio()
            t_now = time.time() - start_time
            force = (raw - zero_offset) * calibration_factor

            times.append(t_now)
            forces.append(force)

            while times and (t_now - times[0] > window_seconds):
                times.popleft()
                forces.popleft()

            # Track peak during active window
            with state_lock:
                if state.window_active:
                    if state.window_peak_force is None or force > state.window_peak_force:
                        state.window_peak_force = force
                        state.window_peak_time = t_now

            # Plot update
            if len(times) > 1:
                line.set_data(times, forces)
                ax.set_xlim(times[0], times[-1])

            # ROBOT stepping
            with state_lock:
                running = state.running

            auto_capture_cycle = None
            report_cycle_to_write = None
            if running and is_idle(robot):
                with state_lock:
                    if state.cycle_count >= state.target_cycles:
                        state.running = False
                        state.paused = False
                        state.stopped = True
                        state.traj_index = 0
                        state.aligned_to_A = False
                        if auto_report_written_cycle != state.cycle_count:
                            report_cycle_to_write = state.cycle_count
                            auto_report_written_cycle = state.cycle_count
                        print("[Robot] Target cycles reached. Stopping.")
                    else:
                        if not state.aligned_to_A:
                            print("[Robot] Safe Align A-above")
                            robot.play(-1, {
                                "cmd": "jmove", "rel": 0,
                                "vel": SAFE_START_VEL,
                                "acc": SAFE_START_ACC,
                                "jerk": SAFE_START_JERK,
                                **BUTTON_POSES["A"]["above"]
                            })
                            state.aligned_to_A = True
                            state.traj_index = 1
                        else:
                            idx = state.traj_index
                            pos = TRAJECTORY[idx]
                            btn, ph = INDEX_TO_META[idx]

                            # CLOSE window before sending retract
                            if ph == "retract" and state.window_active and state.window_button == btn:
                                peak = state.window_peak_force

                                cycle_num = state.cycle_count + 1

                                if peak is None or peak < peak_start_threshold:
                                    state.button_did_not_retract[btn] += 1
                                    report_missed_cycles[btn].append(cycle_num)
                                    peak_events.append({"t": t_now, "y": peak_start_threshold, "btn": btn, "missed": True})
                                    set_alert("red", f"Missed peak on {btn}")
                                    print(f"[ForceWindow] MISS {btn}")
                                else:
                                    peak_t = state.window_peak_time if state.window_peak_time is not None else t_now
                                    report_peak_cycles[btn].append(cycle_num)
                                    report_peak_forces[btn].append(float(peak))
                                    peak_events.append({"t": peak_t, "y": float(peak), "btn": btn, "missed": False})

                                    # baseline collect
                                    if state.cycle_count < state.baseline_cycles:
                                        state.baseline_peaks[btn].append(float(peak))
                                        ready = all(len(state.baseline_peaks[b]) >= state.baseline_cycles for b in BUTTON_ORDER)
                                        if ready and (not state.baseline_ready):
                                            state.baseline_mean = {b: float(np.mean(state.baseline_peaks[b])) for b in BUTTON_ORDER}
                                            state.baseline_ready = True
                                            set_alert("green", "Baseline ready")
                                            print("[Baseline] READY:", state.baseline_mean)

                                    out_of_range = False
                                    baseline_deviation = False

                                    # out of range
                                    if not (state.force_min <= peak <= state.force_max):
                                        out_of_range = True
                                        set_alert("orange", f"Force out of range {btn}: {peak:.2f}")
                                        print(f"[ForceWindow] OOR {btn} peak={peak:.2f}")

                                    # baseline deviation
                                    if state.baseline_ready and btn in state.baseline_mean and state.baseline_mean[btn] > 0:
                                        base = state.baseline_mean[btn]
                                        if abs(peak - base) > base * deviation_threshold:
                                            baseline_deviation = True
                                            set_alert("orange", f"Baseline dev {btn}: {peak:.2f} vs {base:.2f}")
                                            print(f"[ForceWindow] DEV {btn} peak={peak:.2f} base={base:.2f}")

                                    # Increment once per press even if both checks fail.
                                    if out_of_range or baseline_deviation:
                                        state.force_out_of_range[btn] += 1
                                        report_oor_cycles[btn].append(cycle_num)

                                # close window
                                state.window_active = False
                                state.window_button = None
                                state.window_peak_force = None
                                state.window_peak_time = None

                            # OPEN window before press
                            if ph == "press":
                                state.window_active = True
                                state.window_button = btn
                                state.window_peak_force = None
                                state.window_peak_time = None
                                print(f"[ForceWindow] OPEN {btn}")

                            vel = clamp(int(state.vel), VEL_MIN, VEL_MAX)
                            acc = clamp(int(state.acc), ACC_MIN, ACC_MAX)
                            jerk = clamp(int(state.jerk), JERK_MIN, JERK_MAX)

                            print(f"[Robot] Move idx={idx} {btn}-{ph} vel={vel} acc={acc} jerk={jerk}")
                            robot.play(-1, {
                                "cmd": "jmove", "rel": 0,
                                "vel": vel,
                                "acc": acc,
                                "jerk": jerk,
                                **pos
                            })

                            state.traj_index += 1
                            if state.traj_index >= len(TRAJECTORY):
                                state.traj_index = 0
                                state.cycle_count += 1
                                print(f"[Robot] Cycle complete -> {state.cycle_count}")

                                if state.manual_intervention_requested:
                                    state.running = False
                                    state.paused = True
                                    state.manual_mode_active = True
                                    state.manual_intervention_requested = False
                                    state.aligned_to_A = False
                                    print("[Robot] Soft interrupt reached at cycle boundary -> moving to IC checkpoint")
                                    ok = go_ic_home_checkpoint()
                                    if ok:
                                        set_alert("#0ea5e9", "At IC checkpoint. Press Return to Test to resume")
                                    else:
                                        set_alert("red", "IC checkpoint move failed. Check robot state")

                                if state.auto_capture_enabled and state.capture_every_x_cycles > 0:
                                    if state.cycle_count >= state.next_auto_capture_cycle and not state.manual_mode_active:
                                        due_cycle = state.cycle_count
                                        state.running = False
                                        state.paused = True
                                        state.aligned_to_A = False
                                        state.last_capture_result = f"cycle {due_cycle}: auto capture due"
                                        print(f"[Robot] Auto capture due at cycle {due_cycle}")
                                        # leave lock scope and execute below
                                        auto_capture_cycle = due_cycle
                                    else:
                                        auto_capture_cycle = None
                                else:
                                    auto_capture_cycle = None

            if auto_capture_cycle is not None:
                ok_auto = _auto_capture_cycle(auto_capture_cycle)
                if not ok_auto:
                    with state_lock:
                        state.last_capture_result = f"cycle {auto_capture_cycle}: auto capture failed"
                        if state.auto_fail_policy == "continue":
                            state.running = True
                            state.paused = False
                            state.manual_mode_active = False
                            state.manual_intervention_requested = False
                            state.next_auto_capture_cycle = state.cycle_count + max(1, state.capture_every_x_cycles)
                        else:
                            state.running = False
                            state.paused = True
                    if state.auto_fail_policy == "continue":
                        set_alert("orange", f"Auto capture failed at cycle {auto_capture_cycle}; continuing by policy")
                    else:
                        set_alert("orange", f"Auto capture failed at cycle {auto_capture_cycle}; safe-stop by policy")

            if report_cycle_to_write is not None:
                try:
                    path = _auto_report_path()
                    build_report_pdf(path)
                    last_saved_report_path = path
                    set_alert("green", f"Auto report saved: {os.path.basename(path)}")
                    print(f"[Report] Auto-saved: {path}")
                except Exception as exc:
                    set_alert("red", f"Auto report failed: {exc}")
                    print(f"[Report] Auto-save failed: {exc}")

            # UI text / alerts
            with state_lock:
                mode = "RUNNING" if state.running else ("PAUSED" if state.paused else "STOPPED")
                base_color = "green" if state.running else ("orange" if state.paused else "gray")

                now_wall = time.time()
                if now_wall <= state.alert_until_wall:
                    status_dot.set_facecolor(state.alert_color)
                    alert_msg = state.alert_msg
                else:
                    status_dot.set_facecolor(base_color)
                    alert_msg = ""

                manual_state = "Manual: REQUESTED" if state.manual_intervention_requested else (
                    "Manual: ACTIVE" if state.manual_mode_active else "Manual: OFF"
                )
                with camera_lock:
                    camera_txt = camera_status

                idx = state.traj_index % len(TRAJECTORY)
                btn, ph = INDEX_TO_META[idx]

                if state.baseline_ready:
                    baseline_txt = "Baseline: READY"
                else:
                    min_n = min(len(state.baseline_peaks[b]) for b in BUTTON_ORDER)
                    baseline_txt = f"Baseline: {min_n}/{state.baseline_cycles} per button"

                frame = get_latest_camera_frame()
                if frame is not None:
                    camera_im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ax_cam.set_title(f"IC Camera ({camera_txt})", fontsize=20, color=COLOR_TEXT)

                cam_lock_txt = "LOCKED" if camera_settings_locked else "UNLOCKED"
                tare_txt = "Tare@Start:ON" if state.tare_on_start else "Tare@Start:OFF"
                sched_txt = "ON" if state.auto_capture_enabled else "OFF"
                gold_txt = "READY" if state.golden_ready else "NO"
                next_cap = state.next_auto_capture_cycle if state.auto_capture_enabled else "-"
                fail_pol = state.auto_fail_policy
                status_line.set_text(
                    f"State: {mode} | Mode: {robot_mode_text} | {manual_state} | {tare_txt} | Cycle: {state.cycle_count}/{state.target_cycles} | Next: {btn}-{ph} | {baseline_txt} | {alert_msg}"
                )
                roi_txt = f"ROI:LOCKED {list(button_rois.keys())}" if (button_roi_locked and button_rois) else (f"ROI:LOCKED {locked_roi}" if (roi_locked and locked_roi is not None) else "ROI:UNSET")
                auto_status_1.set_text(f"Camera: {camera_txt} / {cam_lock_txt}")
                auto_status_2.set_text(f"AutoCap: {sched_txt} every={state.capture_every_x_cycles} next={next_cap} retries={state.auto_capture_retries}")
                auto_status_3.set_text(f"Golden ready: {gold_txt}")
                auto_status_4.set_text(f"{roi_txt} | FailPolicy: {fail_pol}")
                auto_status_5.set_text(f"Inspection/Capture: {state.last_capture_result}")
                param_line.set_text("")
                fail_line_1.set_text(
                    f"Force out of range  A:{state.force_out_of_range['A']}  B:{state.force_out_of_range['B']}  C:{state.force_out_of_range['C']}  D:{state.force_out_of_range['D']} | "
                    f"Button did not retract  A:{state.button_did_not_retract['A']}  B:{state.button_did_not_retract['B']}  C:{state.button_did_not_retract['C']}  D:{state.button_did_not_retract['D']}"
                )
                fail_line_2.set_text("")

                ax.set_ylim(-0.2, max(2.0, state.force_max + 1.5))
                force_band.remove()
                force_band = ax.axhspan(state.force_min, state.force_max, alpha=0.18, color="#93c5fd")

                while peak_events and (t_now - peak_events[0]["t"] > window_seconds):
                    peak_events.popleft()

                if peak_scatter is not None:
                    peak_scatter.remove()
                    peak_scatter = None

                for txt in peak_texts:
                    txt.remove()
                peak_texts = []

                if peak_events:
                    x_evt = [e["t"] for e in peak_events]
                    y_evt = [e["y"] for e in peak_events]
                    c_evt = ["red" if e["missed"] else "black" for e in peak_events]
                    peak_scatter = ax.scatter(x_evt, y_evt, c=c_evt, s=36, zorder=3)

                    for e in peak_events:
                        y_off = 0.08 if e["missed"] else 0.04
                        label = f"{e['btn']} (MISS)" if e["missed"] else e["btn"]
                        txt = ax.text(e["t"], e["y"] + y_off, label, fontsize=10, color=("red" if e["missed"] else "black"),
                                      ha="center", va="bottom", zorder=4)
                        peak_texts.append(txt)

            plt.pause(update_interval)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if cycle_video_writer is not None:
                cycle_video_writer.release()
        except Exception:
            pass
        try:
            stop_camera_preview()
        except Exception:
            pass
        try:
            bridge.close()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        print("Exited cleanly.")


if __name__ == "__main__":
    main()
