# RoboSpeed Durability Intelligence Platform
### v2.8 · Dark Industrial GUI · PyQt6 + PyQtGraph

> A real-time durability testing dashboard for robotic button-press cycle testing.
> Monitors force profiles, runs visual inspection pipelines, and logs every cycle result.

---

## Table of Contents

1. [File Structure](#1-file-structure)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Running the Software](#4-running-the-software)
5. [GUI Walkthrough](#5-gui-walkthrough)
6. [Connecting Hardware](#6-connecting-hardware)
7. [Controller Integration](#7-controller-integration)
8. [Backend Hooks — ROI & Inspection Signals](#8-backend-hooks--roi--inspection-signals)
9. [Configuration Reference](#9-configuration-reference)
10. [Data & Logging](#10-data--logging)
11. [Troubleshooting](#11-troubleshooting)
12. [Changelog](#12-changelog)

---

## 1. File Structure

```
gui/
├── robospeed_gui_main.py        ← Main GUI window (run this for demo/simulation)
├── robospeed_controller.py      ← Hardware bridge (run this for real hardware)
├── robospeed_logo.png           ← Horizontal logo (robot icon + ROBOSPEED text)
├── favicon_64.png               ← Square robot icon for window titlebar
├── logo_cropped.png             ← Tight-cropped logo (auto-generated on first run)
├── progress_bar_mockup.html     ← Visual reference for progress bar states
├── README.md                    ← This file
├── logs/                        ← Auto-created on first test run
│   └── ProjectName_Profile/
│       ├── report.csv
│       ├── surface_cycle00025_C1.jpg
│       └── ...
└── .venv/                       ← Python virtual environment
```

> **Logo files must be in the same folder as the `.py` files.**

---

## 2. Requirements

### Python version
- **Python 3.10 or higher** (3.11 recommended)

### Python packages

| Package | Version | Purpose |
|---------|---------|---------|
| `PyQt6` | ≥ 6.5 | GUI framework |
| `pyqtgraph` | ≥ 0.13 | Real-time force graph |
| `numpy` | ≥ 1.24 | Signal processing |
| `Pillow` | ≥ 10.0 | Logo background blending |
| `pyserial` | ≥ 3.5 | Hardware serial comms *(controller only)* |
| `opencv-python` | ≥ 4.8 | Camera feed *(controller only, optional)* |

### Operating system
- **Windows 10 / 11** ✓ (primary target)  **macOS 12+** ✓  **Linux (Ubuntu 22.04+)** ✓

---

## 3. Installation

```powershell
cd C:\Users\anilp\OneDrive\Desktop\gui
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install PyQt6 pyqtgraph numpy Pillow
python -c "import PyQt6; import pyqtgraph; import numpy; import PIL; print('All OK')"
```

> **Execution policy error?**  Run once as Administrator:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## 4. Running the Software

### Simulation mode (no hardware)

```powershell
.venv\Scripts\Activate.ps1
python robospeed_gui_main.py
```

### With real hardware

```powershell
python robospeed_controller.py --no-sim --motion-port COM3 --daq-port COM4
python robospeed_controller.py --no-sim --motion-port COM3 --daq-port COM4 --cam-c1 0 --cam-c2 1
```

---

## 5. GUI Walkthrough

### Identity bar (top)
| Element | What it does |
|---------|-------------|
| **Project** field | Name of the current project |
| **Test Profile** field | Name of the test configuration |
| **💾 Save Profile** | Saves all panel settings to a `.rsprofile` JSON file |
| **📂 Open Profile** | Loads settings from a saved `.rsprofile` file |
| **◀ Controls** / **Insights ▶** | Toggle left/right panel visibility |

### Left panel — Test Run Controls

| Button | Colour | Action |
|--------|--------|--------|
| **▶ START** | Deep Emerald | Starts the test |
| **❚❚ PAUSE** | Amber | Pauses mid-cycle |
| **■ STOP** | Deep Red | Immediately stops and ends test |
| **⌂ HOME** | Grey | Sends robot to home position |
| **↺ RESET** | Grey | Clears cycle counter and failure counts |
| **● RECORD TRAJECTORY** | Grey | Teach mode for recording a new path |
| **↓ DOWNLOAD REPORT** | Grey | Generates CSV report to `logs/` |
| **✕ EXIT** | Dark Grey | Closes the application |

### Left panel — Motion Control

| Field | Default | Description |
|-------|---------|-------------|
| Velocity | 300 | Maximum joint velocity |
| Accel | 300 | Acceleration ramp |
| Jerk | 1000 | Jerk limit (smoothness) |
| Cycles | 100 | Total target test cycles |
| Baseline cycles | 30 | Cycles used to learn normal force pattern |
| Force Min | 0.5 lbs | Lower bound of acceptable force range |
| Force Max | 1.8 lbs | Upper bound of acceptable force range |

### Right panel — Inspection Modules

Each module (Cosmetic Defects, LED Performance, Geometry Deviation) has:
- **Enable/Disable toggle** — entire section dims to 35% opacity when disabled; all inputs locked
- **State badge** — IDLE / PROCESSING… / COMPLETE / ERROR
- **ROI Manager** — up to 5 ROIs. Each ROI row:
  - 👁 visible / 🔴 hidden (eye icon toggles UI visibility — does not affect processing)
  - Shape label (Rectangle / Polygon / Circle)
  - Edit / Lock / Delete action buttons
- **Checks grid** — inspection algorithms to enable
- **Capture frequency** — cycles between captures

### Bottom status bar
Two rows: state dot + cycle info + alert message on Row 1; motion params + failure counts on Row 2.

---

## 6. Connecting Hardware

```
PC  ──USB/Serial──►  Motion Controller  ──►  Robot Arm
PC  ──USB/Serial──►  Force DAQ           ──►  Force Sensor
PC  ──USB───────►  Camera C1  (surface + LED inspection)
PC  ──USB───────►  Camera C2  (3D geometry / point cloud)
```

Find COM ports in **Device Manager → Ports (COM & LPT)**.
Find camera indices by running: `import cv2; [print(i) for i in range(5) if cv2.VideoCapture(i).isOpened()]`

---

## 7. Controller Integration

| Location | What to implement |
|----------|-------------------|
| `MotionDriver.home()` | Send HOME command, wait for completion |
| `MotionDriver.set_params()` | Write velocity/accel/jerk to controller |
| `MotionDriver.run_cycle()` | Send CYCLE command, return True/False |
| `MotionDriver.stop()` | Emergency stop |
| `MotionDriver.record_trajectory()` | Enter teach/record mode |
| `ForceDAQ.read_lbs()` | Read one force sample, return float |
| `VisionDriver.capture()` | Read one camera frame, return JPEG bytes |
| `TestLoopThread.run()` | Replace `retract_ok = True` with sensor read |
| `DataLogger.record_cycle()` | Write to DB (InfluxDB, SQLite, CSV) |

---

## 8. Backend Hooks — ROI & Inspection Signals

This section is the primary integration reference for backend developers.

### 8.1 Architecture Principle

```
User clicks button
    → GUI emits *request* signal   (never mutates state)
        → Backend validates + persists
            → Backend calls *confirm_*() on ROIManager
                → GUI updates the row
```

The GUI ships with default self-wired confirm handlers so it works standalone for demos.
The backend disconnects those and connects its own handlers when integrating.

### 8.2 Accessing ROI Managers

```python
win = MainWindow()

cosmetic_roi = win.right.mod_cosmetic.roi_mgr   # Blue  — Camera C1
led_roi      = win.right.mod_led.roi_mgr        # Green — Camera C1
geometry_roi = win.right.mod_geometry.roi_mgr   # Orange — Camera C2
```

### 8.3 ROI Request Signals

#### `sig_add_roi_requested(module_key: str, shape: str)`
Fired on **+ Add ROI** click.

```python
cosmetic_roi.sig_add_roi_requested.connect(backend.on_add_roi)

def on_add_roi(module_key, shape):
    # 1. Enter draw mode on camera view
    # 2. Receive drawn coordinates
    roi_id = db.create_roi(module=module_key, shape=shape, coords=drawn_coords)
    # 3. Confirm to GUI
    cosmetic_roi.confirm_add(module_key, shape, roi_id=roi_id)
```

#### `sig_delete_roi_requested(module_key: str, roi_id: int)`
Fired on **Delete** click.

```python
def on_delete_roi(module_key, roi_id):
    db.delete_roi(roi_id)
    cosmetic_roi.confirm_delete(module_key, roi_id)
```

#### `sig_edit_roi_requested(module_key: str, roi_id: int)`
Fired on **Edit** click. Backend should re-enter draw/edit mode.

```python
def on_edit_roi(module_key, roi_id):
    camera_widget.start_edit_mode(roi_id)
    # After redraw: cosmetic_roi.update_roi_name(roi_id, new_name)
```

#### `sig_lock_roi_requested(module_key: str, roi_id: int, new_locked: bool)`
Fired on **Lock / Unlock** click.

```python
def on_lock_roi(module_key, roi_id, new_locked):
    db.set_roi_locked(roi_id, new_locked)
    cosmetic_roi.confirm_lock(module_key, roi_id, new_locked)
```

#### `sig_visibility_roi_changed(module_key: str, roi_id: int, is_visible: bool)`
Fired on 👁 eye icon click. **UI-only — does not affect processing.**

```python
def on_roi_visibility(module_key, roi_id, is_visible):
    camera_overlay.set_roi_visible(roi_id, is_visible)
    # No DB write needed — visibility is transient UI state
```

### 8.4 Backend Confirm Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `confirm_add` | `(module_key, shape, roi_id=None)` | Appends ROI row. `roi_id` overrides auto-increment. |
| `confirm_delete` | `(module_key, roi_id)` | Removes row and re-indexes remaining. |
| `confirm_lock` | `(module_key, roi_id, new_locked)` | Updates lock state and button label. |
| `confirm_visibility` | `(module_key, roi_id, is_visible)` | Updates eye icon. |
| `update_roi_name` | `(roi_id, name)` | Updates displayed name label on the row. |
| `set_module_enabled` | `(enabled: bool)` | Enables/disables Add ROI + shape picker. |

### 8.5 Taking Over from Default Handlers

```python
# Disconnect default standalone handlers
roi = cosmetic_roi
roi.sig_add_roi_requested.disconnect(roi._confirm_add)
roi.sig_delete_roi_requested.disconnect(roi._confirm_delete)
roi.sig_lock_roi_requested.disconnect(roi._confirm_lock)

# Connect backend handlers
roi.sig_add_roi_requested.connect(backend.on_add_roi)
roi.sig_delete_roi_requested.connect(backend.on_delete_roi)
roi.sig_lock_roi_requested.connect(backend.on_lock_roi)
```

### 8.6 Module Enable/Disable Hook

```python
# Option A: poll state
is_enabled = win.right.mod_cosmetic._enabled

# Option B: subclass (recommended)
class MyCosmetic(DefectModule):
    def _apply_content_state(self, enabled: bool):
        super()._apply_content_state(enabled)
        backend.set_pipeline_active("cosmetic", enabled)
```

### 8.7 Run Control Signals

| Signal | Source | Fired when |
|--------|--------|-----------|
| `sig_start` | `ControlPanel` | ▶ START clicked |
| `sig_pause` | `ControlPanel` | ❚❚ PAUSE clicked |
| `sig_stop` | `ControlPanel` | ■ STOP clicked |
| `sig_home` | `ControlPanel` | ⌂ HOME clicked |
| `sig_reset` | `ControlPanel` | ↺ RESET clicked |
| `sig_apply` | `ControlPanel` | ✔ Apply Settings clicked |
| `sig_camera_changed(camera: str)` | `RightPanel` | Camera radio button changed |
| `sig_camera_request(camera, msg)` | `DefectModule` | ROI added → switch camera |

### 8.8 State Badge Update (Backend → GUI)

```python
# Valid states: "IDLE", "PROCESSING...", "COMPLETE", "ERROR"
win.right.mod_cosmetic.set_state("PROCESSING...")
win.right.mod_led.set_state("COMPLETE")
win.right.mod_geometry.set_state("ERROR")
```

### 8.9 Full Backend Integration Template

```python
from robospeed_gui_main import MainWindow
from PyQt6.QtWidgets import QApplication

class RoboSpeedBackend:
    def __init__(self, win: MainWindow):
        self.win = win
        self._wire()

    def _wire(self):
        cp = self.win.left
        cp.sig_start.connect(self.on_start)
        cp.sig_pause.connect(self.on_pause)
        cp.sig_stop.connect(self.on_stop)
        cp.sig_apply.connect(self.on_apply_settings)
        self.win.right.sig_camera_changed.connect(self.on_camera_changed)

        for mod_key, mod in [
            ("cosmetic",  self.win.right.mod_cosmetic),
            ("led",       self.win.right.mod_led),
            ("geometry",  self.win.right.mod_geometry),
        ]:
            roi = mod.roi_mgr
            # Disconnect standalone handlers
            roi.sig_add_roi_requested.disconnect(roi._confirm_add)
            roi.sig_delete_roi_requested.disconnect(roi._confirm_delete)
            roi.sig_lock_roi_requested.disconnect(roi._confirm_lock)
            # Connect backend handlers
            roi.sig_add_roi_requested.connect(self.on_add_roi)
            roi.sig_delete_roi_requested.connect(self.on_delete_roi)
            roi.sig_edit_roi_requested.connect(self.on_edit_roi)
            roi.sig_lock_roi_requested.connect(self.on_lock_roi)
            roi.sig_visibility_roi_changed.connect(self.on_visibility_changed)

    def on_start(self):          pass  # TODO
    def on_pause(self):          pass  # TODO
    def on_stop(self):           pass  # TODO
    def on_apply_settings(self, fields): pass  # TODO
    def on_camera_changed(self, cam):    pass  # TODO

    def on_add_roi(self, module_key, shape):
        roi = self._roi(module_key)
        # TODO: enter draw mode, persist, then:
        roi._confirm_add(module_key, shape)     # fallback until draw is wired

    def on_delete_roi(self, module_key, roi_id):
        self._roi(module_key)._confirm_delete(module_key, roi_id)

    def on_edit_roi(self, module_key, roi_id):
        pass  # TODO: redraw mode

    def on_lock_roi(self, module_key, roi_id, new_locked):
        self._roi(module_key)._confirm_lock(module_key, roi_id, new_locked)

    def on_visibility_changed(self, module_key, roi_id, is_visible):
        self._roi(module_key)._confirm_visibility(module_key, roi_id, is_visible)

    def _roi(self, module_key):
        return getattr(self.win.right, f"mod_{module_key}").roi_mgr

if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    backend = RoboSpeedBackend(win)
    win.show()
    app.exec()
```

---

## 9. Configuration Reference

### Command-line arguments

```
python robospeed_controller.py [options]

  --sim                 Simulation mode, no hardware (default)
  --no-sim              Connect to real hardware
  --motion-port PORT    Serial port for motion controller  (default: COM3)
  --daq-port PORT       Serial port for force DAQ          (default: COM4)
  --cam-c1 INDEX        OpenCV camera index for C1         (default: 0)
  --cam-c2 INDEX        OpenCV camera index for C2         (default: 1)
```

### Default motion parameters

```python
velocity=300, acceleration=300, jerk=1000
target_cycles=100, baseline_cycles=30
force_min_lbs=0.5, force_max_lbs=1.8
```

### Inspection frequency defaults

```python
surface_capture_every=25, led_capture_every=25, point_cloud_capture_every=50
```

---

## 10. Data & Logging

Logs written to `logs/ProjectName_ProfileName/`. CSV columns: `cycle, timestamp, peak_force_lbs, force_ok, retract_ok`.

Image naming: `surface_cycle00025_C1.jpg`, `led_cycle00025_C1.jpg`, `point_cloud_cycle00050_C2.jpg`

---

## 11. Troubleshooting

| Problem | Fix |
|---------|-----|
| **GUI freezes on Open Profile after START→STOP** | **Fixed in v2.8.** Timer and thread were recreated on every `_set_run_mode_enabled()` call without stopping old ones → resource accumulation → event loop starvation. Now created once in `__init__`. |
| `Unknown property cursor` flood | Fixed in v2.7. Qt CSS does not support `cursor:`. |
| Enable button stuck / can't re-enable | Fixed in v2.7. Removed `setCheckable` (Windows theme conflict). |
| Checkbox black ghosting on hover | Fixed in v2.7. Added `:disabled`/`:hover` rules to `_chk_css()`. |
| `ModuleNotFoundError: No module named 'PyQt6'` | Activate venv + `pip install PyQt6 pyqtgraph numpy Pillow` |
| Window blank / grey | `pip install --upgrade pyqtgraph` |
| Serial port not found | Check Device Manager → Ports, use correct `--motion-port COMX` |
| Force graph not updating | `ForceDAQ.read_lbs()` must return a float — verify implementation |

---

## 12. Changelog

### v2.8 — Critical freeze/crash fix

**Root cause:** `_set_run_mode_enabled()` created a **new** `QTimer` and **new** `MockDataThread` on every call — from `on_start()`, `on_stop()`, `on_home()`, `on_reset()`. Old timers and threads were never stopped, only overwritten by new instance variable assignments. After START → STOP: 2 timers firing every 100 ms + 2 threads running simultaneously. The Qt event loop became overwhelmed, `QFileDialog` couldn't get CPU time, and the GUI froze.

**Fixes applied:**
- Timer and thread created **once** in `MainWindow.__init__()`, live for entire application lifetime
- `_set_run_mode_enabled()` now **only** manages widget enable/disable state — no resource creation
- `closeEvent()` stops both timer and thread on window close
- DefectModule re-enable respects each module's own `_enabled` flag
- Controller updated for v2.8 resource lifecycle (thread reliably exists at startup)

### v2.7 — PDF spec bug fixes (7 issues)

- **Bug 1:** Right panel clipping → `setMinimumWidth` instead of `setFixedWidth`
- **Bug 2:** Enable/disable broken → toggle row always live; only `_content_w` disabled
- **Bug 3:** Geometry button looks red → dark background, only border/text use module colour
- **Bug 4:** ROI addable when module disabled → `ROIManager.set_module_enabled(False)` gates add button
- **Bug 5:** Eye/visibility not working → `_update_eye()` in-place update, no full rebuild
- **Bug 6:** Signal architecture → 5 request signals, confirm pattern, backend takeover
- **Bug 7:** Profile save/load (`.rsprofile` JSON format)

### v2.6

- Pause icon updated (thicker, matches stop)
- Bottom bar two rows
- Save/Open Profile buttons in identity bar

---

## Quick Start Checklist

```
☐  Python 3.10+ installed
☐  venv activated  (.venv\Scripts\Activate.ps1)
☐  pip install PyQt6 pyqtgraph numpy Pillow
☐  robospeed_logo.png in same folder as .py files
☐  python robospeed_gui_main.py
☐  Press START — graph scrolls, cycle counter increments
☐  Press STOP — then Open Profile — no freeze (v2.8 fix)
☐  Disable a module → section dims to 35%, inputs locked
☐  Re-enable → full brightness, inputs unlocked
☐  Add ROI → eye icon toggles visibility per row
☐  (Backend) Copy RoboSpeedBackend template from Section 8.9
☐  (Backend) Disconnect default handlers, connect your own
☐  (Hardware) pip install pyserial opencv-python
☐  (Hardware) Fill TODO sections in robospeed_controller.py
```

---

*RoboSpeed Durability Intelligence Platform · v2.8 · Stage D*