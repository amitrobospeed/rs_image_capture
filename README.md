# RoboSpeed Durability Intelligence Platform
### v2.4 · Dark Industrial GUI · PyQt6 + PyQtGraph

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
8. [Configuration Reference](#8-configuration-reference)
9. [Data & Logging](#9-data--logging)
10. [Troubleshooting](#10-troubleshooting)

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
├── logs/                        ← Auto-created on first test run
│   └── ProjectName_Profile/
│       ├── report.csv
│       ├── surface_cycle00025_C1.jpg
│       └── ...
└── .venv/                       ← Python virtual environment
```

> **Logo files must be in the same folder as the `.py` files.**
> The app searches automatically — no config needed.

---

## 2. Requirements

### Python version
- **Python 3.10 or higher** (3.11 recommended)
- Python 3.12 works, Python 3.9 and below are **not supported**

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
- **Windows 10 / 11** ✓ (primary target)
- **macOS 12+** ✓
- **Linux (Ubuntu 22.04+)** ✓

---

## 3. Installation

### Step 1 — Create a virtual environment

Open PowerShell in your project folder:

```powershell
cd C:\Users\anilp\OneDrive\Desktop\gui

# Create venv (only needed once)
python -m venv .venv

# Activate it (do this every time you open a new terminal)
.venv\Scripts\Activate.ps1
```

You will see `(.venv)` appear at the start of your prompt — this means it is active.

> **Windows execution policy error?** Run this once in PowerShell as Administrator:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 2 — Install packages

With the venv activated:

```powershell
# Core GUI packages (always required)
pip install PyQt6 pyqtgraph numpy Pillow

# Hardware packages (only needed when connecting real hardware)
pip install pyserial opencv-python
```

### Step 3 — Verify installation

```powershell
python -c "import PyQt6; import pyqtgraph; import numpy; import PIL; print('All OK')"
```

You should see: `All OK`

---

## 4. Running the Software

### Simulation mode (no hardware — for development and testing)

This runs the GUI with a live simulated force waveform. No robot or DAQ needed.

```powershell
# Make sure venv is active first
.venv\Scripts\Activate.ps1

# Run the GUI directly
python robospeed_gui_main.py
```

You will see a startup banner in the terminal:

```
╔══════════════════════════════════════════════════════════╗
║      RoboSpeed Durability Intelligence Platform  v2.4    ║
╠══════════════════════════════════════════════════════════╣
║  Logo  : C:\...\gui\robospeed_logo.png                   ║
║  Icon  : C:\...\gui\favicon_64.png                       ║
║  Theme : Dark Industrial  |  PyQt6 + PyQtGraph           ║
╚══════════════════════════════════════════════════════════╝
```

### With real hardware (using the controller)

```powershell
# Default — simulation (same as running gui_main.py directly)
python robospeed_controller.py --sim

# Real hardware on specific COM ports
python robospeed_controller.py --no-sim --motion-port COM3 --daq-port COM4

# Real hardware with cameras
python robospeed_controller.py --no-sim --motion-port COM3 --daq-port COM4 --cam-c1 0 --cam-c2 1
```

---

## 5. GUI Walkthrough

### Identity bar (top)
| Element | What it does |
|---------|-------------|
| **Project** field | Name of the current project (e.g. "Button Toy") |
| **Test Profile** field | Name of the test config (e.g. "1.5lb Cycle Test") |
| **Save** button | Saves current profile name to memory |
| **◀ Controls** | Hides/shows the left panel to give graphs more space |
| **Insights ▶** | Hides/shows the right panel |

### Left panel — Test Run Controls

| Button | Colour | Action |
|--------|--------|--------|
| **▶ START** | Deep Emerald | Starts the test. Applies current settings first. |
| **‖ PAUSE** | Amber | Pauses mid-cycle. Robot holds position. |
| **■ STOP** | Deep Red | Immediately stops robot and ends test. |
| **⌂ HOME** | Grey | Sends robot to home position. |
| **↺ RESET** | Grey | Clears cycle counter and failure counts. |
| **● RECORD TRAJECTORY** | Grey | Puts robot in teach mode to record a new path. |
| **↓ DOWNLOAD REPORT** | Grey | Generates CSV report to the `logs/` folder. |
| **✕ EXIT** | Dark Grey | Closes the application safely. |

### Left panel — Motion Control (collapsible)

Click the **▼ MOTION CONTROL** header to expand/collapse.

| Field | Unit | Default | Description |
|-------|------|---------|-------------|
| Velocity | — | 300 | Maximum joint velocity |
| Accel | — | 300 | Acceleration ramp |
| Jerk | — | 1000 | Jerk limit (smoothness) |
| Cycles | count | 100 | Total target test cycles |
| Baseline cycles | count | 30 | Cycles used to learn normal force pattern |
| Force Min | lbs | 0.5 | Lower bound of acceptable force range |
| Force Max | lbs | 1.8 | Upper bound of acceptable force range |

Click **✔ Apply Settings** (blue) to send values to the controller.

### Centre panels

**Live Force Monitor (left graph)**
- Plots force in lbs vs time or cycle number (toggle with buttons top-right)
- Blue dashed horizontal lines show the force band (min/max)
- Green dots = force within range, red dots = force out of range
- Progress bar at the bottom: blue→green gradient fills as cycles complete
- `Learning Baseline X/30` label shows baseline collection progress

**Visual Inspection System (right panel)**
- Live camera feed from C1, C2, or split view (selected in right panel)
- Status dot top-right: green = connected, red = disconnected
- `Active feed: C1` label at the bottom confirms which camera is shown

### Right panel — Insights & Controls

**DURABILITY INSIGHTS** — toggle which inspection pipelines are active and which analysis types to run.

**VISUAL CONTROLS**
- Camera selection: `C1` / `C2` / `Split View` radio buttons
- Overlays: toggle ROI, Feature Tracking, Defect Highlight, Bounding Boxes, LED Tracking, Mesh Overlay

**INSPECTION FREQUENCY** — how often (in cycles) to capture each type of image.

**AI ANALYST** — type a question about the current test run and press **Ask AI** for analysis.

### Bottom status bar
Shows current state (STOPPED / RUNNING / PAUSED), cycle count, motion parameters, and per-button failure counts.

---

## 6. Connecting Hardware

### Architecture overview

```
PC  ──USB/Serial──►  Motion Controller  ──►  Robot Arm
PC  ──USB/Serial──►  Force DAQ           ──►  Force Sensor
PC  ──USB───────►  Camera C1  (surface + LED inspection)
PC  ──USB───────►  Camera C2  (3D geometry / point cloud)
```

### Motion controller serial connection

1. Connect the motion controller to the PC via USB or RS-232.
2. Open **Device Manager** → **Ports (COM & LPT)** to find the COM port number.
3. Run with the correct port:
   ```powershell
   python robospeed_controller.py --no-sim --motion-port COM3
   ```
4. In `robospeed_controller.py`, find `MotionDriver.run_cycle()` and add your serial protocol:
   ```python
   def run_cycle(self) -> bool:
       self._ser.write(b"CYCLE\r\n")          # send your command
       response = self._ser.readline()         # wait for completion acknowledgement
       return response.strip() == b"OK"
   ```

### Force DAQ serial connection

1. Connect the force DAQ USB.
2. Find its COM port in Device Manager.
3. Run with:
   ```powershell
   python robospeed_controller.py --no-sim --motion-port COM3 --daq-port COM4
   ```
4. In `robospeed_controller.py`, find `ForceDAQ.read_lbs()` and add your read command:
   ```python
   def read_lbs(self) -> float:
       self._ser.write(b"READ\r\n")
       raw = self._ser.readline().strip()
       return float(raw)                       # parse your DAQ output format
   ```

### Camera connection

Cameras are accessed via OpenCV. USB webcams are detected automatically.

```powershell
# C1 on USB index 0, C2 on USB index 1 (default)
python robospeed_controller.py --no-sim --cam-c1 0 --cam-c2 1
```

To find which index your camera is on, run:
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
```

### Retract sensor

The button retract sensor (detects whether the button returned to rest position) is read inside `TestLoopThread.run()`. Find this line and add your sensor read:

```python
retract_ok = True   # TODO: read retract sensor
# Replace with:
retract_ok = self._motion.read_retract_sensor()
```

---

## 7. Controller Integration

The controller file (`robospeed_controller.py`) has three integration points marked with `# TODO`:

| Location | What to implement |
|----------|-------------------|
| `MotionDriver.home()` | Send HOME command via serial, wait for completion |
| `MotionDriver.set_params()` | Write velocity/accel/jerk to controller registers |
| `MotionDriver.run_cycle()` | Send CYCLE command, poll until cycle complete, return True/False |
| `MotionDriver.stop()` | Send emergency stop command |
| `MotionDriver.record_trajectory()` | Put controller in teach/record mode |
| `ForceDAQ.read_lbs()` | Read one force sample and return as float (lbs) |
| `VisionDriver.capture()` | Read one frame from OpenCV cap, return JPEG bytes |
| `TestLoopThread.run()` | Replace `retract_ok = True` with real sensor read |
| `DataLogger.record_cycle()` | Add database write (InfluxDB, SQLite, CSV — your choice) |

All Qt signal wiring between the GUI and controller is already done. You only need to fill in the hardware communication.

---

## 8. Configuration Reference

### Command-line arguments (controller)

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
velocity        = 300     # (0–1000)
acceleration    = 300     # (0–2000)
jerk            = 1000    # (0–10000)
target_cycles   = 100     # (1–99999)
baseline_cycles = 30      # (1–500)
force_min_lbs   = 0.5     # (0–100 lbs)
force_max_lbs   = 1.8     # (0–100 lbs)
```

### Inspection frequency defaults

```python
surface_capture_every       = 25    # capture surface image every 25 cycles
led_capture_every           = 25    # capture LED image every 25 cycles
point_cloud_capture_every   = 50    # capture 3D scan every 50 cycles
```

Set to `0` to disable that capture type entirely.

---

## 9. Data & Logging

### Log folder location

Logs are written automatically to:
```
logs/ProjectName_ProfileName/
```

For example, project "Button Toy" and profile "1.5lb Cycle Test" writes to:
```
logs/Button_Toy_1.5lb_Cycle_Test/
```

### CSV report format

Generated by clicking **↓ DOWNLOAD REPORT** or `DataLogger.generate_report()`:

```csv
cycle,timestamp,peak_force_lbs,force_ok,retract_ok
1,0.312,0.0423,True,True
2,0.624,1.1832,True,True
3,0.936,2.2104,False,True    ← force out of range
...
```

### Inspection image naming

```
surface_cycle00025_C1.jpg       ← surface capture at cycle 25, camera C1
led_cycle00025_C1.jpg           ← LED capture at cycle 25
point_cloud_cycle00050_C2.jpg   ← 3D scan at cycle 50, camera C2
```

---

## 10. Troubleshooting

### `KeyboardInterrupt` on startup (numpy/import)

You pressed **Ctrl+C** while the app was loading. NumPy takes 1–2 seconds to import on first run. Just run the command again cleanly — do not press anything until the window appears.

```powershell
python robospeed_gui_main.py
```

---

### `ModuleNotFoundError: No module named 'PyQt6'`

Your venv is not activated, or packages are not installed in it.

```powershell
# Activate first
.venv\Scripts\Activate.ps1

# Then install
pip install PyQt6 pyqtgraph numpy Pillow
```

---

### Window opens but is blank / grey

PyQtGraph initialisation issue. Try:
```powershell
pip install --upgrade pyqtgraph
```

---

### Logo shows a dark rectangle around it

The logo PNG has a pure black background. The app uses PIL to replace near-black pixels with the panel colour automatically. Make sure **Pillow is installed**:
```powershell
pip install Pillow
```

---

### Serial port not found (COM port error)

1. Open **Device Manager** → expand **Ports (COM & LPT)**
2. Check which COM number your device shows
3. Run with the correct port:
   ```powershell
   python robospeed_controller.py --no-sim --motion-port COM5
   ```
4. If the port shows but gives "Access Denied" — close any other program (Arduino IDE, PuTTY, etc.) that has it open.

---

### Camera not opening (index error)

Run the camera finder script to find the correct index:
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera at index {i}")
        cap.release()
```

Then pass the correct index:
```powershell
python robospeed_controller.py --no-sim --cam-c1 2
```

---

### Force graph not updating

In sim mode the graph always updates. In hardware mode the `ForceDAQ.read_lbs()` method must return a float. Check that your DAQ serial port is correct and the `read_lbs()` method is implemented.

---

### High CPU usage

PyQtGraph is set to 50 Hz sampling. If CPU is high, reduce the sampling rate in `MockDataThread.run()` (sim) or in the `TestLoopThread` force sampling loop:

```python
# In TestLoopThread.run() — change 0.02 to 0.05 for 20 Hz
time.sleep(0.05)
```

---

## Quick Start Checklist

```
☐  Python 3.10+ installed
☐  venv created and activated  (.venv\Scripts\Activate.ps1)
☐  pip install PyQt6 pyqtgraph numpy Pillow
☐  robospeed_logo.png in same folder as .py files
☐  Run:  python robospeed_gui_main.py
☐  Window appears with live force graph and dark theme
☐  (Hardware) pip install pyserial opencv-python
☐  (Hardware) Find COM ports in Device Manager
☐  (Hardware) Fill in TODO sections in robospeed_controller.py
☐  (Hardware) Run:  python robospeed_controller.py --no-sim --motion-port COMX
```

---

*RoboSpeed Durability Intelligence Platform · v2.4 · Stage D*
