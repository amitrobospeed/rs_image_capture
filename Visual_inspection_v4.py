import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

# =============================
# CONFIG
# =============================
WIDTH = 1280
HEIGHT = 720
FPS = 15
CAPTURE_SECONDS = 1.0
OUTPUT_DIR = "inspection_output"

# Glossy white targets
TARGET_LUMA_MEAN = 95
MAX_SAT_PCT = 3.0
TUNE_MAX_ITERS = 10

EXPOSURE_MIN = 2000
EXPOSURE_MAX = 15000
GAIN_MIN = 0
GAIN_MAX = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# REALSENSE SETUP
# =============================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
profile = pipeline.start(config)

time.sleep(1)

device = profile.get_device()
color_sensor = device.first_color_sensor()

color_sensor.set_option(rs.option.enable_auto_exposure, 0)

# Safe starting values for glossy white
color_sensor.set_option(rs.option.exposure, 4500)
color_sensor.set_option(rs.option.gain, 8)

print("RealSense started (Stabilized Glossy White Mode)")

def get_frame(timeout=10000):
    try:
        frames = pipeline.wait_for_frames(timeout)
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())
    except:
        return None

# Warmup
for _ in range(20):
    _ = get_frame()

# =============================
# ROBUST CAPTURE
# =============================
def capture_average(label=None):

    frames = []
    start = time.time()
    timeout_limit = 2.0

    print("Capturing...")

    while time.time() - start < timeout_limit:

        f = get_frame()
        if f is not None:
            frames.append(f)

        if len(frames) >= FPS:
            break

    if len(frames) < 3:
        print("Frame stream unstable. Captured:", len(frames))
        return None

    avg = np.mean(frames, axis=0).astype(np.uint8)

    if label:
        cv2.putText(avg, label, (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0,255,255), 4)

    print("Captured frames:", len(frames))
    return avg

# =============================
# LUMINANCE STATS
# =============================
def compute_luma_stats(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    sat_pct = float(np.mean(gray >= 250) * 100.0)
    return mean_luma, sat_pct

# =============================
# AUTO TUNE
# =============================
def auto_tune_exposure_from_golden(golden):

    print("AUTO-TUNE STARTED")

    exp = int(color_sensor.get_option(rs.option.exposure))
    gain = int(color_sensor.get_option(rs.option.gain))

    for i in range(TUNE_MAX_ITERS):

        mean_l, sat = compute_luma_stats(golden)

        print(f"[Iter {i+1}] mean={mean_l:.1f}, sat%={sat:.2f}, exp={exp}, gain={gain}")

        if sat > MAX_SAT_PCT:
            exp = int(exp * 0.85)
            gain = int(gain * 0.9)
        else:
            if mean_l > TARGET_LUMA_MEAN + 5:
                exp = int(exp * 0.9)
            elif mean_l < TARGET_LUMA_MEAN - 5:
                exp = int(exp * 1.05)
            else:
                break

        exp = int(np.clip(exp, EXPOSURE_MIN, EXPOSURE_MAX))
        gain = int(np.clip(gain, GAIN_MIN, GAIN_MAX))

        color_sensor.set_option(rs.option.exposure, exp)
        color_sensor.set_option(rs.option.gain, gain)

        time.sleep(0.15)

        frame = get_frame()
        if frame is not None:
            golden = frame

    print("AUTO-TUNE COMPLETE")
    print("Final Exposure:", exp)
    print("Final Gain:", gain)

# =============================
# INSPECTION (Stable Version)
# =============================
def run_inspection(golden, cyc, roi):

    if roi is None:
        print("No ROI selected.")
        return

    x,y,w,h = roi

    golden_roi = golden[y:y+h, x:x+w]
    cyc_roi = cyc[y:y+h, x:x+w]

    g_gray = cv2.cvtColor(golden_roi, cv2.COLOR_BGR2GRAY)
    c_gray = cv2.cvtColor(cyc_roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_norm = clahe.apply(g_gray)
    c_norm = clahe.apply(c_gray)

    g_blur = cv2.GaussianBlur(g_norm, (5,5), 0)
    c_blur = cv2.GaussianBlur(c_norm, (5,5), 0)

    diff = cv2.absdiff(g_blur, c_blur)

    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours,_ = cv2.findContours(mask,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    output = cyc.copy()
    MIN_DEFECT_AREA = 15

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > MIN_DEFECT_AREA:
            rx,ry,rw,rh = cv2.boundingRect(largest)
            cv2.rectangle(output,
                          (x+rx,y+ry),
                          (x+rx+rw,y+ry+rh),
                          (0,0,255),1)

    cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),1)

    cv2.imwrite(os.path.join(OUTPUT_DIR,"Diff_mask.png"),mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR,"Frame_anamoly.png"),output)

    print("Inspection complete.")

# =============================
# UI
# =============================
golden_frame=None
cyc_frame=None
roi=None
status_text="Ready"

BTN_GOLDEN=(20,20,150,60)
BTN_CYCLE=(180,20,360,60)
BTN_RUN=(390,20,600,60)

def draw_button(img,rect,text):
    x1,y1,x2,y2=rect
    cv2.rectangle(img,(x1,y1),(x2,y2),(70,70,70),-1)
    cv2.rectangle(img,(x1,y1),(x2,y2),(150,150,150),1)
    cv2.putText(img,text,(x1+10,y1+35),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,
                (255,255,255),2)

def mouse_callback(event,x,y,flags,param):
    global golden_frame, cyc_frame, roi, status_text

    if event == cv2.EVENT_LBUTTONUP:

        if BTN_GOLDEN[0]<x<BTN_GOLDEN[2] and BTN_GOLDEN[1]<y<BTN_GOLDEN[3]:

            status_text="Capturing Golden..."
            golden_frame = capture_average()

            if golden_frame is not None:

                print("Starting auto-tune...")
                auto_tune_exposure_from_golden(golden_frame)

                print("Waiting for stabilization...")
                time.sleep(0.7)

                # Flush unstable frames
                for _ in range(20):
                    _ = get_frame()

                print("Re-capturing stabilized Golden...")
                golden_frame = capture_average()

                if golden_frame is not None:
                    cv2.imwrite(os.path.join(OUTPUT_DIR,"Golden_avg.png"),
                                golden_frame)
                    status_text="Golden saved (stable). Press R to select ROI."

        elif BTN_CYCLE[0]<x<BTN_CYCLE[2] and BTN_CYCLE[1]<y<BTN_CYCLE[3]:
            status_text="Capturing CYC..."
            cyc_frame = capture_average(label="CYC")
            if cyc_frame is not None:
                cv2.imwrite(os.path.join(OUTPUT_DIR,"CYC_avg.png"),
                            cyc_frame)
                status_text="CYC saved."

        elif BTN_RUN[0]<x<BTN_RUN[2] and BTN_RUN[1]<y<BTN_RUN[3]:
            if golden_frame is not None and cyc_frame is not None:
                status_text="Running inspection..."
                run_inspection(golden_frame,cyc_frame,roi)
                status_text="Inspection done."
            else:
                status_text="Capture Golden and CYC first."

cv2.namedWindow("Inspection")
cv2.setMouseCallback("Inspection",mouse_callback)

try:
    while True:
        frame = get_frame()
        if frame is None:
            continue

        display=frame.copy()

        overlay=display.copy()
        cv2.rectangle(overlay,(0,0),(WIDTH,80),(0,0,0),-1)
        display=cv2.addWeighted(overlay,0.6,display,0.4,0)

        draw_button(display,BTN_GOLDEN,"Golden")
        draw_button(display,BTN_CYCLE,"Cycle count")
        draw_button(display,BTN_RUN,"Run Inspection")

        if roi is not None:
            x,y,w,h=roi
            cv2.rectangle(display,(x,y),(x+w,y+h),(255,0,0),1)

        cv2.putText(display,status_text,(20,HEIGHT-20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,
                    (255,255,255),2)

        cv2.imshow("Inspection",display)

        key=cv2.waitKey(1)&0xFF

        if key==27:
            break

        if key==ord('r') and golden_frame is not None:
            cv2.destroyWindow("Inspection")
            roi=cv2.selectROI("ROI Selector",golden_frame,False,False)
            cv2.destroyWindow("ROI Selector")
            cv2.namedWindow("Inspection")
            cv2.setMouseCallback("Inspection",mouse_callback)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
