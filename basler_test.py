from pypylon import pylon
import cv2
import numpy as np

# ----------------------------
# Camera setup
# ----------------------------
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Load UserSet1 if available
try:
    camera.UserSetSelector.SetValue("UserSet1")
    camera.UserSetLoad.Execute()
except:
    pass

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Converter (Bayer12 → BGR)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# ----------------------------
# Zoom / Pan variables
# ----------------------------
zoom = 1.0
pan_x, pan_y = 0, 0
dragging = False
last_x, last_y = 0, 0

WINDOW_NAME = "Basler Zoom Viewer"

# ----------------------------
# Mouse callback
# ----------------------------
def mouse_callback(event, x, y, flags, param):
    global zoom, pan_x, pan_y, dragging, last_x, last_y

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom *= 1.1
        else:
            zoom /= 1.1
        zoom = max(1.0, min(zoom, 10.0))

    elif event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - last_x
        dy = y - last_y
        pan_x -= dx / zoom
        pan_y -= dy / zoom
        last_x, last_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("Controls: Mouse wheel = zoom, drag = pan, q = quit")

# ----------------------------
# Main loop
# ----------------------------
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        frame = image.GetArray()

        h, w = frame.shape[:2]

        # Calculate crop region based on zoom & pan
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        cx = int(w / 2 + pan_x)
        cy = int(h / 2 + pan_y)

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        cropped = frame[y1:y2, x1:x2]

        # Resize to window size
        display = cv2.resize(cropped, (1280, 800))

        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
