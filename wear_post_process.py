#!/usr/bin/env python3
"""Post-process cycle inspection videos to measure print wear trends.

Workflow
1) Open input video and read first frame as baseline.
2) User selects button ROIs (A/B/C/D by default) on baseline frame.
3) For each button ROI, user samples:
   - plastic color patch
   - print color patch
4) For every frame, compute print mask per button ROI with no specular rejection
   and no morphological cleanup, then report non-zero pixels and % drop from baseline.

Outputs
- wear_metrics.csv
- wear_summary.json
- wear_overlay.mp4 (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ButtonCalibration:
    roi: object
    plastic_lab: np.ndarray
    print_lab: np.ndarray
    baseline_nnz: int = 0


def _roi_bounds(roi_like, shape):
    if isinstance(roi_like, dict):
        rshape = roi_like.get("shape", "rect")
        if rshape == "rect":
            return (
                int(roi_like.get("x", 0)),
                int(roi_like.get("y", 0)),
                int(roi_like.get("w", 0)),
                int(roi_like.get("h", 0)),
            )
        if rshape == "circle":
            cx = int(roi_like.get("cx", 0))
            cy = int(roi_like.get("cy", 0))
            r = int(roi_like.get("r", 0))
            return (cx - r, cy - r, 2 * r, 2 * r)
        if rshape == "poly":
            pts = np.array(roi_like.get("points", []), dtype=np.int32)
            if pts.size == 0:
                return (0, 0, 1, 1)
            x, y, w, h = cv2.boundingRect(pts)
            return (int(x), int(y), int(w), int(h))
    return tuple(map(int, roi_like))


def _sanitize_roi(roi_like, shape):
    x, y, w, h = _roi_bounds(roi_like, shape)
    h_img, w_img = shape[:2]
    x = max(0, min(int(x), max(0, w_img - 1)))
    y = max(0, min(int(y), max(0, h_img - 1)))
    w = max(1, min(int(w), w_img - x))
    h = max(1, min(int(h), h_img - y))
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


def _draw_roi(frame: np.ndarray, roi_like, color=(255, 0, 0), label: Optional[str] = None):
    roi = _sanitize_roi(roi_like, frame.shape)
    if isinstance(roi, dict):
        if roi.get("shape") == "circle":
            cv2.circle(frame, (roi["cx"], roi["cy"]), roi["r"], color, 1)
            lx, ly = roi["cx"] - roi["r"], roi["cy"] - roi["r"]
        elif roi.get("shape") == "poly":
            pts = np.array(roi.get("points", []), dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 1)
            lx, ly = int(pts[:, 0].min()), int(pts[:, 1].min())
        else:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            lx, ly = x, y
    else:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        lx, ly = x, y
    if label:
        cv2.putText(frame, label, (lx + 2, max(14, ly - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _roi_mask_from_spec(shape, roi_like):
    roi = _sanitize_roi(roi_like, shape)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if isinstance(roi, dict):
        rshape = roi.get("shape", "rect")
        if rshape == "circle":
            cv2.circle(mask, (roi["cx"], roi["cy"]), roi["r"], 255, -1)
            x, y, w, h = _roi_bounds(roi, shape)
            return mask, (x, y, w, h)
        if rshape == "poly":
            pts = np.array(roi.get("points", []), dtype=np.int32)
            if pts.size > 0:
                cv2.fillPoly(mask, [pts], 255)
                x, y, w, h = cv2.boundingRect(pts)
                return mask, (int(x), int(y), int(w), int(h))
            return mask, (0, 0, 1, 1)
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return mask, (x, y, w, h)
    x, y, w, h = roi
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask, (x, y, w, h)


def _select_rois(frame: np.ndarray, buttons: List[str]) -> Dict[str, object]:
    selected: Dict[str, object] = {}
    shape_mode = "rect"
    start_pt = None
    drag_pt = None
    poly_pts = []
    working_roi = None
    actions = {}
    current_idx = 0
    roi_targets = list(buttons)
    status_msg = "Draw ROI then Next; select all buttons then Lock All"

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
        return roi_targets[current_idx]

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
        missing = [b for b in roi_targets if b not in selected]
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
                    current_idx = (current_idx + 1) % len(roi_targets)
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
            if start_pt is not None and shape_mode in ("rect", "circle"):
                drag_pt = (x, y)
        elif evt == cv2.EVENT_LBUTTONUP:
            if start_pt is None or drag_pt is None or shape_mode == "poly":
                return
            x0, y0 = start_pt
            x1, y1 = drag_pt
            if shape_mode == "rect":
                rx, ry = min(x0, x1), min(y0, y1)
                rw, rh = abs(x1 - x0), abs(y1 - y0)
                working_roi = {"shape": "rect", "x": rx, "y": ry, "w": rw, "h": rh}
            else:
                cx = int((x0 + x1) / 2)
                cy = int((y0 + y1) / 2)
                r = int(max(abs(x1 - x0), abs(y1 - y0)) / 2)
                working_roi = {"shape": "circle", "cx": cx, "cy": cy, "r": r}
            start_pt = None
            drag_pt = None
            status_msg = f"{_target_btn()} ROI ready"

    win = "Wear ROI Selector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_cb)

    while True:
        canvas = frame.copy()
        _draw_buttons(canvas)
        cv2.putText(canvas, f"Target: {_target_btn()} | mode={shape_mode}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        cv2.putText(canvas, status_msg, (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        for b, r in selected.items():
            _draw_roi(canvas, r, color=(80, 220, 120), label=b)

        if shape_mode == "poly" and poly_pts:
            for i, p in enumerate(poly_pts):
                cv2.circle(canvas, p, 3, (0, 215, 255), -1)
                if i > 0:
                    cv2.line(canvas, poly_pts[i - 1], p, (0, 215, 255), 1)
        elif working_roi is not None:
            _draw_roi(canvas, working_roi, color=(0, 215, 255), label=f"{_target_btn()}*")
        elif start_pt is not None and drag_pt is not None and shape_mode in ("rect", "circle"):
            x0, y0 = start_pt
            x1, y1 = drag_pt
            if shape_mode == "rect":
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 215, 255), 1)
            else:
                cx = int((x0 + x1) / 2)
                cy = int((y0 + y1) / 2)
                r = int(max(abs(x1 - x0), abs(y1 - y0)) / 2)
                cv2.circle(canvas, (cx, cy), r, (0, 215, 255), 1)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if shape_mode == "poly":
                _finalize_working_from_poly()
            continue
        if key == 8:
            if shape_mode == "poly" and poly_pts:
                poly_pts.pop()
                working_roi = None
                status_msg = f"{_target_btn()} polygon points: {len(poly_pts)}"
            continue
        if key in (27, ord("q")):
            cv2.destroyWindow(win)
            raise RuntimeError("ROI selection canceled")
        if key in (ord("l"), ord("L")):
            if _try_lock_all():
                break

    cv2.destroyWindow(win)
    return {b: _sanitize_roi(selected[b], frame.shape) for b in roi_targets}


def _mean_lab_from_patch(frame: np.ndarray, title: str) -> np.ndarray:
    x, y, w, h = cv2.selectROI(title, frame, False, False)
    cv2.destroyWindow(title)
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Patch not selected: {title}")
    patch = frame[int(y):int(y + h), int(x):int(x + w)]
    if patch.size == 0:
        raise RuntimeError(f"Empty patch selected: {title}")
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    return patch_lab.reshape(-1, 3).mean(axis=0)


def _collect_calibration(frame0: np.ndarray, rois: Dict[str, object]) -> Dict[str, ButtonCalibration]:
    calib: Dict[str, ButtonCalibration] = {}
    for btn, roi in rois.items():
        _mask, (x, y, w, h) = _roi_mask_from_spec(frame0.shape, roi)
        crop = frame0[y:y + h, x:x + w]
        plastic_lab = _mean_lab_from_patch(crop, f"{btn}: Select PLASTIC color patch")
        print_lab = _mean_lab_from_patch(crop, f"{btn}: Select PRINT color patch")
        calib[btn] = ButtonCalibration(roi=roi, plastic_lab=plastic_lab, print_lab=print_lab)
    return calib


def _compute_print_mask(crop_bgr: np.ndarray, plastic_lab: np.ndarray, print_lab: np.ndarray,
                        print_tol: float, plastic_tol: float) -> np.ndarray:
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d_print = np.linalg.norm(lab - print_lab.reshape(1, 1, 3).astype(np.float32), axis=2)
    d_plastic = np.linalg.norm(lab - plastic_lab.reshape(1, 1, 3).astype(np.float32), axis=2)

    # Wear mask logic: print-like pixels are 1, plastic-like are 0.
    # No specular rejection and no morphology cleanup by design.
    print_like = d_print <= float(print_tol)
    plastic_like = d_plastic <= float(plastic_tol)
    closer_to_print = d_print < d_plastic
    mask = print_like & (~plastic_like) & closer_to_print
    return (mask.astype(np.uint8) * 255)


def _pick_video_path() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        path = filedialog.askopenfilename(
            title="Select inspection video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            return Path(path)
    except Exception:
        pass

    try:
        txt = input("Enter video path: ").strip()
    except Exception:
        txt = ""
    return Path(txt) if txt else None


def _default_output_dir(video_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("inspection_output") / "post_process" / "print_wear_detect" / f"{video_path.stem}_{ts}"


def run(video_path: Path, out_dir: Path, buttons: List[str], print_tol: float,
        plastic_tol: float, wear_threshold_pct: float, max_frames: int,
        save_overlay: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError("Could not read first frame from video")

    print("Select button ROIs on baseline frame...")
    rois = _select_rois(frame0, buttons)
    print("Select plastic/print patches for each button ROI...")
    calib = _collect_calibration(frame0, rois)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    writer = None
    if save_overlay:
        writer = cv2.VideoWriter(str(out_dir / "wear_overlay.mp4"), fourcc, float(max(1.0, fps)),
                                 (frame0.shape[1], frame0.shape[0]))

    rows = []

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        rec = {"frame_idx": frame_idx, "time_s": frame_idx / float(max(fps, 1e-6))}
        overlay = frame.copy()

        for btn in buttons:
            c = calib[btn]
            roi_mask, (x, y, w, h) = _roi_mask_from_spec(frame.shape, c.roi)
            crop = frame[y:y + h, x:x + w]
            local_roi = roi_mask[y:y + h, x:x + w] > 0
            if crop.size == 0 or local_roi.size == 0 or np.count_nonzero(local_roi) == 0:
                nnz = 0
            else:
                mask = _compute_print_mask(crop, c.plastic_lab, c.print_lab, print_tol, plastic_tol)
                nnz = int(np.count_nonzero((mask > 0) & local_roi))

            if frame_idx == 0:
                c.baseline_nnz = max(1, nnz)

            drop_pct = max(0.0, (float(c.baseline_nnz) - float(nnz)) / float(max(c.baseline_nnz, 1)) * 100.0)
            verdict = "FAIL" if drop_pct >= wear_threshold_pct else "PASS"

            rec[f"nnz_{btn}"] = nnz
            rec[f"drop_pct_{btn}"] = round(drop_pct, 4)
            rec[f"wear_{btn}"] = verdict

            _draw_roi(overlay, c.roi, color=(255, 220, 0), label=btn)
            cv2.putText(
                overlay,
                f"{btn} nnz={nnz} drop={drop_pct:.1f}% {verdict}",
                (x + 4, max(16, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255) if verdict == "PASS" else (0, 0, 255),
                1,
            )

        rows.append(rec)
        if writer is not None:
            writer.write(overlay)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    csv_path = out_dir / "wear_metrics.csv"
    fieldnames = ["frame_idx", "time_s"]
    for btn in buttons:
        fieldnames += [f"nnz_{btn}", f"drop_pct_{btn}", f"wear_{btn}"]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "video": str(video_path),
        "buttons": buttons,
        "print_tol": print_tol,
        "plastic_tol": plastic_tol,
        "wear_threshold_pct": wear_threshold_pct,
        "baseline_nnz": {btn: int(calib[btn].baseline_nnz) for btn in buttons},
        "frames_processed": frame_idx,
        "csv": str(csv_path),
        "overlay_video": str(out_dir / "wear_overlay.mp4") if save_overlay else "",
    }
    (out_dir / "wear_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Done. Processed {frame_idx} frames")
    print(f"CSV: {csv_path}")
    print(f"Summary: {out_dir / 'wear_summary.json'}")
    if save_overlay:
        print(f"Overlay: {out_dir / 'wear_overlay.mp4'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Post-process wear detection on inspection video")
    p.add_argument("--video", default="", help="Path to input raw/labeled video (optional; prompts if omitted)")
    p.add_argument("--out-dir", default="", help="Output folder (default: inspection_output/post_process/print_wear_detect/<video>_<timestamp>)")
    p.add_argument("--buttons", nargs="+", default=["A", "B", "C", "D"], help="Buttons to process")
    p.add_argument("--print-tol", type=float, default=32.0, help="LAB distance tolerance for print color")
    p.add_argument("--plastic-tol", type=float, default=24.0, help="LAB distance tolerance for plastic color")
    p.add_argument("--wear-threshold-pct", type=float, default=10.0, help="Fail threshold for nnz drop %")
    p.add_argument("--max-frames", type=int, default=0, help="Limit frames processed (0 = all)")
    p.add_argument("--no-overlay", action="store_true", help="Disable overlay video output")
    args = p.parse_args()

    video_path = Path(args.video) if str(args.video).strip() else _pick_video_path()
    if video_path is None or not str(video_path):
        raise SystemExit("No video selected")
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else _default_output_dir(video_path)

    run(
        video_path=video_path,
        out_dir=out_dir,
        buttons=[str(b).strip().upper() for b in args.buttons],
        print_tol=float(args.print_tol),
        plastic_tol=float(args.plastic_tol),
        wear_threshold_pct=float(args.wear_threshold_pct),
        max_frames=int(args.max_frames),
        save_overlay=not bool(args.no_overlay),
    )


if __name__ == "__main__":
    main()
