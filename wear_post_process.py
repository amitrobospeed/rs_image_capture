#!/usr/bin/env python3
"""Post-process cycle inspection videos to measure print wear trends.

Workflow
1) Pick video + settings (PyQt6 form by default, CLI fallback).
2) Open first frame and select A/B/C/D ROIs using v24-style ROI selector.
3) Select plastic and print color patches for each button ROI.
4) Process all frames and report nnz + drop% per button.

Outputs
- wear_metrics.csv
- wear_summary.json
- wear_overlay.mp4 (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np


@dataclass
class ButtonCalibration:
    roi: object
    plastic_lab: np.ndarray
    print_lab: np.ndarray
    baseline_nnz: int = 0


@dataclass
class WearConfig:
    video_path: Path
    out_dir: Path
    buttons: List[str]
    print_tol: float
    plastic_tol: float
    wear_threshold_pct: float
    max_frames: int
    save_overlay: bool
    use_qt_interactions: bool = True


# ---------------- ROI helpers (v24-style behavior) ----------------
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


def _draw_roi(frame: np.ndarray, roi_like, color=(255, 0, 0), label: Optional[str] = None, thick: int = 2):
    roi = _sanitize_roi(roi_like, frame.shape)
    if isinstance(roi, dict):
        if roi.get("shape") == "circle":
            cv2.circle(frame, (roi["cx"], roi["cy"]), roi["r"], color, thick)
            lx, ly = roi["cx"] - roi["r"], roi["cy"] - roi["r"]
        elif roi.get("shape") == "poly":
            pts = np.array(roi.get("points", []), dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, thick)
            lx, ly = int(pts[:, 0].min()), int(pts[:, 1].min())
        else:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)
            lx, ly = x, y
    else:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)
        lx, ly = x, y
    if label:
        cv2.putText(frame, label, (lx + 2, max(18, ly - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


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
    status_msg = "Draw ROI then Next; when all done click Lock All (or press L)."
    should_exit = False

    help_line = "Keys: N=Next, L=Lock All, Enter=Finalize poly, Backspace=Undo poly, Esc/Q=Cancel"

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
            ("rect", "Rectangle", 160),
            ("circle", "Circle", 140),
            ("poly", "Polygon", 150),
            ("next", "Next Button", 180),
            ("lock", "Lock All", 140),
        ]
        x0, y0 = 20, 40
        for key, title, w in items:
            h = 42
            x1, y1 = x0 + w, y0 + h
            active = key == shape_mode and key in ("rect", "circle", "poly")
            fill = (46, 204, 113) if active else ((192, 57, 43) if key == "lock" else (44, 62, 80))
            cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (236, 240, 241), 2)
            cv2.putText(canvas, title, (x0 + 10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            actions[key] = (x0, y0, x1, y1)
            x0 += w + 10

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
        nonlocal shape_mode, start_pt, drag_pt, poly_pts, working_roi, current_idx, status_msg, should_exit
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
                if _try_lock_all():
                    should_exit = True
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
        cv2.rectangle(canvas, (16, 94), (canvas.shape[1] - 16, 170), (20, 20, 20), -1)
        cv2.putText(canvas, f"Target: {_target_btn()} | mode={shape_mode}", (24, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (240, 240, 240), 2)
        cv2.putText(canvas, status_msg, (24, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2)
        cv2.putText(canvas, help_line, (24, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 255), 1)

        for b, r in selected.items():
            _draw_roi(canvas, r, color=(80, 220, 120), label=b, thick=2)

        if shape_mode == "poly" and poly_pts:
            for i, p in enumerate(poly_pts):
                cv2.circle(canvas, p, 5, (0, 215, 255), -1)
                if i > 0:
                    cv2.line(canvas, poly_pts[i - 1], p, (0, 215, 255), 2)
        elif working_roi is not None:
            _draw_roi(canvas, working_roi, color=(0, 215, 255), label=f"{_target_btn()}*", thick=2)
        elif start_pt is not None and drag_pt is not None and shape_mode in ("rect", "circle"):
            x0, y0 = start_pt
            x1, y1 = drag_pt
            if shape_mode == "rect":
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 215, 255), 2)
            else:
                cx = int((x0 + x1) / 2)
                cy = int((y0 + y1) / 2)
                r = int(max(abs(x1 - x0), abs(y1 - y0)) / 2)
                cv2.circle(canvas, (cx, cy), r, (0, 215, 255), 2)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if should_exit:
            break
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
        if key in (ord("n"), ord("N")):
            if _commit_current_btn():
                current_idx = (current_idx + 1) % len(roi_targets)
                start_pt = None
                drag_pt = None
                poly_pts = []
                working_roi = selected.get(_target_btn())
            continue
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(win)
            raise RuntimeError("ROI selection canceled")
        if key in (ord("l"), ord("L")):
            if _try_lock_all():
                break

    cv2.destroyWindow(win)
    return {b: _sanitize_roi(selected[b], frame.shape) for b in roi_targets}


# ---------------- Calibration and wear computation ----------------
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


def _mean_lab_from_patch_constrained(frame: np.ndarray, local_roi_mask: np.ndarray, title: str) -> np.ndarray:
    while True:
        x, y, w, h = cv2.selectROI(title, frame, False, False)
        cv2.destroyWindow(title)
        if w <= 0 or h <= 0:
            raise RuntimeError(f"Patch not selected: {title}")
        x, y, w, h = int(x), int(y), int(w), int(h)
        patch = frame[y:y + h, x:x + w]
        patch_mask = local_roi_mask[y:y + h, x:x + w]
        if patch.size == 0 or patch_mask.size == 0:
            print(f"[wear_post_process] Invalid patch bounds for {title}. Please select again.")
            continue

        valid = patch_mask > 0
        valid_n = int(np.count_nonzero(valid))
        total_n = int(valid.size)
        if valid_n <= 0:
            print(f"[wear_post_process] Patch must overlap active ROI shape for {title}. Select again.")
            continue

        overlap_ratio = valid_n / float(max(total_n, 1))
        if overlap_ratio < 0.5:
            print(f"[wear_post_process] Patch overlap with active ROI is too low ({overlap_ratio:.2f}). Select again.")
            continue

        patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        vals = patch_lab[valid]
        if vals.size == 0:
            print(f"[wear_post_process] No valid ROI pixels in patch for {title}. Select again.")
            continue
        return vals.reshape(-1, 3).mean(axis=0)


def _collect_calibration(frame0: np.ndarray, rois: Dict[str, object]) -> Dict[str, ButtonCalibration]:
    calib: Dict[str, ButtonCalibration] = {}
    for btn, roi in rois.items():
        full_mask, (x, y, w, h) = _roi_mask_from_spec(frame0.shape, roi)
        crop = frame0[y:y + h, x:x + w]
        local_mask = (full_mask[y:y + h, x:x + w] > 0).astype(np.uint8) * 255

        print(f"[wear_post_process] {btn}: select PLASTIC patch inside ROI")
        plastic_lab = _mean_lab_from_patch_constrained(crop, local_mask, f"{btn}: Select PLASTIC color patch")
        print(f"[wear_post_process] {btn}: select PRINT patch inside ROI")
        print_lab = _mean_lab_from_patch_constrained(crop, local_mask, f"{btn}: Select PRINT color patch")
        calib[btn] = ButtonCalibration(roi=roi, plastic_lab=plastic_lab, print_lab=print_lab)
    return calib


def _compute_print_mask(crop_bgr: np.ndarray, plastic_lab: np.ndarray, print_lab: np.ndarray,
                        print_tol: float, plastic_tol: float) -> np.ndarray:
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d_print = np.linalg.norm(lab - print_lab.reshape(1, 1, 3).astype(np.float32), axis=2)
    d_plastic = np.linalg.norm(lab - plastic_lab.reshape(1, 1, 3).astype(np.float32), axis=2)

    # Wear logic constraints: no specular rejection, no morphology cleanup.
    print_like = d_print <= float(print_tol)
    plastic_like = d_plastic <= float(plastic_tol)
    closer_to_print = d_print < d_plastic
    mask = print_like & (~plastic_like) & closer_to_print
    return (mask.astype(np.uint8) * 255)


def _collect_rois_and_calibration_pyqt6(frame0: np.ndarray, buttons: List[str]) -> Optional[tuple[Dict[str, object], Dict[str, ButtonCalibration]]]:
    try:
        from PyQt6.QtCore import Qt, QPoint
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QWidget, QMessageBox
    except Exception:
        return None

    app = QApplication.instance() or QApplication(sys.argv)

    def bgr_to_pix(bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    class Canvas(QLabel):
        def __init__(self, dlg):
            super().__init__()
            self.dlg = dlg
            self.setMouseTracking(True)
            self.start: Optional[QPoint] = None
            self.drag: Optional[QPoint] = None

        def mousePressEvent(self, e):
            if e.button() != Qt.MouseButton.LeftButton:
                return
            x, y = int(e.position().x()), int(e.position().y())
            self.dlg.on_mouse_down(x, y)
            self.start = QPoint(x, y)
            self.drag = QPoint(x, y)
            self.dlg.render()

        def mouseMoveEvent(self, e):
            if self.start is None:
                return
            self.drag = QPoint(int(e.position().x()), int(e.position().y()))
            self.dlg.on_mouse_drag(self.drag.x(), self.drag.y())
            self.dlg.render()

        def mouseReleaseEvent(self, e):
            if self.start is None:
                return
            x, y = int(e.position().x()), int(e.position().y())
            self.dlg.on_mouse_up(x, y)
            self.start = None
            self.drag = None
            self.dlg.render()

    class Dlg(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Wear ROI + Patch Selector (PyQt6)')
            self.frame = frame0.copy()
            self.buttons = list(buttons)
            self.target_idx = 0
            self.mode = 'rect'  # rect/circle/poly/patch_plastic/patch_print
            self.poly_pts: List[tuple[int,int]] = []
            self.working: Optional[dict] = None
            self.rois: Dict[str, object] = {}
            self.patches: Dict[str, Dict[str, Any]] = {b: {'plastic': None, 'print': None} for b in self.buttons}
            self.result = None
            self.last_down: Optional[tuple[int,int]] = None
            self.last_drag: Optional[tuple[int,int]] = None

            root = QHBoxLayout(self)
            self.canvas = Canvas(self)
            self.canvas.setFixedSize(self.frame.shape[1], self.frame.shape[0])
            root.addWidget(self.canvas)

            side = QVBoxLayout()
            self.status = QLabel('Draw ROI then Next; after all ROIs click Lock All.')
            side.addWidget(self.status)
            row1 = QHBoxLayout()
            self.btn_rect = QPushButton('Rectangle')
            self.btn_circle = QPushButton('Circle')
            self.btn_poly = QPushButton('Polygon')
            row1.addWidget(self.btn_rect); row1.addWidget(self.btn_circle); row1.addWidget(self.btn_poly)
            side.addLayout(row1)

            row2 = QHBoxLayout()
            self.btn_poly_done = QPushButton('Finalize Poly')
            self.btn_next = QPushButton('Next')
            self.btn_lock = QPushButton('Lock All')
            row2.addWidget(self.btn_poly_done); row2.addWidget(self.btn_next); row2.addWidget(self.btn_lock)
            side.addLayout(row2)

            self.btn_pick_plastic = QPushButton('Pick Plastic')
            self.btn_pick_print = QPushButton('Pick Print')
            self.btn_start = QPushButton('Start Analysis')
            self.btn_cancel = QPushButton('Cancel')
            side.addWidget(self.btn_pick_plastic)
            side.addWidget(self.btn_pick_print)
            side.addWidget(self.btn_start)
            side.addWidget(self.btn_cancel)

            self.swatches = {}
            for b in self.buttons:
                lb = QLabel(f'{b}: plastic / print')
                lb.setStyleSheet('background:#111;color:#eee;padding:4px;')
                side.addWidget(lb)
                self.swatches[b] = lb

            side.addStretch(1)
            root.addLayout(side)

            self.btn_rect.clicked.connect(lambda: self._set_mode('rect'))
            self.btn_circle.clicked.connect(lambda: self._set_mode('circle'))
            self.btn_poly.clicked.connect(lambda: self._set_mode('poly'))
            self.btn_poly_done.clicked.connect(self._finalize_poly)
            self.btn_next.clicked.connect(self._next)
            self.btn_lock.clicked.connect(self._lock_all)
            self.btn_pick_plastic.clicked.connect(lambda: self._set_mode('patch_plastic'))
            self.btn_pick_print.clicked.connect(lambda: self._set_mode('patch_print'))
            self.btn_start.clicked.connect(self._start)
            self.btn_cancel.clicked.connect(self.reject)
            self.btn_start.setEnabled(False)
            self.render()

        def _target(self):
            return self.buttons[self.target_idx]

        def _set_mode(self, m):
            self.mode = m
            self.status.setText(f'Target {self._target()} | mode={m}')

        def _is_valid(self, roi):
            if not roi:
                return False
            if roi.get('shape') == 'rect':
                return roi.get('w',0) >= 4 and roi.get('h',0) >= 4
            if roi.get('shape') == 'circle':
                return roi.get('r',0) >= 3
            if roi.get('shape') == 'poly':
                return len(roi.get('points',[])) >= 3
            return False

        def _finalize_poly(self):
            if len(self.poly_pts) >= 3:
                self.working = {'shape':'poly','points':self.poly_pts.copy()}
                self.status.setText(f'{self._target()} polygon ready')
            else:
                self.status.setText('Polygon needs >=3 points')
            self.render()

        def _commit(self):
            if self.mode == 'poly':
                self._finalize_poly()
            if not self._is_valid(self.working or {}):
                self.status.setText(f'Draw valid ROI for {self._target()}')
                return False
            self.rois[self._target()] = _sanitize_roi(self.working, self.frame.shape)
            self.status.setText(f'Saved ROI for {self._target()}')
            return True

        def _next(self):
            if self._commit():
                self.target_idx = (self.target_idx + 1) % len(self.buttons)
                self.working = self.rois.get(self._target())
                self.poly_pts = []
            self.render()

        def _lock_all(self):
            if self.working and self._is_valid(self.working):
                self._commit()
            missing = [b for b in self.buttons if b not in self.rois]
            if missing:
                self.status.setText('Missing ROI: ' + ','.join(missing))
                self.render()
                return
            self.status.setText('ROIs locked. Pick plastic and print patches for each button.')
            self.render()

        def _all_patches_done(self):
            for b in self.buttons:
                if self.patches[b]['plastic'] is None or self.patches[b]['print'] is None:
                    return False
            return True

        def _start(self):
            if not self._all_patches_done():
                QMessageBox.warning(self, 'Incomplete', 'Select plastic and print patches for all buttons.')
                return
            calib = {}
            for b in self.buttons:
                roi = self.rois[b]
                calib[b] = ButtonCalibration(roi=roi, plastic_lab=self.patches[b]['plastic'], print_lab=self.patches[b]['print'])
            self.result = (self.rois, calib)
            self.accept()

        def _update_swatch(self, b):
            p = self.patches[b]['plastic']
            q = self.patches[b]['print']
            ptxt = 'set' if p is not None else 'unset'
            qtxt = 'set' if q is not None else 'unset'
            self.swatches[b].setText(f'{b}: plastic={ptxt} | print={qtxt}')
            if p is not None and q is not None:
                self.swatches[b].setStyleSheet('background:#163;color:#fff;padding:4px;')
            else:
                self.swatches[b].setStyleSheet('background:#111;color:#eee;padding:4px;')

        def on_mouse_down(self, x, y):
            self.last_down = (x, y)
            self.last_drag = (x, y)
            if self.mode == 'poly':
                self.poly_pts.append((x, y))

        def on_mouse_drag(self, x, y):
            self.last_drag = (x, y)

        def _select_patch_point_rect(self, x0,y0,x1,y1):
            rx, ry = min(x0,x1), min(y0,y1)
            rw, rh = abs(x1-x0), abs(y1-y0)
            if rw <= 0 or rh <= 0:
                self.status.setText('Patch invalid size.')
                return
            btn = self._target()
            roi = self.rois.get(btn)
            if roi is None:
                self.status.setText(f'Lock ROI for {btn} first.')
                return
            roi_mask, _ = _roi_mask_from_spec(self.frame.shape, roi)
            pmask = roi_mask[ry:ry+rh, rx:rx+rw] > 0
            if pmask.size == 0 or int(np.count_nonzero(pmask)) <= 0:
                self.status.setText('Patch must overlap active ROI shape.')
                return
            if (np.count_nonzero(pmask) / float(max(pmask.size,1))) < 0.5:
                self.status.setText('Patch overlap too low (<50%).')
                return
            patch = self.frame[ry:ry+rh, rx:rx+rw]
            patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            vals = patch_lab[pmask]
            if vals.size == 0:
                self.status.setText('No valid ROI pixels in patch.')
                return
            mean_lab = vals.reshape(-1,3).mean(axis=0)
            kind = 'plastic' if self.mode == 'patch_plastic' else 'print'
            self.patches[btn][kind] = mean_lab
            self._update_swatch(btn)
            self.status.setText(f'{btn}: {kind} patch captured.')
            self.btn_start.setEnabled(self._all_patches_done())

        def on_mouse_up(self, x, y):
            if self.last_down is None:
                return
            x0,y0 = self.last_down
            if self.mode == 'rect':
                rx, ry = min(x0, x), min(y0, y)
                rw, rh = abs(x - x0), abs(y - y0)
                self.working = {'shape':'rect','x':rx,'y':ry,'w':rw,'h':rh}
                self.status.setText(f'{self._target()} ROI ready')
            elif self.mode == 'circle':
                cx = int((x0 + x) / 2)
                cy = int((y0 + y) / 2)
                r = int(max(abs(x - x0), abs(y - y0)) / 2)
                self.working = {'shape':'circle','cx':cx,'cy':cy,'r':r}
                self.status.setText(f'{self._target()} ROI ready')
            elif self.mode in ('patch_plastic','patch_print'):
                self._select_patch_point_rect(x0,y0,x,y)

        def render(self):
            canvas = self.frame.copy()
            for b,r in self.rois.items():
                _draw_roi(canvas, r, color=(80,220,120), label=b, thick=2)
            if self.mode == 'poly' and self.poly_pts:
                for i,p in enumerate(self.poly_pts):
                    cv2.circle(canvas, p, 5, (0,215,255), -1)
                    if i > 0:
                        cv2.line(canvas, self.poly_pts[i-1], p, (0,215,255), 2)
            elif self.working is not None and self.mode in ('rect','circle','poly'):
                _draw_roi(canvas, self.working, color=(0,215,255), label=f'{self._target()}*', thick=2)
            if self.last_down and self.last_drag and self.mode in ('rect','circle','patch_plastic','patch_print'):
                x0,y0 = self.last_down
                x1,y1 = self.last_drag
                if self.mode in ('rect','patch_plastic','patch_print'):
                    cv2.rectangle(canvas, (x0,y0), (x1,y1), (0,215,255), 2)
                else:
                    cx = int((x0+x1)/2); cy = int((y0+y1)/2); r = int(max(abs(x1-x0),abs(y1-y0))/2)
                    cv2.circle(canvas, (cx,cy), r, (0,215,255), 2)

            self.canvas.setPixmap(bgr_to_pix(canvas))

    dlg = Dlg()
    dlg.resize(min(1800, frame0.shape[1] + 420), min(1100, frame0.shape[0] + 80))
    if dlg.exec() == QDialog.DialogCode.Accepted:
        return dlg.result
    return None


def _default_output_dir(video_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("inspection_output") / "post_process" / "print_wear_detect" / f"{video_path.stem}_{ts}"


def run(config: WearConfig) -> None:
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {config.video_path}")

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError("Could not read first frame from video")

    print("Select button ROIs on baseline frame...")
    rois = None
    calib = None
    if config.use_qt_interactions:
        qt_out = _collect_rois_and_calibration_pyqt6(frame0, config.buttons)
        if qt_out is not None:
            rois, calib = qt_out
        else:
            print("[wear_post_process] PyQt6 ROI/Patch interaction unavailable; using OpenCV selection.")

    if rois is None or calib is None:
        rois = _select_rois(frame0, config.buttons)
        print("Select plastic/print patches for each button ROI...")
        calib = _collect_calibration(frame0, rois)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    writer = None
    if config.save_overlay:
        writer = cv2.VideoWriter(str(out_dir / "wear_overlay.mp4"), fourcc, float(max(1.0, fps)),
                                 (frame0.shape[1], frame0.shape[0]))

    rows = []

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if config.max_frames > 0 and frame_idx >= config.max_frames:
            break

        rec = {"frame_idx": frame_idx, "time_s": frame_idx / float(max(fps, 1e-6))}
        overlay = frame.copy()

        for btn in config.buttons:
            c = calib[btn]
            roi_mask, (x, y, w, h) = _roi_mask_from_spec(frame.shape, c.roi)
            crop = frame[y:y + h, x:x + w]
            local_roi = roi_mask[y:y + h, x:x + w] > 0
            if crop.size == 0 or local_roi.size == 0 or np.count_nonzero(local_roi) == 0:
                nnz = 0
            else:
                mask = _compute_print_mask(crop, c.plastic_lab, c.print_lab, config.print_tol, config.plastic_tol)
                nnz = int(np.count_nonzero((mask > 0) & local_roi))

            if frame_idx == 0:
                c.baseline_nnz = max(1, nnz)

            drop_pct = max(0.0, (float(c.baseline_nnz) - float(nnz)) / float(max(c.baseline_nnz, 1)) * 100.0)
            verdict = "FAIL" if drop_pct >= config.wear_threshold_pct else "PASS"

            rec[f"nnz_{btn}"] = nnz
            rec[f"drop_pct_{btn}"] = round(drop_pct, 4)
            rec[f"wear_{btn}"] = verdict

            _draw_roi(overlay, c.roi, color=(255, 220, 0), label=btn, thick=2)
            cv2.putText(
                overlay,
                f"{btn} nnz={nnz} drop={drop_pct:.1f}% {verdict}",
                (x + 4, max(16, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255) if verdict == "PASS" else (0, 0, 255),
                2,
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
    for btn in config.buttons:
        fieldnames += [f"nnz_{btn}", f"drop_pct_{btn}", f"wear_{btn}"]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "video": str(config.video_path),
        "buttons": config.buttons,
        "print_tol": config.print_tol,
        "plastic_tol": config.plastic_tol,
        "wear_threshold_pct": config.wear_threshold_pct,
        "baseline_nnz": {btn: int(calib[btn].baseline_nnz) for btn in config.buttons},
        "frames_processed": frame_idx,
        "csv": str(csv_path),
        "overlay_video": str(out_dir / "wear_overlay.mp4") if config.save_overlay else "",
    }
    (out_dir / "wear_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Done. Processed {frame_idx} frames")
    print(f"CSV: {csv_path}")
    print(f"Summary: {out_dir / 'wear_summary.json'}")
    if config.save_overlay:
        print(f"Overlay: {out_dir / 'wear_overlay.mp4'}")


# ---------------- PyQt6 settings UI ----------------
def _collect_config_pyqt6(initial: WearConfig) -> Optional[WearConfig]:
    try:
        from PyQt6.QtWidgets import (
            QApplication,
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QFileDialog,
            QDoubleSpinBox,
            QSpinBox,
            QCheckBox,
            QMessageBox,
        )
    except Exception:
        return None

    app = QApplication.instance() or QApplication(sys.argv)

    class SettingsWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Print Wear Post-Process (PyQt6)")
            self.result: Optional[WearConfig] = None

            root = QVBoxLayout(self)

            root.addWidget(QLabel("Video file"))
            row_video = QHBoxLayout()
            self.video_edit = QLineEdit(str(initial.video_path) if str(initial.video_path) else "")
            btn_video = QPushButton("Browse")
            row_video.addWidget(self.video_edit)
            row_video.addWidget(btn_video)
            root.addLayout(row_video)

            root.addWidget(QLabel("Output directory"))
            row_out = QHBoxLayout()
            self.out_edit = QLineEdit(str(initial.out_dir) if str(initial.out_dir) else "")
            btn_out = QPushButton("Browse")
            row_out.addWidget(self.out_edit)
            row_out.addWidget(btn_out)
            root.addLayout(row_out)

            root.addWidget(QLabel("Buttons (space-separated)"))
            self.buttons_edit = QLineEdit(" ".join(initial.buttons))
            root.addWidget(self.buttons_edit)

            row_num1 = QHBoxLayout()
            row_num1.addWidget(QLabel("Print tol"))
            self.print_tol = QDoubleSpinBox()
            self.print_tol.setRange(1.0, 255.0)
            self.print_tol.setValue(float(initial.print_tol))
            row_num1.addWidget(self.print_tol)

            row_num1.addWidget(QLabel("Plastic tol"))
            self.plastic_tol = QDoubleSpinBox()
            self.plastic_tol.setRange(1.0, 255.0)
            self.plastic_tol.setValue(float(initial.plastic_tol))
            row_num1.addWidget(self.plastic_tol)
            root.addLayout(row_num1)

            row_num2 = QHBoxLayout()
            row_num2.addWidget(QLabel("Wear threshold %"))
            self.wear_thr = QDoubleSpinBox()
            self.wear_thr.setRange(0.0, 100.0)
            self.wear_thr.setValue(float(initial.wear_threshold_pct))
            row_num2.addWidget(self.wear_thr)

            row_num2.addWidget(QLabel("Max frames (0=all)"))
            self.max_frames = QSpinBox()
            self.max_frames.setRange(0, 10_000_000)
            self.max_frames.setValue(int(initial.max_frames))
            row_num2.addWidget(self.max_frames)
            root.addLayout(row_num2)

            self.save_overlay = QCheckBox("Save overlay video")
            self.save_overlay.setChecked(bool(initial.save_overlay))
            root.addWidget(self.save_overlay)

            help_lbl = QLabel("Workflow: pick settings -> Start -> ROI selector -> patch selectors -> processing")
            root.addWidget(help_lbl)

            row_btn = QHBoxLayout()
            btn_start = QPushButton("Start")
            btn_cancel = QPushButton("Cancel")
            row_btn.addWidget(btn_start)
            row_btn.addWidget(btn_cancel)
            root.addLayout(row_btn)

            def browse_video():
                path, _ = QFileDialog.getOpenFileName(self, "Select inspection video", "", "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)")
                if path:
                    self.video_edit.setText(path)
                    if not self.out_edit.text().strip():
                        self.out_edit.setText(str(_default_output_dir(Path(path))))

            def browse_out():
                path = QFileDialog.getExistingDirectory(self, "Select output directory", "")
                if path:
                    self.out_edit.setText(path)

            def on_cancel():
                self.result = None
                self.close()

            def on_start():
                video_txt = self.video_edit.text().strip()
                if not video_txt:
                    QMessageBox.warning(self, "Missing video", "Please select a video file.")
                    return
                video_path = Path(video_txt)
                if not video_path.exists():
                    QMessageBox.warning(self, "Video not found", f"Video path does not exist:\n{video_path}")
                    return

                buttons = [b.strip().upper() for b in self.buttons_edit.text().split() if b.strip()]
                if not buttons:
                    QMessageBox.warning(self, "Missing buttons", "Enter at least one button label.")
                    return

                out_txt = self.out_edit.text().strip()
                out_dir = Path(out_txt) if out_txt else _default_output_dir(video_path)

                self.result = WearConfig(
                    video_path=video_path,
                    out_dir=out_dir,
                    buttons=buttons,
                    print_tol=float(self.print_tol.value()),
                    plastic_tol=float(self.plastic_tol.value()),
                    wear_threshold_pct=float(self.wear_thr.value()),
                    max_frames=int(self.max_frames.value()),
                    save_overlay=bool(self.save_overlay.isChecked()),
                )
                self.close()

            btn_video.clicked.connect(browse_video)
            btn_out.clicked.connect(browse_out)
            btn_cancel.clicked.connect(on_cancel)
            btn_start.clicked.connect(on_start)

    win = SettingsWindow()
    win.resize(760, 360)
    win.show()
    app.exec()
    return win.result


def _collect_config_cli(args) -> WearConfig:
    video_txt = str(args.video).strip()
    if not video_txt:
        video_txt = input("Enter video path: ").strip()
    if not video_txt:
        raise SystemExit("No video selected")

    video_path = Path(video_txt)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else _default_output_dir(video_path)
    buttons = [str(b).strip().upper() for b in args.buttons if str(b).strip()]
    if not buttons:
        raise SystemExit("No buttons configured")

    return WearConfig(
        video_path=video_path,
        out_dir=out_dir,
        buttons=buttons,
        print_tol=float(args.print_tol),
        plastic_tol=float(args.plastic_tol),
        wear_threshold_pct=float(args.wear_threshold_pct),
        max_frames=int(args.max_frames),
        save_overlay=not bool(args.no_overlay),
        use_qt_interactions=not bool(args.opencv_ui),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Post-process wear detection on inspection video")
    p.add_argument("--video", default="", help="Path to input raw/labeled video")
    p.add_argument("--out-dir", default="", help="Output folder (default: inspection_output/post_process/print_wear_detect/<video>_<timestamp>)")
    p.add_argument("--buttons", nargs="+", default=["A", "B", "C", "D"], help="Buttons to process")
    p.add_argument("--print-tol", type=float, default=32.0, help="LAB distance tolerance for print color")
    p.add_argument("--plastic-tol", type=float, default=24.0, help="LAB distance tolerance for plastic color")
    p.add_argument("--wear-threshold-pct", type=float, default=10.0, help="Fail threshold for nnz drop %")
    p.add_argument("--max-frames", type=int, default=0, help="Limit frames processed (0 = all)")
    p.add_argument("--no-overlay", action="store_true", help="Disable overlay video output")
    p.add_argument("--cli", action="store_true", help="Use CLI config flow instead of PyQt6 settings UI")
    p.add_argument("--opencv-ui", action="store_true", help="Force OpenCV ROI+patch windows instead of integrated PyQt6 selector")
    args = p.parse_args()

    initial = WearConfig(
        video_path=Path(args.video) if str(args.video).strip() else Path(),
        out_dir=Path(args.out_dir) if str(args.out_dir).strip() else Path(),
        buttons=[str(b).strip().upper() for b in args.buttons if str(b).strip()],
        print_tol=float(args.print_tol),
        plastic_tol=float(args.plastic_tol),
        wear_threshold_pct=float(args.wear_threshold_pct),
        max_frames=int(args.max_frames),
        save_overlay=not bool(args.no_overlay),
        use_qt_interactions=not bool(args.opencv_ui),
    )

    config: Optional[WearConfig] = None
    if not args.cli:
        config = _collect_config_pyqt6(initial)
        if config is None:
            print("[wear_post_process] PyQt6 not available or dialog canceled; falling back to CLI.")

    if config is None:
        config = _collect_config_cli(args)

    run(config)


if __name__ == "__main__":
    main()
