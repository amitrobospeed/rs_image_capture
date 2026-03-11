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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class ButtonCalibration:
    roi: Tuple[int, int, int, int]
    plastic_lab: np.ndarray
    print_lab: np.ndarray
    baseline_nnz: int = 0


def _select_rois(frame: np.ndarray, buttons: List[str]) -> Dict[str, Tuple[int, int, int, int]]:
    rois: Dict[str, Tuple[int, int, int, int]] = {}
    for btn in buttons:
        x, y, w, h = cv2.selectROI(f"Select ROI for {btn} (Enter to accept)", frame, False, False)
        cv2.destroyWindow(f"Select ROI for {btn} (Enter to accept)")
        if w <= 0 or h <= 0:
            raise RuntimeError(f"ROI not selected for button {btn}")
        rois[btn] = (int(x), int(y), int(w), int(h))
    return rois


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


def _collect_calibration(frame0: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, ButtonCalibration]:
    calib: Dict[str, ButtonCalibration] = {}
    for btn, roi in rois.items():
        x, y, w, h = roi
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
            x, y, w, h = c.roi
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                nnz = 0
            else:
                mask = _compute_print_mask(crop, c.plastic_lab, c.print_lab, print_tol, plastic_tol)
                nnz = int(np.count_nonzero(mask))

            if frame_idx == 0:
                c.baseline_nnz = max(1, nnz)

            drop_pct = max(0.0, (float(c.baseline_nnz) - float(nnz)) / float(max(c.baseline_nnz, 1)) * 100.0)
            verdict = "FAIL" if drop_pct >= wear_threshold_pct else "PASS"

            rec[f"nnz_{btn}"] = nnz
            rec[f"drop_pct_{btn}"] = round(drop_pct, 4)
            rec[f"wear_{btn}"] = verdict

            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 220, 0), 2)
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
    p.add_argument("--video", required=True, help="Path to input raw/labeled video")
    p.add_argument("--out-dir", default="wear_post_output", help="Output folder")
    p.add_argument("--buttons", nargs="+", default=["A", "B", "C", "D"], help="Buttons to process")
    p.add_argument("--print-tol", type=float, default=32.0, help="LAB distance tolerance for print color")
    p.add_argument("--plastic-tol", type=float, default=24.0, help="LAB distance tolerance for plastic color")
    p.add_argument("--wear-threshold-pct", type=float, default=10.0, help="Fail threshold for nnz drop %")
    p.add_argument("--max-frames", type=int, default=0, help="Limit frames processed (0 = all)")
    p.add_argument("--no-overlay", action="store_true", help="Disable overlay video output")
    args = p.parse_args()

    run(
        video_path=Path(args.video),
        out_dir=Path(args.out_dir),
        buttons=[str(b).strip().upper() for b in args.buttons],
        print_tol=float(args.print_tol),
        plastic_tol=float(args.plastic_tol),
        wear_threshold_pct=float(args.wear_threshold_pct),
        max_frames=int(args.max_frames),
        save_overlay=not bool(args.no_overlay),
    )


if __name__ == "__main__":
    main()
