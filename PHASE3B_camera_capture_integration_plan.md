# Phase 3B Plan: Camera Capture Integration

## Context from current code review
- `robospeed_stageD_force_window_v23.py` already includes a background camera preview thread, hardware lock separation, exposure/gain state, and placeholders for golden/capture artifacts. This indicates the app is in a partial integration state where camera plumbing exists but process-level capture workflow still needs hardening.
- `Visual_inspection_v4.py` contains the reference implementation pattern for stable average-frame capture, golden recapture after auto-tune, ROI selection, and diff-mask based anomaly detection.

## Phase 3B objective
Integrate production-grade image capture and inspection flow into `robospeed_stageD_force_window_v23.py` so that force-cycle execution and inspection captures can run reliably in one operator workflow, without camera contention or non-deterministic settings drift.

## Scope for Phase 3B
1. **Capture workflow unification**
   - Add explicit commands/actions for:
     - Capture Golden (with stabilization)
     - Capture Cycle frame
     - Run inspection against selected ROI
   - Persist session artifact paths and statuses for operator traceability.

2. **Camera state machine hardening**
   - Define camera states (`not_started`, `warming`, `ready`, `capturing`, `error`) and transition guards.
   - Ensure all hardware operations (`set_option`, frame waits) run behind a single hardware lock.

3. **Deterministic exposure tuning pipeline**
   - Reuse/port auto-tune logic from `Visual_inspection_v4.py` into main app flow.
   - Lock exposure/gain after initial successful tune and prevent accidental retune unless operator explicitly resets.

4. **Stable frame capture contract**
   - Standardize a `capture_average(timeout_s, min_frames, label)` helper that:
     - Flushes stale frames
     - Collects a minimum sample count
     - Returns explicit failure reason on insufficient frames
   - Use helper for both golden and cycle captures.

5. **ROI and inspection integration**
   - Introduce persistent ROI handling in main UI:
     - Select ROI from golden frame
     - Validate bounds against incoming frame size
   - Port diff pipeline (CLAHE + blur + absdiff + threshold + morphology + contour filter).

6. **Concurrency and failure behavior**
   - Ensure force-test run loop does not deadlock when capture is requested mid-cycle.
   - Add non-blocking status reporting for camera errors and recoverable retries.

7. **Operator outputs and reporting hooks**
   - Save `Golden_avg.png`, `CYC_avg.png`, `Diff_mask.png`, `Frame_anomaly.png` with timestamped naming.
   - Add summary fields for image capture count and latest inspection result to final report payload.

## Implementation sequence (recommended)
1. Refactor camera helpers into a cohesive section in `robospeed_stageD_force_window_v23.py` (no behavior change).
2. Introduce `capture_average` and `auto_tune_exposure_from_golden` integration.
3. Wire UI controls and status updates for Golden/Cycle/Inspect actions.
4. Add ROI selection persistence and validation.
5. Add inspection artifact saving + report linkage.
6. Run dry-run validation paths and hardware-available smoke checks.

## Validation checklist for Phase 3B
- [ ] Camera can start preview and remain responsive for >10 minutes.
- [ ] Golden capture succeeds with at least configured minimum frame count.
- [ ] Exposure/gain lock remains unchanged after tuning during normal operation.
- [ ] ROI selection persists across multiple captures.
- [ ] Inspection run generates expected mask/output artifacts.
- [ ] Force-cycle loop remains responsive while camera actions occur.
- [ ] Error injection (camera disconnect / frame timeout) reports actionable status without crashing app.

## Out-of-scope for this planning pass
- Full migration to a modular package layout.
- Deep algorithmic changes to anomaly scoring thresholds.
- Cloud/off-device artifact upload.
