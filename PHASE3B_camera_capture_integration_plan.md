# Phases 3A-4 Plan: Automated Camera Capture Integration to Project Completion

## Context from current code review
- `robospeed_stageD_force_window_v23.py` contains camera preview plumbing and session placeholders, but the periodic auto-capture flow is not yet fully automated and policy-driven.
- `Visual_inspection_v4.py` remains the reference for stable averaging capture, tuning behavior, and baseline inspection processing.
- The existing v23 PDF report layout already contains baseline summary + per-button force pages and must remain intact.

## Phase 3 objective
Automate periodic cycle captures and inspection flow **without manual button presses**, while preserving motion safety, deterministic behavior, and full traceability.

---

## Phase 3A — Auto-capture scheduler (every X cycles)

### 3A plan
1. Add configurable `capture_every_x_cycles` as:
   - UI textbox input
   - validated runtime state field (positive integer, with disable mode if empty/0 by policy)
2. Trigger auto-capture only at cycle boundaries (where `cycle_count` increments).
3. When capture is due:
   - pause run,
   - move to IC checkpoint (existing helper),
   - capture image,
   - log result,
   - return to test through home-first restart path.

### 3A acceptance criteria
- [ ] No capture occurs mid-cycle.
- [ ] Checkpoint move is always called before auto-capture.
- [ ] Run resumes only after successful restart path.

---

## Phase 3B — Golden vs cycle capture flow

### 3B plan
1. Add `golden_ready` run-scoped state (exactly one active golden per run unless explicitly reset/re-baselined).
2. Support configuration that allows first post-start capture to be designated golden.
3. Route all subsequent scheduled captures to CYC type and bind each to cycle index metadata.
4. Ensure all CYC captures reference the active golden baseline ID/path in manifest/report metadata.
5. Keep manual capture path available but clearly tagged (`manual`) and separated from scheduler logic.

### 3B acceptance criteria
- [ ] Exactly one active golden per run unless operator resets/re-baselines.
- [ ] CYC captures reference the same golden baseline in metadata.

---

## Phase 3C — File naming + run manifest

### 3C plan
1. Use deterministic, non-overwriting artifact naming:
   - `inspection_output/golden/golden_<run_id>.png`
   - `inspection_output/cyc/cycle_<n>_<ts>.png`
   - `inspection_output/anomaly/frame_anamoly_<cycle>_<ts>.png` (saved for every inspected CYC frame, regardless of verdict)
   - `inspection_output/video/cycle_inspection_<run_id>.mp4`
2. Write a capture manifest (`manifest.csv` or JSON) containing at minimum:
   - `run_id`
   - cycle number
   - capture type (`golden` / `cyc` / `manual`)
   - timestamp
   - camera status
   - pass/fail or error message
   - file path
   - inspection score/verdict when available
   - cycle inspection video path (or run-level reference)
3. Append manifest rows for both successes and failures.

### 3C acceptance criteria
- [ ] No artifact overwrites.
- [ ] Every capture attempt has a manifest row.
- [ ] Failures are logged, never silent.
- [ ] `frame_anamoly` image is saved for each inspected cycle capture even when no anomaly is detected.

---

## Phase 3D — Auto inspection hook (basic)

### 3D plan
1. Add a lightweight inspection callback after each auto CYC capture.
2. If golden exists, compute diff score + threshold verdict.
3. Persist score/verdict in manifest/report fields.
4. Always save `frame_anamoly` output for each inspected CYC frame (anomaly found or not).
5. Build/update a run-level `cycle_inspection` video that:
   - starts with a labeled golden frame (`GOLDEN`, run id, timestamp),
   - appends each saved `frame_anamoly` frame,
   - overlays cycle label + timestamp on each appended frame.
6. If golden is missing, emit warning state and continue based on policy.
7. Keep advanced anomaly pipeline deferred to Phase 4+.

### 3D acceptance criteria
- [ ] Each auto CYC capture yields a scored result when golden exists.
- [ ] Missing golden is handled gracefully with warning state.
- [ ] `cycle_inspection` video contains golden intro frame plus cycle-labeled `frame_anamoly` frames in capture order.

---

## Phase 3E — Failure behavior policy

### 3E plan
1. Add explicit retry policy for auto-capture at IC checkpoint (`N` retries).
2. If retries fail, support configurable terminal policy:
   - **Option A:** continue cycle test and log warning
   - **Option B:** safe-stop and require operator action
3. Surface exact failure reason in UI status + manifest.

### 3E acceptance criteria
- [ ] No deadlocks on camera/frame timeout failures.
- [ ] Operator can see exactly why auto-capture failed.

---

## Phase 3F — UI/status upgrades

### 3F plan
Add explicit operator-visible fields in status line/panel:
- next auto-capture cycle
- last capture result
- golden-ready flag
- scheduler enabled/disabled
- capture mode (`manual` vs `auto`)

### 3F acceptance criteria
- [ ] Operator can determine capture state at a glance.
- [ ] Manual and auto captures are clearly differentiated.

---

## Phase 3G — Validation checklist (before Phase 4)

### 3G validation scenarios
- [ ] `capture_every_x=1` stability test (capture every cycle).
- [ ] `capture_every_x=5` cadence correctness.
- [ ] Manual IC Home + auto scheduler interaction.
- [ ] Camera timeout simulation and policy result verification.
- [ ] Restart-after-failure path validation.
- [ ] `cycle_inspection` video validation (golden first frame + ordered cycle anomaly frames with labels/timestamps).

### 3G acceptance criteria
- [ ] Deterministic behavior in all 6 scenarios.
- [ ] No unsafe motion transitions.

---

## Phase 4 — Integration hardening and project completion

### 4 scope
1. **System integration completion**
   - Verify robot motion + force pipeline + auto-capture + inspection coexist under full-cycle runs.
   - Resolve discovered race conditions/lock contention.

2. **Reliability and restart recovery**
   - Harden reset/reconnect flows for camera, scheduler, and inspection state.
   - Validate safe recovery after interruption without undefined state.

3. **Anomaly detection during cycling (expanded from 3D)**
   - Run anomaly evaluation on each scheduled CYC capture while cycling.
   - Track per-cycle anomaly indicators (score, threshold, verdict, reason code).
   - Aggregate anomaly statistics for run-level summary (counts, first-fail cycle, worst score).

4. **Reporting closure (append-only to original v23 layout)**
   - Preserve all original v23 report sections unchanged.
   - Append a new **Anomaly Detection During Cycling** report section after existing sections.
   - Include cycle-indexed anomaly table, `frame_anamoly` artifact links, and `cycle_inspection` video reference in the appended section only.

5. **Release readiness**
   - Execute nominal + stress + failure acceptance matrix.
   - Freeze defaults and publish operator runbook.

### 4 acceptance criteria
- [ ] End-to-end run from start to report completes with no manual workaround.
- [ ] Long-duration stress run completes without camera deadlock.
- [ ] Recovery from camera interruption succeeds by policy.
- [ ] Final report preserves original v23 sections and appends anomaly section only.
- [ ] Operator workflow is documented and repeatable.

---

## Done criteria (project completion)
- [ ] Phase 3A, 3B, 3C, 3D, 3E, 3F, and 3G acceptance criteria all complete.
- [ ] Phase 4 acceptance criteria complete.
- [ ] Open critical camera-integration defects: zero.
- [ ] Team sign-off for safety, operations, and maintainability complete.
