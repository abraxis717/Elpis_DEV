# R124 — R123 above-clamp reversal forensics

**Retrospective, consumed-evidence-only. RESEARCH_ONLY.** The sole authority is the published R123 K4 25% result at `53057654cd8f11a4e8d95d0dcad469d294ca9498`. This phase does not spend a new holdout, change frozen weights, fit policy parameters, or modify a production runtime. **R93 remains `ENERGY_NORMALIZED_DENOMINATOR_HOLD`.**

## Purpose

Replay the 40 published `2^2` K4-quarter readout reversals and determine whether each reversal is already present when the K4 transition is applied to the **native incoming state**, or is already present when a native transition is applied to the **K4 incoming state**. These are one-step conditional counterfactuals; an interaction-only result does not prove a unique causal mechanism over the full trajectory.

At every verified reversal event `(graph, overlap, checkpoint, seed, case, step, pair)` record:

- NN: native step on native incoming state (must be bit-identical to native trajectory)
- NK: K4-quarter step on native incoming state
- KN: native step on K4-quarter incoming state
- KK: K4-quarter step on K4-quarter incoming state (must be bit-identical to K4 trajectory)
- Direction class, cosine, norm ratio, predicted/teacher readout norm and relative squared error for each path
- First induced reversal for the same pair, checkpoint and fixture
- Compact postreconciliation relation-state norms and how many are near the clamp radius (proxy only)
- Topology hop distance, overlap, cohort, seed, case and step

**Limits:** The `post_relation_vectors_near_4` field is *not* a pre-clamp-hit counter, especially when MEAN reconciliation combines multiple window proposals. The test does not remove/alter the clamp, and therefore cannot by itself prove clamp causation. There is no performance qualification for a policy that depends on the teacher.

## Registered integrity conditions

Published R123 raw records are the sole fixture selector; the selected R123 `K4=0.25, scale=2^2, induced>0` records must sum to exactly 40. Every replay's H8/H20 full-state NSE and induced-reversal count must match the published record. Native/native and K4/K4 one-step counterfactuals must match their autonomous trajectories *bitwise*. All selected fixtures replay at scale=1 with zero induced reversals, and a bounded subset of trajectories compares R119 EXACT_GUARDED against EXACT_R118 bitwise. The six checkpoint SHA pins and the frozen manifest must remain unchanged. Invalid or nonfinite data or resource overrun is NONQUAL.

The accepted forensic disposition is `FORENSIC_REPLAY_PASS_BOUNDED` when all integrity conditions pass, **not** `MECHANISM_PROVED` or production admission. The class histogram is reported descriptively, without adjusting any thresholds or fractions. A negative/inconclusive mechanism distribution is a valid observation.

## Publication workflow

The package `RUN.sh` uses the user's already-cached Git objects in `/dev/shm/elpis-r123-reuse/repo` (or compatible volatile fallback) and fetches only the missing R123 commit history when needed. It makes a new volatile Git worktree, commits R124 source and preregistered protocol on `research/ym-r124-stress-reversal-forensics-r0`, pushes and verifies that commit *before* any R123 trajectory replay, runs the diagnostic once, and publishes its evidence as a separate commit. It does **not** modify `main`, the active `/mnt/primesauce/Elpis_DEV` checkout, or any frozen source. Read-only Git access to the local workspace is allowed for remote selection. Evidence is staged in `/dev/shm` and explicitly published to GitHub only after the run.

If the run fails, the volatile workspace and preregistration remain for a deliberate diagnosis; do not automatically rerun an incomplete phase. The final printout contains `=== SEND BACK ===` with dispositions and SHA256 hashes.
