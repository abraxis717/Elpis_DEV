# H-ECS R3 successor C2 — QUAL Q2: `WORLD_MODEL_VALID` (12/12)

`RESEARCH_ONLY` · `C2_WORLD_MODEL_VALID` · `CONSTITUENT_BLOCKER_CLOSED` · `HIERARCHY_NOT_ADJUDICATED` · `NO_INTEGRATION`

The unchanged C2 candidate ran once, in order, on the fresh, never-executed 12-seed Q2 held-out registry (`../q2_freeze/`). It ran under release R1 `49860123…`, after correction R1, which was found and committed before any seed. There was no fault, retry or replay.

**Per seed:** TASK, S1 and M passed at every binding and bound horizon on all 12 seeds (`seeds/seed_*/C2_OBSERVER_STREAM.log`).

**Cohort (R3-BP law):** TASK_LOCAL, V2, V3, V4, S1 and M all pass.
- V2 means: 0.9965 separated, 1.0 matched.
- V4 means: 31.16 separated, 1.424 matched.
- Disposition `WORLD_MODEL_VALID` (`QUAL2_DISPOSITION.txt`, `R3_BP_FULL_COHORT_C2_Q2.log`).

**Margins (`BINDING_SUMMARY.tsv`):**
- Minimum bound HD/L2 M capture per seed: 0.9742–1.0086.
- Minimum bound M capture over all bindings and seeds: 0.9742, against a frozen threshold of 0.90.
- Minimum S1 capture: 0.9909, against a frozen threshold of 0.95.
- Maximum bound escape: 0.

**Lineage:**
1. Successor 8/8 closure: C0/C1 invalid (`d26c78b`).
2. C2 DESIGN and freeze (`8a98d9c`).
3. Fresh DEV 8/8 pass (`b64e32f`).
4. Q1 incomplete after an infrastructure interruption; registry spent; no verdict (`24fed33`).
5. Q2 pre-registration (`39e8c33`) and correction R1 (`39ea422`).
6. **Q2 QUAL 12/12 `WORLD_MODEL_VALID`** (this record).

**Scope of the claim.** C2 is a valid constituent world model under the frozen R3 validity hierarchy: TASK, S1 and M at every binding and horizon, on fresh DEV and on fresh held-out QUAL. That makes the hierarchy question scientifically admissible again; it does not answer it. The hierarchy needs its own frozen adjudication. C2 is not compute-matched to C0/C1, and no superiority claim is made. Raw telemetry is not in Git; each `COMMIT.txt` binds its sha256.
