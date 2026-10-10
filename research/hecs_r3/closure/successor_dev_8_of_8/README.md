# H-ECS R3 successor — DEV 8/8 closure (2026-10-09)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `DEV_COMPLETE_8_OF_8` · `C0_DEV_VALIDITY=FAIL` · `C1_DEV_VALIDITY=FAIL` · `NO_QUAL` · `NO_R4` · `NO_INTEGRATION`

This record closes the preregistered successor DEV cohort that `research/hecs_r3/unresolved/successor_dev_3_of_8/` left at 3/8. That archive is not modified. Pairs 0–2 are its committed evidence. Pairs 3–7 ran in order under the frozen DEV law, using a cold rebuild of the frozen sources from source-closure commit `6776cd4c18c83f9bf75b6ddabfa22428cb2ff090`.

## Disposition

Both candidates pass task validity (TASK_LOCAL, V2, V3, V4) and S1 on all eight seeds. Both fail the frozen multistep gate M at one bound cell: seed index 2 (`5526746150047008937`), HD/L2 rollout horizon 4. The R3-BP full-cohort observer therefore returns `WORLD_MODEL_INVALID_ON_DEV` for C0 and for C1. Seed 2 had already fixed this universal result at 3/8. Seeds 3–7 add no further failure.

| Candidate | TASK_LOCAL | S1 pass | M pass | V2 mean (sep / matched) | V4 mean (sep / matched) | Disposition |
|---|---:|---:|---:|---|---|---|
| C0 | 8/8 | 8/8 | 7/8 | 0.98958 / 0.99479 | 30.4198 / 1.47803 | `WORLD_MODEL_INVALID_ON_DEV` |
| C1 | 8/8 | 8/8 | 7/8 | 0.98958 / 0.99479 | 30.4198 / 1.47803 | `WORLD_MODEL_INVALID_ON_DEV` |

Failures by seed, binding and horizon, over every bound ROLLOUT cell (M) and every ROLLOUT cell (S1):

- C0: index 2, seed `5526746150047008937`, HD/L2, h=4, M capture 0.8529768527391456 (< 0.90), escape 0.
- C1: index 2, seed `5526746150047008937`, HD/L2, h=4, M capture 0.8685551895096904 (< 0.90), escape 0.
- No S1 failure.

## HD/L2 rollout, horizon 4 (the C1 treatment cell)

| Pair | Seed | Commit | C0 M capture | C1 M capture | C1 − C0 | M gate |
|---|---:|---|---:|---:|---:|---|
| 0 | 15007341021985616615 | `a731f84c…` | 0.9777302523 | 0.9772411281 | −0.000489 | both pass |
| 1 | 12334837942893278080 | `5c48e9ec…` | 0.9180048262 | 0.9336881712 | +0.015683 | both pass |
| 2 | 5526746150047008937 | `f5992e83…` | **0.8529768527** | **0.8685551895** | +0.015578 | **both fail** |
| 3 | 2644957107391819373 | `77b215ea…` | 0.9779619380 | 0.9776520370 | −0.000310 | both pass |
| 4 | 14747431184225333660 | `2b2a14fc…` | 0.9723810521 | 0.9754280782 | +0.003047 | both pass |
| 5 | 2338945401063472337 | `5a35946d…` | 0.9698641648 | 0.9701328700 | +0.000269 | both pass |
| 6 | 2859738232811479868 | `07fa16ad…` | 0.9243173628 | 0.9412841461 | +0.016967 | both pass |
| 7 | 229129266962718901 | `7741e2a4…` | 0.9320948697 | 0.9292518786 | −0.002843 | both pass |

Paired C1 − C0 at HD/L2 h=4: mean +0.005988, with C1 > C0 in 5 of 8 seeds. At HD/L2 h=1 the mean is −0.001277 and at h=2 it is −0.000551. Every non-HD/L2 metric row is byte-identical between C0 and C1 on all eight seeds. M escape is 0 in every bound cell. All 112 bound cells are in `RECORD.json` (`paired_C1_minus_C0`).

These are **descriptive paired differences only**. C0 and C1 are **not compute-matched**: C1 uses 8 windows per update plus a detached horizon-4 term with lambda4 = 0.1. No superiority, causal or treatment-effect claim is made, and none is supported.

## How pairs 3–7 were produced (reconstruction, not the historical binaries)

1. **Cold rebuild.** The frozen sources were compiled unmodified in the original crate layout (`native_source/src/bin/r3_bn_c{0,1}_dev_engine.rs` plus `src/bin/support/`, with the `research/hecs_r2/rust` crate and the supplied `Cargo.toml`/`Cargo.lock`, cargo dev profile `--offline --locked`). The sources are the C0/C1 engines, the support modules, the R3-BP observer, the R3-BQ supervisor and native K1 from `native/ECS`. The hard-coded historical project root was resolved by a private mount namespace onto a scratch export of the source-closure commit. No source byte was changed.
2. **Binary identity.** The rebuilt binaries do **not** reproduce the historical binary digests: the toolchain differs, and the historical one is not recorded. They are recorded as reconstructed artifacts next to the historical digests in `reconstruction/DEV_RECONSTRUCTED_RELEASE.txt`. They are never presented as the historical binaries.
3. **Qualification.** Before any unused seed was consumed, the rebuilt engines and observer re-executed committed pairs 0–2 with the supervisor's exact argv, cwd and stream capture, into a separate directory. The result (`reconstruction/QUALIFICATION.json`) is **exact raw-output identity**:
   - all 30 committed raw TSVs are byte-identical;
   - all 6 observer streams are byte-identical;
   - all 6 child streams are line-identical per stream. Four differ in bytes only by stdout/stderr pipe interleaving;
   - all six known HD/L2 h=4 captures are reproduced exactly.
4. **Supervision.** Pairs 3–7 ran one at a time, C0 then C1, under `reconstruction/r3_bq_dev_supervisor_recon_adapter.rs`. This is the frozen R3-BQ supervisor (source `35df429c…`). Child invocation, stream capture, START/COMMIT formats, the prefix chain, write-once/no-replay and FAULT handling are kept verbatim. Only the release identity is replaced:
   - pairs 0–2 stay bound to the frozen release `cffe1fff…` and to their archived commit digests;
   - pairs 3–7 bind the reconstructed release R′ (`a459a994…`). R′ binds the frozen authority, approval, protocol, the source closure, all source digests, the reconstructed and historical binary digests, the toolchain and the qualification record.
   No pair faulted, retried or was replayed. No seed was replaced. No parameter, threshold, gate order or observer arithmetic was changed.
5. **Audit.** The full chain `seed_0 → seed_7` was re-verified (START and COMMIT recomputed from the files, predecessor links, release per index, and pairs 0–2 byte-identical to the archive). The R3-BP observer `--cohort` then ran on the 8/8 cohort for C0 and C1 (`cohort/`).

## Contents

- `RECORD.json` — machine-readable closure: frozen law, rebuild and qualification identities, the 8-pair commit chain, observer per-seed and full-cohort results, pass counts, failures, all paired deltas, disposition and authorization flags.
- `evidence/seed_{3..7}/COMMITTED/` — `START.txt`, `COMMIT.txt` and the two one-line observer streams per pair. `COMMIT.txt` binds the sha256 of the raw five-file TSV sets and child streams, which are **not** in Git.
- `cohort/R3_BP_FULL_COHORT_{C0,C1}.log` — the observer's full-cohort output.
- `reconstruction/` — qualification record, reconstructed release R′, and the adapter supervisor source (reconstruction mechanics, not frozen science).
- `SHA256SUMS` — content checksums of this directory.

**Not authorized by this record:** QUAL release or access to reserved QUAL seeds, R4, C1 promotion, production integration, or any superiority claim. `QUAL_AUTHORIZED=NO`, `R4_AUTHORIZED=NO`, `INTEGRATION_AUTHORIZED=NO`.
