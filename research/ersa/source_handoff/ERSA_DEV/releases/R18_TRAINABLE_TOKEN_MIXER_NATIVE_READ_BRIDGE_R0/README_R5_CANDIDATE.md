# ERSA R5 — substrate retrieval boundary / direct qualification

Surgical successor to **ERSA R4**; no other Elpis branch is changed.

## What is actually improved

1. `discovery='score_complete'` performs a **full, finite scan over admitted live records** before score-order truncation. It repairs *candidate omission* caused by zero-overlap product-key subkeys. At capacity 8192 its O(N) CPU cost is explicitly measured. It cannot guarantee that a latent task-required record receives a top-K relevance score.
2. `aggregate_scoped(relation_ids, ...)` provides atomic, exact, fixed-ceiling evidence reads for a **caller-supplied identity set**, optionally pinned to epoch and per-record revision. Invalid, stale, denied, unbound or missing members reject the whole set and erase previous output. This is an addressing API, **not** authorization to demand arbitrary IDs; caller permission is not attested.
3. `aggregate_reciprocal_contradiction(seed_relation_id, ...)` reads only a live, mutually referencing pair. A one-sided, absent, expired, unauthorized or invalid partner fails closed. No hidden premise/answer is passed through semantic ranking.
4. The prior 205-test ERSA R4 suite remains unchanged; new adversarial tests distinguish scoped precision from open-world semantic retrieval and detect stale or partial reads.

## Reproducible scientific boundaries

- **R2 frozen families, 8 candidates:** complete discovery finds all task-required records among candidates (candidate recall=1 for all 12); dense ranking parity 12/12, **task-grounded selection still fails** for sparse evidence and the two-sided contradiction, and an unscoped query still leaks a forbidden neighbor. Matching the dense scoring rule is not task correctness.
- Explicit evidence read on *three known-ID* frozen prospective tasks reaches required recall 1/1 for each, without forbidden-neighbor leakage. Caller supplies identity: no open-world semantic recognition is proved.
- Inherited R2 prospective gauge harness transforms participant frame connections without also transforming the participant-vector representation; it exhibits an O(1) covariance residual even with dense mode. An additional physically consistent SO(2)xSO(2) gauge-input test passes at float64 tolerance. **Do not silently correct the frozen test** and do not claim general gauge conformance.
- `SSC` predecessor identity-space binding remains explicitly nonconformant/fail-closed. No trainable neural runtime exists in this package. The diagnostic `score_complete` scan is intended to expose the accuracy/cost ceiling, **not** as the target edge kernel.

## Scope

All mutable inference state in the reference bank is volatile and capacity bounded. Candidate source is original NumPy + Python; no NVIDIA GDN source, GPU kernel or trained weights. Production trust, external model rights, semantic relevance, SSC parity and native neural implementation remain **HOLD**.

## Run on Ouroboros

Verify ZIP SHA-256 before extraction. Extract under `/dev/shm`, verify `FILES.sha256`, set `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONPATH` to this ZIP's `src`, then `python3 -B tools/qualify_ersa_r5.py`. The runner verifies four read-only frozen `NativeAttention_R2` file pins, runs **246** tests (205 predecessor + 41 new), replays the R4 metrics, repeats the 8192-slot stress, performs a diagnostic cost comparison, and prints `=== SEND BACK ===`.

Disposition ceiling: `PASS_COMPLETE_DISCOVERY_AND_EXPLICIT_RELATION_SET_MECHANICS_ONLY`.
