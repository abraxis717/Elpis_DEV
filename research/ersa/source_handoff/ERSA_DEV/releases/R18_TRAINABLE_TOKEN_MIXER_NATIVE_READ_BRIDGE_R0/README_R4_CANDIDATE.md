# ERSA R4 — direct retrieval semantics on the repaired R3 reference bank

**Scope:** A surgical read-path enhancement to the R3 integrity-qualified reference substrate; **not** a new architecture or GDN fork. Frozen R2 fixtures and independent oracle remain byte-identical. No NVIDIA source, GPU, trained controller or external permissions are needed.

## Changes

1. Add `discovery='score_indexed'`: enumerate the bounded product-key/adjacency candidate union, score *all admitted index hits*, then truncate to `max_candidates` by the canonical relevance order `(-round(score,12), relation_id)`. The original indexed/dense modes remain available and unchanged. This is **not** a universal sparse retrieval guarantee: relations with no index hit remain undiscoverable.
2. Add `exact_relation_id=` (keyword-only in usage) to `query_candidates`, `select`, and `aggregate`. Exact reads resolve the distinct logical identity using the admitted RID index, never physical slot, key similarity or hash equality. Missing, expired, malformed and unbound identities fail closed, clearing prior selection/evidence. Exact identity is an **addressing operation**, not a semantic prediction, and does not establish permission for arbitrary untrusted callers to choose any admitted identity.
3. Preserve the R3 staged-update/integrity fixes, protected eviction, fixed-capacity arrays, bounded pending stages, and fail-closed SSC adapter.
4. Expose diagnostics for how many index hits were scored; bounded above by live capacity. Selection with no verified identity is explicitly **heuristic**. A high dense-parity score is not sufficient to certify task-grounded relevance.

## Confirmed results in the local build

- Frozen R2 original tests + R3 BREAK + new R4: **205/205**.
- R0 K8 dense parity: original `indexed=15/18`, new `score_indexed=16/18`, dense `18/18`.
- R2 K8 dense parity: original `indexed=7/12`, new `score_indexed=12/12`, dense `12/12`.
- R2 **task-grounded failures still present in unscoped retrieval**: one forbidden-neighbour leak (also in the dense oracle); sparse no-overlap target candidate recall zero; contradiction partner incomplete despite matched dense ranking. **Do not report 12/12 as full retrieval qualification.**
- Exact-ID R2 forbidden-neighbour control: request relation 0 → only relation 0 selected and aggregated; record 1 absent. Invalid IDs reject without reusing stale evidence.
- 20/20 randomized full-index-coverage cases agree with independent dense oracle and remain insertion-order invariant.
- 8,192-slot stress with 64 replacement inserts, all 16 protected records retained, fixed arrays remain bounded.

## Execute

Unzip this package under `/dev/shm` and verify its internal `FILES.sha256`. With `PYTHONPATH` pointing at its `src`, run `python3 -B tools/qualify_ersa_r4.py`. The qualification runner reads (and hashes) four frozen files from `/mnt/primesauce/RALPH/DEV/Hermes/NativeAttention_R2` without importing/executing them and runs only copied source from tmpfs. It prints `=== SEND BACK ===`. Do not extract into persistent development tree or mutate frozen R2/R3.

### Qualification honesty

Maximum disposition: `PASS_EXACT_IDENTITY_AND_CONDITIONAL_INDEX_RANKING_MECHANICS_ONLY`.

**Remaining before any full-substrate/native qualification:** independent *task-grounded* retrieval success including sparse unseen-key recall and contradiction coverage, proven SSC identity mapping (current adapter is intentionally nonconformant), performance/memory scaling against the actual cost envelope, native/learned recurrent runtime, and external rights/trust where applicable. Current experiment is a **reference substrate** result, not a neural replacement for GDN.
