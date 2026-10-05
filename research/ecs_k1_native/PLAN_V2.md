# Native K1 differential qualification — plan v2 (write-once)

`ecsg-k1-native.v2`. Machine-readable authority: `specs/ecsg-k1-native.v2.plan.json`, whose sha256 is pinned in
`tests/research/ecs_k1_native/test_plan_v2.py`. K1N-v1 stays `NOT_QUALIFIED`, unchanged.

**Question.** Does the corrected native K1 reproduce every decision-bearing Retention R3 quantity and every required
runtime/state invariant, without demanding long-horizon coordinate identity between numerically different
implementations?

## Gates

All gates are fixed here, before any v2 execution.

**Exact invariants:**
- **E1:** same-host exact invariants on every world of Q (the 32 R3 QUAL worlds) and F (32 fresh worlds `k1n2-0000..0031`).
- **E2:** determinism.
- **E3:** clean-process transplant.
- **E4:** the native K1 test suite passes.
- **E5:** structural checks (one native crossing per operation whatever K; no hot allocation).

**Bounded numerics:**
- **L1:** 16-step local horizon from an identical state, ≤ 1e-10.
- **L2:** consolidation from an identical state, ≤ 1e-12.

**Decision equivalence:**
- **D1_Q:** the recorded R3 decision record is reproduced exactly, with OUTCOME_A.
- **D1_F:** native decision record equals the same-host laboratory decision record on every key and world.

Long-horizon W, H, a and nmse differences are descriptive only.

## Verdict and rules

- **QUALIFIED** if every gate holds; otherwise **NOT_QUALIFIED**, recorded as found.
- The plan runs once from a clean harness commit that follows this plan.
- No rerun, no threshold change.
- No per-world tolerance.
- No use of any v1 world or outcome.
