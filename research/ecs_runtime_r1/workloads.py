"""Runtime R1 workloads and pre-registered gates (fixed before any measurement). PERFORMANCE_ONLY.

R0 workloads mirror the qualified Cognitive R0 shape. Scaling workloads exist
only for engineering scaling and carry NO_SCIENTIFIC_CLAIM: their widths are not
"supported cognition".
"""
from __future__ import annotations

SEED = 20261004
LEARNING_RATE = 0.002
DIM = 6

R0 = {"dim": 6, "width": 36, "rows": 64}
R0_QUERY_ROWS = (64, 256)
R0_STEPS = (1, 10, 100, 1000, 4000)

SCALING_WIDTHS = (72, 288, 1152, 4608)
SCALING_ROWS = (64, 256)
SCALING_STEPS = 10

# Per measurement: an explicit warmup, then individually timed samples.
BUDGET_SECONDS = 1.5
MIN_SAMPLES, MAX_SAMPLES = 20, 2000
MIN_WARMUP, MAX_WARMUP = 3, 200

# Data regime (values do not change the work: no data-dependent early exit).
# W0 ~ N(0, (0.18 * sqrt(36 / N))^2), X ~ N(0, 0.5^2), y = 0.9 * f_W0(X).
W_SCALE_AT_36 = 0.18
X_SCALE = 0.5
TARGET_FACTOR = 0.9


def workloads():
    out = []
    d, n, r = R0["dim"], R0["width"], R0["rows"]
    for rows in R0_QUERY_ROWS:
        out.append({"id": f"r0-query-R{rows}", "scope": "R0", "kind": "query", "dim": d, "width": n, "rows": rows,
                    "steps": 0})
    for k in R0_STEPS:
        out.append({"id": f"r0-learn-K{k}", "scope": "R0", "kind": "learn", "dim": d, "width": n, "rows": r,
                    "steps": k})
    for n in SCALING_WIDTHS:
        for rows in SCALING_ROWS:
            out.append({"id": f"scale-query-N{n}-R{rows}", "scope": "PERFORMANCE_ONLY", "kind": "query", "dim": DIM,
                        "width": n, "rows": rows, "steps": 0})
            out.append({"id": f"scale-learn-N{n}-R{rows}-K{SCALING_STEPS}", "scope": "PERFORMANCE_ONLY",
                        "kind": "learn", "dim": DIM, "width": n, "rows": rows, "steps": SCALING_STEPS})
    return out


# Pre-registered gates (docs/ECS_RUNTIME_R1.md). Structural gates are pass/fail;
# the jitter ratio triggers an investigation that must be reported.
GATES = {
    "single_ffi": "CognitiveCore.learn(K) issues exactly one native learn call and no per-step native call; its "
                  "native call counts are identical for K=1 and K=4000",
    "no_hot_allocation": "no heap allocation inside executor query, learn or transaction operations after creation "
                         "(native allocation-interposition test)",
    "python_overhead_k_independent": "wrapper overhead = Python p50 - native-direct p50 for R0 learn; overhead at "
                                     "K=4000 <= 2 x overhead at K=10 + 100 us",
    "material_speedup": "new CognitiveCore.learn p50 at least 2x faster than the pre-nativeization baseline p50 for "
                        "every R0 learn workload with K >= 10",
    "scalar_parity": "executor forward and K-step learn are bitwise equal to the scalar reference "
                     "(elpis_ecsg_forward_f64, K x elpis_ecsg_state_gd_step_f64) on randomized cases; Cognitive R0 "
                     "QUAL measurements reproduce exactly on all 8 QUAL worlds",
    "sanitizers": "executor native tests pass under ASan and UBSan",
    "jitter_investigation": "steady-state p99 / p50 > 3 for any measured workload is investigated and its source "
                            "reported",
}
THRESHOLDS = {"speedup_min": 2.0, "overhead_factor": 2.0, "overhead_slack_us": 100.0, "jitter_ratio": 3.0}
