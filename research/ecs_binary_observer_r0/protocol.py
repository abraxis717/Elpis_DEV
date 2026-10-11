"""ECS Binary Observation R0: the preregistered protocol (RESEARCH_ONLY, NO_RUNTIME_AUTHORITY).

Committed before any DESIGN world is generated. DESIGN calibrates the two laws and the probe decoders and nothing
else; DEV is a single rehearsal on fresh worlds with the frozen laws; the freeze binds laws, decoders, this file,
the source and the DEV evidence; QUAL runs once on fresh worlds. Gate thresholds below are fixed here and are never
moved by DESIGN or DEV results. DEV may only expose mechanical defects, repaired in the smallest layer and recorded.

Worlds reuse the Cognitive R0 numerical regime (dim 6, width 36, init 0.18, rate 0.002, teachers of width 4 and
scale 0.5, inputs N(0, 0.5^2) offset +-0.5 on the first axis) so every LEARN is inside the qualified recurrence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

SPEC = {
    "name": "ebo-r0",
    "version": 1,
    "seed": 20261011,
    "dim": 6,
    "width": 36,
    "init_scale": 0.18,
    "learning_rate": 0.002,
    "teacher_width": 4,
    "teacher_scale": 0.5,
    "teachers_per_world": 3,
    "input_scale": 0.5,
    "region_offset": 0.5,
    "episodes": 12,                       # states W_0 .. W_12 per world
    "experience_rows": 64,
    "step_choices": [250, 500, 1000],     # K1 G1 steps per episode, drawn per episode
    "workload_rows": [1, 16, 64, 256],    # the second observed query of each state, drawn per state
    "ref_rows": 64,                       # X_REF: the standing observed query (global)
    "held_rows": 64,                      # X_HELD: lab-only ground-truth inputs, never observed (global)
    "max_rows": 256,
    "registries": {"DESIGN": 24, "DEV": 12, "QUAL": 24},
}

# --- calibration (DESIGN only) ---------------------------------------------------------------------------------
CALIBRATION = {
    "GRAY4": "t_k = numpy.quantile(pooled DESIGN X_REF answers, k/16, method='linear'), k = 1..15; refused unless "
             "strictly increasing",
    "SPROJ": "mu = mean, inv_scale = 1/std (ddof 0) of pooled DESIGN X_REF answers; for bit k, 4 distinct window "
             "rows and signs from SplitMix64 seeded by SHA-256('ebo-r0/sproj/v1')[:8] (index = next % 64 rejecting "
             "repeats; sign = +1 if next >> 63 else -1); tau_k = numpy.median of s_k over DESIGN X_REF records",
    "decoders": "ridge on DESIGN only: features standardized by DESIGN mean/std (zero-variance columns dropped), "
                "targets centred by DESIGN mean; lambda per feature family by leave-one-world-out CV on DESIGN over "
                "10^-4 .. 10^4 (9 values), smallest mean squared error, ties to the larger lambda",
}
RIDGE_GRID = [10.0 ** e for e in range(-4, 5)]

# --- feature families -------------------------------------------------------------------------------------------
ENCODERS = ("GRAY4", "SPROJ")                 # the two candidate laws (the record payload)
COMPARATORS = ("RAW16", "RAW4")               # matched 256-bit raw readouts (science only, not encoders)
CONTROLS = ("DIGEST", "RANDOM")               # identity bits (64) and state-independent random bits (256)
DERIVED_CONTROLS = ("SHUFFLED", "RESET", "ROWPERM")   # evaluated with each encoder's own decoder

# --- preregistered gates (QUAL; DEV reports the same quantities) ------------------------------------------------
GATES = {
    "M1_repeatability": "every observed record: encoded twice in process, bitwise identical; native == pure-Python "
                        "reference bitwise; the first QUAL world re-encoded in a clean process, bitwise identical",
    "M2_non_interference": "every QUAL world run with the observer OFF and ON: identical answers (bytes), digests, "
                           "retained-state snapshots at every state, and K1 counters (forward, learn, commits, "
                           "refusals, heap allocations)",
    "M3_refusal": "the native mechanics test (ctest research.test_ebo_r0) and the Python refusal tests pass: pinned "
                  "codes, no record written, inputs unchanged; an observation failure leaves the query result intact",
    "M4_fixed_bounded": "every record is 64 bytes and admitted by ebo_record_check; rows > 64 are TRUNCATED; the "
                        "encoder reads only the window (native test)",
    "M5_no_io_alloc": "the observer library imports no symbol outside {memcpy, memset, __stack_chk_fail} and weak "
                      "runtime symbols; ebo_encode / ebo_ring_push / ebo_ring_pop call nothing but the stack-protector "
                      "failure path; zero allocations across QUERY + encode + push (native test)",
    "M6_no_learn_continuity": "the observer library imports no ECS/runtime symbol; observer.py imports no elpis module",
    "M7_handoff": "fixed-capacity SPSC ring, DROP_NEWEST, never blocks; FIFO and loss accounting under threads (native "
                  "test, TSan in CI)",
    "S1_equivalence": "per encoder over QUAL states vs the neuron-permuted copy W[:, pi]: digest prefixes differ in "
                      "100% of states (identity separates), and payloads are identical in >= 0.99 of states",
    "S2a_distinction": "per encoder over QUAL transitions t -> t+1: payload differs in >= 0.95 of transitions",
    "S2b_graded": "per encoder over within-world QUAL state pairs: Spearman rho(payload Hamming, L2 of f on X_HELD) "
                  ">= 0.6, and DIGEST rho is within [-0.2, 0.2] (control validity)",
    "S3_state": "per encoder: pooled R^2 on QUAL of f(X_HELD) from the record (DESIGN-trained decoder) >= 0.5, and "
                "exceeds every control (DIGEST, RANDOM, SHUFFLED, RESET, ROWPERM) by >= 0.25",
    "S4_transition": "per encoder: pooled R^2 on QUAL of f_{t+1}(X_HELD) - f_t(X_HELD) from (record_t, record_t+1) "
                     ">= 0.3, and exceeds every control (DIGEST, RANDOM, SHUFFLED, RESET) by >= 0.2",
}
REPORTED = {
    "matched_budget": "S2b/S3/S4 for RAW16 and RAW4 beside the encoders; class BINARY_AT_OR_ABOVE_RAW when the "
                      "encoder's S3 R^2 >= RAW16 S3 R^2 - 0.05, else RAW_SUPERIOR (not a gate)",
    "information": "collisions (distinct-digest QUAL state pairs with identical payload), per-bit entropy (mean, min, "
                   "constant bits), GRAY4 midpoint-decode RMSE / std and outer-level share, boundary proximity "
                   "(|v - t| <= 1e-9 |t| or SPROJ |s - tau| <= 1e-9), window truncation share",
}
USEFUL = "an encoder passing S1, S2a, S2b and S3"

# --- performance gate (direct native; run once after the freeze) -------------------------------------------------
PERF = {
    "workload": "K1 dim 6 width 36, GCC Release, rows 1, 16, 64, 256, 4096; OFF = query_identity; ON = "
                "query_identity + ebo_encode + ebo_ring_push; pairs interleaved in fixed pseudo-random order; ring "
                "drained outside the timed region",
    "pairs": {"1": 20000, "16": 20000, "64": 20000, "256": 20000, "4096": 4000},
    "theta": "theta(rows) = min(0.01 * p99_OFF, 1500 ns)",
    "estimator": "delta_p99 = p99(ON) - p99(OFF); 95% CI by block bootstrap over 20 consecutive blocks of pairs, "
                 "2000 resamples, seed 20261011",
    "rule": "per rows: PASS if CI_high <= theta and isolated (encode + push/pop) <= 0.01 * median_OFF; NONPASS if "
            "CI_low > theta or isolated > 0.01 * median_OFF; else INCONCLUSIVE. Encoder PASS iff PASS at every rows; "
            "NONPASS if any rows NONPASS; else INCONCLUSIVE",
    "disclosed": "isolated cost relative to the bare forward median (not a gate)",
}
PERF_THETA_CEILING_NS = 1500.0
PERF_RELATIVE = 0.01
PERF_BLOCKS = 20
PERF_RESAMPLES = 2000

DISPOSITION = (
    "MECHANICS_BLOCKED if any M gate fails; else BINARY_OBSERVER_NOT_USEFUL if no encoder is USEFUL; else "
    "PERFORMANCE_NONPASS if every USEFUL encoder is perf NONPASS; else INCONCLUSIVE if no USEFUL encoder is perf PASS; "
    "else BINARY_OBSERVER_FEASIBLE"
)
BFM_FOLLOWUP = (
    "a conditional Bernoulli-transition experiment is justified only if the disposition is BINARY_OBSERVER_FEASIBLE "
    "and a USEFUL perf-PASS encoder also passes S4 with mean per-bit entropy >= 0.5 bit and <= 10% constant bits"
)


def disposition(mechanics_pass: bool, useful: dict, perf: dict) -> str:
    """useful: encoder -> bool; perf: encoder -> PASS / NONPASS / INCONCLUSIVE."""
    if not mechanics_pass:
        return "MECHANICS_BLOCKED"
    chosen = [e for e in ENCODERS if useful.get(e)]
    if not chosen:
        return "BINARY_OBSERVER_NOT_USEFUL"
    if all(perf.get(e) == "NONPASS" for e in chosen):
        return "PERFORMANCE_NONPASS"
    if not any(perf.get(e) == "PASS" for e in chosen):
        return "INCONCLUSIVE"
    return "BINARY_OBSERVER_FEASIBLE"


def world_ids(registry: str) -> list:
    return [f"{registry.lower()}-{i:04d}" for i in range(SPEC["registries"][registry])]


def _code(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def rng(world: str, stream: str) -> np.random.Generator:
    seq = np.random.SeedSequence(entropy=SPEC["seed"], spawn_key=(_code(SPEC["name"]), _code(world), _code(stream)))
    return np.random.Generator(np.random.PCG64(seq))


def source_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(ROOT.glob("*.py")) + sorted((ROOT / "native").glob("*.[ch]")):
        h.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def protocol_digest() -> str:
    return hashlib.sha256((ROOT / "protocol.py").read_bytes()).hexdigest()
