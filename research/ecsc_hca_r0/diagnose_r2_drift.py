"""R2 source-drift diagnosis. Mechanical reproduction only; never changes frozen evidence."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from .verify_r2_local import verify_inputs, r2_hashes, strip_time

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def atomic_json(path, obj):
    # Only write new, explicitly secondary output names, never overwrite the original evidence.
    raw = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    if len(raw.encode("utf-8")) > 2_000_000:
        raise ValueError("RESULT_SIZE_OVER_BOUND")
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return digest(path)


def cmp_diff(a, b, label, differences, count_limit=120):
    if len(differences) >= count_limit:
        return
    if type(a) is not type(b):
        differences.append((label, a, b))
    elif isinstance(a, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                differences.append((label + "." + str(key), a.get(key, "<ABSENT>"), b.get(key, "<ABSENT>")))
            else:
                cmp_diff(a[key], b[key], label + "." + str(key), differences, count_limit)
            if len(differences) >= count_limit:
                return
    elif isinstance(a, list):
        if len(a) != len(b):
            differences.append((label + ".length", len(a), len(b)))
        for i, (av, bv) in enumerate(zip(a, b)):
            cmp_diff(av, bv, label + f"[{i}]", differences, count_limit)
            if len(differences) >= count_limit:
                return
    elif a != b:
        differences.append((label, a, b))


def semantic(row):
    # Retain proof-bound input identity, all predictions, constraints, source/status labels, refusals.
    identity_keys = ("arity", "name", "task", "evidence", "policy", "view", "axioms", "relations", "incidence_entries", "dropped")
    result_keys = ("outcome", "code", "predictions", "constraints", "omitted", "score", "transition", "equality_to_frozen_B0_excluding_costs")
    return {
        "identity": {k: row.get(k) for k in identity_keys},
        "methods": {m: {k: rec.get(k) for k in result_keys if k in rec}
                    for m, rec in sorted(row["methods"].items())},
    }


def damage_semantic(d):
    return {"name": d.get("name"), "task": d.get("task"), "methods": {
        m: {k: rec.get(k) for k in ("outcome", "code", "predictions", "constraints", "omitted", "score", "transition") if k in rec}
        for m, rec in sorted(d["methods"].items())}}


def miniature(value, maxlen=160):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return encoded if len(encoded) <= maxlen else encoded[: maxlen - 4] + " ..."


def main():
    print("===== R2 FINAL-SOURCE DIAGNOSIS (NO ASTRA) =====", flush=True)
    baseline = json.loads((DIR / "BASELINE_R1_FREEZE.json").read_text())
    before = verify_inputs(baseline)
    source = r2_hashes()
    old_path = DIR / "R2_DEV_RESULTS.json"
    original = json.loads(old_path.read_text())
    prior_check = json.loads((DIR / "R2_LOCAL_VERIFICATION.json").read_text())
    if (prior_check.get("source_pin") != before["head"] or
        prior_check.get("current_r2_source_hashes") != source or
        prior_check.get("original_result_sha256") != digest(old_path) or
        prior_check.get("tests") != "UNITTEST_AND_CANONICAL_SELECTED_PASS" or
        prior_check.get("events") != 40):
        raise ValueError("PRIOR_TEST_RECORD_INVALID_OR_SOURCE_CHANGED")
    prior_drift = sorted(k for k, h in source.items() if original.get("r2_source_hashes", {}).get(k) != h)
    print("PRIOR_LOCAL_TESTS=PASS; SOURCE_UNCHANGED_SINCE_TESTS=PASS", flush=True)
    print("RECORDED_SOURCE_HASH_DRIFT=", prior_drift, flush=True)

    from .r2_run import run_trajectory, damage_control
    events = run_trajectory(4) + run_trajectory(8)
    damage = damage_control()
    if len(events) != 40:
        raise ValueError("R2_EVENT_COUNT_CHANGED")
    if verify_inputs(baseline) != before or r2_hashes() != source:
        raise ValueError("SOURCE_CHANGED_DURING_REPRO")

    actual_semantics = [semantic(row) for row in events]
    old_semantics = [semantic(row) for row in original["events"]]
    semantic_diffs = []
    cmp_diff(old_semantics, actual_semantics, "events", semantic_diffs)
    cmp_diff(damage_semantic(original["cycle_damage"]), damage_semantic(damage), "cycle_damage", semantic_diffs)
    full_diffs = []
    cmp_diff(strip_time(original["events"]), strip_time(events), "events", full_diffs)
    cmp_diff(strip_time(original["cycle_damage"]), strip_time(damage), "cycle_damage", full_diffs)

    semantic_equal = not semantic_diffs
    nontiming_equal = not full_diffs
    status = ("EXACT_REPRODUCED" if nontiming_equal
              else "SEMANTICS_PRESERVED_DIAGNOSTICS_CHANGED" if semantic_equal
              else "SEMANTIC_OR_AUTHORITY_DRIFT_REVIEW_REQUIRED")
    # This is a NEW CURRENT-SOURCE observation, not a rewrite or retroactive validation of R2_DEV_RESULTS.
    current = {
        "phase": "R2_FINAL_SOURCE_DESCRIPTIVE_REPRODUCTION",
        "scientific_status": "NOT_RUN_NOT_QUALIFIED",
        "original_result_sha256": digest(old_path),
        "source_pin": before["head"],
        "frozen_input_hashes": before["frozen"],
        "canonical_context_sha256": before["canonical"],
        "r2_source_hashes": source,
        "prior_tests": prior_check["tests"],
        "events": events,
        "cycle_damage": damage,
        "recorded_source_hash_drift": prior_drift,
        "diagnostic_status": status,
        "notice": "Current-source descriptive data; no source repair, scientific qualification, or historical record overwrite",
    }
    result_path = DIR / "R2_CURRENT_SOURCE_RESULTS.json"
    result_sha = atomic_json(result_path, current)
    diagnosis = {
        "source_pin": before["head"],
        "original_result_sha256": digest(old_path),
        "current_result_sha256": result_sha,
        "r2_source_hashes": source,
        "recorded_source_hash_drift": prior_drift,
        "original_events": len(original["events"]),
        "current_events": len(events),
        "semantic_or_authority_equal": semantic_equal,
        "all_nontiming_equal": nontiming_equal,
        "status": status,
        "semantic_differences": [{"path": p, "recorded": miniature(a), "current": miniature(b)}
                                 for p, a, b in semantic_diffs[:25]],
        "nontiming_differences": [{"path": p, "recorded": miniature(a), "current": miniature(b)}
                                  for p, a, b in full_diffs[:25]],
        "note": "Diffs truncated to first 25 per category; original R2 evidence is immutable",
    }
    diag_path = DIR / "R2_DRIFT_DIAGNOSIS.json"
    atomic_json(diag_path, diagnosis)
    print("===== SOURCE-DRIFT RESULTS =====", flush=True)
    print("SEMANTIC_OR_AUTHORITY_EQUAL=", semantic_equal, flush=True)
    print("ALL_NONTIMING_EQUAL=", nontiming_equal, flush=True)
    print("DISPOSITION=", status, flush=True)
    for item in diagnosis["nontiming_differences"][:12]:
        print("DIFF", item["path"], "OLD=", item["recorded"], "CURRENT=", item["current"], flush=True)
    print("DIAGNOSIS=", diag_path, flush=True)
    print("FINAL_SOURCE_RESULTS=", result_path, flush=True)
    # Successful diagnosis is not a scientific/mechanical PASS. Fail specifically on changed semantics.
    return 4 if not semantic_equal else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, OSError, ValueError, KeyError, TypeError) as e:
        print("BLOCKED_R2_SOURCE_DIAGNOSIS", type(e).__name__, str(e), flush=True)
        sys.exit(2)
