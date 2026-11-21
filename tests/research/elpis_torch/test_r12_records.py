"""Elpis Torch R12 records: provenance, integrity and the frozen verdicts recomputed independently.

These tests read JSON only (no PyTorch). They re-implement the decision rules of ``PREREGISTRATION_R12.md`` rather
than calling the runner. They check that:
- every confirmation record was produced by the frozen plan and sources;
- the plan predates the records;
- T1 rebuilt R11's Stage C models exactly;
- every planned group has exactly one record;
- the integrity gates hold;
- the recorded dispositions follow from the per-episode rows;
- the P5 amendment predates its records, its rebuilds are the T2 models, and its readings follow from the rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess

import pytest

from .conftest import REPO

RESULTS = REPO / "research" / "elpis_torch" / "results" / "r12"
PLAN = RESULTS / "preregistration.json"
TORCH = REPO / "research" / "elpis_torch"
SOURCES = {"runner": TORCH / "experiments" / "run_r12.py", "fast_state": TORCH / "fast_state.py",
           "fast_episodes": TORCH / "fast_episodes.py", "session": TORCH / "session.py",
           "lastblock": TORCH / "lastblock.py", "stack": TORCH / "specialist_stack.py",
           "tasks": TORCH / "specialist_tasks.py", "profile": TORCH / "cgc_specialist.py",
           "sparse_channel": TORCH / "sparse_channel.py", "cgc": TORCH / "cgc.py", "model": TORCH / "model.py",
           "config": TORCH / "config.py", "ersa": TORCH / "ersa.py", "ersa_model": TORCH / "ersa_model.py",
           "r11_stage_c_runner": TORCH / "experiments" / "run_r11_stage_c.py"}
FAMILIES = ("smooth", "conditional")
STEPS = ("0", "1", "2", "4", "8")


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan():
    return _load(PLAN)


@pytest.fixture(scope="module")
def records():
    out = {"T1": {}, "T2": {}}
    for path in sorted(RESULTS.glob("t[12]_*.json")):
        record = _load(path)
        out[record["regime"]][(record["family"], record["seed"])] = record
    return out


@pytest.fixture(scope="module")
def study():
    return _load(RESULTS / "study_r12.json")


# -- provenance and integrity ------------------------------------------------------------------------------------

def test_every_confirmation_record_was_produced_by_the_frozen_plan_and_sources(records):
    expected = {"preregistration_sha256": _sha(PLAN), **{f"{k}_sha256": _sha(v) for k, v in SOURCES.items()}}
    assert records["T1"] and records["T2"]
    for regime in records.values():
        for key, record in regime.items():
            assert record["stage"] == "confirmation" and record["status"] == "COMPLETED", key
            assert {k: record["provenance"][k] for k in expected} == expected, key
            assert not record["provenance"]["git_revision"].endswith("+dirty"), key


def test_the_plan_predates_every_record():
    relative = PLAN.relative_to(REPO).as_posix()
    revisions = {_load(p)["provenance"]["git_revision"].split("+")[0] for p in RESULTS.glob("t[12]_*.json")}
    assert revisions
    for revision in revisions:
        shown = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=REPO, capture_output=True)
        if shown.returncode != 0:
            pytest.skip("the recorded revision is not in this checkout's history (shallow clone)")
        assert hashlib.sha256(shown.stdout).hexdigest() == _sha(PLAN), revision


def test_every_planned_group_has_one_record_and_t1_rebuilt_r11_exactly(plan, records):
    for regime in ("T1", "T2"):
        assert set(records[regime]) == {(f, s) for f in FAMILIES for s in plan["regimes"][regime]["seeds"]}
        for record in records[regime].values():
            assert record["namespace"] == plan["regimes"][regime]["namespace"]
            assert record["eval_split"] == plan["regimes"][regime]["eval_split"]
            assert set(record["arms"]) == set(plan["arms"])
    for key, record in records["T1"].items():
        r11 = _load(REPO / "research" / "elpis_torch" / "results" / "r11" / f"c_{key[0]}_s{key[1]}.json")
        for arm, check in record["r11_verification"].items():
            assert check["rows_identical"] and check["selected_step_equal"] and check["trace_equal"], (key, arm)
            assert check["curve_max_abs_diff"] == 0.0 and record["arms"][arm]["selected_step"] == r11["arms"][arm]["selected_step"]


def test_development_records_are_labelled_and_named_by_the_plan(plan):
    named = plan["development"]["records"]
    assert {p.name for p in RESULTS.glob("dev_*.json")} == set(named)
    for name, digest in named.items():
        record = _load(RESULTS / name)
        assert _sha(RESULTS / name) == digest and record["stage"] == "development", name
        assert record["namespace"] in ("R11C_DEV", "R12C_DEV") and record["label"] == "DEVELOPMENT ONLY", name


def _gates(rows) -> dict:
    return {"reset_exact": all(r["reset"]["exact"] for r in rows),
            "disabled_exact": all(r["disabled"]["exact"] for r in rows),
            "disabled_refused": all(set(r["disabled"]["status"]) == {"REFUSED_LEARNING_DISABLED"} for r in rows),
            "no_leakage": all(r["leakage"]["probe_tokens_carry_no_target"] and r["leakage"]["query_support_disjoint"]
                              for r in rows),
            "states_unchanged_by_the_core": all(r["parity"]["states_unchanged_by_the_core"] for r in rows),
            "full_vs_cached_parity": max(r["parity"]["full_vs_cached_max_abs"] for r in rows) <= 1e-5,
            "every_update_applied": all(t["status"] == "APPLIED" for r in rows for t in r["trace"]),
            "bound_never_binds": not any(t["projected"] for r in rows for t in r["trace"]),
            "finite": all(math.isfinite(v) for r in rows for v in (*r["fast"].values(), r["shuffled"], r["fast_ood"],
                                                                   r["fast_context_free"]))}


# -- the frozen rules, re-implemented ---------------------------------------------------------------------------

def _sign_p(values) -> float:
    negative = sum(v < 0 for v in values)
    n = sum(v != 0 for v in values)
    return sum(math.comb(n, k) for k in range(negative, n + 1)) / 2 ** n if n else 1.0


def _better(groups, rule) -> dict:
    pooled = [v for g in groups for v in g]
    gm = math.exp(statistics.fmean(pooled))
    below = sum(statistics.fmean(g) < 0 for g in groups) / len(groups)
    p = _sign_p(pooled)
    return {"ratio": gm, "p": p, "below": below,
            "better": gm <= rule["max_ratio"] and p < rule["alpha"] and below >= rule["min_group_fraction"]}


def _measure(row, name):
    if name == "frozen":
        return row["frozen"]["query"]
    if name.startswith("fast") and name[4:].isdigit():
        return row["fast"][name[4:]]
    return row["shuffled"]


@pytest.mark.parametrize("regime", ["T1", "T2"])
def test_the_recorded_dispositions_follow_from_the_rows(plan, records, study, regime):
    rule = plan["rules"]["better"]
    groups = [(f, s) for s in plan["regimes"][regime]["seeds"] for f in FAMILIES]
    recs = records[regime]

    def rows(arm):
        return [recs[g]["arms"][arm]["rows"] for g in groups]

    def compare(x, xm, y, ym):
        return _better([[math.log(max(_measure(a, xm), 1e-300) / max(_measure(b, ym), 1e-300)) for a, b in zip(ga, gb)]
                        for ga, gb in zip(rows(x), rows(y))], rule)

    g0 = [statistics.median(r["frozen"]["query"] for r in grp) for grp in rows("G0")]
    null = [statistics.median(r["null"] for r in grp) for grp in rows("G0")]
    learned = all(a <= plan["rules"]["max_g0_over_null"] * b for a, b in zip(g0, null))
    recorded = study[regime]
    assert recorded["assay"]["learned"] is learned
    gates = {arm: _gates([r for grp in rows(arm) for r in grp]) for arm in ("G1", "G2", "G3")}
    for arm in ("G1", "G2"):
        c = {"beats_ffn": compare(arm, "fast8", "G0", "frozen"), "beats_fast_gelu": compare(arm, "fast8", "G3", "fast8"),
             "adaptation_curve": compare(arm, "fast8", arm, "frozen"),
             "target_driven": compare(arm, "fast8", arm, "shuffled")}
        for key, name in (("beats_ffn", "fast_vs_ffn"), ("beats_fast_gelu", "fast_vs_fast_gelu"),
                          ("adaptation_curve", "fast_vs_frozen_self"), ("target_driven", "fast_vs_shuffled")):
            r = recorded["comparisons"][arm][name]
            assert math.isclose(r["pooled_geomean_ratio"], c[key]["ratio"], rel_tol=1e-9), (arm, key)
            assert math.isclose(r["sign_test_p"], c[key]["p"], rel_tol=1e-9, abs_tol=1e-300), (arm, key)
            assert r["better"] is c[key]["better"], (arm, key)
        integrity = all(gates[arm].values()) and all(gates["G3"].values())
        criteria = {"assay_learned": learned, **{k: v["better"] for k, v in c.items()}, "integrity": integrity}
        label = ("NONQUAL_INTEGRITY" if not integrity else "ASSAY_NOT_LEARNED" if not learned else
                 "INTEGRATED_FAST_CGC_SPECIALIST_QUALIFIED" if all(criteria.values()) else "NOT_ESTABLISHED")
        assert recorded["dispositions"][arm] == {"disposition": label, "criteria": criteria}, arm


def test_the_reading_follows_from_the_dispositions(study):
    q = "INTEGRATED_FAST_CGC_SPECIALIST_QUALIFIED"
    d = {r: {arm: study[r]["dispositions"][arm]["disposition"] for arm in ("G1", "G2")} for r in ("T1", "T2")}
    if q in d["T1"].values():
        reading = "DIRECT_RESTORATION_QUALIFIED"
    elif q in d["T2"].values():
        reading = "QUALIFIED_ONLY_WITH_ADAPTATION_TRAINED_INITIALIZATION"
    elif any(v.startswith("NONQUAL") for r in d.values() for v in r.values()):
        reading = "NONQUAL"
    else:
        reading = "NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED"
    assert study["verdict"]["dispositions"] == d and study["verdict"]["reading"] == reading
    assert study["refused"] == []


def test_the_combined_study_names_every_input_by_digest(study):
    assert set(study["inputs"]) == {p.name for p in RESULTS.glob("t[12]_*.json")}
    for name, digest in study["inputs"].items():
        assert _sha(RESULTS / name) == digest, name


def test_the_manifest_lists_every_r12_record_with_its_digest():
    manifest = _load(RESULTS / "MANIFEST.json")
    files = {p.name for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json"}
    assert set(manifest["files"]) == files
    for name, digest in manifest["files"].items():
        assert _sha(RESULTS / name) == digest, name


# -- P5: the representation seam (DIAGNOSIS ONLY) ----------------------------------------------------------------

AMENDMENT = RESULTS / "amendment_p5.json"
P5_SOURCES = {**SOURCES, "p5_runner": TORCH / "experiments" / "run_r12_p5.py",
              "r11_runner": TORCH / "experiments" / "run_r11.py"}


@pytest.fixture(scope="module")
def amendment():
    return _load(AMENDMENT)


@pytest.fixture(scope="module")
def p5_records():
    return {p.name: _load(p) for p in sorted(RESULTS.glob("p5_*.json"))}


@pytest.fixture(scope="module")
def study_p5():
    return _load(RESULTS / "study_p5.json")


def test_p5_was_frozen_on_the_study_that_triggered_it(amendment, study, plan):
    reading = "NOT_ESTABLISHED_REPRESENTATION_STAGE_TRIGGERED"
    assert study["verdict"]["reading"] == amendment["trigger"]["reading"] == reading
    assert amendment["trigger"]["study_r12_sha256"] == _sha(RESULTS / "study_r12.json")
    assert amendment["trigger"]["preregistration_sha256"] == _sha(PLAN)
    r11 = _load(REPO / "research" / "elpis_torch" / "results" / "r11" / "preregistration.json")
    assert amendment["stage_b"]["plan_b"] == r11["stage_b"]["training"]
    for regime in ("T1", "T2"):
        assert amendment["groups"][regime]["seeds"] == plan["regimes"][regime]["seeds"]
        assert amendment["groups"][regime]["namespace"] == plan["regimes"][regime]["namespace"]


def test_every_p5_record_was_produced_by_the_frozen_amendment_and_sources(amendment, p5_records):
    expected = {"amendment_sha256": _sha(AMENDMENT), "preregistration_sha256": _sha(PLAN),
                "study_r12_sha256": _sha(RESULTS / "study_r12.json"),
                **{f"{k}_sha256": _sha(v) for k, v in P5_SOURCES.items()}}
    names = {f"p5_b_{r}_s{s}.json" for r in ("T1", "T2") for s in amendment["groups"][r]["seeds"]} | \
        {f"p5_c_s{s}.json" for s in amendment["groups"]["T2"]["seeds"]}
    assert set(p5_records) == names
    for name, record in p5_records.items():
        assert record["stage"] == "diagnosis" and record["status"] == "COMPLETED", name
        assert {k: record["provenance"][k] for k in expected} == expected, name
        assert not record["provenance"]["git_revision"].endswith("+dirty"), name


def test_the_amendment_predates_every_p5_record():
    relative = AMENDMENT.relative_to(REPO).as_posix()
    revisions = {_load(p)["provenance"]["git_revision"].split("+")[0] for p in RESULTS.glob("p5_*.json")}
    assert revisions
    for revision in revisions:
        shown = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=REPO, capture_output=True)
        if shown.returncode != 0:
            pytest.skip("the recorded revision is not in this checkout's history (shallow clone)")
        assert hashlib.sha256(shown.stdout).hexdigest() == _sha(AMENDMENT), revision


def test_p5_development_records_are_labelled_and_named_by_the_amendment(amendment):
    named = amendment["development"]["records"]
    assert {p.name for p in RESULTS.glob("p5dev_*.json")} == set(named)
    for name, digest in named.items():
        record = _load(RESULTS / name)
        assert _sha(RESULTS / name) == digest and record["stage"] == "development", name
        assert record["label"] == "DEVELOPMENT ONLY", name
    assert all(_load(RESULTS / "p5dev_stage_b_r11b_s201.json")["identical_to_r11"].values())


def test_the_p5_stack_models_are_the_t2_models_and_every_gate_held(p5_records, study_p5):
    stacks = {n: r for n, r in p5_records.items() if n.startswith("p5_c_")}
    assert stacks
    for name, record in stacks.items():
        for family, entry in record["families"].items():
            check = entry["rebuilt"]["verification"]
            assert check["rows_identical"] and check["selected_step_equal"] and check["trace_equal"], (name, family)
            t2 = _load(RESULTS / f"t2_{family}_s{record['seed']}.json")["arms"]["G2"]["rows"]
            assert [r["fast"] for r in entry["rebuilt"]["rows"]] == [r["fast"] for r in t2], (name, family)
            assert entry["fixed_projection"]["projection_unchanged"], (name, family)
            for label in ("rebuilt", "fixed_projection", "misrouted_branch", "misrouted_unit"):
                assert all(r["integrity"]["all"] for r in entry[label]["rows"]), (name, family, label)
    assert study_p5["integrity"]["all"] and study_p5["disposition"] == "COMPLETED" and study_p5["refused"] == []
    assert set(study_p5["inputs"]) == set(p5_records)
    for name, digest in study_p5["inputs"].items():
        assert _sha(RESULTS / name) == digest, name


def _three(x_vs_y, y_vs_x, yes, no, neither):
    return yes if x_vs_y["better"] else (no if y_vs_x["better"] else neither)


def test_the_p5_readings_follow_from_the_rows(plan, amendment, p5_records, study, study_p5):
    rule = plan["rules"]["better"]

    def logs(xs, ys):
        return [[math.log(max(a, 1e-300) / max(b, 1e-300)) for a, b in zip(gx, gy)] for gx, gy in zip(xs, ys)]

    def check(recorded, xs, ys):
        mine = _better(logs(xs, ys), rule)
        assert math.isclose(recorded["pooled_geomean_ratio"], mine["ratio"], rel_tol=1e-9)
        assert math.isclose(recorded["sign_test_p"], mine["p"], rel_tol=1e-9, abs_tol=1e-300)
        assert recorded["better"] is mine["better"]
        return mine

    for regime in ("T1", "T2"):
        groups = [(f, s) for s in amendment["groups"][regime]["seeds"] for f in FAMILIES]

        def confirmation(arm, value, regime=regime, groups=groups):
            return [[value(r) for r in _load(RESULTS / f"{regime.lower()}_{f}_s{s}.json")["arms"][arm]["rows"]]
                    for f, s in groups]

        def stage_b(arm, key="rows", regime=regime, groups=groups):
            return [[r["curve"]["8"][0] for r in p5_records[f"p5_b_{regime}_s{s}.json"]["groups"][f][arm][key]]
                    for f, s in groups]

        g0 = confirmation("G0", lambda r: r["frozen"]["query"])
        stack_g2 = confirmation("G2", lambda r: r["fast"]["8"])
        sb = {arm: stage_b(arm) for arm in amendment["stage_b"]["arms"]}
        c = study_p5["comparisons"][regime]
        assay = check(c["stage_b_g2_vs_stage_b_g0"], sb["G2"], sb["G0"])
        check(c["stage_b_g2_vs_stage_b_g3"], sb["G2"], sb["G3"])
        up = check(c["stage_b_g2_vs_in_context_g0"], sb["G2"], g0)
        down = check(c["in_context_g0_vs_stage_b_g2"], g0, sb["G2"])
        identity = check(c["stage_b_g2_vs_stage_b_misrouted"], sb["G2"], stage_b("G2", "misrouted_rows"))
        access = check(c["stage_b_g2_vs_stack_g2"], sb["G2"], stack_g2)
        reverse = check(c["stack_g2_vs_stage_b_g2"], stack_g2, sb["G2"])
        assert study_p5["readings"][regime] == {
            "stage_b_assay": "REPRODUCED" if assay["better"] else "NOT_REPRODUCED",
            "benefit_ceiling": _three(up, down, "HEADROOM_UNDER_COORDINATE_ACCESS", "IN_CONTEXT_INFERENCE_BETTER",
                                      "INDETERMINATE"),
            "task_feature_access": _three(access, reverse, "COORDINATE_ACCESS_BETTER", "STACK_REPRESENTATION_BETTER",
                                          "NO_DIFFERENCE_ESTABLISHED"),
            "stage_b_identity": "IDENTITY_MATTERS" if identity["better"] else "NOT_ESTABLISHED"}, regime

    groups = [(f, s) for s in amendment["groups"]["T2"]["seeds"] for f in FAMILIES]

    def stack(label):
        return [[r["fast"]["8"] for r in p5_records[f"p5_c_s{s}.json"]["families"][f][label]["rows"]] for f, s in groups]

    c = study_p5["comparisons"]["T2_stack"]
    learned, fixed = stack("rebuilt"), stack("fixed_projection")
    lf = check(c["learned_vs_fixed_projection"], learned, fixed)
    fl = check(c["fixed_vs_learned_projection"], fixed, learned)
    branch = check(c["correct_vs_misrouted_branch"], learned, stack("misrouted_branch"))
    unit = check(c["correct_vs_misrouted_unit"], learned, stack("misrouted_unit"))
    assert study_p5["readings"]["T2_stack"] == {
        "projection": _three(lf, fl, "LEARNED_PROJECTION_BETTER", "FIXED_PROJECTION_BETTER", "NO_DIFFERENCE_ESTABLISHED"),
        "identity_branch": "IDENTITY_MATTERS" if branch["better"] else "NOT_ESTABLISHED",
        "identity_unit": "IDENTITY_MATTERS" if unit["better"] else "NOT_ESTABLISHED"}
    t1 = study["T1"]["dispositions"]["G2"]["criteria"]["adaptation_curve"]
    t2 = study["T2"]["dispositions"]["G2"]["criteria"]["adaptation_curve"]
    assert study_p5["attribution"]["initialization"] == (
        "EXPLAINS_THE_T1_CURVE" if t2 and not t1 else "NOT_THE_EXPLANATION" if t1 else "NOT_SUFFICIENT")
    assert study_p5["attribution"]["task_feature_access"] == study_p5["readings"]["T2"]["task_feature_access"]
    assert study_p5["attribution"]["absence_of_benefit"] == {r: study_p5["readings"][r]["benefit_ceiling"]
                                                              for r in ("T1", "T2")}
