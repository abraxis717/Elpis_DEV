"""Elpis Torch R11 records: provenance, integrity and the frozen verdicts recomputed independently.

These tests read JSON only (no PyTorch). They re-implement the decision rules of ``PREREGISTRATION_R11.md`` §6 and
of the Stage C amendment (``PREREGISTRATION_R11_STAGE_C.md`` §3) rather than calling the runners, and check that:
- every confirmation record was produced by the frozen plan and sources (byte digests);
- the plans predate the records they govern;
- every planned group has exactly one record, and no unplanned one exists;
- development records are labelled and use development namespaces only;
- the recorded dispositions follow from the per-episode rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess

import pytest

from .conftest import REPO

RESULTS = REPO / "research" / "elpis_torch" / "results" / "r11"
PLAN = RESULTS / "preregistration.json"
AMENDMENT = RESULTS / "amendment_stage_c.json"
TORCH = REPO / "research" / "elpis_torch"
SOURCES = {"runner": TORCH / "experiments" / "run_r11.py", "tasks": TORCH / "specialist_tasks.py",
           "profile": TORCH / "cgc_specialist.py", "sparse_channel": TORCH / "sparse_channel.py",
           "cgc": TORCH / "cgc.py", "model": TORCH / "model.py", "config": TORCH / "config.py",
           "upstream_e0": TORCH / "upstream" / "cgc_ecsg_e0.py"}
STAGE_C_SOURCES = {**SOURCES, "stage_c_runner": TORCH / "experiments" / "run_r11_stage_c.py",
                   "stack": TORCH / "specialist_stack.py"}
FAMILIES = ("smooth", "conditional")


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a, b) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


@pytest.fixture(scope="module")
def plan():
    return _load(PLAN)


@pytest.fixture(scope="module")
def amendment():
    return _load(AMENDMENT)


@pytest.fixture(scope="module")
def stage_a():
    return {(r["family"], r["seed"]): r for r in map(_load, sorted(RESULTS.glob("stage_a_*.json")))}


@pytest.fixture(scope="module")
def stage_b():
    return {r["seed"]: r for r in map(_load, sorted(RESULTS.glob("stage_b_*.json")))}


@pytest.fixture(scope="module")
def stage_c():
    runs, routers = {}, {}
    for path in sorted(RESULTS.glob("c_*.json")):
        if path.name == "c_study.json":
            continue
        record = _load(path)
        if record["command"] == "run":
            runs[(record["family"], record["seed"])] = record
        else:
            routers[record["seed"]] = record
    return runs, routers


# -- the statistics, re-implemented ------------------------------------------------------------------------------------

def _log_ratios(rows_x, rows_y, step, key="curve"):
    out = []
    for x, y in zip(rows_x, rows_y, strict=True):
        assert x["episode"] == y["episode"]
        a = x[key][step] if key == "curve" else x[key]      # Stage A/B rows hold [query, OOD]; Stage C, query
        a = a[0] if isinstance(a, list) else a
        out.append(math.log(max(a, 1e-300) / max(y["curve"][step][0], 1e-300)))
    return out


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


def _check(recorded: dict, mine: dict, label: str):
    assert _close(recorded["pooled_geomean_ratio"], mine["ratio"]), label
    assert _close(recorded["sign_test_p"], mine["p"]), label
    assert _close(recorded["groups_below_one"], mine["below"]), label
    assert recorded["better"] is mine["better"], label


# -- provenance and integrity ------------------------------------------------------------------------------------------

def test_every_confirmation_record_was_produced_by_the_frozen_plan_and_sources(stage_a, stage_b):
    expected = {"preregistration_sha256": _sha(PLAN), **{f"{k}_sha256": _sha(v) for k, v in SOURCES.items()}}
    assert stage_a and stage_b
    for key, record in [*stage_a.items(), *stage_b.items()]:
        assert record["stage"] == "confirmation" and record["status"] == "COMPLETED", key
        provenance = record["provenance"]
        assert {k: provenance[k] for k in expected} == expected, key
        assert not provenance["git_revision"].endswith("+dirty"), key


def test_every_stage_c_record_was_produced_by_the_frozen_amendment_and_sources(stage_c):
    runs, routers = stage_c
    expected = {"preregistration_sha256": _sha(PLAN), "amendment_sha256": _sha(AMENDMENT),
                **{f"{k}_sha256": _sha(v) for k, v in STAGE_C_SOURCES.items()}}
    assert runs
    dirty = set()
    for key, record in [*runs.items(), *routers.items()]:
        assert record["stage"] == "confirmation" and record["status"] == "COMPLETED", key
        assert {k: record["provenance"][k] for k in expected} == expected, key
        if record["provenance"]["git_revision"].endswith("+dirty"):
            dirty.add(key)
    # The two documented exceptions (EVIDENCE_R11.md section 11), admitted by the test below; every other is clean.
    assert dirty <= {("smooth", 301), ("conditional", 301)}


@pytest.mark.parametrize("pattern, frozen", [("stage_*.json", PLAN), ("c_*_s*.json", AMENDMENT)])
def test_each_plan_predates_every_record_it_governs(pattern, frozen):
    relative = frozen.relative_to(REPO).as_posix()
    revisions = {_load(p)["provenance"]["git_revision"].split("+")[0] for p in RESULTS.glob(pattern)}
    assert revisions
    for revision in revisions:
        shown = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=REPO, capture_output=True)
        if shown.returncode != 0:
            pytest.skip("the recorded revision is not in this checkout's history (shallow clone)")
        assert hashlib.sha256(shown.stdout).hexdigest() == _sha(frozen), revision


def test_a_record_written_beside_uncommitted_edits_ran_unchanged_code():
    """``+dirty`` marks a record written while the working tree held uncommitted edits (outside the results). Such a
    record is admissible only if the commit that added it changed no package module since the recorded revision,
    except the performance-only benchmark supplement, which no runner imports (``EVIDENCE_R11.md`` §11)."""
    allowed = {"research/elpis_torch/experiments/bench_r11_operators.py", "research/elpis_torch/PROVENANCE.md"}
    for path in sorted(RESULTS.glob("*.json")):
        revision = (_load(path).get("provenance") or {}).get("git_revision", "")
        if not revision.endswith("+dirty"):
            continue
        added = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", path.relative_to(REPO).as_posix()],
                               cwd=REPO, capture_output=True, text=True)
        if added.returncode != 0 or not added.stdout.split():
            pytest.skip("the record's history is not in this checkout (shallow clone)")
        changed = subprocess.run(["git", "diff", "--name-only", revision.split("+")[0], added.stdout.split()[-1], "--",
                                  "research/elpis_torch", "tests/research/elpis_torch/conftest.py"],
                                 cwd=REPO, capture_output=True, text=True)
        if changed.returncode != 0:
            pytest.skip("the recorded revision is not in this checkout's history (shallow clone)")
        assert all(c in allowed or c.startswith("research/elpis_torch/results/") for c in changed.stdout.split()), \
            path.name
    for runner in (SOURCES["runner"], STAGE_C_SOURCES["stage_c_runner"], STAGE_C_SOURCES["stack"]):
        assert "bench_r11_operators" not in runner.read_text(encoding="utf-8")


def test_every_planned_group_has_one_record_and_no_unplanned_record_exists(plan, amendment, stage_a, stage_b,
                                                                            stage_c):
    assert set(stage_a) == {(f, s) for f in FAMILIES for s in plan["stage_a"]["seeds"]}
    assert set(stage_b) == set(plan["stage_b"]["seeds"])
    runs, routers = stage_c
    assert set(runs) == {(f, s) for f in FAMILIES for s in amendment["seeds"]}
    assert set(routers) <= set(amendment["seeds"])
    for record in stage_a.values():
        assert record["namespace"] == plan["stage_a"]["namespace"] and record["lr_proj"] == plan["stage_a"]["lr_proj"]
        assert set(record["arms"]) == set(plan["stage_a"]["arms"])
    for record in stage_b.values():
        assert record["namespace"] == plan["stage_b"]["namespace"] and record["split"] == "TEST"
        assert record["plan_b"] == plan["stage_b"]["training"]
        for family in FAMILIES:
            assert set(record["groups"][family]) == set(plan["stage_b"]["arms"])
    for record in runs.values():
        assert set(record["arms"]) == set(amendment["arms"]) and record["profile"] == amendment["profile"]


def test_development_records_are_labelled_and_named_by_the_plans(plan, amendment):
    confirmation = set(plan["stage_a"]["seeds"]) | set(plan["stage_b"]["seeds"]) | set(amendment["seeds"])
    named = {**plan["development"]["records"], **amendment["development"]["records"]}
    found = {p.name for p in RESULTS.glob("dev_*.json")}
    assert found == set(named)
    for name, digest in named.items():
        record = _load(RESULTS / name)
        assert _sha(RESULTS / name) == digest, name
        if record["command"] == "dev":
            assert record["label"].startswith("DEVELOPMENT ONLY") and "R11C_DEV" in record["label"], name
        else:
            assert record["stage"] == "development" and record["namespace"].endswith("_DEV"), name
            assert record["seed"] not in confirmation, name
    assert _sha(RESULTS / "calibration.json") == plan["development"]["calibration_sha256"]
    assert _load(RESULTS / "calibration.json")["label"].startswith("DEVELOPMENT ONLY")


def test_the_parity_record_qualifies_the_e0_reference(plan):
    record = _load(RESULTS / "parity.json")
    assert _sha(RESULTS / "parity.json") == plan["parity"]["record_sha256"]
    assert record["e0_reference_forward_max_abs"] <= record["tolerance_e0"] == 1e-12
    assert record["e0_reference_grad_autograd_vs_analytic_max_abs"] <= 1e-12
    assert all(row["taylor_bound_holds"] for row in record["bounded_vs_e0"])


# -- Stage A, recomputed ------------------------------------------------------------------------------------------------

STAGE_A_UTILITY = (("E0_META", "E0_COLD"), ("GELU_META", "GELU_COLD"), ("PROJ_E0_FIXED_META", "PROJ_E0_COLD"),
                   ("PROJ_E0_LEARNED_META", "PROJ_E0_COLD"), ("PROJ_GELU_FIXED_META", "PROJ_GELU_COLD"),
                   ("PROJ_GELU_LEARNED_META", "PROJ_GELU_COLD"), ("PROJ_B3_FIXED_META", "PROJ_B3_COLD"),
                   ("PROJ_B3_LEARNED_META", "PROJ_B3_COLD"))


def _auc(rows):
    return statistics.median(statistics.fmean(r["curve"][k][0] for k in ("0", "1", "2", "4", "8")) for r in rows)


def test_the_stage_a_verdicts_follow_from_the_records(plan, stage_a):
    study = _load(RESULTS / "study_r11.json")["stage_a"]
    util, rule = plan["analysis"]["stage_a_utility"], plan["analysis"]["better"]
    keys = sorted(stage_a)
    passes = {}
    for meta, cold in STAGE_A_UTILITY:
        verdicts = []
        for key in keys:
            arms = stage_a[key]["arms"]
            logs = _log_ratios(arms[meta]["rows"], arms[cold]["rows"], "8")
            forget = [f["ratio"] for f in arms[meta]["forget"]]
            assert forget, (meta, key)
            verdicts.append(math.exp(statistics.fmean(logs)) <= util["max_geomean_meta_vs_cold"]
                            and sum(v < 0 for v in logs) / len(logs) >= util["min_paired_positive_fraction"]
                            and _auc(arms[meta]["rows"]) / _auc(arms[cold]["rows"]) <= util["max_auc_ratio"]
                            and statistics.median(forget) >= util["min_reset_to_fast_ratio"])
        passes[meta] = all(verdicts)
        assert study["utility"][meta]["all_groups_pass"] is passes[meta], meta
        assert [g["pass"] for g in study["utility"][meta]["groups"]] == verdicts, meta
    for name, recorded in study["comparisons"].items():
        x, y = name.split("/")
        mine = _better([_log_ratios(stage_a[k]["arms"][x]["rows"], stage_a[k]["arms"][y]["rows"], "8")
                        for k in keys], rule)
        _check(recorded, mine, name)
    parity = _load(RESULTS / "parity.json")["e0_reference_qualified"]
    assert study["dispositions"]["MICRO_REFERENCE"] == (
        "MICRO_REFERENCE_QUALIFIED" if passes["E0_META"] and parity else "NOT_ESTABLISHED")
    for arm in ("PROJ_E0_FIXED_META", "PROJ_E0_LEARNED_META", "PROJ_B3_FIXED_META", "PROJ_B3_LEARNED_META"):
        assert study["dispositions"][arm] == ("PROJECTED_ADAPTATION_QUALIFIED" if passes[arm] else "NOT_ESTABLISHED")
    limit = plan["analysis"]["bound_destroys"]["max_ratio_bounded_over_unbounded"]
    destroys = any((not passes[f"PROJ_B3_{m}_META"] and passes[f"PROJ_E0_{m}_META"])
                   or study["comparisons"][f"PROJ_B3_{m}_META/PROJ_E0_{m}_META"]["pooled_geomean_ratio"] > limit
                   for m in ("FIXED", "LEARNED"))
    assert study["bound_destroys_behavior"] is destroys
    assert study["stage_b_profile"] == ("E0_REFERENCE" if destroys else "BOUNDED_SPECIALIST")


# -- Stage B, recomputed ------------------------------------------------------------------------------------------------

def test_the_stage_b_verdicts_follow_from_the_records(plan, stage_b):
    study = _load(RESULTS / "study_r11.json")
    b = study["stage_b"]
    rule, b_rule = plan["analysis"]["better"], plan["analysis"]["stage_b"]
    groups = [(s, f) for s in sorted(stage_b) for f in FAMILIES]
    assert all(stage_b[s]["profile"] == study["stage_a"]["stage_b_profile"] for s in stage_b)

    def rows(arm, kind="rows"):
        return [stage_b[s]["groups"][f][arm][kind] for s, f in groups]

    def compare(x_rows, y_rows, step):
        return _better([_log_ratios(a, c, step) for a, c in zip(x_rows, y_rows, strict=True)], rule)

    mine = {}
    for name, recorded in b["comparisons"].items():
        x, y = name.split("/")
        mine[name] = compare(rows(x), rows(y), "8")
        _check(recorded, mine[name], name)
        _check(b["mode_b"][name], compare(rows(x), rows(y), "0"), "mode_b " + name)
    causal = {}
    for arm in ("G1", "G2", "G3", "G2_POOLED"):
        pooled = [v for g in rows(arm) for v in _log_ratios(g, g, "8", key="ablated")]
        ratio = math.exp(statistics.fmean(pooled))
        causal[arm] = ratio >= b_rule["min_ablation_ratio"]
        assert _close(b["controls"][arm]["ablation"]["pooled_geomean_ablated_over_intact"], ratio), arm
        assert b["controls"][arm]["ablation"]["causal"] is causal[arm], arm
        _check(b["controls"][arm]["specialist_swap_over_correct"],
               compare(rows(arm, "specialist_swap_rows"), rows(arm), "8"), f"swap {arm}")
        _check(b["controls"][arm]["expert_misrouted_over_correct"],
               compare(rows(arm, "expert_misrouted_rows"), rows(arm), "8"), f"misrouted {arm}")
    learned = all(statistics.median(r["curve"]["0"][0] for r in g)
                  <= b_rule["max_g0_over_null"] * statistics.median(r["null"][0] for r in g) for g in rows("G0"))
    assert b["assay"]["ffn_learned_shared_function"] is learned is True
    dispositions, slot = {}, {}
    for arm in ("G1", "G2"):
        q = {"adaptive_benefit": mine[f"{arm}/G0"]["better"] and mine[f"{arm}/G0"]["ratio"] <= b_rule[
            "max_ratio_over_ffn"], "causal": causal[arm], "specialist_objective": mine[f"{arm}/G2_POOLED"]["better"],
             "cgc_specific_over_gelu": mine[f"{arm}/G3"]["better"]}
        assert b["qualification"][arm] == q, arm
        slot[arm] = q["adaptive_benefit"] and q["causal"]
        dispositions[arm] = ("EXPERT_SPECIALIZATION_QUALIFIED" if all(q.values()) else
                             "SPECIALIST_SLOT_FUNCTIONAL_NO_CGC_SPECIFIC_ADVANTAGE"
                             if slot[arm] and q["specialist_objective"] else "NOT_ESTABLISHED")
    dispositions["gate_identifiability"] = {(True, False): "GATE_IDENTIFIABILITY_FAILURE",
                                            (True, True): "GATE_NOT_LIMITING",
                                            (False, False): "SPECIALIST_OPERATION_NOT_ESTABLISHED"}.get(
        (slot["G2"], slot["G1"]), "LEARNED_GATE_ONLY")
    dispositions["mode_b"] = {arm: "FROZEN_BENEFIT" if compare(rows(arm), rows("G0"), "0")["better"] else
                              "ADAPTIVE_SPECIALIST_REQUIRES_INNER_UPDATES" if slot[arm] else "NOT_ESTABLISHED"
                              for arm in ("G1", "G2")}
    assert b["dispositions"] == dispositions
    triggered = "EXPERT_SPECIALIZATION_QUALIFIED" in (dispositions["G1"], dispositions["G2"])
    assert b["stage_c_triggered"] is triggered


def test_the_combined_study_names_every_confirmation_input_by_digest(stage_a, stage_b):
    study = _load(RESULTS / "study_r11.json")
    assert study["refused"] == []
    names = {p.name for p in RESULTS.glob("stage_*.json")}
    assert set(study["inputs"]) == names
    for name, digest in study["inputs"].items():
        assert _sha(RESULTS / name) == digest, name


# -- Stage C, recomputed ------------------------------------------------------------------------------------------------

def test_stage_c_ran_only_because_stage_b_triggered_it_and_its_amendment_names_that_analysis(amendment):
    study = _load(RESULTS / "study_r11.json")
    assert study["stage_b"]["stage_c_triggered"] is True
    assert amendment["trigger"]["study_r11_sha256"] == _sha(RESULTS / "study_r11.json")
    assert amendment["trigger"]["stage_b_dispositions"] == study["stage_b"]["dispositions"]
    assert amendment["preregistration_sha256"] == _sha(PLAN)


def test_the_stage_c_verdicts_follow_from_the_records(amendment, stage_c):
    runs, routers = stage_c
    study = _load(RESULTS / "c_study.json")
    rules = amendment["rules"]
    groups = [(f, s) for s in amendment["seeds"] for f in FAMILIES]
    assert study["refused"] == [] and "missing" not in study

    def rows(arm):
        return [runs[g]["arms"][arm]["rows"] for g in groups]

    def compare(x, y, step="32"):
        return _better([_log_ratios(a, c, step) for a, c in zip(rows(x), rows(y), strict=True)], rules["better"])

    mine = {}
    for name, recorded in study["comparisons_k32"].items():
        mine[name] = compare(*name.split("/"))
        _check(recorded, mine[name], name)
    for name, recorded in study["comparisons_k1"].items():
        _check(recorded, compare(*name.split("/"), step="1"), "k1 " + name)
    controls = {}
    for arm in amendment["arms"]:
        controls[arm] = {}
        for key in ("ablate_all", "ablate_block0", "ablate_block1", "ersa_reset_after_support"):
            pooled = [v for g in rows(arm) for v in _log_ratios(g, g, "32", key=key)]
            controls[arm][key] = math.exp(statistics.fmean(pooled))
            assert _close(study["controls"][arm][f"{key.replace('_after_support', '')}_over_intact"],
                          controls[arm][key]), (arm, key)
    learned = all(statistics.median(r["curve"]["32"][0] for r in g)
                  <= rules["max_g0_k32_over_null"] * statistics.median(r["null"] for r in g) for g in rows("G0"))
    assert study["assay"]["in_context_learned"] is learned
    expected = {}
    for arm in ("G1", "G2"):
        if not learned:
            expected[arm] = "ASSAY_NOT_LEARNED"
        elif (mine[f"{arm}/G0"]["better"] and mine[f"{arm}/G3"]["better"]
              and controls[arm]["ablate_all"] >= rules["min_ablation_ratio"]):
            expected[arm] = "FROZEN_INFERENCE_TRANSFER_QUALIFIED"
        elif mine[f"G0/{arm}"]["better"]:
            expected[arm] = "NOT_ESTABLISHED_FROZEN_SPECIALIST_WORSE_THAN_FFN"
        else:
            expected[arm] = "NOT_ESTABLISHED"
    assert study["dispositions"] == expected
    if routers:
        separated, purities = [], []
        for record in routers.values():
            for block in record["purity"].values():
                counts = {f: block["counts"][f] for f in FAMILIES}
                majority = {f: max(range(len(c)), key=lambda e, c=c: (c[e], -e)) for f, c in counts.items()}
                assert majority == block["majority_expert"]
                separated.append(majority["smooth"] != majority["conditional"])
                purities.extend(c[majority[f]] / sum(c) for f, c in counts.items())
        mean = statistics.fmean(purities)
        assert study["router"]["blocks_separating_families"] == sum(separated)
        assert math.isclose(study["router"]["mean_purity"], mean, rel_tol=1e-6)   # the runner divides in float32
        assert study["router"]["disposition"] == ("LEARNED_ROUTING_SEPARATES_FAMILIES"
                                                  if all(separated) and mean >= rules["min_router_purity"]
                                                  else "INTEGRATION_GAP_LEARNED_ROUTING")


def test_the_manifest_lists_every_r11_record_with_its_digest():
    manifest = _load(RESULTS / "MANIFEST.json")
    files = {p.name for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json"}
    assert set(manifest["files"]) == files
    for name, digest in manifest["files"].items():
        assert _sha(RESULTS / name) == digest, name
