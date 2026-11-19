"""Elpis Torch R9 records: provenance, the preregistered verdict recomputed independently, and the manifest.

These tests read JSON only (no PyTorch), so they also run where PyTorch is absent. They re-implement the frozen
classification of ``PREREGISTRATION_R9.md`` section 5 rather than calling the runner, and they check that every
confirmatory record was produced by the committed preregistration and runner (byte digests).
"""
from __future__ import annotations

import hashlib
import json
import statistics

import pytest

from .conftest import REPO

RESULTS = REPO / "research" / "elpis_torch" / "results" / "r9"
PREREGISTRATION = RESULTS / "preregistration.json"
RUNNER = REPO / "research" / "elpis_torch" / "experiments" / "run_r9.py"
DOMAINS = ("code", "prose")


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan():
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs(plan):
    out = {}
    for path in sorted(RESULTS.glob("cgc_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[(record["arm"], record["seed"])] = record
    return out


def test_every_confirmatory_record_was_produced_by_the_committed_plan_and_runner(runs, plan):
    assert runs, "no confirmatory records"
    expected = (_sha(PREREGISTRATION), _sha(RUNNER))
    for key, record in runs.items():
        p = record["provenance"]
        assert (p["preregistration_sha256"], p["runner_sha256"]) == expected, key
        assert record["corpus"]["digest"] == plan["corpus"]["digest"], key
    for name in ("integrity.json", "latency.json", "study_cgc_attribution.json"):
        p = _load(name)["provenance"]
        assert (p["preregistration_sha256"], p["runner_sha256"]) == expected, name


def test_the_preregistration_predates_every_confirmatory_record():
    """The frozen plan was committed before the runs: every record names a revision that contains it unchanged."""
    import subprocess
    revisions = {_load(p.name)["provenance"]["git_revision"].split("+")[0] for p in RESULTS.glob("cgc_*.json")}
    for revision in revisions:
        shown = subprocess.run(["git", "show", f"{revision}:research/elpis_torch/results/r9/preregistration.json"],
                               cwd=REPO, capture_output=True)
        if shown.returncode != 0:
            pytest.skip("the recorded revision is not in this checkout's history (shallow clone)")
        assert hashlib.sha256(shown.stdout).hexdigest() == _sha(PREREGISTRATION), revision


def test_every_primary_run_completed_exactly_the_preregistered_steps(runs, plan):
    for arm in ("A", "B", "C", "C_mac"):
        for seed in plan["seeds"]:
            record = runs[(arm, seed)]
            assert record["status"] == "COMPLETED" and record["train"]["steps"] == plan["training"]["steps"]
            assert [p["step"] for p in record["curve"]] == list(range(250, 1501, 250))


def _classify(diffs, delta):
    mean = statistics.fmean(diffs)
    if all(v > 0 for v in diffs) and mean >= delta:
        return "A_BETTER"
    if all(v < 0 for v in diffs) and mean <= -delta:
        return "X_BETTER"
    return "NOT_DISTINGUISHED"


def test_the_recorded_verdict_follows_from_the_run_records_under_the_frozen_rule(runs, plan):
    delta = plan["analysis"]["delta_min_bits_per_byte"]
    study = _load("study_cgc_attribution.json")

    def classes(other):
        out = {}
        for d in DOMAINS:
            key = f"val_{d}_bits_per_byte"
            diffs = [runs[(other, s)]["final"][key] - runs[("A", s)]["final"][key] for s in plan["seeds"]]
            out[d] = _classify(diffs, delta)
            assert study["comparisons"][f"{other}_vs_A"][d]["per_seed"] == pytest.approx(diffs, abs=1e-15)
            assert study["comparisons"][f"{other}_vs_A"][d]["class"] == out[d]
        return out

    vs_c, vs_mac, vs_b = classes("C"), classes("C_mac"), classes("B")
    if all(v == "A_BETTER" for v in vs_c.values()) and all(v == "A_BETTER" for v in vs_mac.values()):
        verdict = "CGC_BENEFIT_QUALIFIED"
    elif any(v == "X_BETTER" for v in vs_c.values()) and not any(v == "A_BETTER" for v in vs_c.values()):
        verdict = "FFN_PREFERRED"
    else:
        verdict = "CGC_NOT_ESTABLISHED"
    assert study["primary_verdict"] == verdict
    b = list(vs_b.values())
    assert study["secondary_verdict_B"] == ("CGC_ADDS_OVER_SMALLER_FFN" if all(v == "A_BETTER" for v in b) else
                                            "SMALLER_FFN_NOT_WORSE" if not any(v == "A_BETTER" for v in b) else "MIXED")


def test_arm_d_is_the_trained_arm_a_with_cgc_removed_and_is_never_a_trained_arm(runs, plan):
    assert "kind" not in plan["arms"]["D"] and plan["arms"]["D"]["from"] == "A"
    for seed in plan["seeds"]:
        record = runs[("A", seed)]
        assert set(record["gates_zeroed"]) == {f"val_{d}_bits_per_byte" for d in DOMAINS}
        assert len(record["cgc_gates_final"]) == 16
    assert not any(arm == "D" for arm, _ in runs)


def test_the_r7_reproduction_and_construction_identity_passed():
    record = _load("integrity.json")
    assert record["status"] == "PASS" and all(record["construction_identity"].values())
    assert record["train"]["steps"] == 1432                       # R7's protocol as executed: one epoch
    assert max(record["deviation"].values()) <= 0.005


def test_the_manifest_lists_every_r9_record_with_its_digest():
    manifest = _load("MANIFEST.json")
    files = {p.name for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json"}
    assert set(manifest["files"]) == files
    for name, digest in manifest["files"].items():
        assert _sha(RESULTS / name) == digest, name


# -- P3: whole-model accounting records ---------------------------------------------------------------------------

def _configuration(account, name, ple=False):
    return next(c for c in account["configurations"] if c["configuration"] == name and c["ple"] is ple)


def test_the_history_state_is_allocated_at_capacity_and_live_history_stays_inside_it():
    account = _load("p3_account.json")
    for config in account["configurations"]:
        rows = config["rows"]
        totals = {r["exact"]["fixed_state_bytes_allocated_total"] for r in rows}
        assert len(totals) == 1, config["configuration"]            # allocated once, independent of the length
        for r in rows:
            e = r["exact"]
            for tier in ("r5_token_ring", "descriptor", "fine_key", "window_kv"):
                if f"{tier}_bytes_allocated" in e:
                    assert e[f"{tier}_bytes_live"] <= e[f"{tier}_bytes_allocated"], (config["configuration"], tier)
        if config["history_capacity"]:
            last = rows[-1]["exact"]                                  # at 1M live history fills the allocation
            assert last["r5_token_ring_bytes_live"] == last["r5_token_ring_bytes_allocated"]


def test_stored_fine_keys_add_exactly_their_tier_and_cut_the_read_flops():
    account = _load("p3_account.json")
    for scale in ("r6_500m", "r6_1b"):
        reread, stored = _configuration(account, f"{scale}_reread"), _configuration(account, f"{scale}_stored")
        a, b = reread["rows"][-1], stored["rows"][-1]
        tier = b["exact"]["fine_key_bytes_allocated"]
        assert tier == (1 << 20) // 32 * 32 * 64 * 4                 # 256 MiB of float32 keys at 1M
        assert b["exact"]["fixed_state_bytes_allocated_total"] - a["exact"]["fixed_state_bytes_allocated_total"] == tier
        assert b["projected_per_decoded_token"]["flops"]["total"] < a["projected_per_decoded_token"]["flops"]["total"] / 5


def test_a_lookup_table_is_never_counted_as_matrix_weights():
    """The 614 MiB draft figure counted the encoder's delta table (65,536 x 256) as weights read every token."""
    account = _load("p3_account.json")
    row = next(r for r in _configuration(account, "r6_500m_stored")["rows"] if r["length"] == 1 << 20)
    tables = row["exact"]["row_read_tables"]
    assert tables["encoder.delta"] == {"rows": 65536, "width": 256}
    per = row["projected_per_decoded_token"]["bytes"]
    gathered = sum(t["rows"] * t["width"] for t in tables.values())
    assert per["weights"] == 2 * (row["exact"]["weights_active_parameters_per_token"] - gathered)
    assert 0 < per["encoder_rows_gathered"] < (1 << 20)                # rows read, not the 32 MiB table


def test_the_wrapped_steady_state_copies_the_live_index():
    account = _load("p3_account.json")
    for config in account["configurations"]:
        if not config["history_capacity"]:
            continue
        last = config["rows"][-1]
        assert last["steady_state_wrapped"] and last["length"] > config["history_capacity"]
        b = last["projected_per_decoded_token"]["bytes"]
        assert b["r5_wrapped_index_copy"] == 2 * b["r5_descriptor_scan"]


def test_measured_state_allocation_agrees_with_the_exact_accounting():
    for mode in ("reread", "stored"):
        record = _load(f"p3_rss_{mode}.json")["states"][mode]
        assert 0.9 <= record["state_rss_over_exact"] <= 1.05, (mode, record["state_rss_over_exact"])


def test_the_roofline_is_derived_from_the_committed_account_and_the_frozen_assumptions(plan):
    roofline = _load("p3_roofline.json")
    assert roofline["account_sha256"] == _sha(RESULTS / "p3_account.json")
    assert roofline["assumptions"]["ddr3"] == plan["p3"]["ddr3_scenario"]
    assert roofline["assumptions"]["gpu"] == plan["p3"]["gpu_scenario"]
    assert "PROJECTED" in roofline["labels"]["all"] and "CUDA NOT_RUN" in roofline["labels"]["all"]


# -- P3: the executed decode trace ----------------------------------------------------------------------------------

def _trace_steps(name, mode, ple=None, offset=16):
    trace = _load("p3_trace.json")
    config = next(c for c in trace["configurations"] if c["configuration"] == name and c["ple"] == ple)
    return {s["length"]: s for s in config["modes"][mode]["steps"] if s["position"] - s["length"] == offset}


def _class_total(step, name):
    return sum(step["classes"].get(name, {}).values())


def test_the_synthetic_fill_moves_exactly_the_bytes_of_a_real_prefill():
    trace = _load("p3_trace.json")
    checked = 0
    for config in trace["configurations"]:
        for mode, data in config["modes"].items():
            v = data["validation"]
            if not v:
                continue
            assert v["equal_totals"] and v["counters_equal"], (config["configuration"], mode)
            assert v["differing_keys_not_routed_expert_weights"] == [], (config["configuration"], mode)
            checked += 1
    assert checked >= 4


def test_ersa_reads_no_history_at_any_length():
    for mode in ("stored", "reread"):
        steps = _trace_steps("r6_500m", mode)
        ersa = {L: (_class_total(s, "ersa_recurrent_state"), s["stages"]["ersa_mixer"]) for L, s in steps.items()}
        assert len(set(ersa.values())) == 1, ersa                     # byte-identical from 1K to the wrapped ring
        state = steps[1 << 20]["classes"]["ersa_recurrent_state"]
        assert state["ersa_state_read"] == 3 * state["ersa_state_write"]       # S k, S q, retain S; one write


def test_only_the_r5_scan_grows_with_context_beyond_the_window():
    for mode in ("stored", "reread"):
        steps = _trace_steps("r6_500m", mode)
        small, large = steps[8192], steps[1 << 20]
        for cls in ("weight_traffic", "ersa_recurrent_state", "evidence_reencoding", "embedding_lookups"):
            assert _class_total(small, cls) == _class_total(large, cls), (mode, cls)
        growth = large["total"] - small["total"]
        scan = large["classes"]["kv_history_reads"]["r5_descriptor_scan"] - \
            small["classes"]["kv_history_reads"]["r5_descriptor_scan"]
        other = large["classes"]["other"]
        explained = scan + sum(other.get(k, 0) - small["classes"]["other"].get(k, 0)
                               for k in ("r5_per_block_bookkeeping_and_wrapped_index_copy",
                                         "r5_coarse_scan_intermediates"))
        assert abs(growth - explained) < 0.1 * (1 << 20), (mode, growth, explained)
        for L, s in steps.items():                                    # 1 KiB per live 32-token block
            assert abs(s["classes"]["kv_history_reads"]["r5_descriptor_scan"] - min(L, 1 << 20) // 32 * 1024) <= 1024


def test_the_window_kv_read_saturates_at_the_window():
    steps = _trace_steps("r6_500m", "stored")
    full = 5 * 2 * 2 * 4096 * 128 * 4                                   # 5 layers, K and V, 2 KV heads, float32
    assert all(s["classes"]["kv_history_reads"]["window_kv"] == full for L, s in steps.items() if L >= 4096)
    assert steps[1024]["classes"]["kv_history_reads"]["window_kv"] < full


def test_the_wrapped_ring_adds_one_index_copy_per_decoded_token():
    for mode in ("stored", "reread"):
        steps = _trace_steps("r6_500m", mode)
        at_capacity = steps[1 << 20]
        wrapped = steps[max(steps)]
        scan = wrapped["classes"]["kv_history_reads"]["r5_descriptor_scan"]
        assert abs((wrapped["total"] - at_capacity["total"]) - 2 * scan) < 0.01 * (1 << 20), mode


def test_the_unique_traffic_of_a_traced_step_matches_the_compulsory_account():
    account = _load("p3_account.json")
    row = next(r for r in _configuration(account, "r6_500m_stored")["rows"] if r["length"] == 1 << 20)
    projected = row["projected_per_decoded_token"]["bytes"]
    unique = _trace_steps("r6_500m", "stored")[1 << 20]["unique_persistent"]
    # float32 as executed against the account's bfloat16 weights; the account counts every active weight, the step
    # leaves out the descriptor projection used only when a block completes (17,408 parameters)
    assert abs(unique["weights"] - 2 * projected["weights"]) <= 17408 * 4
    # the trace decodes at 1M + 16, where the edge block has lost its left context: one live block fewer
    assert 0 <= projected["r5_descriptor_scan"] - unique["state_tiers"]["descriptors_read"] <= 4 * 64 * 4


def test_dram_traffic_is_not_claimed_where_no_counters_exist():
    trace = _load("p3_trace.json")
    assert trace["pmu"]["hardware_pmu"] is False
    assert "Not DRAM transfers" in trace["labels"]["all"] and "CUDA NOT_RUN" in trace["labels"]["all"]
