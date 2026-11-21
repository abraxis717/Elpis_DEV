"""Elpis Torch R13 records: provenance, integrity and every gate recomputed from the raw rows.

These tests read JSON only (no PyTorch). They re-derive each gate from the recorded rows rather than trusting the
runner's summary strings, then recompute the dispositions of ``verdict.json``. They check that:
- every record was produced by the committed sources, on a clean revision, against the vendored Geodome bytes;
- the manifest lists every record with its digest;
- the identity, seam, parity, gradient, bridge and resource rows support their gates;
- the dispositions follow from those rows, and the competence fields say what was NOT done.
"""
from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from .conftest import REPO

RESULTS = REPO / "research" / "elpis_torch" / "results" / "r13"
TORCH = REPO / "research" / "elpis_torch"
VENDORED = TORCH / "upstream" / "geodome_v1"
COGNITIVE = ("source", "topology", "plan", "state", "kernels", "activation", "binding", "reference")
SOURCES = {"runner": TORCH / "experiments" / "run_r13.py",
           **{name: TORCH / "geodome_cognitive" / f"{name}.py" for name in COGNITIVE},
           **{name: TORCH / f"{name}.py" for name in ("evidence_model", "evidence_ring", "ersa_model", "lastblock",
                                                     "sparse_channel", "context_runtime", "model", "config", "scroll",
                                                     "ple")}}
GATES = ("identity", "audit", "parity", "gradients", "bridge", "resources")
RECORDS = (*GATES, "verdict")
GEODOME_COMMIT = "4e5fe56670c61a0d8906770b03033658c3087c3a"
GEODOME_MANIFEST = "e44d288fec19a09c17be0ce503f011d400b499e4d393f1275d305e2b23d31cb8"
ELPIS_BASE = "d73cdde3a5050de50cce098472ee966eec103000"
VENDORED_FILES = {"PUBLIC.sha256", "LICENSE", "NOTICE", "Hexaflag.py", "GEO144K/g1c_compact.py",
                  "GEO144K/g2b_bridge.py", "geodome.py", "geodome_index.py", "geodome_state.py", "geodome_scheduler.py"}
IDENTITY = {"vertices": 144002, "edges": 432000, "arcs": 864000, "capacity": 823543, "depth": 7,
            "mapping_sha256": "abe96202b554608bd8f5e702b70459daaaf4ab33eb7dfd058723fd4d1724f51b",
            "edge_sha256": "712a8fc3d94685aeb614456ba8c88f6df7e6c6ea8a4c19e86d4d3865ad014231",
            "index_payload_bytes": 11502238}
TOLERANCE = 1e-12
KERNEL_TENSORS = 19


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records():
    return {name: _load(RESULTS / f"{name}.json") for name in RECORDS}


# -- provenance and integrity ------------------------------------------------------------------------------------

def test_every_record_was_produced_by_the_committed_sources_and_the_vendored_geodome(records):
    expected = {f"{k}_sha256": _sha(v) for k, v in SOURCES.items()}
    for name, record in records.items():
        assert record["phase"] == "ELPIS_TORCH_R13_GEODOME_SUBSTRATE" and record["command"] == name
        provenance = record["provenance"]
        assert {k: provenance[k] for k in expected} == expected, name
        assert provenance["git_revision"] and not provenance["git_revision"].endswith("+dirty"), name
        assert (provenance["elpis_base"], provenance["geodome_commit"]) == (ELPIS_BASE, GEODOME_COMMIT), name
        files = provenance["geodome_files_sha256"]
        assert set(files) == VENDORED_FILES and files["PUBLIC.sha256"] == GEODOME_MANIFEST, name
        assert files == {local: _sha(VENDORED / local) for local in files}, name


def test_the_records_were_produced_on_a_revision_that_holds_these_sources(records):
    revisions = {record["provenance"]["git_revision"] for record in records.values()}
    for revision in revisions:
        for key in ("runner", "binding", "activation"):
            relative = SOURCES[key].relative_to(REPO).as_posix()
            shown = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=REPO, capture_output=True)
            if shown.returncode != 0:
                pytest.skip("the recorded revision is not in this checkout's history (shallow clone)")
            assert hashlib.sha256(shown.stdout).hexdigest() == _sha(SOURCES[key]), (revision, key)


def test_the_manifest_lists_every_r13_record_with_its_digest():
    manifest = _load(RESULTS / "MANIFEST.json")
    files = {p.name for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json"}
    assert set(manifest["files"]) == files == {f"{name}.json" for name in RECORDS}
    for name, digest in manifest["files"].items():
        assert _sha(RESULTS / name) == digest, name
    assert (manifest["geodome"]["commit"], manifest["geodome"]["license"]) == (GEODOME_COMMIT, "Apache-2.0")
    assert manifest["sources"] == {k: _sha(v) for k, v in SOURCES.items()}


# -- the gates, from the rows --------------------------------------------------------------------------------------

def test_gate_1_identity(records):
    r = records["identity"]
    for key, value in IDENTITY.items():
        assert r["identity"][key] == value == r["source"]["identity"][key], key
    assert r["identity"]["degree_five_vertices"] == 12
    checks = r["checks"]
    assert all(v["ok_address"] and v["ok_coordinate"] for v in checks["roundtrips"].values())
    assert all(a["ok_endpoints"] and a["ok_reverse"] and a["ok_port"] for a in checks["arcs"].values())
    assert checks["nonedge"]["code"] == "NONEDGE" and checks["out_of_range_arc"]["code"] == "ARC_RANGE"
    assert checks["elpis_base_sources"]["modified_or_removed"] == []
    assert checks["elpis_base_sources"]["prior_records_modified_or_removed"] == []
    assert all(v is None or v.startswith("research/elpis_torch/upstream/geodome_v1/")
               for v in r["modules_loaded_from"].values())
    disconnected = sum(g["components"] > 1 for g in r["prefix_groups"])
    assert r["logical_prefixes_disconnected"] == disconnected > 0       # a logical prefix is not a region
    assert r["gate_1"] == "PASS"


def test_the_p0_seam(records):
    c = records["audit"]["checks"]
    assert c["return_pre"]["captured_rows_shape"] == [1, 70, 32] and c["return_pre"]["equal_final_row"]
    assert c["tail"]["equal_T_row_recomputation"]
    assert c["evidence_modes"]["equal_rows_before_final"] and c["evidence_modes"]["equal_state"]
    assert c["evidence_modes"]["ok_read_present"]
    for name in ("chunk_limit", "invalid_tokens", "oracle_without_labels"):
        assert c[name]["refused"] and c[name]["equal_state_unchanged"], name
    assert c["placement"] == {"ok_last_block": True, "read_layer": 1}
    assert records["audit"]["gate_seam"] == "PASS"


def test_gates_3_4_and_hop_causality_against_geodomes_scheduler(records):
    r = records["parity"]
    statuses = set()
    for name, w in r["workloads"].items():
        for residency in ("sparse", "dense"):
            row = w[residency]
            assert row["equal_status"] and row["equal_rounds"] and row["equal_visits"] and row["equal_records"], name
            assert row["rows_max_abs_diff"] <= TOLERANCE, name
            statuses.add(row["status"])
        assert w["equal_dense_sparse_tensor_rows"] and w["sparse"]["status"] == w["dense"]["status"], name
    assert {"QUIESCENT", "ACTIVE_BUDGET", "ARC_BUDGET", "ROUND_BUDGET"} <= statuses
    matched = r["synchronous_matched"]
    assert matched["ok_in_matched_class"] and matched["equal_visits"] and matched["rows_max_abs_diff"] <= TOLERANCE
    assert r["divergence_witness"]["claimed_equal"] is False
    assert r["divergence_witness"]["max_abs_difference"] > 1e-6        # different policies really differ
    for rounds, hop in r["hop_causality"].items():
        assert hop["moved_hops"] == list(range(1, int(rounds) + 1)), rounds
    assert (r["gate_3"], r["gate_4_executor"], r["gate_7_executor"]) == ("PASS", "PASS", "PASS")


def test_gates_5_6_7_and_the_differentiable_path(records):
    g = records["gradients"]
    c = g["checks"]
    grads = c["multi_hop"]["grad_l1"]
    assert len(grads) == KERNEL_TENSORS and all(v is not None and 0 < v < float("inf") for v in grads.values())
    assert c["multi_hop"]["hops_evidence_to_readout"] == 5
    assert c["zero_gate"]["nonzero_gradients"] == ["gate"]
    bound = c["hop_bound"]
    assert bound["4"]["evidence_ingress_grad_l1"] == 0.0 < bound["5"]["evidence_ingress_grad_l1"]
    assert c["gradcheck_executor_float64"]["ok_gradcheck"] and c["gradcheck_bridge_float64"]["ok_gradcheck"]
    assert "inject.weight" in c["gradcheck_bridge_float64"]["inputs"]
    graph = c["bounded_graph"]
    assert len(set(graph["graph_nodes_per_call"])) == 1 and graph["ok_working_state_holds_no_graph"]
    host = c["host_extraction"]
    assert host["calls_on_cognitive_path"] == []
    assert {"item", "tolist", "numpy", "__bool__"} <= set(host["detector_positive_control"])
    for module in ("kernels", "activation", "state", "plan", "binding"):
        assert not {"numpy", "ctypes"} & set(host["imports"][module]), module
    assert all(g[k] == "PASS" for k in ("gate_5", "gate_6", "gate_7", "gate_differentiable_path"))


def test_the_bridge_gates_on_actual_evidence_lm(records):
    r = records["bridge"]
    seen = {(v["variant"]["mode"], v["variant"]["fine_keys"], v["variant"]["ple"], v["variant"]["origin"],
             v["variant"]["window"]) for v in r["variants"]}
    matrix = {(m, f, p, o, False) for m in ("off", "learned") for f in ("reread", "stored") for p in (False, True)
              for o in (0, 64)}
    assert matrix <= seen and any(key[4] for key in seen) and len(seen) == len(r["variants"]) == 18
    for v in r["variants"]:
        name = v["variant"]
        assert v["disabled_path"]["equal_logits_every_chunk"] and v["disabled_path"]["equal_state_every_chunk"], name
        assert v["zero_gate"]["equal_refined_logits"], name
        clock = v["context_clock"]
        assert clock["equal_state_after_admit_reason_readout"] and clock["equal_state_after_refusals"], name
        assert clock["counter_deltas"] == {"blocks_reencoded": 2, "ring_rereads": 2}, name
        assert v["ring"]["ok_wrapped"] and v["ring"]["ok_evicted"], name
        admission = v["admission"]
        assert all(x["refused"] for x in admission.values() if isinstance(x, dict)), name
        assert admission["own_block"]["code"] == admission["future_block"]["code"] == "EVIDENCE_CAUSALITY", name
        reset = v["reset"]
        assert all(reset[k] for k in ("equal_fresh_session_logits", "equal_fresh_session_state",
                                      "equal_fresh_cognitive_latent", "equal_fresh_cognitive_logits",
                                      "ok_working_empty")), name
        assert reset["stale_handle"]["refused"] and reset["stale_address_of_the_previous_epoch"]["refused"], name
        assert v["training_mode"]["ok_backbone_without_grad"], name
        if name["fine_keys"] == "stored" and name["mode"] == "learned":
            assert v["training_mode"]["fine_tier_reads_total"] > 0, name   # R8's tier served the backbone only
        assert v["pass"] is True, name
    battery = r["refusals"]
    for name, x in battery.items():
        assert x.get("refused", True) and x.get("ok_code", True) and x.get("equal_unchanged", True), name
    for name, code in (("unauthorized_arc", "ARC_AUTHORITY"), ("foreign_handle", "FOREIGN_OR_STALE_HANDLE"),
                       ("capacity_overflow", "STATE_CAPACITY"), ("nonfinite_computation", "NONFINITE"),
                       ("rounds_over_cap", "ROUND_RANGE"), ("visit_budget", "VISIT_BUDGET")):
        assert battery[name]["code"] == code, name
    assert battery["nonfinite_computation"]["equal_committed_answer"]
    assert battery["before_readout"]["ok_bypass"] and battery["after_readout"]["ok_cognitive"]
    assert r["residency_bridge"] == {"equal_latent": True, "equal_logits": True}
    assert all(r["immutability"].values())
    assert set(r["gates"].values()) == {"PASS"}


def test_resources_are_labelled_measurements_and_bounded(records):
    r = records["resources"]
    assert r["label"].startswith("MEASURED")
    by = r["bytes"]
    assert len(set(by["gradient_saved_tensors_per_training_step"])) == 1 and by["ok_bounded_scratch"]
    projection = by["session_projection"]
    assert projection["total"] == projection["backbone_state"] + projection["working_state"] + \
        projection["kernel_parameters"]
    assert projection["working_state"] == by["working_state"]["total"]
    assert by["topology_index_payload_shared"] == projection["shared_index_payload"] == IDENTITY["index_payload_bytes"]
    assert r["plan"]["message_visits_per_reason"] == r["plan"]["rounds"] * r["plan"]["admitted_arcs"]
    assert r["parameters"]["kernels"] == sum(r["parameters"]["by_tensor"].values())
    assert len(r["parameters"]["by_tensor"]) == KERNEL_TENSORS
    assert len(set(r["graph_nodes_per_training_step"])) == 1


# -- the verdict ---------------------------------------------------------------------------------------------------

def test_the_dispositions_follow_from_the_rows(records):
    identity, audit, parity, gradients, bridge, resources, verdict = (records[name] for name in RECORDS)
    assert verdict["inputs"] == {f"{name}.json": _sha(RESULTS / f"{name}.json") for name in GATES}
    gates = bridge["gates"]

    def passed(*values) -> str:
        return "PASS" if all(v == "PASS" for v in values) else "FAIL"
    battery = bridge["refusals"]
    bounded = (all(battery[k]["refused"] and battery[k]["equal_unchanged"]
                   for k in ("rounds_over_cap", "visit_budget", "capacity_overflow", "too_much_evidence",
                             "too_many_tokens"))
               and len(set(gradients["checks"]["bounded_graph"]["graph_nodes_per_call"])) == 1
               and resources["bytes"]["ok_bounded_scratch"])
    expected = {
        "ELPIS_BASE_PIN": ELPIS_BASE,
        "SUBSTRATE_IDENTITY_PARITY": passed(identity["gate_1"], parity["gate_3"], parity["gate_4_executor"],
                                            gates["gate_4_bridge"]),
        "DIFFERENTIABLE_EXECUTION": passed(gradients["gate_5"], gradients["gate_6"], gradients["gate_7"],
                                           gradients["gate_differentiable_path"], parity["gate_7_executor"]),
        "DISABLED_PATH_PARITY": passed(gates["gate_2"]),
        "CONTEXT_STATE_COHERENCE": passed(audit["gate_seam"], gates["gate_11"], gates["gate_12"],
                                          gates["training_mode"]),
        "EVIDENCE_ADMISSION": passed(gates["evidence_admission"]),
        "SESSION_ISOLATION": passed(gates["gate_8"], gates["gate_9"], gates["gate_10"], gates["variants"]),
        "BOUNDED_COMPUTATION": "PASS" if bounded else "FAIL",
        "DISTRIBUTED_COMPETENCE": "HOLD",
        "LEARNED_ADDRESSING": "NOT_STARTED",
        "CGC_ADMISSION": "NOT_STARTED",
        "HCA88_ADMISSION": "NOT_ADMITTED",
        "COGNITIVE_ADVANTAGE": "NOT_CLAIMED",
    }
    dispositions = dict(verdict["dispositions"])
    pin = dispositions.pop("GEODOME_SOURCE_PIN")
    assert GEODOME_COMMIT in pin and GEODOME_MANIFEST in pin and "v1.0.0" in pin
    assert dispositions == expected
    assert all(dispositions[k] == "PASS" for k in ("SUBSTRATE_IDENTITY_PARITY", "DIFFERENTIABLE_EXECUTION",
                                                   "DISABLED_PATH_PARITY", "CONTEXT_STATE_COHERENCE",
                                                   "EVIDENCE_ADMISSION", "SESSION_ISOLATION", "BOUNDED_COMPUTATION"))
    for field in ("DISTRIBUTED_COMPETENCE", "LEARNED_ADDRESSING", "CGC_ADMISSION", "HCA88_ADMISSION",
                  "COGNITIVE_ADVANTAGE"):
        assert verdict["reasons"][field], field
