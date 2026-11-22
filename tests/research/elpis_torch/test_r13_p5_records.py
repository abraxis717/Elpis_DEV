"""Elpis Torch R13 P5 records (JSON only): the frozen plan, the preflight and the DEVELOPMENT-ONLY pilot.

These tests read JSON only (no PyTorch). They check that:
- the plan is frozen at its pinned digest and says what the preregistration says;
- the preflight and the pilot were produced under that plan by the committed sources on a clean revision;
- the pilot is labelled DEVELOPMENT ONLY, decides nothing, and records what the owner asked a pilot to record;
- no confirmation record exists yet (confirmation needs the owner's further authorization).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from .conftest import REPO

TORCH = REPO / "research" / "elpis_torch"
RESULTS = TORCH / "results" / "r13_p5"
PLAN = RESULTS / "preregistration.json"
PLAN_SHA256 = "094adb4500857103bbac6f7d215e1873c4167d2dbfb86daa1844a05999dcd432"
SOURCES = {"runner": TORCH / "experiments" / "run_r13_p5.py", "tasks": TORCH / "composition_tasks.py",
           **{name: TORCH / "geodome_cognitive" / f"{name}.py"
              for name in ("arms", "controls", "regions", "binding", "kernels", "activation", "plan", "state",
                           "topology", "source")},
           **{name: TORCH / f"{name}.py" for name in ("evidence_model", "evidence_ring", "ersa_model",
                                                     "sparse_channel", "model", "config", "lastblock")}}
ARMS = ("NATIVE", "GEODOME", "EQUAL_BUDGET", "COMPUTE_MATCHED", "FLAT_RECURRENT", "RELABEL_ISOMORPHIC", "REWIRED",
        "DIAGNOSTIC_ORACLE")


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_plan_is_frozen_at_its_pinned_digest_and_says_what_was_decided():
    assert _sha(PLAN) == PLAN_SHA256
    plan = _load(PLAN)
    assert plan["status"] == "FROZEN" and plan["arms"] == list(ARMS)
    assert plan["seeds"] == {"confirmation": [1301, 1302, 1303, 1304, 1305], "development": [9001]}
    assert plan["substrate"]["region_size"] == 48 and plan["substrate"]["rounds"] == 8
    assert plan["substrate"]["evidence_bindings"] == 8 and plan["substrate"]["training_regions"] == 3
    assert (plan["kernels"]["width"], plan["kernels"]["hidden"], plan["kernels"]["gate_init"]) == (32, 64, 0.0)
    assert plan["training"]["steps"] == 2000 and plan["training"]["batch"] == 1
    assert plan["pilot"]["max_updates_per_trainable_arm"] == 200 and plan["pilot"]["label"] == "DEVELOPMENT ONLY"
    assert plan["compute"] == {"device": "cpu", "threads": 2, "time_limit_minutes_per_arm_seed": 60,
                               "ram_limit_bytes": 5 * (1 << 30), "on_exhaustion": "stop and report; never retried "
                                                                                 "silently"}
    stats = plan["statistics"]
    assert (stats["alpha"], stats["margin"], stats["positive_seeds"], stats["recall_threshold"]) == (0.05, 0.05, 4, 0.9)
    assert stats["bootstrap"]["draws"] == 10000 and stats["bootstrap"]["kind"] == "hierarchical, seed-aware"
    assert [r.split(":")[0] for r in plan["readings"]] == ["GEOMETRY_SPECIFIC_GAIN",
                                                          "DISTRIBUTED_COMPETENCE_ESTABLISHED",
                                                          "SUBSTRATE_AGNOSTIC_GAIN", "RETRIEVAL_LIMITED",
                                                          "NOT_ESTABLISHED"]
    assert "never a hop-2 or hop-3 fact" in plan["backbone"]["training"]["evidence_label"]
    doc = (REPO / "docs" / "research" / "elpis_torch" / "PREREGISTRATION_R13_P5.md").read_text(encoding="utf-8")
    assert PLAN_SHA256 in doc and "FROZEN" in doc


def test_no_confirmation_record_exists_without_authorization():
    assert not list(RESULTS.glob("confirm_*.json")) and not (RESULTS / "study.json").exists()


def _records():
    return {p.name: _load(p) for p in sorted(RESULTS.glob("*.json")) if p.name not in ("preregistration.json",
                                                                                         "MANIFEST.json")}


def test_every_record_was_produced_under_the_frozen_plan_by_the_committed_sources():
    records = _records()
    if not records:
        pytest.skip("no P5 record yet")
    expected = {f"{k}_sha256": _sha(v) for k, v in SOURCES.items()}
    for name, record in records.items():
        assert record["plan_sha256"] == PLAN_SHA256, name
        provenance = record["provenance"]
        assert {k: provenance[k] for k in expected} == expected, name
        assert provenance["plan_sha256"] == PLAN_SHA256, name
        assert provenance["r13_supplement_manifest_sha256"] == _sha(TORCH / "results" / "r13_hardening" /
                                                                    "MANIFEST.json"), name
        assert provenance["git_revision"] and not provenance["git_revision"].endswith("+dirty"), name


def test_the_pilot_is_development_only_and_records_what_was_asked():
    path = RESULTS / "dev_pilot.json"
    if not path.is_file():
        pytest.skip("no pilot record yet")
    pilot = _load(path)
    assert pilot["label"] == "DEVELOPMENT ONLY" and pilot["seed"] == 9001
    assert pilot["decides"] == "nothing (DEVELOPMENT ONLY)"
    if pilot["status"] != "COMPLETED":
        assert pilot["status"].startswith(("TIME_LIMIT", "RAM_LIMIT"))
        return
    assert set(pilot["arms"]) == set(ARMS[1:])
    for name, arm in pilot["arms"].items():
        training = arm["training"]
        assert training["steps"] <= 200, name
        first = training["log"][0]
        assert first["step"] == 0 and first["gate"] == 0.0 and first["grad_norm"]["gate"] > 0, name
        later = training["log"][1]
        assert later["gate"] != 0.0, name
        groups = ("ingress", "readout", "inject") if name == "EQUAL_BUDGET" else \
            ("ingress", "message", "update", "readout", "inject")
        assert all(later["grad_norm"][g] > 0 for g in groups), name           # the gate opened; gradients flow
        assert training["loss_first_fifth"] is not None and training["loss_last_fifth"] is not None
        for key in ("primary", "iid", "heldout_region"):
            assert 0.0 <= arm[key]["accuracy"] <= 1.0
    recall = pilot["recall"]["primary"]
    assert 0.0 <= recall["complete_set"] <= 1.0 and set(recall["complete_set_by_hops"]) == {"2", "3"}
    assert pilot["interventions"]["disabled_injection"]["bit_exact_native"]
    assert pilot["backbone"]["ok_unchanged_after_every_arm"]
    assert pilot["peak_rss_bytes"] < 5 * (1 << 30)
    projection = pilot["confirmation_projection"]
    assert projection["seconds_per_seed"] > 0 and set(projection["per_arm"]) == set(ARMS[1:])


def test_the_preflight_said_go_before_the_pilot():
    path = RESULTS / "preflight.json"
    if not path.is_file():
        pytest.skip("no preflight record yet")
    preflight = _load(path)
    assert preflight["go"] is True and all(preflight["checks"].values())
    assert preflight["confirmation"]["projected_peak_rss_bytes"] < 5 * (1 << 30)


def test_the_manifest_lists_every_p5_record():
    path = RESULTS / "MANIFEST.json"
    if not path.is_file():
        pytest.skip("no manifest yet")
    manifest = _load(path)
    files = {p.name for p in RESULTS.glob("*.json") if p.name != "MANIFEST.json"}
    assert set(manifest["files"]) == files
    for name, digest in manifest["files"].items():
        assert _sha(RESULTS / name) == digest, name
