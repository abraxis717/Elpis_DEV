"""The K1N-v2 plan is write-once and bound to the frozen R3 authority and the unchanged K1N-v1 record."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "research" / "ecs_k1_native"
PINS = {"specs/ecsg-k1-native.v2.plan.json": "6720e607d887f1a49545e3e5a6cfdc8b641991795a990f8fdd2157b0b86ea7eb", "PLAN_V2.md": "02475255b0d26e3ba3e7699310c7c6a25a7f467068a0e28f3519daabc162352d"}


def test_v2_plan_files_are_write_once():
    for name, sha in PINS.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name


def test_v2_plan_is_bound_to_r3_and_keeps_v1_not_qualified():
    plan = json.loads((ROOT / "specs" / "ecsg-k1-native.v2.plan.json").read_text(encoding="utf-8"))
    for key in ("r3_spec", "r3_frozen", "r3_qual"):
        entry = plan["authority"][key]
        assert hashlib.sha256((REPO / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], key
    v1 = plan["authority"]["k1n_v1"]
    assert hashlib.sha256((REPO / v1["record"]).read_bytes()).hexdigest() == v1["record_sha256"]
    assert json.loads((REPO / v1["record"]).read_bytes())["body"]["verdict"] == "NOT_QUALIFIED"
    fresh = plan["world_sets"]["F"]["ids"]
    assert len(fresh) == 32 and not any(w.startswith(("r3dev-", "r3qual-", "test-")) for w in fresh)
    assert plan["bounded_numerics"]["L1_local_horizon"]["rule"].endswith("<= 1e-10")
