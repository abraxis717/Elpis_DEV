"""R118 closure verification: independently re-check a checkout of the published R118 component.

    python -B research/yang_mills_hca/r118_scalable_awr/evidence/verify_closure.py \
        [--repo CHECKOUT] [--torch] [--tests] [--out PATH]

Lives in ``evidence/`` on purpose: the declared R118 component digest covers every
.py/.c/.json file of the component *outside* ``evidence/``, so this verifier (and its
output record) cannot perturb the digest it verifies. Checks, all against values
declared in R118 evidence at publication commit f901cf0:

1. frozen manifest, ledger, snapshot digest, six checkpoint pins (stdlib);
2. component source digest, performance protocol hash, native kernel source hash;
3. evidence index hashes, and that A/C/E evidence records carry the same digest;
4. tracked-file inventory against the repository boundary policy;
5. with ``--torch``: historical R78 run.sh protocol (byte-identical archive), full R77
   evaluation replay (576 rows bitwise), six models through the historical r78 path;
6. with ``--tests``: the R118 unittest suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

DECLARED = {
    "publication_commit": "f901cf0e4db7d93945b590d1b19262e3af37aeff",
    "component_digest": "770411b1f6788f9593f5f7adf1d4f6efb57a564f8c207d13dd7f55352af0df71",
    "protocol_sha256": "7ac1a0f7377772af1e9a0a891dfdbe01104d6621e10ca8dff1310fa476710fbc",
    "kernel_source_sha256": "87b53392c475fe2bb15290a52f73c9dfb6acb04294d2c22a859e5fc2d0e31e54",
    "frozen_manifest_sha256": "b0415964e83c2dba01d2a1b6bd9f2f55dad53f147a3747119433a3e07524d855",
    "frozen_snapshot_digest": "2776a730b0ec3ce0f9af22cb2cf5e7c7bc90d91a32d064cd2bc774f6dc9c24ef",
    "r78_archive_sha256": "7646e113fa8dcef48ff2b0ae724504ddc745d68fe7a8fd6699cb73ef662ab40e",
    "r77_rows_canonical_sha256": "051fa3bc48eba4852749c6f8de6eb34a9edd46a33973c33a8f7201563e12d38b",
}


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parents[4]))
    ap.add_argument("--torch", action="store_true")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    repo = pathlib.Path(args.repo).resolve()
    comp = repo / "research/yang_mills_hca/r118_scalable_awr"
    ev = comp / "evidence"
    sys.path.insert(0, str(repo))
    t0 = time.time()
    from research.yang_mills_hca.r118_scalable_awr import provenance as P

    if pathlib.Path(P.__file__).resolve().parent != comp:
        raise SystemExit("IMPORTED_WRONG_CHECKOUT")
    rec = {"schema": "elpis.ym_hca.r118.closure.v1", "declared": DECLARED, "checks": {}}
    chk = rec["checks"]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    rec["checkout_head"] = head.stdout.strip() if head.returncode == 0 else "NOT_A_GIT_CHECKOUT"
    anc = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", DECLARED["publication_commit"], "HEAD"])
    chk["publication_commit_in_history"] = anc.returncode == 0
    # 1. frozen sources and checkpoints (stdlib)
    man = P.verify_manifest()
    chk["frozen_manifest_files"] = man["files_verified"] == 39 and man["manifest_sha256"] == DECLARED["frozen_manifest_sha256"]
    chk["frozen_ledger_entries"] = man["ledger_entries_verified"] == 26
    chk["frozen_snapshot_digest"] = P.snapshot_digest() == DECLARED["frozen_snapshot_digest"]
    pins = {k: v["sha256"] for k, v in P.checkpoint_records().items()}
    chk["checkpoint_pins_6_of_6"] = pins == P.CHECKPOINT_SHA256 and len(pins) == 6
    rec["checkpoint_sha256"] = pins
    # 2. declared source, protocol and kernel hashes
    digest = P.component_digest()
    rec["component_digest"] = digest
    rec["component_sources"] = P.component_sources()
    chk["component_digest"] = digest == DECLARED["component_digest"]
    chk["protocol_sha256"] = sha(comp / "PROTOCOL_R118D.json") == DECLARED["protocol_sha256"]
    chk["kernel_source_sha256"] = sha(comp / "csrc/awr_kernel.c") == DECLARED["kernel_source_sha256"]
    # 3. evidence
    index = json.loads((ev / "INDEX.json").read_text())
    chk["evidence_index_hashes"] = all(sha(ev / n) == r["sha256"] for n, r in index["files"].items())
    chk["evidence_protocol_hash"] = index["performance_protocol_sha256"] == DECLARED["protocol_sha256"]
    digests = {n: json.loads((ev / n).read_text()).get("component_digest") for n in ("r118a.json", "r118c.json", "r118e.json")}
    chk["evidence_component_digests"] = set(digests.values()) == {DECLARED["component_digest"]}
    d_runs = {n: json.loads((ev / n).read_text())["protocol_sha256"] for n in ("r118d_run1.json", "r118d_run2.json")}
    chk["evidence_d_runs_protocol"] = set(d_runs.values()) == {DECLARED["protocol_sha256"]}
    c_ev = json.loads((ev / "r118c.json").read_text())
    chk["evidence_kernel_hash"] = c_ev["native_kernel"]["source_sha256"] == DECLARED["kernel_source_sha256"]
    a_ev = json.loads((ev / "r118a.json").read_text())
    chk["evidence_gates"] = (a_ev["gate"] == "R118A_PASS" and c_ev["gate"] == "R118C_PASS"
                             and index["r93_gate"] == "ENERGY_NORMALIZED_DENOMINATOR_HOLD")
    # 4. tracked inventory vs boundary policy
    ls = subprocess.run(["git", "-C", str(repo), "ls-files", "research/yang_mills_hca"], capture_output=True, text=True)
    files = ls.stdout.split()
    policy = json.loads((repo / "ELPIS_SYSTEM.json").read_text())["boundary_policy"]
    pats = [re.compile(p) for p in policy["forbidden_directory_name_patterns"]]
    bad = [f for f in files if f.endswith(tuple(policy["forbidden_file_suffixes"]))
           or (repo / f).stat().st_size > 2 * 1024 * 1024
           or any(p.match(part) for part in pathlib.PurePosixPath(f).parts[:-1] for p in pats)]
    chk["boundary_policy_inventory"] = not bad and len(files) > 0
    rec["tracked_r118_files"] = {f: sha(repo / f) for f in files if "/r118_scalable_awr/" in f}
    rec["boundary_violations"] = bad
    # 5. historical parity (torch)
    if args.torch:
        from research.yang_mills_hca.r118_scalable_awr import frozen_oracle, historical_replay

        rep = historical_replay.replay(("verify", "spot", "run"))
        chk["r78_archive_byte_identical"] = (rep["disposition"] == "HISTORICAL_R78_REPLAY_PASS"
                                             and rep["modes"]["run"]["replayed_sha256"] == DECLARED["r78_archive_sha256"])
        o = frozen_oracle.get_oracle()
        models = o.models()
        chk["six_models_historical_path"] = o.parameter_pins() == P.CHECKPOINT_SHA256
        rows = o.r77.evaluate(models)
        canon = hashlib.sha256(P.canonical_json(rows)).hexdigest()
        chk["r77_576_rows_bitwise"] = len(rows) == 576 and canon == DECLARED["r77_rows_canonical_sha256"]
        rec["torch"] = o.torch.__version__
    # 6. tests
    if args.tests:
        t = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s",
                            "research/yang_mills_hca/r118_scalable_awr/tests", "-t", "."],
                           cwd=repo, capture_output=True, text=True)
        tail = [line for line in t.stderr.strip().splitlines() if line.startswith(("Ran ", "OK", "FAILED"))]
        rec["tests"] = {"exit": t.returncode, "summary": tail}
        chk["r118_test_suite"] = t.returncode == 0
    chk["frozen_tree_unchanged_after"] = P.verify_manifest()["files_verified"] == 39
    rec["verifier_sha256"] = sha(pathlib.Path(__file__).resolve())
    rec["seconds"] = round(time.time() - t0, 1)
    rec["disposition"] = "R118_CLOSURE_PASS" if all(chk.values()) else "R118_CLOSURE_FAIL"
    blob = json.dumps(rec, indent=1, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(blob + "\n")
    print(blob)
    return 0 if rec["disposition"] == "R118_CLOSURE_PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
