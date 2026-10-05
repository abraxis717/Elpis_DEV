"""Write-once K1N-v2 qualification. RESEARCH_ONLY; PLAN_V2.md is immutable.

    python -m research.ecs_k1_native.run_v2 --library <libelpis_ecsg_math.so>

No F preview/check mode. Tests use synthetic or disclosed Q worlds. The record is
exclusively reserved before science, including on interruption or failure.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from elpis.ECS_G.k1 import K1Library, K1State
from research.ecs_retention_r3 import engine as E
from research.ecs_retention_r3 import experiment as X
from research.ecs_retention_r3 import numerics
from research.ecs_retention_r3 import protocol as P3
from research.ecs_retention_r3 import run as R3
from . import differential as D
from . import regression as G
from . import run as V1
from . import samehost as S

ROOT, REPO = V1.ROOT, V1.REPO
PLAN_FILE = ROOT / "specs/ecsg-k1-native.v2.plan.json"
PLAN_MD = ROOT / "PLAN_V2.md"
RECORD = ROOT / "evidence/ecsg-k1-native.v2.qualification.json"
PLAN_SHA = "6720e607d887f1a49545e3e5a6cfdc8b641991795a990f8fdd2157b0b86ea7eb"
PLAN_MD_SHA = "02475255b0d26e3ba3e7699310c7c6a25a7f467068a0e28f3519daabc162352d"
PLAN_COMMIT = "ed9701b2"
GATE_NAMES = ("E1", "E2", "E3", "E4", "E5", "L1", "L2", "D1_Q", "D1_F")
NATIVE_TESTS = {"ECS_G.test_ecsg_k1" + suffix for suffix in
                ("", "_fms", "_alloc", "_fms_alloc", "_fms_faults", "_performance")}
E1_CHECKS = {"k1_disabled_equals_samehost_g1", "w_a_equals_samehost_canonical",
             "query_equals_runtime_r1_forward", "epoch_and_provenance", "envelope_round_trip",
             "resident_equals_standalone", "reset_mechanics", "w_only_semantics",
             "transplant_in_process", "state_semantics", "nothing_refused"}
_STATE = {}


def authority() -> tuple[dict, dict]:
    if P3.sha256_file(PLAN_FILE) != PLAN_SHA or P3.sha256_file(PLAN_MD) != PLAN_MD_SHA:
        raise V1.QualificationError("v2 plan bytes changed")
    p = json.loads(PLAN_FILE.read_bytes())
    for key in ("r3_spec", "r3_frozen", "r3_qual"):
        pin = p["authority"][key]
        if P3.sha256_file(REPO / pin["path"]) != pin["sha256"]:
            raise V1.QualificationError(f"authority mismatch: {key}")
    for key, kind in (("r3_frozen", "frozen"), ("r3_qual", "qual")):
        pin = p["authority"][key]
        if P3.load(REPO / pin["path"], kind)["digest"] != pin["digest"]:
            raise V1.QualificationError(f"internal authority mismatch: {key}")
    v1 = p["authority"]["k1n_v1"]
    if P3.sha256_file(REPO / v1["record"]) != v1["record_sha256"]:
        raise V1.QualificationError("v1 record changed")
    if V1.load_record()["body"]["verdict"] != "NOT_QUALIFIED":
        raise V1.QualificationError("v1 verdict changed")
    qual = G.r3_qual()
    if qual["outcome"] != "OUTCOME_A":
        raise V1.QualificationError("R3 outcome changed")
    return p, qual


def decode(nat, lab, envelope: bytes):
    """Decode through native restore/getters; binary64 values are copied exactly."""
    with K1State.restore(nat.k1, envelope, max_rows=lab.rows) as state:
        m = E.K1(lab.idx, provenance=state.provenance)
        m.H = E.unpack_upper(np.asarray(state.h_packed()), lab.idx.size)
        m.a = np.asarray(state.a()).copy()
        return m, np.asarray(state.w()).reshape(lab.dim, lab.width).copy(), state.epoch


def bounded(nat, lab, world, main) -> dict:
    """L1 from complete A/B/C boundaries; L2 uses previous native H and current W.

    Learning leaves H unchanged, so previous consolidated H is pre-consolidation H
    (zero before A). Consolidation leaves W unchanged. Save both operands.
    """
    local, consolidation = {}, {}
    previous_h = np.zeros((lab.idx.size, lab.idx.size))
    for i, t in enumerate(S.STAGES):
        stage = main["stages"][t]
        m, W, epoch = decode(nat, lab, stage["envelope"])
        ref = E.K1(lab.idx)
        ref.H = previous_h.copy()
        ref.consolidate(W, world.tasks[t].x_train)
        consolidation[t] = {"previous_native_H": E.pack_upper(previous_h).tolist(),
                            "native_H": E.pack_upper(m.H).tolist(), "lab_H": E.pack_upper(ref.H).tolist(),
                            "native_a": m.a.tolist(), "lab_a": ref.a.tolist(),
                            "h_relative": D._rel(m.H, ref.H), "a_relative": D._rel(m.a, ref.a)}
        previous_h = m.H.copy()
        if i < 3:
            exp = world.tasks[S.STAGES[i + 1]]
            W_lab, refused, _ = E.engine_learn(lab.api, W, exp.x_train, exp.y_train, lab.rate, 16, m)
            with K1State.restore(nat.k1, stage["envelope"], max_rows=lab.rows) as state:
                state.learn(D._mv(exp.x_train), D._mv(exp.y_train), lab.rate, 16)
                W_native = np.asarray(state.w()).reshape(W.shape)
                local[t] = {"next": S.STAGES[i + 1], "steps": 16, "source_epoch": epoch,
                            "native_epoch": state.epoch, "lab_refused": refused,
                            "native_W": W_native.tolist(), "lab_W": W_lab.tolist(),
                            "w_relative": D._rel(W_native, W_lab)}
    return {"L1": local, "L2": consolidation}


def structural(library, lab, world) -> dict:
    # A separate bound namespace: counting cannot leak into scientific runs.
    k1 = K1Library(ctypes.CDLL(str(library.with_name("libelpis_ecsg_k1.so"))))
    calls, operations = {}, {}
    def wrap(name, fn):
        def call(*args):
            calls[name] = calls.get(name, 0) + 1
            return fn(*args)
        return call
    for name, fn in vars(k1._k).copy().items():
        setattr(k1._k, name, wrap(name, fn))
    exp = world.tasks["A"]
    x, y = D._mv(exp.x_train), D._mv(exp.y_train)
    with K1State.create(k1, *world.w0.shape, D._mv(world.w0), max_rows=lab.rows) as state:
        before = state.stats()["heap_allocations"]
        def measure(name, op, symbol):
            calls.clear()
            result = op()
            operations[name] = {"calls": dict(calls), "expected": {symbol: 1}}
            return result
        for steps in (1, 10, 4000):
            measure(f"learn_{steps}", lambda: state.learn(x, y, lab.rate, steps), "learn")
        measure("query", lambda: state.query(x), "forward")
        measure("consolidate", lambda: state.consolidate(x), "consolidate")
        txn = measure("txn_begin", state.transaction, "txn_begin")
        for steps in (1, 10, 4000):
            measure(f"txn_learn_{steps}", lambda: txn.learn(x, y, lab.rate, steps), "txn_learn")
        measure("txn_query", lambda: txn.query(x), "txn_forward")
        measure("txn_epoch", txn.epoch, "txn_epoch")
        measure("txn_consolidate", lambda: txn.consolidate(x), "txn_consolidate")
        measure("txn_commit", txn.commit, "txn_commit")
        txn = measure("txn_begin_abort", state.transaction, "txn_begin")
        measure("txn_abort", txn.abort, "txn_abort")
        after = state.stats()["heap_allocations"]
    return {"world": world.world, "operations": operations,
            "heap_allocations_before": before, "heap_allocations_after": after}


def _init(library: str):
    nat = D.Native(library)
    _STATE.update(nat=nat, lab=X.Lab(nat.api, R3.SPEC), qual=G.r3_qual())


def world_evidence(nat, lab, wid, reference):
    native = S.world_samehost(nat, lab, wid, reference)
    native["bounded"] = bounded(nat, lab, lab.world(wid), native.pop("_main"))
    native["_b_envelope"] = native["_b_envelope"].hex()
    native["descriptive_nmse_difference"] = {
        t: {s: native["rows"]["K1"]["stages"][t]["nmse"][s]
               - reference["P"]["mechanisms"]["K1"]["stages"][t]["nmse"][s] for s in S.STAGES}
        for t in S.STAGES}
    return P3.plain(native)


def _job(job):
    group, wid = job
    try:
        lab, nat = _STATE["lab"], _STATE["nat"]
        reference = _STATE["qual"]["per_world"][wid] if group == "Q" else X.qual_world(lab, wid)
        return {"native": world_evidence(nat, lab, wid, reference),
                "reference": P3.plain(reference) if group == "F" else None}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def native_determinism(library, wid, reference):
    nat = D.Native(str(library))
    lab = X.Lab(nat.api, R3.SPEC)
    runs = [{t: D._sha(s["envelope"]) for t, s in
             D.standalone_run(nat, lab.world(wid), lab.rate, lab.steps)["stages"].items()} for _ in range(2)]
    return {"world": wid, "runs": runs, "reference": reference,
            "identical": all(r == reference for r in runs)}


def laboratory_determinism(library, wid, reference):
    nat = D.Native(str(library))
    runs = [X.determinism_digest(lambda: X.Lab(nat.api, R3.SPEC), wid) for _ in range(2)]
    recorded = {"stages": {t: {k: reference["P"]["mechanisms"]["K1"]["stages"][t][k]
                              for k in ("W_digest", "state_digest", "nmse")} for t in S.STAGES}}
    return {"world": wid, "runs": runs, "reference": recorded, "identical": all(r == recorded for r in runs)}


def native_suite(build: Path) -> dict:
    """Save test names, exit status and individual outputs; refuse an empty suite."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "ctest.xml"
        out = subprocess.run(["ctest", "--test-dir", str(build), "-R", r"^ECS_G\.test_ecsg_k1",
                              "--output-on-failure", "--output-junit", str(report)],
                             capture_output=True, text=True)
        def portable(text):
            return text.replace(str(build), "<build>").replace(str(REPO), "<repo>").replace(tmp, "<temporary>")
        cases = []
        if report.exists():
            for case in ET.parse(report).getroot().iter("testcase"):
                cases.append({"name": case.get("name"), "status": case.get("status"),
                              "failed": case.find("failure") is not None,
                              "skipped": case.find("skipped") is not None,
                              "output": portable(case.findtext("system-out", ""))})
        return {"returncode": out.returncode, "tests": cases, "stdout": portable(out.stdout),
                "stderr": portable(out.stderr)}


def decision(reference, native, det, transplant, *, qual=None, lab_det=None):
    rec = S.decision_record(reference, native, R3.SPEC, det["identical"], transplant["bitwise"])
    if qual is not None:
        expected = qual["pre_robustness_decision_record"]
        gates = dict(rec["gates"], L_numerical_robustness=qual["gates"]["L_numerical_robustness"])
        sub = rec["substituted"]
        final = X.disposition(rec["validity"], rec["mechanics"], gates,
                              *[[sub[w]["P"]["mechanisms"][k] for w in sub] for k in ("K1", "M0", "M1")],
                              R3.SPEC["thresholds"])
    else:
        lab_transplant = all(r["P"]["causality"]["full_state_transplant_at_B"]["continuation_bitwise"]
                             for r in reference.values())
        expected = X.gates(reference, R3.SPEC, lab_det, {"bitwise": lab_transplant})["decision_record"]
        final = None
    actual, expected = P3.plain(rec["decision_record"]), P3.plain(expected)
    return {"native": actual, "reference": expected,
            "equal": {k: actual.get(k) == expected.get(k) for k in sorted(set(actual) | set(expected))},
            "final_with_recorded_L": final}


def structural_pass(st):
    expected = {**{f"learn_{k}": "learn" for k in (1, 10, 4000)},
                **{f"txn_learn_{k}": "txn_learn" for k in (1, 10, 4000)},
                "query": "forward", "consolidate": "consolidate", "txn_begin": "txn_begin",
                "txn_query": "txn_forward", "txn_epoch": "txn_epoch", "txn_consolidate": "txn_consolidate",
                "txn_commit": "txn_commit", "txn_begin_abort": "txn_begin", "txn_abort": "txn_abort"}
    return set(st["operations"]) == set(expected) and all(
        st["operations"][k]["calls"] == {symbol: 1} for k, symbol in expected.items()) and (
        st["heap_allocations_before"] == st["heap_allocations_after"])


def exact_checks(rec):
    """Recompute E1 using saved operands, counters and the causal continuation rows."""
    e, rows = rec["evidence"], rec["rows"]
    full, reset, imported = e["full_B"], e["reset_B"], e["w_only_B"]
    sha = rec["envelopes_sha256"]
    return {
        "k1_disabled_equals_samehost_g1": all(
            e["zero"][t]["W_digest"] == e["canonical"][t]["W_digest"]
            == rows["controls"]["M0"]["stages"][t]["W_digest"]
            and not e["canonical"][t]["refused"] and e["zero"][t]["stats"]["corrected_steps"] == 0
            for t in S.STAGES),
        "w_a_equals_samehost_canonical": rows["K1"]["W_A_digest"] == e["canonical"]["A"]["W_digest"],
        "query_equals_runtime_r1_forward": all(
            b["native"] == b["canonical"] for run in e["queries"].values() for b in run.values()),
        "epoch_and_provenance": all(e["boundaries"][t]["epoch"] == 4000 * (i + 1)
                                    and e["boundaries"][t]["provenance"] == "COMPLETE"
                                    for i, t in enumerate(S.STAGES)),
        "envelope_round_trip": all(e["boundaries"][t]["restored_sha256"] == sha[t] for t in S.STAGES),
        "resident_equals_standalone": all(e["boundaries"][t]["resident_sha256"] == sha[t] for t in S.STAGES),
        "reset_mechanics": reset["W"] == full["W"] and reset["epoch"] == full["epoch"]
        and not any(reset["H"]) and not any(reset["a"]) and reset["provenance"] == "RESET"
        and e["reset_corrected_steps"] == 0 and set(rows["reset_challenge"]["stages"]) == {"C"},
        "w_only_semantics": imported["provenance"] == "UNCONSOLIDATED_IMPORT" and not any(imported["H"])
        and not any(imported["a"]) and imported["W"] == full["W"]
        and imported["envelope_sha256"] != full["envelope_sha256"] and rows["w_only"]["reproduces_reset_at_C"],
        "transplant_in_process": rows["transplant"]["restored_digest_equal"]
        and rows["transplant"]["digests"] == {t: sha[t] for t in ("C", "D")},
        "state_semantics": full["query"] == reset["query"],
        "nothing_refused": all(not e["canonical"][t]["refused"] for t in S.STAGES),
    }


def local_pass(bounded):
    return set(bounded["L1"]) == {"A", "B", "C"} and all(
        b["steps"] == 16 and not b["lab_refused"] and b["native_epoch"] == b["source_epoch"] + 16
        and D._rel(b["native_W"], b["lab_W"]) <= 1e-10 for b in bounded["L1"].values())


def consolidation_pass(bounded):
    return set(bounded["L2"]) == set(S.STAGES) and all(
        D._rel(b["native_H"], b["lab_H"]) <= 1e-12 and D._rel(b["native_a"], b["lab_a"]) <= 1e-12
        for b in bounded["L2"].values())


def recompute_gates(body) -> dict:
    """Derive every gate from saved observations and R3 rows, without rerunning science."""
    gates = dict.fromkeys(GATE_NAMES, False)
    sets = body.get("sets", {})
    if set(sets) != {"Q", "F"} or body.get("errors"):
        return gates
    p, qual = authority()
    expected_ids = {"Q": qual["world_ids"], "F": p["world_sets"]["F"]["ids"]}
    if body["world_ids"] != expected_ids or any(
            g["world_ids"] != expected_ids[name] or set(g["per_world"]) != set(expected_ids[name])
            for name, g in sets.items()):
        return gates
    worlds = [r for group in sets.values() for r in group["per_world"].values()]
    gates["E1"] = all(set(r["checks"]) == E1_CHECKS and exact_checks(r) == r["checks"]
                     and all(r["checks"].values()) for r in worlds)
    gates["E2"] = all(all(r == g["determinism"]["reference"] for r in g["determinism"]["runs"])
                       and len(g["determinism"]["runs"]) == 2 for g in sets.values())
    gates["E3"] = all(g["clean_transplant"]["child"] == g["clean_transplant"]["reference"]
                       and set(g["clean_transplant"]["child"]) == set(g["world_ids"])
                       and g["clean_transplant"]["bitwise"] and not g["clean_transplant"]["failed"]
                       and g["clean_transplant"]["worlds"] == len(g["world_ids"]) for g in sets.values())
    suite = body["native_suite"]
    gates["E4"] = suite["returncode"] == 0 and {t["name"] for t in suite["tests"]} == NATIVE_TESTS and all(
        t["status"] == "run" and not t["failed"] and not t["skipped"] for t in suite["tests"])
    gates["E5"] = structural_pass(body["structural"])
    gates["L1"] = all(local_pass(r["bounded"]) for r in worlds)
    gates["L2"] = all(consolidation_pass(r["bounded"]) for r in worlds)
    for name, group in sets.items():
        reference = qual["per_world"] if name == "Q" else group["laboratory"]
        rec = decision(reference, group["per_world"], group["determinism"], group["clean_transplant"],
                       qual=qual if name == "Q" else None, lab_det=group.get("laboratory_determinism"))
        gates[f"D1_{name}"] = rec["native"] == rec["reference"]
        if name == "Q":
            gates["D1_Q"] &= rec["native"]["outcome"] == "OUTCOME_A" and (
                rec["final_with_recorded_L"]["outcome"] == "OUTCOME_A")
        else:
            det = group["laboratory_determinism"]
            gates["D1_F"] &= len(det["runs"]) == 2 and all(r == det["reference"] for r in det["runs"]) and all(
                r["P"]["causality"]["full_state_transplant_at_B"]["continuation_bitwise"]
                for r in group["laboratory"].values())
    return gates


def implementation(library):
    if V1._git("status", "--porcelain", "--untracked-files=all") != "":
        raise V1.QualificationError("qualification requires the entire tree clean")
    if V1._git("branch", "--show-current") != "corrective/k1n-v2-closure-r0":
        raise V1.QualificationError("wrong qualification branch")
    subprocess.run(["git", "merge-base", "--is-ancestor", PLAN_COMMIT, "HEAD"], cwd=REPO, check=True)
    if "OPENBLAS_CORETYPE" in os.environ:
        raise V1.QualificationError("forced kernel forbidden")
    if any(os.environ.get(v) != "1" for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        raise V1.QualificationError("all registered thread constraints must be 1")
    P3.require_single_thread()
    build = next(p for p in library.parents if (p / "CMakeCache.txt").is_file())
    cache = (build / "CMakeCache.txt").read_text()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={REPO}" not in cache:
        raise V1.QualificationError("native build must belong to this checkout")
    files = V1._git("ls-files", "src", "native", "research", "tests", "CMakeLists.txt").splitlines()
    sources = {f: P3.sha256_file(REPO / f) for f in files if "/evidence/" not in f and "/frozen/" not in f}
    out = {"base_commit": V1._git("rev-parse", "HEAD"), "tree": V1._git("rev-parse", "HEAD^{tree}"),
           "dirty": False, "plan_sha256": PLAN_SHA, "plan_md_sha256": PLAN_MD_SHA,
           "files": sources, "source_digest": V1.digest("source", sources),
           "libraries": {p.name: P3.sha256_file(p) for p in sorted(library.parent.glob("libelpis*.so"))},
           "native_test_binaries": {p.name: P3.sha256_file(p) for p in sorted(build.rglob("test_ecsg_k1*"))
                                    if p.is_file() and os.access(p, os.X_OK)},
           "build": P3._build(library), "numerical_profile": numerics.profile()}
    return out, build


def qualify(library):
    if RECORD.exists():
        raise V1.QualificationError("v2 already started; no retry is permitted")
    p, qual = authority()
    impl, build = implementation(library)
    body = {"experiment": "ecsg-k1-native.v2", "plan_digest": V1.digest("plan", p),
            "r3_qual_digest": p["authority"]["r3_qual"]["digest"], "authority": p["authority"],
            "k1n_v1_plan_sha256": P3.sha256_file(REPO / p["authority"]["k1n_v1"]["plan"]),
            "implementation": impl, "world_ids": {"Q": qual["world_ids"], "F": p["world_sets"]["F"]["ids"]},
            "sets": {}, "errors": [], "verdict": "NOT_QUALIFIED", "run_complete": False}
    def write(fh):
        record = {"body": P3.plain(body), "digest": V1.digest("qualification", body)}
        fh.seek(0)
        fh.write(P3.canonical_json(record) + b"\n")
        fh.truncate()
        fh.flush()
        os.fsync(fh.fileno())
    # Durable reservation before the first F construction, even on interruption.
    with RECORD.open("x+b") as fh:
        write(fh)
        try:
            body["native_suite"] = native_suite(build)
            for group, ids in body["world_ids"].items():
                print(f"Evaluating {group}: {len(ids)} registered worlds", file=sys.stderr, flush=True)
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=G.WORKERS, mp_context=ctx, initializer=_init,
                                         initargs=(str(library),)) as pool:
                    rows = dict(zip(ids, pool.map(_job, [(group, w) for w in ids])))
                errors = {w: r["error"] for w, r in rows.items() if "error" in r}
                if errors:
                    body["errors"].append({group: errors})
                    body["sets"][group] = {"world_ids": ids, "observations": rows}
                    continue
                native = {w: r["native"] for w, r in rows.items()}
                det = native_determinism(library, ids[0], native[ids[0]]["envelopes_sha256"])
                transplant = G.clean_transplant(library, native)
                rec = {"world_ids": ids, "per_world": native, "determinism": det, "clean_transplant": transplant}
                body["sets"][group] = rec
                if group == "F":
                    rec["laboratory"] = {w: r["reference"] for w, r in rows.items()}
                    rec["laboratory_determinism"] = laboratory_determinism(library, ids[0], rows[ids[0]]["reference"])
                    nat = D.Native(str(library))
                    lab = X.Lab(nat.api, R3.SPEC)
                    body["structural"] = structural(library, lab, lab.world(ids[0]))
                rec["decision"] = decision(qual["per_world"] if group == "Q" else rec["laboratory"], native,
                                           det, transplant, qual=qual if group == "Q" else None,
                                           lab_det=rec.get("laboratory_determinism"))
                write(fh)
            body["gates"] = recompute_gates(body)
            body["run_complete"] = True
        except BaseException as exc:
            body["errors"].append(f"{type(exc).__name__}: {exc}")
            body["gates"] = dict.fromkeys(GATE_NAMES, False)
        body["verdict"] = "QUALIFIED" if all(body["gates"].values()) else "NOT_QUALIFIED"
        write(fh)
    return {"verdict": body["verdict"], "gates": body["gates"], "errors": body["errors"],
            "sha256": P3.sha256_file(RECORD)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(qualify(args.library.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
