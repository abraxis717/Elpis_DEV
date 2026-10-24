"""Three-way differential: H1D0 port vs REAL Python reference substrate vs REAL native C HCA.

Uses the mirror's own ctypes bridge (tests/research/ecsc_hca_native/_native.py) against
the library built from native/ECS_C/hca/src/ecs_c_hca.c.  Nothing in the mirror is
modified; the .so is built into this phase's own native_build directory.
"""
import json, os, sys, time
from pathlib import Path

# Publication derivative: both packages are inside the same checkout.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_H1D0_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src", _H1D0_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from generator import build_split
from oracle import gold_for_case
from engines import HcaNativeSemantics
from native_ref import build_graph, task_for, reference_predict
from tests.research.ecsc_hca_native._native import bind, NativeGraph, Result

# Build the actual native C ABI v1 library separately; no implicit build.
LIB = os.environ.get("ELPIS_HCA_NATIVE_LIB")
if not LIB or not Path(LIB).is_file():
    raise RuntimeError("Set ELPIS_HCA_NATIVE_LIB to the compiled native C ABI v1 library")
lib = bind(LIB)
assert lib.elpis_hca_abi_version() == 1, "ABI version mismatch"

TRAIN_SEED, TUNE_SEED = 12345, 67890
tr, truth_tr = build_split("TRAIN", TRAIN_SEED, 4)
tu, truth_tu = build_split("TUNE", TUNE_SEED, 2)

out = {"native_abi_version": lib.elpis_hca_abi_version(), "comparisons": 0,
       "mismatches": {}, "native_build_seconds": 0.0}
t0 = time.perf_counter()
for split, truth in ((tr, truth_tr), (tu, truth_tu)):
    for w in split["worlds"]:
        for case in w["cases"]:
            gold = gold_for_case(case, truth[case["case_id"]])
            for pref in (0, len(case["events"]) // 2, len(case["events"])):
                graph, cells = build_graph(case, pref)
                for q in case["queries"]:
                    task = task_for(case, q, pref, cells, rounds=4)
                    ng = NativeGraph(lib, graph, budget=10_000_000)
                    if ng.code != "OK":
                        out["mismatches"].setdefault("native_graph_create", 0)
                        out["mismatches"]["native_graph_create"] += 1
                        ng.close()
                        continue
                    for method in ("B0", "B1", "B2"):
                        code, res, scores = ng.predict(task, method)
                        if code != "OK":
                            out["mismatches"].setdefault("native_" + method + "_refusal", 0)
                            out["mismatches"]["native_" + method + "_refusal"] += 1
                            continue
                        target_key = __import__("native_ref").cell_key(q["to"], "L3")
                        native_pred = 1 if any(p[0] == target_key for p in res[0]) else 0
                        ref_pred = reference_predict(case, pref, q, method, rounds=4)[0]
                        port_pred = HcaNativeSemantics(case, method=method, rounds=4)
                        for e in case["events"][:pref]:
                            port_pred.apply(e)
                        port_val = port_pred.snapshot()[q["id"]]
                        out["comparisons"] += 3
                        for name, a, b in (("native_vs_gold", native_pred, gold[pref][q["id"]]),
                                          ("native_vs_reference", native_pred, ref_pred),
                                          ("port_vs_reference", port_val, ref_pred)):
                            if a != b:
                                out["mismatches"][name] = out["mismatches"].get(name, 0) + 1
                    ng.close()
out["native_build_seconds"] = round(time.perf_counter() - t0, 2)
print(json.dumps(out, indent=1))
# Do not write into the Git checkout by default. An explicit evidence
# directory is required; existing results are never overwritten.
if os.environ.get("ELPIS_HCA_RESULT_DIR"):
    _outdir = Path(os.environ["ELPIS_HCA_RESULT_DIR"]).resolve()
    if not _outdir.is_dir() or _outdir.is_symlink():
        raise RuntimeError("ELPIS_HCA_RESULT_DIR must be an existing real directory")
    _target = _outdir / "native_differential.json"
    with _target.open("x", encoding="utf-8") as _handle:
        json.dump(out, _handle, indent=1)
        _handle.write("\n")
