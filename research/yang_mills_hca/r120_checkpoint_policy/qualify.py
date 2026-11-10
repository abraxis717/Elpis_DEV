"""Prospective R120 fresh-seed test of a checkpoint-only correction policy.

Reuse frozen R118-E execution *unchanged* for both NATIVE and R114, changing
only independent evaluation seeds and structure seed. Because the R120 policy
is static per checkpoint and all 20 HCA passes in a cell use that checkpoint,
its trajectory is exactly one of the two already-executed control trajectories.
This derivation does not splice or mix trajectories, and it does not query the
teacher for selection. Teacher remains evaluation-only inside R118-E.

R119 EXACT_GUARDED was registered bit-identical to R118 for H1/H20; this
scientific source comparison uses R118's frozen exact engine for historical
harness parity. Independently test create_engine in integration before hot use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from .policy import controller_for_checkpoint

ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT / "PROTOCOL_R120.json"


def load_protocol():
    obj = json.loads(PROTOCOL.read_text())
    if obj["status"] != "PREREGISTRATION_SOURCE_ONLY_NOT_MEASURED":
        raise RuntimeError("NOT_PREREGISTRATION")
    return obj


def source_guard(proto, harness):
    actual = hashlib.sha256(pathlib.Path(harness.__file__).read_bytes()).hexdigest()
    if actual != proto["r118_characterize_e_sha256"]:
        raise RuntimeError("R118E_SOURCE_DRIFT:" + actual)
    from research.yang_mills_hca.r118_scalable_awr import provenance as P
    if P.component_digest() != "770411b1f6788f9593f5f7adf1d4f6efb57a564f8c207d13dd7f55352af0df71":
        raise RuntimeError("R118_COMPONENT_DIGEST_DRIFT")
    kernel = ROOT.parent / "r119_numerical_hotpath" / "csrc" / "awr_kernel_r119.c"
    if hashlib.sha256(kernel.read_bytes()).hexdigest() != proto["r119_kernel_sha256"]:
        raise RuntimeError("R119_KERNEL_DRIFT")
    return P


def derive(frozen_evidence, controller_lookup=controller_for_checkpoint):
    src = frozen_evidence["by_graph"]
    dest = dict(src)
    for key in src:
        if "/NATIVE/" not in key:
            continue
        prefix, suffix = key.split("/NATIVE/", 1)
        ck = suffix.rsplit("/H", 1)[0]
        mode = controller_lookup(ck)
        lookup = prefix + "/" + mode + "/" + suffix
        if lookup not in src:
            raise RuntimeError("MISSING_MATCHED_CONTROL:" + lookup)
        dest[prefix + "/SIGNED_ONLY/" + suffix] = src[lookup]
    out = {}
    for h in frozen_evidence["horizons"]:
        for k in frozen_evidence["k_values"]:
            for mode in ("NATIVE", "R114", "SIGNED_ONLY"):
                rows = [v for name, v in dest.items()
                        if "/k" + str(k) + "/" + mode + "/" in name
                        and name.endswith("/H" + str(h))]
                if len(rows) != 24:
                    raise RuntimeError("CELL_COUNT:" + str((h, k, mode, len(rows))))
                out[f"k{k}/{mode}/H{h}"] = {
                    "n": len(rows),
                    "nse": sum(r["nse_vs_composed_teacher"] for r in rows) / len(rows),
                    "matched_nse": sum(r["matched_nse_vs_composed_teacher"] for r in rows) / len(rows),
                    "energy_ratio": sum(r["energy_ratio_vs_composed_teacher"] for r in rows) / len(rows),
                }
    criteria = {}
    for k in frozen_evidence["k_values"]:
        selected = out[f"k{k}/SIGNED_ONLY/H20"]["nse"]
        native = out[f"k{k}/NATIVE/H20"]["nse"]
        r114 = out[f"k{k}/R114/H20"]["nse"]
        criteria[f"k{k}"] = {"delta_vs_native": native - selected,
                              "delta_vs_r114": r114 - selected,
                              "pass": selected < native and selected < r114}
    worst = []
    for key, rec in dest.items():
        if "/SIGNED_ONLY/" not in key or not key.endswith("/H20"):
            continue
        native = src[key.replace("/SIGNED_ONLY/", "/NATIVE/")]["nse_vs_composed_teacher"]
        r114 = src[key.replace("/SIGNED_ONLY/", "/R114/")]["nse_vs_composed_teacher"]
        current = rec["nse_vs_composed_teacher"]
        worst.append({"cell": key, "harm_vs_native": current - native,
                      "harm_vs_r114": current - r114})
    worst.sort(key=lambda x: max(x["harm_vs_native"], x["harm_vs_r114"]), reverse=True)
    return {"phase": "R120", "gate": "FRESH_SEED_MEASURED_PASS_BOUNDED" if all(
                x["pass"] for x in criteria.values()) else "FRESH_SEED_NONQUAL",
            "summary": out, "primary": criteria, "worst_h20": worst[:12],
            "source_harness_phase": frozen_evidence["phase"],
            "seed_leak": frozen_evidence["seed_leak"],
            "weights_unchanged": frozen_evidence["weights_unchanged"],
            "r93": "ENERGY_NORMALIZED_DENOMINATOR_HOLD"}


def run():
    proto = load_protocol()
    from research.yang_mills_hca.r118_scalable_awr import characterize_e as e
    source_guard(proto, e)
    orig_seeds, orig_struct = e.EVAL_SEEDS, e.STRUCTURE_SEED
    try:
        e.EVAL_SEEDS = tuple(proto["evaluation_seeds"])
        e.STRUCTURE_SEED = proto["structure_seed"]
        if set(e.EVAL_SEEDS) & e.historical_seeds():
            raise RuntimeError("EVAL_SEED_LEAK")
        if len(set(e.EVAL_SEEDS)) != len(e.EVAL_SEEDS):
            raise RuntimeError("DUPLICATE_SEEDS")
        data = e.run()
    finally:
        e.EVAL_SEEDS, e.STRUCTURE_SEED = orig_seeds, orig_struct
    if not data["weights_unchanged"] or data["seed_leak"]:
        raise RuntimeError("EVIDENCE_INTEGRITY")
    out = derive(data)
    out["protocol_sha256"] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    out["source_digest"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    out["eval_seeds"] = list(proto["evaluation_seeds"])
    out["structure_seed"] = proto["structure_seed"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", help="Optional result file under /dev/shm only")
    args = p.parse_args()
    if args.out and not pathlib.Path(args.out).resolve().is_relative_to("/dev/shm"):
        raise SystemExit("VOLATILE_RESULTS_ONLY")
    out = run()
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, sort_keys=True, indent=1) + "\n")
    print("=== SEND BACK ===")
    print("PHASE=R120_FRESH_SEED_STATIC_CHECKPOINT_SELECTION")
    print("GATE=" + out["gate"])
    for k, v in out["primary"].items():
        print(f"{k}_NATIVE_MINUS_SELECTED={v['delta_vs_native']:.9f} R114_MINUS_SELECTED={v['delta_vs_r114']:.9f} PASS={v['pass']}")
    print("CHECKPOINT_PINS_UNCHANGED=" + str(out["weights_unchanged"]))
    print("PROTOCOL_SHA256=" + out["protocol_sha256"])
    print("SOURCE_SHA256=" + out["source_digest"])
    print("R93=" + out["r93"])
    print("BLOCKER=NONE CHILD_EXIT=0")


if __name__ == "__main__":
    main()
