"""R122 registered runner: energy-normalized fidelity, directional safety, recurrent error localization.

    python -B -m research.yang_mills_hca.r122_energy_direction_forensics.qualify --registered --out /dev/shm/r122
    python -B -m research.yang_mills_hca.r122_energy_direction_forensics.qualify --consumed-companion --out /dev/shm/r122c

``--registered``         the prospective evaluation of PROTOCOL_R122.json. Refuses to run unless
                         the component source and protocol are committed, HEAD equals its upstream
                         (the preregistration is published) and no registered result exists yet.
``--consumed-companion`` the identical pipeline on the already-consumed R121 graphs and seeds. It
                         carries no decision authority; it rehearses the pipeline and discloses the
                         discovery material that preceded the registration.

Order of work (a failure stops before the next stage):

  0 source guard   pins of the frozen host, R118 / R119 kernels, R120 / R121 evidence and this source
  1 controls       synthetic classifier cases; consumed R121 regression fixture
  2 measurement    autonomous, teacher-forced, restart and controller tensors (measure.py)
  3 gauge + linear joint-gauge / negative controls; decayed-regime operators
  4 analysis       registered decompositions and decisions (analyze.py)
  5 evidence       written under /dev/shm only; nothing is written into the repository

Teacher states never enter an autonomous trajectory (see harness.py / measure.py). No
training, no parameter change, no persistent inference mutation; R93 stays on HOLD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import resource
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
YM = HERE.parent
REPO = YM.parent.parent
PROTOCOL = HERE / "PROTOCOL_R122.json"
SOURCE_FILES = ("__init__.py", "metrics.py", "topology.py", "harness.py", "measure.py", "analyze.py", "linear.py",
                "controls.py", "qualify.py")
REGISTERED_STATUS = "PREREGISTERED_SOURCE_NOT_MEASURED"
R93 = "ENERGY_NORMALIZED_DENOMINATOR_HOLD"


def sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def source_sha256() -> dict:
    return {name: sha(HERE / name) for name in SOURCE_FILES}


def component_digest() -> str:
    lines = "".join(f"{h}  {n}\n" for n, h in sorted(source_sha256().items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def consumed_seeds() -> set:
    """Every seed used by the frozen host protocols and by R118-R121 (sources cited inline)."""
    from research.yang_mills_hca.r118_scalable_awr import characterize_e as e118
    seeds = set(e118.historical_seeds())                       # FROZEN_PROTOCOL_R*.json and the R92 helper protocol
    seeds |= set(e118.EVAL_SEEDS) | {e118.STRUCTURE_SEED + i for i in range(4)}                  # R118-E
    seeds |= {11801, 11802, 11803, 118500, 119500}                                               # R118-D / R119 registries, states
    seeds |= set(range(119000, 119100)) | {119100 + a for a in (64, 256, 1024)}                  # R119 active sets
    seeds |= {120101, 120103, 120107, 120109} | {120300 + i for i in range(4)}                   # R120
    seeds |= {121101, 121103, 121107, 121109} | {121300 + i for i in range(4)} | {121400}        # R121
    return seeds


def pair_state_generator_seeds(seeds, cases=3):
    """torch generator seeds that ``portable_mechanics.state_pair`` derives: seed * 139 + case_index * 83."""
    return {s * 139 + ci * 83 for s in seeds for ci in range(cases)}


def seed_audit(proto) -> dict:
    old = consumed_seeds()
    new = set(proto["evaluation_seeds"]) | {proto["structure_seed"] + i for i in range(len(proto["graphs"]))}
    aux = {proto["graph_generation"]["seed"], proto["thresholds"]["bootstrap_seed"], proto["controls"]["gauge_seed"],
           proto["controls"]["classifier_seed"]}
    distinct = len(new) == len(proto["evaluation_seeds"]) + len(proto["graphs"])
    clash = sorted((new | aux) & old)
    derived_clash = sorted(pair_state_generator_seeds(proto["evaluation_seeds"]) & pair_state_generator_seeds(old))
    return {"pass": bool(distinct and not clash and not derived_clash and not (new & aux)), "consumed_seed_count": len(old),
            "new_seed_count": len(new | aux), "clashes": clash, "derived_generator_seed_clashes": derived_clash}


def git_state() -> dict:
    def run(*args):
        p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    rel = str(HERE.relative_to(REPO))
    dirty = run("status", "--porcelain", "--", rel)
    return {"head": run("rev-parse", "HEAD"), "upstream": run("rev-parse", "@{u}"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "component_uncommitted": None if dirty is None else [ln for ln in dirty.splitlines() if "/evidence/" not in ln]}


def source_guard(proto, oracle, registered: bool) -> dict:
    """Authenticate everything the measurement depends on. Raises on any drift."""
    from research.yang_mills_hca.r118_scalable_awr import provenance as P
    from . import metrics as M
    from . import topology as T
    pins = proto["pins"]
    checks = {}

    def need(name, ok, detail=""):
        checks[name] = bool(ok)
        if not ok:
            raise RuntimeError("SOURCE_GUARD:" + name + (":" + str(detail) if detail else ""))

    need("protocol_status", proto["status"] == REGISTERED_STATUS)
    need("r122_source_sha256", source_sha256() == proto["source_sha256"],
         sorted(k for k, v in source_sha256().items() if proto["source_sha256"].get(k) != v))
    manifest = P.verify_manifest()
    need("frozen_manifest", manifest["files_verified"] == 39 and manifest["manifest_sha256"] == pins["frozen_manifest_sha256"])
    need("checkpoint_pins", oracle.parameter_pins() == P.CHECKPOINT_SHA256 and P.CHECKPOINT_SHA256 == pins["checkpoint_sha256"])
    need("r118_component_digest", P.component_digest() == pins["r118_component_digest"])
    for key, rel in (("r118_kernel_sha256", "r118_scalable_awr/csrc/awr_kernel.c"),
                     ("r119_kernel_sha256", "r119_numerical_hotpath/csrc/awr_kernel_r119.c"),
                     ("r119_engine_sha256", "r119_numerical_hotpath/engine_r119.py"),
                     ("r119_guards_sha256", "r119_numerical_hotpath/guards.py"),
                     ("r118_characterize_e_sha256", "r118_scalable_awr/characterize_e.py"),
                     ("controller_reference_sha256", "reference.py"),
                     ("r120_raw_sha256", "r120_checkpoint_policy/evidence/R120_RAW_FRESH.json"),
                     ("r121_protocol_sha256", "r121_temporal_topology/PROTOCOL_R121.json"),
                     ("r121_policy_sha256", "r121_temporal_topology/policy.py"),
                     ("r121_qualify_sha256", "r121_temporal_topology/qualify.py"),
                     ("r121_result_sha256", "r121_temporal_topology/evidence/R121_RESULT.json")):
        need(key, sha(YM / rel) == pins[key], rel)
    c = proto["classifier"]
    need("classifier_constants", (c["r_floor"], c["cos_aligned"], c["cos_deviated"], c["cos_reversed"], c["cos_antiparallel"],
                                  c["near_zero_exponent"], c["pair_condition_limit"]) ==
         (M.R_FLOOR, M.COS_ALIGNED, M.COS_DEVIATED, M.COS_REVERSED, M.COS_ANTIPARALLEL, M.NEAR_ZERO_EXPONENT, M.PAIR_CONDITION_LIMIT))
    # topology: regenerate the registered graphs from the registered scan and audit their novelty
    gen = proto["graph_generation"]
    regenerated = T.generate(gen["seed"], gen["degree_schedule"], set(T.historical_classes()), attempts=gen["attempts_per_graph"])
    need("graph_regeneration", regenerated == list(proto["graphs"].values()))
    audit = T.audit(proto["graphs"])
    need("graph_novelty", audit["all_novel"])
    seeds = seed_audit(proto)
    need("seed_disjointness", seeds["pass"], seeds)
    git = git_state()
    if registered:
        need("git_component_committed", git["component_uncommitted"] == [], git["component_uncommitted"])
        need("git_preregistration_published", git["head"] is not None and git["head"] == git["upstream"], git)
        need("no_prior_registered_result", not (HERE / "evidence" / "R122_RESULT.json").exists())
    return {"checks": checks, "topology_audit": audit, "seed_audit": seeds, "git": git, "frozen_manifest": manifest}


def config_from_protocol(proto) -> dict:
    return {"graphs": dict(proto["graphs"]), "structure_seed": proto["structure_seed"],
            "evaluation_seeds": list(proto["evaluation_seeds"]), "cases": list(proto["cases"]),
            "k_values": list(proto["k_values"]), "scales": dict(proto["scales"]), "ratio": proto["source_ratio"],
            "batch": proto["batch_per_seed"], "horizons": list(proto["horizons"]),
            "deep_check_stride": proto["controls"]["deep_check_stride"],
            "reference_teacher_steps": list(proto["controls"]["reference_teacher_steps"])}


def companion_config(proto) -> dict:
    """The consumed R121 graphs, seeds and structure with the R122 scales and horizons."""
    from . import controls as C
    r121 = json.loads(C.R121_PROTOCOL.read_bytes())
    cfg = config_from_protocol(proto)
    cfg.update(graphs={n: r121["graphs"][n] for n in sorted(r121["graphs"])}, structure_seed=r121["structure_seed"],
               evaluation_seeds=list(r121["evaluation_seeds"]), cases=list(r121["cases"]), k_values=list(r121["k_values"]))
    return cfg


def gauge_block(oracle, cfg, proto, progress) -> dict:
    from . import controls as C
    from . import harness as H
    scale = cfg["scales"][proto["thresholds"]["primary_scale"]]
    out, ok = {}, True
    for gi, (name, graph) in enumerate(cfg["graphs"].items()):
        ks = (cfg["k_values"][0], cfg["k_values"][-1])
        systems, parts = H.build_systems(oracle, graph, cfg["structure_seed"] + gi, ks=ks)
        st0 = oracle.r64.pair_state(cfg["evaluation_seeds"][0], cfg["cases"][-1], cfg["ratio"], batch=cfg["batch"])
        for k in ks:
            S = systems[k]
            X0 = S.x0(st0[0].numpy(), st0[1].numpy(), scale)
            T = S.teacher(oracle, X0)
            rng = np.random.default_rng(proto["controls"]["gauge_seed"] + 10 * gi + k)
            res = C.gauge_controls(oracle, S, parts, X0, T, proto["controls"]["gauge_checkpoints"], rng, proto["tolerances"])
            out[f"{name}/k{k}"] = res
            ok = ok and res["pass"]
        progress(f"R122_GAUGE graph={name} pass={all(out[f'{name}/k{k}']['pass'] for k in ks)}")
    return {"pass": bool(ok), "systems": out,
            "max_joint_covariance_abs": max(v["joint_covariance_max_abs"] for v in out.values()),
            "max_metric_invariance": max(v["metric_invariance_max"] for v in out.values()),
            "min_state_only_negative_control": min(v["state_only_negative_control_min"] for v in out.values()),
            "inconsistent_gauge_detected": [sum(v["half_turn_detected_as_reversed"] for v in out.values()),
                                            sum(v["quarter_turn_detected_as_orthogonal"] for v in out.values()),
                                            sum(v["inconsistent_gauge_samples"] for v in out.values())]}


def linear_block(oracle, cfg, proto, progress) -> dict:
    from research.yang_mills_hca.r118_scalable_awr import provenance as P
    from . import harness as H
    from . import linear as L
    tol = proto["tolerances"]
    lin_label = proto["thresholds"]["linear_limit_scale"]
    scale = cfg["scales"][lin_label]
    systems_out, worst_h, worst_a, worst_t = {}, 0.0, 0.0, 0.0
    ratios = {"SIGNED": [], "UNSIGNED": [], "UNSIGNED_R104": []}
    phase = {"SIGNED": [], "UNSIGNED": [], "UNSIGNED_R104": []}
    for gi, (name, graph) in enumerate(cfg["graphs"].items()):
        systems, _ = H.build_systems(oracle, graph, cfg["structure_seed"] + gi, ks=tuple(cfg["k_values"]))
        st0 = oracle.r64.pair_state(cfg["evaluation_seeds"][0], cfg["cases"][0], cfg["ratio"], batch=cfg["batch"])
        for k in cfg["k_values"]:
            S = systems[k]
            rec, ops = L.analyze_system(S, oracle)
            X0 = S.x0(st0[0].numpy(), st0[1].numpy(), scale)
            T = S.teacher(oracle, X0)
            real = {}
            for ck in P.CHECKPOINTS:
                real[ck + "|NATIVE"] = S.rollout(X0, ck, "NATIVE")
                if not H.is_signed(ck):
                    real[ck + "|R114"] = S.rollout(X0, ck, "R114")
            resid = L.linear_trajectory_residual(S, ops, X0, T, real)
            rec["linear_limit_trajectory_max_relative_residual"] = max(resid.values())
            worst_h = max(worst_h, rec["max_probe_homogeneity_abs"])
            worst_a = max(worst_a, rec["max_analytic_vs_probe_rel"])
            worst_t = max(worst_t, rec["linear_limit_trajectory_max_relative_residual"])
            for ck, item in rec["checkpoints"].items():
                coh = "SIGNED" if H.is_signed(ck) else "UNSIGNED"
                ratios[coh].append(item["native"]["radius_ratio_vs_teacher"])
                phase[coh].append(item["native"]["phase_minus_teacher_deg_per_step"])
                if "r104" in item:
                    ratios["UNSIGNED_R104"].append(item["r104"]["radius_ratio_vs_teacher"])
                    phase["UNSIGNED_R104"].append(item["r104"]["phase_minus_teacher_deg_per_step"])
            systems_out[f"{name}/k{k}"] = rec
        progress(f"R122_LINEAR graph={name} done")
    summary = {}
    for key in ratios:
        r, p = np.array(ratios[key]), np.array(phase[key])
        summary[key] = {"n": int(r.size), "radius_ratio_median": float(np.median(r)), "radius_ratio_min": float(r.min()),
                        "radius_ratio_max": float(r.max()), "fraction_slower_decay_than_teacher": float((r > 1).mean()),
                        "phase_minus_teacher_median_deg": float(np.median(p)),
                        "fraction_faster_phase_than_teacher": float((p > 0).mean())}
    ok = (worst_h <= tol["linear_probe_homogeneity_abs"] and worst_a <= tol["linear_analytic_vs_probe_rel"]
          and worst_t <= tol["linear_limit_trajectory_rel"])
    return {"pass": bool(ok), "probe_exponent": L.PROBE_EXPONENT, "linear_limit_scale": lin_label,
            "max_probe_homogeneity_abs": worst_h, "max_analytic_vs_probe_rel": worst_a,
            "max_linear_limit_trajectory_rel": worst_t, "teacher_gate_limits": list(L.teacher_limits()),
            "checkpoint_gate_limits": {ck: list(L.gate_limits(ck)) for ck in P.CHECKPOINTS},
            "summary": summary, "systems": systems_out}


def raw_digest(raw) -> str:
    h = hashlib.sha256()
    for name in sorted(raw):
        arr = np.ascontiguousarray(raw[name])
        h.update(name.encode() + str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def environment() -> dict:
    from research.yang_mills_hca.r118_scalable_awr import native as N118
    from research.yang_mills_hca.r118_scalable_awr import provenance as P
    from research.yang_mills_hca.r119_numerical_hotpath import native_r119 as N119
    env = P.environment_fingerprint()
    env["libc"] = "-".join(platform.libc_ver())
    env["r118_kernel"] = {k: N118.build().info[k] for k in ("source_sha256", "compiler_version", "cflags")}
    env["r119_kernel"] = {k: N119.kernel()[1][k] for k in ("source_sha256", "compiler_version", "cflags")}
    return env


def evaluate(mode: str, out_dir: pathlib.Path, progress=print) -> dict:
    t0 = time.monotonic()
    registered = mode == "registered"
    proto = json.loads(PROTOCOL.read_bytes())
    from research.yang_mills_hca.r118_scalable_awr import provenance as P
    from research.yang_mills_hca.r118_scalable_awr.frozen_oracle import get_oracle
    from . import analyze as A
    from . import controls as C
    from . import measure as MS
    oracle = get_oracle()
    result = {"phase": "R122", "schema": "elpis.ym_hca.r122.result.v1", "r93": R93,
              "mode": "REGISTERED_PROSPECTIVE" if registered else "CONSUMED_COMPANION_NO_DECISION_AUTHORITY",
              "protocol_sha256": sha(PROTOCOL), "source_sha256": source_sha256(), "component_digest": component_digest(),
              "base_commit": proto["base_commit"]}
    progress("R122_STAGE=0 source guard")
    result["source_guard"] = source_guard(proto, oracle, registered)
    progress("R122_STAGE=1 controls (synthetic classifier cases, consumed R121 regression fixture)")
    result["classifier_controls"] = C.synthetic_classifier_controls(proto["controls"]["classifier_seed"])
    result["regression_fixture"] = C.regression_fixture(oracle, proto["tolerances"], progress=lambda s: None)
    if not (result["classifier_controls"]["pass"] and result["regression_fixture"]["pass"]):
        raise RuntimeError("CONTROLS_FAILED_BEFORE_MEASUREMENT")
    cfg = config_from_protocol(proto) if registered else companion_config(proto)
    budget = proto["budgets"]
    if MS.raw_bytes(cfg) > budget["max_raw_tensor_bytes"] or MS.engine_step_bound(cfg) > budget["max_engine_steps"]:
        raise RuntimeError("RESOURCE_BUDGET_REFUSED_BEFORE_WORK")
    progress(f"R122_STAGE=2 measurement fixtures={len(cfg['graphs']) * len(cfg['k_values']) * len(cfg['scales']) * len(cfg['cases']) * len(cfg['evaluation_seeds'])}")
    raw, info = MS.run(cfg, oracle, progress=progress)
    result["measurement"] = info
    result["raw_tensor_sha256"] = raw_digest(raw)
    result["raw_tensor_bytes"] = int(sum(v.nbytes for v in raw.values()))
    progress("R122_STAGE=3 gauge controls and decayed-regime operators")
    result["gauge_controls"] = gauge_block(oracle, cfg, proto, progress)
    linear = linear_block(oracle, cfg, proto, progress)
    progress("R122_STAGE=4 registered analysis")
    thr = dict(proto["thresholds"])
    analysis, tables = A.analyze(raw, cfg, thr)
    result.update(analysis)
    tables["linear_systems"] = linear.pop("systems")
    result["linear"] = linear
    result["checkpoints_unchanged"] = oracle.parameter_pins() == P.CHECKPOINT_SHA256
    result["frozen_tree_unchanged"] = P.verify_manifest()["files_verified"] == 39
    tol = proto["tolerances"]
    c, mx = info["counters"], info["maxima"]
    da, ta = analysis["decomposition_audit"], analysis["telescoping_audit"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    seconds = time.monotonic() - t0
    gates = {
        "SOURCE_PINS": all(result["source_guard"]["checks"].values()),
        "FROZEN_PARITY": mx["teacher_k0_vs_historical_max_abs"] <= tol["teacher_k0_parity_abs"] and c["teacher_k0_checks"] > 0,
        "CHECKPOINTS_UNCHANGED": result["checkpoints_unchanged"] and result["frozen_tree_unchanged"],
        "GRAPH_NOVELTY": result["source_guard"]["topology_audit"]["all_novel"],
        "SEED_DISJOINTNESS": result["source_guard"]["seed_audit"]["pass"],
        "DETECTOR_CONTROLS": result["classifier_controls"]["pass"] and result["regression_fixture"]["pass"] and result["gauge_controls"]["pass"],
        "AUTONOMOUS_GATE": (c["nonfinite"] == 0 and c["autonomous_recheck_mismatch"] == 0 and c["fresh_state_mismatch"] == 0
                            and c["delayed_spliced_vs_full_mismatch"] == 0 and c["delayed_h8_vs_native_mismatch"] == 0
                            and c["autonomous_rechecks"] > 0 and c["fresh_state_checks"] > 0 and c["delayed_checks"] > 0
                            and all(v == 0 for v in info["raw_nan_entries"].values())),
        "LOCAL_TRANSITION_GATE": (ta["one_step_max_residual"] <= tol["telescoping_residual"]
                                  and ta["restart_max_residual"] <= tol["telescoping_residual"]),
        "ENERGY_DECOMPOSITION_GATE": (da["max_identity_residual"] <= tol["identity_residual"] and da["nonfinite_samples"] == 0
                                      and mx["exact_audit_max_relative"] <= tol["exact_audit_relative"]
                                      and c["exact_identity_nonzero"] == 0 and c["exact_audit_samples"] > 0),
        "NUMERICAL_PARITY": (c["exact_r118_mismatch"] == 0 and c["exact_r118_checks"] > 0
                             and mx["reference_teacher_max_rel"] <= tol["reference_teacher_relative"]),
        "LINEAR_OPERATOR_CONTROLS": linear["pass"],
        "RESOURCE_BOUND": (result["raw_tensor_bytes"] <= budget["max_raw_tensor_bytes"]
                           and c["engine_steps"] <= MS.engine_step_bound(cfg) and rss <= budget["max_rss_bytes"]
                           and seconds <= budget["max_seconds"]),
    }
    if not registered:
        # the companion graphs are consumed by construction; novelty and seed checks describe the registered set
        gates["GRAPH_NOVELTY_OF_COMPANION"] = "NOT_APPLICABLE_CONSUMED"
    result["gates"] = gates
    integrity = all(v is True for k, v in gates.items() if k != "GRAPH_NOVELTY_OF_COMPANION")
    sci = result["scientific_disposition_before_integrity"]
    result["disposition"] = sci if integrity else "NONQUAL"
    if not registered:
        result["disposition"] = "CONSUMED_COMPANION:" + result["disposition"]
    result["resources"] = {"peak_rss_bytes": int(rss), "seconds": round(seconds, 1), "engine_steps": int(c["engine_steps"]),
                           "engine_step_bound": int(MS.engine_step_bound(cfg)), "raw_tensor_bytes": result["raw_tensor_bytes"]}
    result["environment"] = environment()
    result["limitations"] = proto["limitations"]
    # evidence (volatile only)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "R122" if registered else "R122_COMPANION"
    files = {}
    blob = (json.dumps(tables, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    (out_dir / f"{tag}_TABLES.json").write_bytes(blob)
    files[f"{tag}_TABLES.json"] = hashlib.sha256(blob).hexdigest()
    result["tables_sha256"] = files[f"{tag}_TABLES.json"]
    blob = (json.dumps(result, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()
    (out_dir / f"{tag}_RESULT.json").write_bytes(blob)
    files[f"{tag}_RESULT.json"] = hashlib.sha256(blob).hexdigest()
    evidence_bytes = sum((out_dir / n).stat().st_size for n in files)
    index = {"schema": "elpis.ym_hca.r122.evidence_index.v1", "phase": "R122", "mode": result["mode"],
             "disposition": result["disposition"], "r93": R93, "protocol_sha256": result["protocol_sha256"],
             "component_digest": result["component_digest"], "source_commit": result["source_guard"]["git"]["head"],
             "raw_tensor_sha256": result["raw_tensor_sha256"], "files_sha256": files, "evidence_bytes": evidence_bytes,
             "evidence_within_budget": evidence_bytes <= budget["max_evidence_bytes"]}
    (out_dir / f"{tag}_INDEX.json").write_text(json.dumps(index, sort_keys=True, indent=1) + "\n")
    np.savez(out_dir / f"{tag}_RAW.npz", **raw)
    result["_index"] = index
    return result


def send_back(result) -> list:
    g = result.get("gates", {})
    d = result.get("directional", {})
    pc = result.get("policy_comparison", {})
    git = result.get("source_guard", {}).get("git", {})

    def gate(name):
        v = g.get(name)
        return "NOT_MEASURED" if v is None else ("PASS" if v is True else ("FAIL" if v is False else str(v)))

    mech = result.get("mechanism", {}).get("decision", {})
    lines = ["=== SEND BACK ===", "PHASE=R122_ENERGY_DIRECTION_FORENSICS", "MODE=" + result.get("mode", "NOT_MEASURED"),
             "DISPOSITION=" + result.get("disposition", "NOT_MEASURED"),
             "BASE_COMMIT=" + str(result.get("base_commit")),
             "PREREG_COMMIT=" + str(git.get("head")),
             "RESULT_COMMIT=NOT_MEASURED (assigned when the evidence is committed)",
             "SOURCE_SHA256=" + str(result.get("component_digest")),
             "PROTOCOL_SHA256=" + str(result.get("protocol_sha256")),
             "RESULT_SHA256=" + str(result.get("_index", {}).get("files_sha256", {}).get(
                 ("R122" if result.get("mode") == "REGISTERED_PROSPECTIVE" else "R122_COMPANION") + "_RESULT.json", "NOT_MEASURED")),
             "FROZEN_PARITY=" + gate("FROZEN_PARITY") + " teacher_k0_max_abs=" + str(result.get("measurement", {}).get("maxima", {}).get("teacher_k0_vs_historical_max_abs")),
             "CHECKPOINTS_UNCHANGED=" + gate("CHECKPOINTS_UNCHANGED"),
             "GRAPH_NOVELTY=" + gate("GRAPH_NOVELTY"), "SEED_DISJOINTNESS=" + gate("SEED_DISJOINTNESS"),
             "AUTONOMOUS_GATE=" + gate("AUTONOMOUS_GATE"), "LOCAL_TRANSITION_GATE=" + gate("LOCAL_TRANSITION_GATE"),
             "ENERGY_DECOMPOSITION_GATE=" + gate("ENERGY_DECOMPOSITION_GATE"),
             "DIRECTIONAL_SAFETY_GATE=DETECTOR_" + gate("DETECTOR_CONTROLS") + " " + " ".join(
                 f"{p}:{d[p]['verdict_in_range_scales']}(n={d[p]['induced_in_range_all_steps']})" for p in d if p != "NATIVE"),
             "WORST_GRAPH_HARM=" + json.dumps(pc.get("worst_graph_harm_h20_over_k", "NOT_MEASURED"), sort_keys=True),
             "NUMERICAL_PARITY=" + gate("NUMERICAL_PARITY"), "RESOURCE_BOUND=" + gate("RESOURCE_BOUND"),
             "MECHANISM_COUNTS=" + json.dumps(mech.get("ALL", {}).get("counts", "NOT_MEASURED"), sort_keys=True),
             "COHORTS=" + json.dumps(result.get("cohort_dispositions", "NOT_MEASURED"), sort_keys=True),
             "R93=" + R93,
             "BLOCKER=" + ("NONE" if all(v is True for k, v in g.items() if not isinstance(v, str)) and g else
                           ",".join(k for k, v in g.items() if v is False) or "NOT_MEASURED"),
             "CHILD_EXIT=0", "=== END SEND BACK ==="]
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--registered", action="store_true")
    grp.add_argument("--consumed-companion", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out).resolve()
    if not out.is_relative_to("/dev/shm"):
        raise SystemExit("VOLATILE_OUTPUT_ONLY")
    try:
        result = evaluate("registered" if args.registered else "companion", out, progress=lambda s: print(s, flush=True))
    except Exception as exc:  # report the blocker; never manufacture a pass
        import traceback
        traceback.print_exc()
        print("=== SEND BACK ===\nPHASE=R122_ENERGY_DIRECTION_FORENSICS\nDISPOSITION=NONQUAL_NOT_COMPLETED\nR93=" + R93 +
              "\nBLOCKER=" + type(exc).__name__ + ":" + str(exc)[:300] + "\nCHILD_EXIT=2\n=== END SEND BACK ===", flush=True)
        return 2
    for line in send_back(result):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
