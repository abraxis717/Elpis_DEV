"""ECS Binary Observation R0 runner: design -> dev -> freeze -> qual -> perf -> disposition (write-once evidence).

    python -m research.ecs_binary_observer_r0.run <phase> --build <native build dir> [--raw-dir <dir>]

Every phase refuses to overwrite its evidence. dev, qual and perf rebuild the DESIGN decoders from the DESIGN
worlds and refuse unless the DESIGN observations reproduce bitwise. qual and perf-qual refuse unless the freeze
(protocol, source, laws, DEV evidence) still holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import lab
from . import law as L
from . import protocol as P
from .observer import Observer

ROOT = P.ROOT
EVIDENCE = ROOT / "evidence"
REPO = ROOT.parents[1]


def _find(build: Path, name: str) -> Path:
    found = sorted(p for p in build.rglob(name) if p.is_file())
    if not found:
        raise SystemExit(f"{name} not found under {build}")
    return found[0]


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_once(phase: str, payload: dict, files: dict | None = None) -> Path:
    out = EVIDENCE / phase
    if out.exists():
        raise SystemExit(f"evidence/{phase} exists: write-once")
    out.mkdir(parents=True)
    for name, data in (files or {}).items():
        (out / name).write_bytes(data)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    (out / "summary.json").write_text(text)
    print(f"evidence/{phase}/summary.json sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    return out


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _environment(build: Path) -> dict:
    cpu = ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(), "cpu": cpu,
            "k1_library_sha256": _sha_file(_find(build, "libelpis_ecsg_k1.so")),
            "observer_library_sha256": _sha_file(_find(build, "libelpis_ebo_r0.so")),
            "git_head": _git_head(), "protocol_sha256": P.protocol_digest(), "source_sha256": P.source_digest()}


def _laws_from(phase_dir: Path) -> dict:
    return {"GRAY4": (phase_dir / "law_gray4.bin").read_bytes(), "SPROJ": (phase_dir / "law_sproj.bin").read_bytes()}


def _design_bundle(k1, build: Path):
    """Rebuild the DESIGN decoders from the DESIGN worlds; refuse unless DESIGN reproduces bitwise."""
    design_summary = json.loads((EVIDENCE / "design" / "summary.json").read_text())
    blobs = _laws_from(EVIDENCE / "design")
    observers = {name: Observer(_find(build, "libelpis_ebo_r0.so"), blobs[name]) for name in P.ENCODERS}
    laws = {name: obs.law for name, obs in observers.items()}
    worlds = {w: lab.World(w) for w in P.world_ids("DESIGN")}
    design = [lab.run_world(k1, worlds[w]) for w in P.world_ids("DESIGN")]
    if lab.observations_digest(design) != design_summary["observations_sha256"]:
        raise SystemExit("DESIGN observations do not reproduce: refusing")
    lab.encode_post_hoc(design, observers)
    fitted = lab.fit_decoders(design, worlds, laws, design_summary["lambdas"])
    return observers, laws, worlds, fitted


def phase_design(args) -> None:
    k1 = lab.load_k1(_find(args.build, "libelpis_ecsg_k1.so"))
    worlds = {w: lab.World(w) for w in P.world_ids("DESIGN")}
    design = [lab.run_world(k1, worlds[w]) for w in P.world_ids("DESIGN")]
    cal = lab.calibrate(design)
    observers = {name: Observer(_find(args.build, "libelpis_ebo_r0.so"), cal[name]) for name in P.ENCODERS}
    laws = {name: obs.law for name, obs in observers.items()}
    lab.encode_post_hoc(design, observers)
    lambdas = lab.train_decoders(design, worlds, laws)
    probe = {}
    for name in P.ENCODERS:   # decoder-free diagnostics only (DESIGN never evaluates its own decoders)
        recs = [s["records"][name] for r in design for s in r["states"]]
        probe[name] = {"S1_payload_identical": sum(r["ref"][32:] == r["perm"][32:] for r in recs) / len(recs)}
    g = laws["GRAY4"]
    sp = laws["SPROJ"]
    payload = {
        "phase": "DESIGN", "registry": P.world_ids("DESIGN"), "environment": _environment(args.build),
        "observations_sha256": lab.observations_digest(design), "records_sha256": lab.records_digest(design),
        "laws": {"GRAY4": {"sha256": g.sha256, "tag": g.tag.hex(), "thresholds": [float(t).hex() for t in g.thresholds]},
                 "SPROJ": {"sha256": sp.sha256, "tag": sp.tag.hex(), "mu": float(sp.mu).hex(),
                           "inv_scale": float(sp.inv_scale).hex(),
                           "tau_summary": {"min": min(sp.tau), "max": max(sp.tau)}}},
        "pooled_values": cal["pooled_values"], "lambdas": lambdas, "diagnostics": probe,
        "failures": [f for r in design for f in r["failures"]],
    }
    _write_once("design", payload, {"law_gray4.bin": cal["GRAY4"], "law_sproj.bin": cal["SPROJ"]})


def _evaluate(k1, build: Path, registry: str, observers, laws, fitted) -> dict:
    worlds = {w: lab.World(w) for w in P.world_ids(registry)}
    on, interference, failures = [], [], []
    for w in P.world_ids(registry):
        off_result = lab.run_world(k1, worlds[w])
        on_result = lab.run_world(k1, worlds[w], observers, parity=True)
        same = lab.interference_witness(off_result) == lab.interference_witness(on_result)
        interference.append({"world": w, "identical": same})
        failures += on_result["failures"] + off_result["failures"]
        on.append(on_result)
    probe = lab.probes(on, worlds, laws, fitted)
    return {"results": on, "probe": probe, "gates": lab.gates(probe),
            "mechanics": {"M1_parity_and_repeatability_failures": [f for f in failures if "parity" in f or "flag" in f],
                          "M2_non_interference": all(i["identical"] for i in interference),
                          "M2_worlds": interference, "other_failures": [f for f in failures
                                                                        if "parity" not in f and "flag" not in f],
                          "records": sum(len(s["records"][n]) for r in on for s in r["states"] for n in s["records"]),
                          "records_sha256": lab.records_digest(on)}}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def phase_dev(args) -> None:
    k1 = lab.load_k1(_find(args.build, "libelpis_ecsg_k1.so"))
    observers, laws, _, fitted = _design_bundle(k1, args.build)
    ev = _evaluate(k1, args.build, "DEV", observers, laws, fitted)
    _write_once("dev", _clean({"phase": "DEV", "registry": P.world_ids("DEV"), "environment": _environment(args.build),
                               "probe": ev["probe"], "gates": ev["gates"], "mechanics": ev["mechanics"]}))


def phase_freeze(args) -> None:
    dev = EVIDENCE / "dev" / "summary.json"
    design = EVIDENCE / "design"
    if not dev.is_file():
        raise SystemExit("freeze requires DEV evidence")
    status = subprocess.run(["git", "status", "--porcelain", "--", str(ROOT)], cwd=REPO, capture_output=True,
                            text=True).stdout
    if status.strip():
        raise SystemExit("freeze requires a committed laboratory:\n" + status)
    payload = {"phase": "FREEZE", "source_commit": _git_head(), "protocol_sha256": P.protocol_digest(),
               "source_sha256": P.source_digest(), "dev_summary_sha256": _sha_file(dev),
               "design_summary_sha256": _sha_file(design / "summary.json"),
               "laws": {"GRAY4": _sha_file(design / "law_gray4.bin"), "SPROJ": _sha_file(design / "law_sproj.bin")},
               "qual_registry": P.world_ids("QUAL"), "gates": P.GATES, "perf": P.PERF, "disposition": P.DISPOSITION,
               "bfm_followup": P.BFM_FOLLOWUP}
    _write_once("freeze", payload)


def _verify_freeze() -> dict:
    freeze = json.loads((EVIDENCE / "freeze" / "summary.json").read_text())
    checks = {"protocol": freeze["protocol_sha256"] == P.protocol_digest(),
              "source": freeze["source_sha256"] == P.source_digest(),
              "dev": freeze["dev_summary_sha256"] == _sha_file(EVIDENCE / "dev" / "summary.json"),
              "design": freeze["design_summary_sha256"] == _sha_file(EVIDENCE / "design" / "summary.json"),
              "laws": freeze["laws"] == {"GRAY4": _sha_file(EVIDENCE / "design" / "law_gray4.bin"),
                                         "SPROJ": _sha_file(EVIDENCE / "design" / "law_sproj.bin")}}
    if not all(checks.values()):
        raise SystemExit(f"freeze does not hold: {checks}")
    return freeze


def audit_library(build: Path) -> dict:
    lib = _find(build, "libelpis_ebo_r0.so")
    allowed = {"memcpy", "memset", "__stack_chk_fail"}
    nm = shutil.which("nm")
    objdump = shutil.which("objdump")
    if nm is None or objdump is None:
        return {"available": False}
    undefined = subprocess.run([nm, "-D", "--undefined-only", str(lib)], capture_output=True, text=True).stdout
    imports = sorted({line.split()[-1].split("@")[0] for line in undefined.splitlines() if line.strip()
                      and line.split()[0] == "U"})
    weak = sorted({line.split()[-1].split("@")[0] for line in undefined.splitlines() if line.strip()
                   and line.split()[0] == "w"})
    disasm = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(lib)], capture_output=True, text=True).stdout
    def reached(fn: str, seen: set) -> set:
        """External (PLT) calls reachable from fn, following local calls transitively; MISSING if unresolved."""
        body = re.search(rf"<{re.escape(fn)}>:\n(.*?)\n\n", disasm, re.S)
        if body is None:
            return {f"MISSING:{fn}"}
        out = set()
        for callee in set(re.findall(r"call\s+\S+\s+<([^>+]+)", body.group(1))):
            if callee.endswith("@plt"):
                out.add(callee)
            elif callee not in seen:
                seen.add(callee)
                out |= reached(callee, seen)
        return out

    calls = {fn: sorted(reached(fn, {fn})) for fn in ("ebo_encode", "ebo_ring_push", "ebo_ring_pop")}
    hot_ok = all(set(c) <= {"__stack_chk_fail@plt"} for c in calls.values())
    return {"available": True, "imports": imports, "weak": weak, "observation_path_calls": calls,
            "M5": set(imports) <= allowed and hot_ok,
            "M6": not any(s.startswith(("elpis_ecsg_", "elpis_runtime", "elpis_continuity")) for s in imports)}


def _ctest(build: Path) -> dict:
    exe = _find(build, "test_ebo_r0")
    done = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    return {"exit": done.returncode, "stdout": done.stdout.strip()}


def phase_qual(args) -> None:
    freeze = _verify_freeze()
    if (EVIDENCE / "qual").exists():
        raise SystemExit("evidence/qual exists: QUAL runs once")
    k1 = lab.load_k1(_find(args.build, "libelpis_ecsg_k1.so"))
    observers, laws, _, fitted = _design_bundle(k1, args.build)
    ev = _evaluate(k1, args.build, "QUAL", observers, laws, fitted)
    first = ev["results"][0]
    clean = subprocess.run([sys.executable, "-m", "research.ecs_binary_observer_r0.run", "reencode", "--build",
                            str(args.build), "--world", first["world"]], cwd=REPO, capture_output=True, text=True,
                           env={"PYTHONPATH": f"{REPO}:{REPO / 'src'}", "PATH": "/usr/bin:/bin"})
    mech = ev["mechanics"]
    mech["M1_clean_process"] = clean.stdout.strip() == lab.records_digest([first])
    mech["M3_M4_M7_native_test"] = _ctest(args.build)
    mech["M5_M6_audit"] = audit_library(args.build)
    mech["pass"] = (not mech["M1_parity_and_repeatability_failures"] and mech["M1_clean_process"]
                    and mech["M2_non_interference"] and not mech["other_failures"]
                    and mech["M3_M4_M7_native_test"]["exit"] == 0 and mech["M5_M6_audit"].get("M5") is True
                    and mech["M5_M6_audit"].get("M6") is True)
    _write_once("qual", _clean({"phase": "QUAL", "freeze_commit": freeze["source_commit"],
                                "registry": P.world_ids("QUAL"), "environment": _environment(args.build),
                                "probe": ev["probe"], "gates": ev["gates"], "mechanics": mech}))


def phase_reencode(args) -> None:
    k1 = lab.load_k1(_find(args.build, "libelpis_ecsg_k1.so"))
    blobs = _laws_from(EVIDENCE / "design")
    observers = {n: Observer(_find(args.build, "libelpis_ebo_r0.so"), blobs[n]) for n in P.ENCODERS}
    print(lab.records_digest([lab.run_world(k1, lab.World(args.world), observers)]))


# --- performance ---------------------------------------------------------------------------------------------------

def _p99(x: np.ndarray) -> float:
    return float(np.percentile(x, 99, method="linear"))


def _bootstrap(off: np.ndarray, on: np.ndarray, seed: int) -> tuple:
    blocks = np.array_split(np.arange(len(off)), P.PERF_BLOCKS)
    g = np.random.Generator(np.random.PCG64(seed))
    diffs = []
    for _ in range(P.PERF_RESAMPLES):
        idx = np.concatenate([blocks[i] for i in g.integers(0, P.PERF_BLOCKS, P.PERF_BLOCKS)])
        diffs.append(_p99(on[idx]) - _p99(off[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _read_raw(path: Path) -> dict:
    data = path.read_bytes()
    out, pos = {}, 0
    while pos < len(data):
        rows, pairs = np.frombuffer(data, dtype="<u4", count=2, offset=pos)
        pos += 8
        s = np.frombuffer(data, dtype="<u8", count=2 * int(pairs), offset=pos).reshape(-1, 2).astype(float)
        pos += 16 * int(pairs)
        out[int(rows)] = s
    return out


def perf_decide(summary_rows: dict, raw: dict) -> dict:
    decided = {}
    for rows, s in raw.items():
        off, on = s[:, 0], s[:, 1]
        facts = summary_rows[rows]
        p99_off = _p99(off)
        theta = min(P.PERF_RELATIVE * p99_off, P.PERF_THETA_CEILING_NS)
        lo, hi = _bootstrap(off, on, P.SPEC["seed"])
        aa_lo, aa_hi = _bootstrap(off[0::2], off[1::2], P.SPEC["seed"] + 1)
        isolated = facts["encode_ns"] + facts["push_pop_ns"]
        median_off = float(np.median(off))
        if hi <= theta and isolated <= P.PERF_RELATIVE * median_off:
            verdict = "PASS"
        elif lo > theta or isolated > P.PERF_RELATIVE * median_off:
            verdict = "NONPASS"
        else:
            verdict = "INCONCLUSIVE"
        stats = {}
        for arm, x in (("off", off), ("on", on)):
            stats[arm] = {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "p99": _p99(x),
                          "max": float(x.max()), "mean": float(x.mean()), "throughput_per_s": 1e9 / float(x.mean())}
        decided[str(rows)] = {
            "pairs": int(len(off)), "stats_ns": stats, "delta_median_ns": stats["on"]["median"] - stats["off"]["median"],
            "delta_p99_ns": stats["on"]["p99"] - stats["off"]["p99"], "delta_p99_ci95_ns": [lo, hi],
            "aa_off_p99_ci95_ns": [aa_lo, aa_hi], "theta_ns": theta, "isolated_ns": isolated,
            "isolated_over_median_off": isolated / median_off,
            "isolated_over_bare_forward": isolated / facts["bare_forward_ns"],
            "encode_ns": facts["encode_ns"], "push_pop_ns": facts["push_pop_ns"], "drop_ns": facts["drop_ns"],
            "bare_forward_ns": facts["bare_forward_ns"],
            "k1_forward_calls_timed": facts["k1_forward_calls_timed"],
            "k1_crossings_added_by_observer": facts["k1_forward_calls_timed"] - 2 * int(len(off)),
            "k1_learn_calls_timed": facts["k1_learn_calls_timed"], "verdict": verdict}
    verdicts = [d["verdict"] for d in decided.values()]
    overall = "PASS" if all(v == "PASS" for v in verdicts) else ("NONPASS" if "NONPASS" in verdicts
                                                                  else "INCONCLUSIVE")
    return {"rows": decided, "verdict": overall}


def phase_perf(args) -> None:
    phase = f"perf_{args.stage}"
    if args.stage == "qual":
        _verify_freeze()
    if (EVIDENCE / phase).exists():
        raise SystemExit(f"evidence/{phase} exists: write-once")
    if args.raw_dir is None:
        raise SystemExit("--raw-dir is required (raw telemetry stays out of Git; its SHA-256 is recorded)")
    bench = _find(args.build, "ebo_r0_bench")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    out = {"phase": phase.upper(), "environment": _environment(args.build), "config": P.PERF, "encoders": {},
           "accounting": {"k1_crossings_per_observation": 0, "allocations_per_observation": 0,
                          "device_syncs": 0, "copies_per_observation":
                              "2 x 64 bytes (staged record -> caller record; record -> ring slot); inputs read in place",
                          "allocation_evidence": "native test test_ebo_r0 (interposed allocator, zero in window)"}}
    groups = {}
    for rows, pairs in P.PERF["pairs"].items():
        groups.setdefault(pairs, []).append(rows)
    for name in P.ENCODERS:
        blob = EVIDENCE / "design" / f"law_{name.lower()}.bin"
        rows_facts, raw_all, raw_files = {}, {}, {}
        for pairs, rows in groups.items():
            raw_path = args.raw_dir / f"{phase}_{name}_{pairs}.bin"
            done = subprocess.run([str(bench), str(blob), str(raw_path), str(pairs), *rows], capture_output=True,
                                  text=True, timeout=3600)
            if done.returncode != 0:
                raise SystemExit(done.stderr)
            facts = json.loads(done.stdout)
            for r in facts["rows"]:
                rows_facts[r["rows"]] = r
            raw_all.update(_read_raw(raw_path))
            raw_files[raw_path.name] = _sha_file(raw_path)
            out.setdefault("affinity_cpu", facts["affinity_cpu"])
        decided = perf_decide(rows_facts, raw_all)
        decided["raw_sha256"] = raw_files
        out["encoders"][name] = decided
    _write_once(phase, out)


def phase_disposition(args) -> None:
    freeze = _verify_freeze()
    qual = json.loads((EVIDENCE / "qual" / "summary.json").read_text())
    perf = json.loads((EVIDENCE / "perf_qual" / "summary.json").read_text())
    useful = {n: qual["gates"][n]["USEFUL"] for n in P.ENCODERS}
    verdicts = {n: perf["encoders"][n]["verdict"] for n in P.ENCODERS}
    result = P.disposition(qual["mechanics"]["pass"], useful, verdicts)
    bfm = False
    for n in P.ENCODERS:
        info = qual["probe"]["information"][n]
        if (result == "BINARY_OBSERVER_FEASIBLE" and useful[n] and verdicts[n] == "PASS" and qual["gates"][n]["S4"]
                and info["bit_entropy_mean"] >= 0.5 and info["constant_bits"] <= 0.1 * L.PAYLOAD_BITS):
            bfm = True
    _write_once("disposition", {"phase": "DISPOSITION", "freeze_commit": freeze["source_commit"],
                                "qual_summary_sha256": _sha_file(EVIDENCE / "qual" / "summary.json"),
                                "perf_qual_summary_sha256": _sha_file(EVIDENCE / "perf_qual" / "summary.json"),
                                "mechanics_pass": qual["mechanics"]["pass"], "useful": useful, "perf": verdicts,
                                "disposition": result, "rule": P.DISPOSITION,
                                "bfm_followup_justified": bfm, "bfm_rule": P.BFM_FOLLOWUP})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["design", "dev", "freeze", "qual", "reencode", "perf", "disposition"])
    ap.add_argument("--build", type=Path, default=REPO / "build")
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--world")
    ap.add_argument("--stage", choices=["dev", "qual"], default="dev")
    args = ap.parse_args(argv)
    args.build = args.build.resolve()
    {"design": phase_design, "dev": phase_dev, "freeze": phase_freeze, "qual": phase_qual,
     "reencode": phase_reencode, "perf": phase_perf, "disposition": phase_disposition}[args.phase](args)


if __name__ == "__main__":
    main()
