#!/usr/bin/env python3
"""research/volatile_runtime_v1: zero persistent writes on the composed runtime (closure condition 11).

    python3 -I research/volatile_runtime_v1/zero_write_composed.py --repo . --build build \\
        --out research/volatile_runtime_v1/results

Extends research/volatile_runtime_v0, whose recorded evidence covered the volatile ECS_C session's own test paths
only. It reuses V0's tracer, detector and confinement (imported from its harness, unchanged) and runs the composed
runtime. The harness prepares every input before any traced run, and the traced processes write nothing:

  host_session      the native host (production entry point): RuntimeCore ABI v8 over the volatile session, ECS_G +
                    ECS_C composed, sources and operator events, typed evidence and LOOKUP, managed numeric LEARN and
                    QUERY, in-memory refusals (fuel, rate, invalid typed relationships, a dormant source's evidence, a
                    stale coordinate), retired commands, session close and a fresh session in the same process
  host_long         a randomized long session (seeded), for boundedness: its peak RSS against a ten times shorter one
  host_restarts     repeated startup/shutdown: every process starts at the lineage's initial identity
  python_cli        the Python reference interface over the same script (an independent differential: same answers)
  python_fms_hacf   the Python reference's FMS-resident K1 under RuntimeCore, a RAM-only FMS context at its bound, two
                    sessions in a row, a volatile HACF retrieval epoch and query ingress (python_fms_hacf.py)
  runtime_core      RuntimeCore's C suites over the real composite (session, composite, ABI, lifecycle, hot I/O)
  ecs_c_session     the session's C ABI and the relational view over it (V0's P1 and P2, rerun)
  fms_cold          FMS ABI v3 over an immutable COLD asset the harness made read-only: COLD -> WARM -> HOT, volatile
                    mutation, dematerialization, reload by SHA-256, budgets, failed materialization, legacy writers
  hgram_read        a retained H-gram (a read-only static resource): probe and association reads

Each path runs in three modes:
  plain                 no tracer: exit status, wall time, peak RSS (V0's peak_rss launcher)
  observed              under strace -f -y, with every write permitted. Root for the C suites and the Python
                        reference; uid 65534 for the host and the H-gram reader, which refuse a privileged process or
                        a foreign owner. The claim: no write is even attempted.
  confined              V0's confinement under strace: a private mount namespace with every mount but /proc
                        read-only, uid 65534, no capabilities, elpis_autonomous_run with no grant, RLIMIT_CORE 0

A legitimate run passes when it exits as expected with zero write-class events on storage (attempts count).

Adversarial runs (traced, each expecting a refusal or a contained outcome and zero write events):
  - standard output on storage;
  - an inherited writable descriptor;
  - active swap (the testing host's /proc/swaps override);
  - an unlimited core limit;
  - 6, 10 and 14 MiB address-space limits (failed allocations: refused, never a signal);
  - a privileged start;
  - a missing library and a drifted library;
  - legacy persistent commands and options.

Negative controls, kept apart from legitimate runs:
  - V0's detector and containment-layer controls, run again;
  - the testing host's deliberate write attempts under confinement: each refused, each detected as an attempt.

Static checks:
  - no binary the runtime loads imports a write-class C library function;
  - every input file's SHA-256 is unchanged after all runs: host, libraries, bridges, the COLD asset, the H-gram;
  - the harness's work tree holds exactly the entries it made.

The results carry no host-local path (<repo>, <build>, <work>). Needs root, Linux x86-64, strace, unshare, setpriv,
prlimit, nm, cc and Landlock ABI >= 5; it fails closed otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("zero_write_v0", HERE.parent / "volatile_runtime_v0" / "zero_write.py")
V0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V0)

SCHEMA = "elpis.volatile-runtime-v1.zero-write.v1"
NOBODY = 65534
SETPRIV = [shutil.which("setpriv") or "/usr/bin/setpriv", f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", "--inh-caps=-all",
           "--bounding-set=-all", "--"]


class Fail(SystemExit):
    def __init__(self, message: str):
        super().__init__(f"VOLATILE_V1_REFUSED: {message}")


def traced(argv: list[str], mode: str, launcher: Path, timeout: int, *, env: dict | None = None, stdout=None,
           extra_fds: tuple[int, ...] = ()) -> tuple[subprocess.CompletedProcess, str, float]:
    """V0's traced runner with an environment, a standard output and inherited descriptors. mode: observed (as
    the caller), uid_only (uid 65534, writable mounts, no launcher) or confined (V0's full confinement)."""
    r, w = os.pipe()
    chunks: list[bytes] = []

    def drain():
        while True:
            block = os.read(r, 1 << 16)
            if not block:
                return
            chunks.append(block)

    thread = threading.Thread(target=drain)
    thread.start()
    environment = {"PATH": "/usr/bin:/bin", "TRACED": ",".join(V0.TRACED)}
    environment.update(env or {})
    if mode == "observed":
        cmd = ["strace", "-f", "-qq", "-y", "-s", "96", "-e", "trace=" + ",".join(V0.TRACED), "-e", "signal=none",
               "-o", f"/proc/self/fd/{w}", "--", *argv]
    else:
        cmd = ["unshare", "--mount", "--propagation", "private", "--", "/bin/sh", "-c", V0.CONFINE, "confine", str(w),
               "none" if mode == "uid_only" else str(launcher), "1" if mode == "confined" else "0", *argv]
    start = time.monotonic()
    try:
        run = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=stdout if stdout is not None else subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, pass_fds=(w, *extra_fds), env=environment, cwd="/",
                             timeout=timeout)
    finally:
        os.close(w)
        thread.join()
        os.close(r)
    return run, b"".join(chunks).decode("utf-8", "replace"), time.monotonic() - start


def plain(argv: list[str], peak_rss: Path, timeout: int, env: dict | None = None,
          extra_fds: tuple[int, ...] = ()) -> dict:
    environment = {"PATH": "/usr/bin:/bin"}
    environment.update(env or {})
    start = time.monotonic()
    run = subprocess.run([str(peak_rss), *argv], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                         env=environment, cwd="/", timeout=timeout, pass_fds=extra_fds)
    seconds = round(time.monotonic() - start, 3)
    last = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else ""
    m = re.fullmatch(r"peak_rss_kib=(\d+) exit=(-?\d+)", last)
    if not m:
        raise Fail(f"peak_rss reported nothing for {argv[0]}")
    return {"exit": int(m.group(2)), "seconds": seconds, "peak_rss_kib": int(m.group(1)), "stdout": run.stdout}


def lines(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


# ---- inputs the harness prepares ------------------------------------------------------------------------------------

N1, N2, N3 = (hashlib.sha256(f"v1-node:{n}".encode()).hexdigest() for n in ("alice", "acme", "engineer"))


def prov(tag) -> str:
    return hashlib.sha256(f"v1-provenance:{tag}".encode()).hexdigest()


X = [[((i * 7 + j * 3) % 11 - 5) / 16.0 for j in range(6)] for i in range(4)]
EDGE = {"relation": "hyperedge:1", "participants": [{"node": N1, "role": "incidence:1"},
                                                    {"node": N2, "role": "incidence:2"},
                                                    {"node": N3, "role": "incidence:3"}]}
LOOKUP = {"op": "lookup", "relation": "hyperedge:1", "target_role": "incidence:3",
          "cues": [{"role": "incidence:1", "node": N1}, {"role": "incidence:2", "node": N2}]}


def learn(seed: int, **extra) -> dict:
    r = random.Random(seed)
    x = [[round(r.uniform(-0.4, 0.4), 6) for _ in range(6)] for _ in range(4)]
    y = [round(r.uniform(-0.5, 0.5), 6) for _ in range(4)]
    return dict({"op": "learn", "x": x, "y": y, "schedule": [[2, 3], [2, 5]], "rate": 0.002,
                 "grant": "qualification: numeric experience"}, **extra)


def evidence(tick: int, source: int = 1, **extra) -> dict:
    item = dict(EDGE, provenance=prov(tick), authority=source, polarity="ASSERT", tick=tick)
    return dict({"op": "evidence", "items": [item], "grant": "qualification: source", "sources": [source]}, **extra)


def composed_script() -> list[dict]:
    """The composed session: every lifecycle step and the in-memory refusals (directive, section 9)."""
    source = {"op": "source_found", "label": "qualification source 1", "source_authority": 1, "per_turn": 8,
              "scope_label": "qualification scope"}
    second = dict(source, label="qualification source 2", source_authority=2)
    bad_role = dict(evidence(90), items=[dict(evidence(90)["items"][0], participants=[
        {"node": N1, "role": "incidence:1"}, {"node": N1, "role": "incidence:1"}])])
    bad_polarity = dict(evidence(91), items=[dict(evidence(91)["items"][0], polarity="MAYBE")])
    return [{"op": "status"}, {"op": "live"}, source, second,
            *(learn(s) for s in range(5)), evidence(1), {"op": "query", "x": X}, LOOKUP,
            {"op": "source_lifecycle", "source_authority": 2, "to": "DORMANT"},
            evidence(2, source=2, expect="RUNTIME_OBSERVATION_REFUSED"),                  # a dormant source
            {"op": "source_version", "source_authority": 1, "expected": 5, "version": 6,
             "expect": "ECS_C_SOURCE_STALE"},                                            # a stale coordinate
            dict(bad_role, expect="OBSERVATION_UNSUPPORTED"),                             # invalid typed relationships
            dict(bad_polarity, expect="OBSERVATION_INVALID"),
            learn(77, schedule=[[4, 1048576]], expect="COGNITION_FUEL_EXCEEDED"),         # refused before anything
            learn(78, rate=-1, expect="LEARNING_RATE"),
            {"op": "publish", "expect": "OPERATION_RETIRED"}, {"op": "reconcile", "expect": "OPERATION_RETIRED"},
            {"op": "candidates", "expect": "OPERATION_RETIRED"},
            {"op": "counters"}, {"op": "authority"}, {"op": "live"},
            {"op": "close"}, LOOKUP, {"op": "live"}]


def differential_script() -> list[dict]:
    """The operations the host and the Python reference both offer, for the independent differential."""
    source = {"op": "source_found", "label": "qualification source 1", "source_authority": 1, "per_turn": 8,
              "scope_label": "qualification scope"}
    return [{"op": "live"}, source, {"op": "query", "x": X}, *(learn(s) for s in range(5)), {"op": "query", "x": X},
            evidence(1), evidence(2), LOOKUP, learn(77, schedule=[[4, 1048576]], expect="COGNITION_FUEL_EXCEEDED"),
            learn(78, rate=-1, expect="LEARNING_RATE"), {"op": "publish", "expect": "OPERATION_RETIRED"},
            {"op": "live"}, {"op": "close"}, LOOKUP, {"op": "live"}]


def long_script(n: int, seed: int) -> list[dict]:
    r = random.Random(seed)
    ops = [{"op": "source_found", "label": "long source", "source_authority": 1, "per_turn": 8,
            "scope_label": "long"}]
    for i in range(n):
        k = r.random()
        if k < 0.35:
            ops.append(learn(1000 + i))
        elif k < 0.6:
            ops.append({"op": "query", "x": learn(5000 + i)["x"]})
        elif k < 0.85:
            ops.append(evidence(10 + i))
        else:
            ops.append(LOOKUP)
    ops.append({"op": "live"})
    return ops


def hgram_file(path: Path, records: dict[int, tuple[int, int]], mib: int = 1) -> None:
    """A retained H-gram in the documented format (the harness is its author; Elpis writes none): records maps a
    key byte to (generation, first value byte), each written as one valid copy of its bucket."""
    def fnv(data: bytes) -> int:
        h = 14695981039346656037
        for b in data:
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    size = mib << 20
    header = bytearray(4096)
    header[0:8] = b"ELPHGRA1"
    header[8:12] = (1).to_bytes(4, "little")
    header[12:16] = (4096).to_bytes(4, "little")
    header[16:24] = size.to_bytes(8, "little")
    header[24:32] = (size - 4096).to_bytes(8, "little")
    header[32:36] = (256).to_bytes(4, "little")
    header[40:48] = ((size - 4096) // 256).to_bytes(8, "little")
    header[48:56] = fnv(bytes(header[:48])).to_bytes(8, "little")
    data = bytearray(size)
    data[:4096] = header
    buckets = ((size - 4096) // 256) // 2
    for key_byte, (generation, value0) in records.items():
        key = bytes([key_byte]) * 32
        bucket = fnv(key) % buckets
        rec = bytearray(256)
        rec[0:8] = b"ELPHAS01"
        rec[8:12] = (1).to_bytes(4, "little")
        rec[12:16] = (1).to_bytes(4, "little")
        rec[16:24] = bucket.to_bytes(8, "little")
        rec[24:32] = generation.to_bytes(8, "little")
        rec[32:64] = key
        rec[64:192] = bytes([value0]) * 128
        rec[248:256] = fnv(bytes(rec[:248])).to_bytes(8, "little")
        off = 4096 + bucket * 512
        data[off:off + 256] = rec
    path.write_bytes(bytes(data))
    with open(path, "rb+") as f:                       # fully allocated, as the probe requires
        os.posix_fallocate(f.fileno(), 0, size)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args(argv)
    repo, build, out = args.repo.resolve(), args.build.resolve(), args.out.resolve()

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise Fail("Linux x86-64 is required")
    if os.geteuid() != 0:
        raise Fail("root is required to build the read-only mount namespace and to drop to uid 65534")
    for tool in ("strace", "unshare", "setpriv", "prlimit", "nm", "cc", "git", "mount", "awk"):
        if shutil.which(tool) is None:
            raise Fail(f"{tool} is not available")
    abi = V0.landlock_abi()
    if abi < 5:
        raise Fail(f"Landlock ABI >= 5 is required (found {abi})")
    native = build / "native"
    host, testing_host = native / "runtime/elpis", native / "runtime/elpis_testing"
    launcher = native / "runtime/elpis_autonomous_run"
    libraries = {name: native / rel for name, rel in (
        ("libelpis_ecsg_math.so", "ECS_G/libelpis_ecsg_math.so"), ("libelpis_ecsg_k1.so", "ECS_G/libelpis_ecsg_k1.so"),
        ("libelpis_ecs_memory.so", "ECS_C/libelpis_ecs_memory.so"),
        ("libelpis_ecs_composite.so", "runtime/composite/libelpis_ecs_composite.so"),
        ("libelpis_hacf_observation.so", "runtime/libelpis_hacf_observation.so"),
        ("libelpis_runtime.so", "runtime/libelpis_runtime.so"))}
    loaded = [host, *libraries.values(), native / "ECS_G/libelpis_ecsg_k1_fms.so", native / "ECS_G/libelpis_ecsg_fms.so",
              native / "ECS_C/session/libelpis_ecs_c_session.so", native / "substrate/libelpis_fms_file_assets.so",
              native / "structure/libelpis_retrieval_bridge.so", native / "pipeline/libelpis_ingress_bridge.so",
              native / "substrate/libelpis_execution.so"]
    suites = {"runtime_core.session": native / "runtime/test_runtime_session",
              "runtime_core.composite": native / "runtime/test_runtime_composite",
              "runtime_core.abi": native / "runtime/test_runtime_abi",
              "runtime_core.lifecycle": native / "runtime/test_runtime_lifecycle",
              "runtime_core.hot_io": native / "runtime/test_runtime_hot_io",
              "ecs_c_session.abi": native / "ECS_C/session/test_ecs_c_session_abi",
              "ecs_c_session.relational": native / "ECS_C/test_ecs_c_relational_session"}
    fms_suite = native / "substrate/test_fms_volatile_cold"
    for path in [testing_host, launcher, fms_suite, *loaded, *suites.values()]:
        if not path.is_file():
            raise Fail(f"{path.relative_to(build)} is not built")

    # The work tree: every input made here, before any traced run, readable by uid 65534.
    work = build / "research/volatile_runtime_v1/work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for d in (build / "research", build / "research/volatile_runtime_v1", work):
        d.chmod(0o755)
    clean = V0.Sanitizer(repo, build)
    tool_dir = build / "research/volatile_runtime_v1"
    peak_rss = tool_dir / "peak_rss"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(repo / "research/volatile_runtime_v0/peak_rss.c"), "-o", str(peak_rss)], check=True)
    hgram_read = tool_dir / "hgram_read"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I",
                    str(repo / "native/structure/hacf/include"), str(HERE / "hgram_read.c"),
                    str(native / "structure/libelpis_cascade.a"), "-o", str(hgram_read)], check=True)
    for tool in (peak_rss, hgram_read):
        tool.chmod(0o755)
    scripts = work / "scripts"
    scripts.mkdir()
    composed = composed_script()
    (scripts / "composed.jsonl").write_text("\n".join(json.dumps(o) for o in composed) + "\n")
    (scripts / "differential.jsonl").write_text("\n".join(json.dumps(o) for o in differential_script()) + "\n")
    (scripts / "long_300.jsonl").write_text("\n".join(json.dumps(o) for o in long_script(300, 11)) + "\n")
    (scripts / "long_3000.jsonl").write_text("\n".join(json.dumps(o) for o in long_script(3000, 11)) + "\n")
    (scripts / "restart.jsonl").write_text("\n".join(json.dumps(o) for o in (
        {"op": "live"}, learn(5), {"op": "query", "x": X}, {"op": "close"}, {"op": "live"})) + "\n")
    (scripts / "probes.jsonl").write_text("")   # filled below, once the probe targets exist
    cold = work / "cold"
    cold.mkdir()
    asset = cold / "cold.asset"
    page, pages = 4096, 16
    size = pages * page - 333
    asset.write_bytes(bytes(((i * 167 + (i >> 11) + 3) & 0xFF) for i in range(size)))
    asset.chmod(0o444)
    hgram_dir = work / "hgram"
    hgram_dir.mkdir()
    hgram = hgram_dir / "hgram.bin"
    hgram_records = {0x17: (2, 0x72), 0x42: (7, 0x45)}
    hgram_file(hgram, hgram_records)
    os.chown(hgram_dir, NOBODY, NOBODY)
    os.chown(hgram, NOBODY, NOBODY)
    hgram.chmod(0o400)
    swaps = work / "swaps"
    swaps.write_text("Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n"
                     "/swapfile                               file\t\t4194300\t\t0\t\t-2\n")
    missing = work / "missing-library"
    drifted = work / "drifted-library"
    for d, skip, flip in ((missing, "libelpis_ecs_memory.so", None), (drifted, None, "libelpis_ecs_composite.so")):
        d.mkdir()
        for name, path in libraries.items():
            if name == skip:
                continue
            data = bytearray(path.read_bytes())
            if name == flip:
                data[-1] ^= 1
            (d / name).write_bytes(bytes(data))
            (d / name).chmod(0o444)
    probe_dir = work / "probe"
    probe_dir.mkdir()
    (probe_dir / "existing").write_text("existing")
    (probe_dir / "existing").chmod(0o666)
    probe_dir.chmod(0o777)
    probes = [{"op": "testing_write_probe", "kind": kind, "path": str(probe_dir / name), **extra}
              for kind, name, extra in (("create", "created", {}), ("write", "existing", {}),
                                        ("extend", "existing", {"offset": 4096}), ("remove", "existing", {}),
                                        ("rename", "existing", {}), ("mkdir", "made", {}))]
    (scripts / "probes.jsonl").write_text("\n".join(json.dumps(o) for o in probes) + "\n")
    script_bytes = {p.name: p.stat().st_size for p in scripts.glob("*.jsonl")}
    for d in work.rglob("*"):
        if d.is_dir() and d not in (hgram_dir, probe_dir):
            d.chmod(0o755)
    sink = work / "stdout.sink"
    sink.write_text("")
    sink.chmod(0o666)
    rw_target = work / "inherited.rw"
    rw_target.write_text("inherited")
    rw_target.chmod(0o666)
    inputs = sorted([host, testing_host, launcher, fms_suite, *loaded, *suites.values(), asset, hgram,
                     *scripts.glob("*.jsonl"), *(p for p in missing.iterdir()), *(p for p in drifted.iterdir()),
                     peak_rss, hgram_read], key=str)
    hashes_before = {str(p): V0.sha256(p) for p in inputs}
    tree_before = tree(work)

    def pin(binary: Path, source: list[str]) -> str:
        run = subprocess.run([*SETPRIV, str(binary), *source, "catalog"], capture_output=True, text=True, cwd="/",
                             env={"PATH": "/usr/bin:/bin"}, timeout=300)
        out = lines(run.stdout)
        if run.returncode != 0 or not out or not out[0].get("ok"):
            raise Fail(f"catalog of {binary.name}: {run.stdout[-300:]} {run.stderr[-300:]}")
        return out[0]["catalog_sha256"]

    source = ["--native-build", str(build)]
    host_pin, testing_pin = pin(host, source), pin(testing_host, source)
    host_cmd = [str(host), *source, "--catalog-sha256", host_pin, "--seed", "7"]
    py_env = {"PYTHONPATH": str(repo / "src"), "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C"}
    py_catalog = subprocess.run([sys.executable, "-B", "-m", "elpis.runtime.cli", *source, "catalog"],
                                capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **py_env}, cwd="/",
                                timeout=300)
    py_pin = lines(py_catalog.stdout)[0]["catalog_sha256"]

    def session(script: str, cmd=None) -> list[str]:
        return [*(cmd or host_cmd), "session", "--script", str(scripts / script)]

    # ---- legitimate composed runs --------------------------------------------------------------------------------
    rw_fd = os.open(asset, os.O_RDWR)       # the harness's, inherited by the FMS suite; never written through
    os.set_inheritable(rw_fd, True)
    paths: dict[str, dict] = {
        "host_session": {"argv": session("composed.jsonl"), "unprivileged": True},
        "host_long": {"argv": session("long_3000.jsonl"), "unprivileged": True},
        "host_long_short": {"argv": session("long_300.jsonl"), "unprivileged": True, "plain_only": True},
        "host_differential": {"argv": session("differential.jsonl"), "unprivileged": True, "plain_only": True},
        "python_cli": {"argv": [sys.executable, "-B", "-m", "elpis.runtime.cli", *source, "--catalog-sha256", py_pin,
                                "--seed", "7", "session", "--script", str(scripts / "differential.jsonl")],
                       "env": py_env},
        "python_fms_hacf": {"argv": [sys.executable, "-I", "-B", str(HERE / "python_fms_hacf.py"), str(repo),
                                     str(build)], "env": {"LC_ALL": "C"}},
        # The writable descriptor case 10 must refuse is the harness's (inherited): the suite opens none. Under
        # confinement the launcher closes it first, and the suite then has none to offer.
        "fms_cold": {"argv": [str(fms_suite), "--asset", str(asset), "--rw-fd", str(rw_fd)], "fds": (rw_fd,)},
        "hgram_read": {"argv": [str(hgram_read), str(hgram), "17", "42", "99"], "unprivileged": True},
        **{name: {"argv": [str(path)]} for name, path in suites.items()},
    }
    runs: dict[str, dict] = {}
    for name, spec in paths.items():
        prefix = SETPRIV if spec.get("unprivileged") else []
        entry = {"argv": [clean(a) for a in spec["argv"]], "sha256": V0.sha256(Path(spec["argv"][0])),
                 "plain": plain([*prefix, *spec["argv"]], peak_rss, args.timeout, spec.get("env"),
                                spec.get("fds", ()))}
        entry["plain_output"] = lines(entry["plain"].pop("stdout"))
        if not spec.get("plain_only"):
            observed_mode = "uid_only" if spec.get("unprivileged") else "observed"
            for label, mode in (("observed", observed_mode), ("confined", "confined")):
                run, log, seconds = traced(spec["argv"], mode, launcher, args.timeout, env=spec.get("env"),
                                           extra_fds=spec.get("fds", ()))
                events = V0.write_events(log)
                entry[label] = {"mode": mode, "exit": run.returncode, "seconds": round(seconds, 3),
                                "traced_calls": V0.syscall_counts(log), "write_events": [clean(e) for e in events],
                                "stdout_tail": V0.tail(run.stdout, clean, 3), "stderr_tail": V0.tail(run.stderr, clean, 3)}
                entry[label]["output"] = lines(run.stdout) if name in ("host_session", "python_cli", "hgram_read",
                                                                       "python_fms_hacf") else None
        runs[name] = entry
    os.close(rw_fd)

    # Repeated startup/shutdown: plain processes (and the traced ones above); each starts at the initial identity.
    restarts = []
    for _ in range(25):
        p = plain([*SETPRIV, *session("restart.jsonl")], peak_rss, args.timeout)
        said = lines(p["stdout"])
        restarts.append({"exit": p["exit"], "peak_rss_kib": p["peak_rss_kib"],
                         "initial_live": said[0].get("live", {}).get("identity") if said else None,
                         "after_close_live": said[-1].get("live", {}).get("identity") if said else None})
    for _ in range(3):
        run, log, _ = traced(session("restart.jsonl"), "uid_only", launcher, args.timeout)
        restarts.append({"exit": run.returncode, "traced": True, "write_events": [clean(e) for e in V0.write_events(log)],
                         "initial_live": (lines(run.stdout) or [{}])[0].get("live", {}).get("identity")})
    runs["host_restarts"] = {"processes": restarts}

    # ---- adversarial runs -----------------------------------------------------------------------------------------
    adversarial: dict[str, dict] = {}

    def adverse(name: str, argv: list[str], mode: str = "uid_only", **kw) -> None:
        run, log, seconds = traced(argv, mode, launcher, args.timeout, **kw)
        adversarial[name] = {"mode": mode, "exit": run.returncode, "write_events": [clean(e) for e in V0.write_events(log)],
                             "output": lines(run.stdout or ""), "stderr_tail": V0.tail(run.stderr or "", clean, 2)}

    with open(sink, "w") as f:
        adverse("storage_stdout", session("composed.jsonl"), stdout=f)
    adversarial["storage_stdout"]["sink_bytes"] = sink.stat().st_size
    fd = os.open(rw_target, os.O_RDWR)
    try:
        os.set_inheritable(fd, True)
        adverse("inherited_writable_descriptor", [*host_cmd, "status"], extra_fds=(fd,))
    finally:
        os.close(fd)
    adverse("swap_active", [str(testing_host), *source, "--catalog-sha256", testing_pin, "--seed", "7", "status"],
            env={"ELPIS_TESTING_PROC_SWAPS": str(swaps)})
    # Raised as root before the drop to uid 65534 (a confined shell would already have lowered the hard limit).
    adverse("core_unlimited", [shutil.which("prlimit"), "--core=unlimited:unlimited", "--", *SETPRIV, *host_cmd,
                               "status"], mode="observed")
    for mib in (6, 10, 14):                          # failed allocations: from a refused start to a constrained run
        adverse(f"address_space_{mib}mib", [shutil.which("prlimit"), f"--as={mib << 20}:{mib << 20}", "--",
                                            *session("composed.jsonl")])
    adverse("privileged_start", [*host_cmd, "status"], mode="observed")
    adverse("missing_library", [str(host), "--libraries", str(missing), "--catalog-sha256", host_pin, "status"])
    adverse("drifted_library", [str(host), "--libraries", str(drifted), "--catalog-sha256", host_pin, "status"])
    for retired in (["init"], ["publish"], ["reconcile"], ["--home", str(work / "home"), "status"]):
        adverse(f"legacy:{' '.join(r for r in retired if not r.startswith('/'))}", [str(host), *source, *retired])

    # ---- negative controls (apart) ---------------------------------------------------------------------------------
    controls: dict[str, dict] = {}
    control_dir = work / "control"
    control_dir.mkdir()
    os.chown(control_dir, NOBODY, NOBODY)
    target = control_dir / "probe"
    observed, log, _ = traced(["/bin/sh", "-c", f'echo x > "{target}.a" && mv "{target}.a" "{target}.b" && sync && '
                                                f'rm "{target}.b"'], "observed", launcher, args.timeout)
    controls["root_unconfined"] = {"exit": observed.returncode, "write_events": [clean(e) for e in V0.write_events(log)]}
    for mode in ("uid_only", "landlock_only", "confined"):
        name = mode if mode != "confined" else "full_confinement"
        run, log, _ = traced(["/usr/bin/touch", str(control_dir / f"{name}.file")], mode, launcher, args.timeout)
        controls[name] = {"exit": run.returncode, "write_events": [clean(e) for e in V0.write_events(log)],
                          "file_created": (control_dir / f"{name}.file").exists()}
    run, log, _ = traced([str(testing_host), *source, "--catalog-sha256", testing_pin, "--seed", "7", "session",
                          "--script", str(scripts / "probes.jsonl")], "confined", launcher, args.timeout)
    controls["testing_host_write_probes"] = {"exit": run.returncode, "output": lines(run.stdout),
                                             "write_events": [clean(e) for e in V0.write_events(log)],
                                             "probe_dir": sorted(p.name for p in probe_dir.iterdir())}
    shutil.rmtree(control_dir)

    # ---- static checks ---------------------------------------------------------------------------------------------
    write_class = {"rename", "renameat", "renameat2", "unlink", "unlinkat", "mkdir", "mkdirat", "rmdir", "ftruncate",
                   "ftruncate64", "truncate", "truncate64", "fallocate", "fallocate64", "posix_fallocate",
                   "posix_fallocate64", "fsync", "fdatasync", "sync", "syncfs", "sync_file_range", "pwrite", "pwrite64",
                   "pwritev", "pwritev2", "link", "linkat", "symlink", "symlinkat", "creat", "creat64", "msync",
                   "mkstemp", "mkstemp64", "mkstemps", "mkostemp", "mkdtemp", "tmpfile", "tmpfile64", "chmod", "fchmod",
                   "fchmodat", "chown", "fchown", "lchown", "fchownat", "utimensat", "futimens", "setxattr",
                   "fsetxattr", "lsetxattr", "copy_file_range", "sendfile", "splice", "mknod", "mknodat", "remove"}
    imports = {}
    for path in loaded:
        table = subprocess.run(["nm", "-D", "--undefined-only", str(path)], capture_output=True, text=True,
                               check=True).stdout
        names = {line.split()[-1].split("@")[0] for line in table.splitlines() if line.split()}
        imports[str(path.relative_to(build))] = sorted(names & write_class)
    hashes_after = {str(p): V0.sha256(p) for p in inputs}
    tree_after = tree(work)

    # ---- verdict ------------------------------------------------------------------------------------------------
    problems = []
    for name, entry in runs.items():
        if name == "host_restarts":
            continue
        if entry["plain"]["exit"] != 0:
            problems.append(f"{name} plain exit {entry['plain']['exit']}")
        for label in ("observed", "confined"):
            if label in entry:
                if entry[label]["exit"] != 0:
                    problems.append(f"{name} {label} exit {entry[label]['exit']}")
                if entry[label]["write_events"]:
                    problems.append(f"{name} {label}: {len(entry[label]['write_events'])} write-class events")
    # The composed session: every operation as expected; LOOKUP before close, nothing after; a fresh identity.
    hs = runs["host_session"]["plain_output"]
    if not hs or not all(o.get("met", o.get("ok")) for o in hs):
        problems.append("host_session: an operation was neither ok nor refused as expected")
    else:
        status, first_live = hs[0], hs[1]
        initial = status.get("lineage", {}).get("initial_identity")
        lookups = [o for o in hs if o.get("op") == "lookup"]
        if not (lookups[0].get("found") and lookups[0].get("answer") == N3 and not lookups[-1].get("found")):
            problems.append("host_session: LOOKUP did not answer in the session, or answered after close")
        if not (first_live.get("live", {}).get("identity") == initial == hs[-1].get("live", {}).get("identity")):
            problems.append("host_session: the session after close does not start at the initial identity")
        containment = status.get("containment", {})
        if not (containment.get("restricted") and containment.get("write_grants") == 0 and
                containment.get("core_limit") == 0 and containment.get("dumpable") is False):
            problems.append(f"host_session: containment not established: {containment}")
        if status.get("host", {}).get("persistence") != "NONE":
            problems.append("host_session: the host does not report persistence NONE")
    # The independent differential: the Python reference answers as the host does.
    py, hd = runs["python_cli"]["plain_output"], runs["host_differential"]["plain_output"]
    if not hd or not all(o.get("met", o.get("ok")) for o in hd) or not all(o.get("met", o.get("ok")) for o in py):
        problems.append("the differential script was not answered as expected by the host or the Python reference")
    if len(py) != len(hd):
        problems.append(f"python_cli: {len(py)} answers for the host's {len(hd)}")
    else:
        for n, p in zip(hd, py):
            for key in ("ok", "code", "identity", "values_sha256", "answer", "found", "epoch", "generation"):
                if n.get(key) != p.get(key) and n.get("op") not in ("status", "counters", "authority", "catalog"):
                    problems.append(f"python_cli differs at {n.get('op')}: {key} {n.get(key)!r} != {p.get(key)!r}")
                    break
    # Boundedness: a ten times longer randomized session. The host holds its whole script in memory, so its peak may
    # grow by the script's extra bytes; what remains (the session's state) must stay within 512 KiB.
    long_peak, short_peak = runs["host_long"]["plain"]["peak_rss_kib"], runs["host_long_short"]["plain"]["peak_rss_kib"]
    script_delta_kib = (script_bytes["long_3000.jsonl"] - script_bytes["long_300.jsonl"]) // 1024
    state_delta_kib = long_peak - short_peak - script_delta_kib
    if state_delta_kib > 512:
        problems.append(f"host_long: beyond its script, peak RSS grew {state_delta_kib} KiB over a 10x longer session")
    initial_ids = {r.get("initial_live") for r in restarts}
    if any(r["exit"] != 0 for r in restarts) or len(initial_ids) != 1 or any(r.get("write_events") for r in restarts):
        problems.append("host_restarts: a restart failed, wrote, or did not start at the one initial identity")
    if any(r.get("after_close_live") not in (None, *initial_ids) for r in restarts):
        problems.append("host_restarts: a closed session was not discarded")
    # H-gram reads: exactly the records the harness made.
    hg = runs["hgram_read"]["plain"]
    hg_text = subprocess.run([*SETPRIV, str(hgram_read), str(hgram), "17", "42", "99"], capture_output=True, text=True,
                             cwd="/", env={"PATH": "/usr/bin:/bin"}).stdout
    want = ["probe=1 mib=1", "key=17 status=0 generation=2 value0=114", "key=42 status=0 generation=7 value0=69",
            "key=99 status=1 generation=0 value0=0"]
    if hg["exit"] != 0 or hg_text.strip().splitlines() != want:
        problems.append(f"hgram_read: {hg_text.strip().splitlines()} != {want}")
    # Adversarial expectations.
    expect_exit = {"storage_stdout": 4, "inherited_writable_descriptor": 0, "swap_active": 0, "core_unlimited": 0,
                   "privileged_start": 4, "missing_library": 4, "drifted_library": 4}
    for name, code in expect_exit.items():
        if adversarial[name]["exit"] != code:
            problems.append(f"adversarial {name}: exit {adversarial[name]['exit']}, expected {code}")
    for name, a in adversarial.items():
        if a["write_events"]:
            problems.append(f"adversarial {name}: {len(a['write_events'])} write-class events")
        if name.startswith("legacy:") and (a["exit"] != 2 or (a["output"] or [{}])[0].get("code") != "OPERATION_RETIRED"):
            problems.append(f"adversarial {name}: not refused OPERATION_RETIRED")
    if adversarial["storage_stdout"]["sink_bytes"] != 0:
        problems.append("adversarial storage_stdout: bytes were written onto the storage standard output")
    constrained = [adversarial[f"address_space_{mib}mib"] for mib in (6, 10, 14)]
    if not any(a["exit"] != 0 for a in constrained):
        problems.append("adversarial address_space: no limit made an allocation fail (the case proved nothing)")
    if any(a["exit"] < 0 for a in constrained):
        problems.append("adversarial address_space: the host died of a signal instead of refusing")
    swap = (adversarial["swap_active"]["output"] or [{}])[0].get("containment", {})
    if not (swap.get("swap_areas") == 1 and swap.get("memory_locked") is True):
        problems.append(f"adversarial swap_active: memory not locked under active swap: {swap}")
    core = (adversarial["core_unlimited"]["output"] or [{}])[0].get("containment", {})
    if not (core.get("core_limit") == 0 and core.get("dumpable") is False):
        problems.append(f"adversarial core_unlimited: core dumps not disabled: {core}")
    # Controls: the detector sees real writes; each containment layer refuses on its own; the probes are refused.
    root = controls["root_unconfined"]
    names = {V0.CALL.match(e).group(2) for e in root["write_events"] if V0.CALL.match(e)}
    if not (root["exit"] == 0 and {"rename", "renameat", "renameat2"} & names and {"unlink", "unlinkat"} & names
            and "sync" in names):
        problems.append("control root_unconfined: the detector missed a real write")
    if not (controls["uid_only"]["exit"] == 0 and controls["uid_only"]["file_created"]):
        problems.append("control uid_only: the uid layer alone did not write as expected")
    for name, errno_name in (("landlock_only", "EACCES"), ("full_confinement", "EROFS")):
        c = controls[name]
        if c["exit"] == 0 or c["file_created"] or not any(errno_name in e for e in c["write_events"]):
            problems.append(f"control {name}: expected a refused, recorded attempt ({errno_name})")
    probe = controls["testing_host_write_probes"]
    if not (probe["write_events"] and all(o.get("errno", 0) != 0 for o in probe["output"] if
                                          o.get("op") == "testing_write_probe")
            and probe["probe_dir"] == ["existing"]):
        problems.append("control testing_host_write_probes: a deliberate write was not refused or not detected")
    for rel, names in imports.items():
        if names:
            problems.append(f"{rel} imports {names}")
    changed = sorted(clean(p) for p in hashes_before if hashes_before[p] != hashes_after.get(p))
    if changed:
        problems.append(f"inputs changed: {changed}")
    if tree_after != tree_before:
        problems.append(f"the work tree changed: {sorted(set(tree_after) ^ set(tree_before))}")

    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "native", "src",
                            "research/volatile_runtime_v1", "research/volatile_runtime_v0"],
                           capture_output=True, text=True, check=True).stdout.strip()
    environment = {
        "kernel": platform.release(), "machine": platform.machine(), "landlock_abi": abi,
        "strace": V0.first_line(["strace", "-V"]), "cc": V0.first_line(["cc", "--version"]),
        "rustc": V0.first_line(["rustc", "-V"]), "python": platform.python_version(),
        "swap_total_kib": int(re.search(r"SwapTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)),
        "core_pattern": Path("/proc/sys/kernel/core_pattern").read_text().strip(),
        "root_fs": V0.mount_of("/"), "build_fs": V0.mount_of(str(build)),
    }
    document = {
        "schema": SCHEMA,
        "classification": "QUALIFICATION evidence for closure condition 11: the composed runtime (native host, "
                          "RuntimeCore, the Python reference, FMS COLD/WARM/HOT, HACF, the H-gram read path)",
        "source_commit": head, "source_tree_clean": not dirty,
        "environment": environment, "traced_syscalls": list(V0.TRACED),
        "catalog_pins": {"host": host_pin, "testing_host": testing_pin, "python_reference": py_pin},
        "runs": runs, "adversarial": adversarial, "controls": controls,
        "static": {"write_class_imports": imports, "inputs_unchanged": not changed,
                   "input_sha256": {clean(k): v for k, v in hashes_after.items()}},
        "boundedness": {"host_long_peak_rss_kib": long_peak, "host_long_short_peak_rss_kib": short_peak,
                        "script_delta_kib": script_delta_kib, "state_delta_kib": state_delta_kib,
                        "composite_envelope_bytes": (runs["host_session"]["plain_output"] or [{}])[0].get(
                            "lineage", {}).get("envelope_bytes"),
                        "restart_peak_rss_kib": sorted({r["peak_rss_kib"] for r in restarts if "peak_rss_kib" in r})},
        "verdict": "ZERO_WRITE_PASS" if not problems else "FAIL", "problems": problems,
    }
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    text = text.replace(str(work), "<work>")
    text = clean(text)
    assert str(repo) not in text and str(build) not in text, "a host-local path leaked into the results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "zero_write_v1.json").write_text(text)
    env_text = "\n".join(f"{k}: {v}" for k, v in environment.items()) + \
        f"\nsource_commit: {head} (tree clean for native/, src/ and the harnesses: {not dirty})\n" + \
        f"run: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n"
    (out / "ENVIRONMENT.txt").write_text(env_text)
    sources = sorted(p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".c", ".md"))
    manifest = {"schema": "elpis.volatile-runtime-v1.manifest.v1", "source_commit": head,
                "sources": {str(p.relative_to(repo)): V0.sha256(p) for p in [*sources, HERE.parent /
                                                                            "volatile_runtime_v0/zero_write.py"]},
                "results": {name: V0.sha256(out / name) for name in ("ENVIRONMENT.txt", "zero_write_v1.json")}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    shutil.rmtree(work)
    print(f"{document['verdict']}: {len(runs)} composed paths, {len(adversarial)} adversarial runs, "
          f"{len(controls)} controls; problems: {problems or 'none'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
