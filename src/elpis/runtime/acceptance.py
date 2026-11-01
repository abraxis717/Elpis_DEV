"""System acceptance of an installed Elpis: ``python -m elpis.runtime.acceptance --native-build BUILD --work DIR``.

It exercises the supported product, not the test suite: every operator step runs the installed CLI
(``python -m elpis.runtime.cli``) as a separate process over the build's libraries, read-only and pinned, and the
adversarial checks use only the public runtime API and the installation's own admitted libraries. No test fixture or
research module is imported. The result is one machine-readable JSON report (``--report``); the exit status is 0 only
when every check passed. ``DIR`` (which must not exist or be empty) holds only what this harness itself writes: its
tamper copies and the ABI stub. Elpis writes nothing (docs/ELPIS_MISSION.md, Amendment 3), which the harness checks.

Positive sequence (one lineage, several processes, RuntimeCore ABI v8): the catalog measured read-only and pinned;
pinned identity of ABI and libraries; a session anchored at the lineage's initial identity at a fresh genesis; QUERY;
an authorized LEARN (a live transition, volatile); validated HACF evidence recorded by the observe-only turn (the K1
component unchanged) and LOOKUP; a contextual (WARM) view that changes no ECS state; ``close`` discarding the session
and the next one starting at the initial identity; a new process starting there too (nothing resumed); a killed
process leaving nothing; and no file created or changed by any Elpis process.

Negative sequence: no codec authority, QUERY-to-LEARN escalation, unauthorized and unadmitted observations, an
invalid source identity, an unsupported memory capability, stale context, a wrong runtime ABI, pin drift, budget
exhaustion, a K1 FMS lineage at the WARM budget (refused, never spilled), and every retired persistent command: each
must be refused with its exact code, with nothing changed.

Mechanics only: the stimuli and the evidence are acceptance fixtures (TRAINING=NONE SEMANTICS=NONE); passing
establishes that the installed system's interfaces and laws hold here, not any cognitive capability.
"""
from __future__ import annotations

import argparse
from array import array
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

REPORT_SCHEMA = "elpis.acceptance-report.v2"
_LIBS = ("native/runtime/libelpis_runtime.so", "native/runtime/libelpis_hacf_observation.so",
         "native/ECS_G/libelpis_ecsg_math.so", "native/ECS_G/libelpis_ecsg_k1.so",
         "native/ECS_G/libelpis_ecsg_k1_fms.so", "native/ECS_C/libelpis_ecs_memory.so",
         "native/runtime/composite/libelpis_ecs_composite.so")


def node(label: str) -> str:
    """An acceptance fixture node identity (SHA-256 of a labelled domain string; not a structure identity scheme)."""
    return hashlib.sha256(b"elpis.acceptance.fixture-node.v1:" + label.encode()).hexdigest()


def provenance(label: str) -> str:
    return hashlib.sha256(b"elpis.acceptance.fixture-provenance.v1:" + label.encode()).hexdigest()


X = [[((i * 7 + j * 3) % 11 - 5) / 16.0 for j in range(6)] for i in range(4)]
Y = [0.1, -0.2, 0.05, 0.3]


def learn(grant="acceptance operator: bounded fixture experience", **extra):
    return dict({"op": "learn", "x": X, "y": Y, "schedule": [[2, 3], [2, 5]], "rate": 0.002, "grant": grant}, **extra)


EDGE = {"relation": "hyperedge:1", "participants": [{"node": node("alice"), "role": "incidence:1"},
                                                    {"node": node("acme"), "role": "incidence:2"},
                                                    {"node": node("engineer"), "role": "incidence:3"}]}


def evidence(label, source=1, polarity="ASSERT", tick=1, grant="acceptance operator: fixture sources", **extra):
    item = dict(EDGE, provenance=provenance(label), authority=source, polarity=polarity, tick=tick)
    return dict({"op": "evidence", "items": [item], "grant": grant, "sources": [1, 2]}, **extra)


# The operator founds the two fixture sources the evidence names (ACTIVE in the lineage's ECS_C authority).
SOURCES = [{"op": "source_found", "label": f"acceptance source {n}", "source_authority": n, "per_turn": 8,
            "scope_label": "acceptance fixture scope"} for n in (1, 2)]

LOOKUP = {"op": "lookup", "relation": "hyperedge:1", "target_role": "incidence:3",
          "cues": [{"role": "incidence:1", "node": node("alice")}, {"role": "incidence:2", "node": node("acme")}]}


class Acceptance:
    def __init__(self, native_build: Path, work: Path, source_commit: str | None):
        self.build, self.work, self.source_commit = native_build, work, source_commit
        self.checks: list[dict] = []
        self.pin: str | None = None

    # -- recording ------------------------------------------------------------------------------------------------
    def check(self, cid: str, what: str, ok: bool, **evidence) -> bool:
        self.checks.append({"id": cid, "check": what, "ok": bool(ok), "evidence": evidence})
        return bool(ok)

    # -- the installed CLI, as a separate process ------------------------------------------------------------------
    def cli(self, *args, script=None, source: Path | None = None, pin: str | None = None) -> tuple[int, list]:
        cmd = [sys.executable, "-m", "elpis.runtime.cli", "--native-build", str(source or self.build)]
        if pin or self.pin:
            cmd += ["--catalog-sha256", pin or self.pin]
        cmd += ["--seed", "11", *args]
        if script is not None:
            cmd += ["session", "--script", "-"]
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        run = subprocess.run(cmd, input=json.dumps(script) if script is not None else None, capture_output=True,
                             text=True, env=env, cwd=str(self.work), timeout=600)
        out = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
        if run.returncode not in (0, 2, 3, 4):
            raise RuntimeError(f"cli {args} crashed ({run.returncode}): {run.stderr[-2000:]}")
        return run.returncode, out

    def tree(self) -> dict:
        """Every file under the work directory and every library of the build, with its SHA-256."""
        files = {str(p.relative_to(self.work)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(self.work.rglob("*")) if p.is_file()}
        files.update({r: hashlib.sha256((self.build / r).read_bytes()).hexdigest() for r in _LIBS})
        return files

    # -- the positive sequence --------------------------------------------------------------------------------------
    def positive(self):
        before = self.tree()
        rc, catalog = self.cli("catalog")
        self.pin = catalog[0].get("catalog_sha256")
        self.check("P01", "the catalog is measured read-only and its pin printed; nothing is written",
                   rc == 0 and catalog[0]["ok"] and len(self.pin or "") == 64 and self.tree() == before,
                   catalog_sha256=self.pin)
        rc, status = self.cli("status")
        st = status[0]
        build_digests = {Path(r).name[3:-3]: hashlib.sha256((self.build / r).read_bytes()).hexdigest() for r in _LIBS}
        pinned = {i: v["sha256"] for i, v in st["libraries"].items()}
        self.check("P02", "ABI and every admitted library are the pinned ones (the build's bytes); no persistence",
                   rc == 0 and st["abi"] == {"runtime": 8, "composite": 1} and pinned == build_digests
                   and st["installation"]["persistence"] == "NONE", abi=st["abi"], libraries=pinned)
        s1 = [{"op": "open"}, *SOURCES, {"op": "live"}, {"op": "query", "x": X}, {"op": "counters", "reset": True},
              learn(), {"op": "counters"}, {"op": "query", "x": X}, {"op": "live"},
              evidence("record-1"), {"op": "counters"}, {"op": "live"}, LOOKUP,
              {"op": "context_axiom", "key": node("ctx-1"),
               "references": [{"chunk": node("chunk"), "corpus": node("corpus"), "provenance": provenance("p")}]},
              {"op": "context_view", "wait": 1}, {"op": "live"}, {"op": "close"}, {"op": "open"}, {"op": "live"},
              LOOKUP]
        rc, o = self.cli(script=s1)
        (opened, src1, src2, live0, q0, _, learned, after_learn, q1, live1, ev, after_ev, live2, found, axiom, view,
         live3, closed, reopened, live4, lost) = o
        initial = opened["initial_identity"]
        self.check("P03", "a session: a fresh genesis, anchored at the lineage's initial identity, sources founded",
                   opened["ok"] and opened["anchored"]["identity"] == initial
                   and opened["anchored"]["state_format"] == "COMPOSITE" and opened["session"]["generation"] == 2
                   and src1["ok"] and src2["ok"] and live0["live"]["identity"] == initial,
                   initial=initial, session=opened["session"]["session"])
        self.check("P04", "QUERY answers from the initial live state", q0["ok"] and q0["live_generation"] == 0,
                   values_sha256=q0.get("values_sha256"))
        self.check("P05", "an authorized LEARN runs the K1 schedule natively, volatile",
                   learned["ok"] and learned["durability"] == "VOLATILE" and learned["epoch"] == [0, 8]
                   and after_learn["counters"]["k1_run_schedules"] == 1, counters=after_learn.get("counters"))
        self.check("P06", "the live state moved: new answers, generation + 1, new identity",
                   q1["values"] != q0["values"] and q1["live_generation"] == 1
                   and live1["live"]["identity"] != initial, values_sha256=q1["values_sha256"])
        entry = ev["entries"][0] if ev["ok"] else {}
        self.check("P07", "the evidence is validated and identified by structure and recorded by the observe-only "
                          "managed turn (no K1 schedule)",
                   ev["ok"] and len(entry.get("hyperedge", "")) == 64 and ev["k1_steps"] == 0
                   and after_ev["counters"]["k1_run_schedules"] == 1 and after_ev["counters"]["k1_observes"] == 1,
                   entry=entry, counters=after_ev["counters"])
        self.check("P08", "LOOKUP answers the recorded evidence in the session",
                   found["found"] and found["answer"] == node("engineer"), lookup=found)
        self.check("P09", "the K1 component changed only with the authorized LEARN, never with the evidence",
                   live2["live"]["k1_identity"] == live1["live"]["k1_identity"]
                   and live2["live"]["memory_identity"] != live1["live"]["memory_identity"]
                   and live1["live"]["k1_identity"] != live0["live"]["k1_identity"],
                   k1=[live0["live"]["k1_identity"], live1["live"]["k1_identity"], live2["live"]["k1_identity"]])
        self.check("P10", "contextual WARM work publishes a complete versioned view and changes no ECS state",
                   axiom["ok"] and view["ok"] and view["waited"] == "APPLIED" and view["version"] == 1
                   and live3["live"] == live2["live"], view={k: view[k] for k in ("version", "digest")})
        self.check("P11", "close discards the session; the next starts at a fresh genesis and the initial identity",
                   closed["ok"] and closed["discarded"]["learned"] and reopened["ok"]
                   and reopened["session"]["session"] != opened["session"]["session"]
                   and reopened["session"]["generation"] == 2 and live4["live"]["identity"] == initial
                   and lost["ok"] and not lost["found"], discarded=closed.get("discarded"))
        rc, o2 = self.cli(script=[{"op": "live"}, {"op": "query", "x": X}, LOOKUP])
        live_r, q_r, lookup_r = o2
        self.check("P12", "a new process starts at the initial identity: nothing was kept to resume",
                   rc == 0 and live_r["live"]["identity"] == initial and q_r["values_sha256"] == q0["values_sha256"]
                   and not lookup_r["found"])
        killed = self.kill_mid_session()
        rc, o3 = self.cli(script=[{"op": "live"}])
        self.check("P13", "a process killed mid-session leaves nothing; the next starts at the initial identity",
                   killed is not None and killed != initial and rc == 0 and o3[0]["live"]["identity"] == initial,
                   killed_live=killed)
        rc, with_ctx = self.cli(script=[
            {"op": "context_axiom", "key": node("ctx-2"),
             "references": [{"chunk": node("c2"), "corpus": node("corpus"), "provenance": provenance("p2")}]},
            {"op": "context_view", "wait": 1}, learn(consume_context=True)])
        rc2, without = self.cli(script=[learn()])
        self.check("P14", "a LEARN that consumed a context view equals its twin without one (no context-to-ECS path)",
                   with_ctx[2]["ok"] and with_ctx[2]["context"]["version"] == 1
                   and with_ctx[2]["identity"] == without[0]["identity"], identity=without[0].get("identity"))
        self.check("P15", "no Elpis process wrote anything: the work directory and the build's libraries are "
                          "unchanged", self.tree() == before, files=len(before))

    _KILL = r"""
import json, sys, time
from array import array
from pathlib import Path
from elpis.runtime.cognition import LearnAuthority, Stimulus
from elpis.runtime.deployment import Installation, LineageConfig
i = Installation(Path(sys.argv[1]), sys.argv[2], build_tree=True)
s = i.session(LineageConfig(seed=11))
x = array("d", [((r * 7 + c * 3) % 11 - 5) / 16.0 for r in range(4) for c in range(6)])
s.runtime.run_numeric_learn(s.state, Stimulus(x, array("d", [0.1, -0.2, 0.05, 0.3]), array("Q", [4, 3]), dim=6),
                            LearnAuthority(0.002, "acceptance operator: kill fixture"))
print(json.dumps({"live": s.runtime.live_identity(s.state)[0].hex()}), flush=True)
time.sleep(3600)
"""

    def kill_mid_session(self) -> str | None:
        """A process that learned and is killed (SIGKILL) mid-session: its live identity, or None."""
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.Popen([sys.executable, "-c", self._KILL, str(self.build), self.pin], stdout=subprocess.PIPE,
                                text=True, env=env, cwd=str(self.work))
        try:
            line = proc.stdout.readline()
            return json.loads(line)["live"] if line else None
        finally:
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=60)

    def negative(self):
        from elpis.ECS.composite import CompositeError, CompositeState
        from elpis.runtime.cognition import QueryRequest, QueryStimulus
        from elpis.runtime.composition import CompositionError
        from elpis.runtime.deployment import Installation, LineageConfig
        from elpis.runtime.context import ContextAxiom, ContextLane, ContextReference, Observation

        def refused(call):
            try:
                call()
            except (CompositionError, CompositeError) as exc:
                return exc.code
            return None

        d = Installation(self.build, self.pin, build_tree=True)
        lineage = LineageConfig(seed=11)
        with d.session(lineage) as s:
            before = s.state.state_digest()
            codes = [refused(lambda: s.runtime.run_query(s.state, QueryRequest("text", object()))),
                     refused(lambda: s.runtime.run_query(s.state, QueryRequest("text", object(), codec=object())))]
            self.check("N01", "no qualified codec: the text QUERY fails closed; a non-admitted codec is refused",
                       codes == ["ECS_CODEC_UNQUALIFIED", "CODEC_UNADMITTED"], codes=codes)
            code = refused(lambda: s.runtime.run_learn(s.state, QueryRequest("text", object())))
            self.check("N02", "a QUERY request cannot become a LEARN", code == "OPERATION", code=code)
            # A superseded context view is detectable from its coordinate and changes nothing (context carries no
            # authority, so nothing needs refusing).
            lane = ContextLane()
            try:
                ref = (ContextReference(node("c"), node("corpus"), provenance("p")),)
                seq, _ = lane.submit(Observation(axiom=ContextAxiom(node("old"), ref)))
                lane.wait(seq, 30)
                old = lane.view()
                seq, _ = lane.submit(Observation(axiom=ContextAxiom(node("new"), ref)))
                lane.wait(seq, 30)
                q = QueryStimulus(array("d", [v for row in X for v in row]), dim=6)
                stale = s.runtime.run_numeric_query(s.state, q, context=old)
                fresh = s.runtime.run_numeric_query(s.state, q, context=lane)
            finally:
                lane.close()
            self.check("N06", "stale context: the superseded coordinate is reported and the answer is unchanged",
                       stale.context.version < fresh.context.version and stale.values == fresh.values
                       and s.state.state_digest() == before,
                       stale=stale.context.version, current=fresh.context.version)
            # Unsupported memory capability: the configuration never carries one; the native memory LEARN refuses.
            caps = s.state.inspect()["capabilities"]
        with CompositeState.create(d.libraries.composite, lineage.composite(),
                                   memoryview(array("d", [0.01] * (6 * 36)))) as scratch:
            tx = scratch.transaction()
            fn = d.libraries.composite._lib.elpis_ecs_composite_leased_txn_memory_learn
            fn.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_char_p, C.c_uint64, C.POINTER(C.c_int)]
            fn.restype = C.c_int
            memory_status = C.c_int(0)
            rc = fn(scratch._handle, 0, tx._token, bytes(32), 1, C.byref(memory_status))
            tx.abort()
        self.check("N05", "an unsupported memory capability: never configured; the native memory LEARN refuses",
                   caps == 0 and rc == -9, composite_status=rc, memory_status=memory_status.value)
        # CLI refusals in one session (nothing may change).
        rc, o = self.cli(script=[
            *SOURCES, {"op": "live"},
            learn(grant=None, expect="LEARN_UNAUTHORIZED"),
            evidence("unauthorized", grant=None, expect="OBSERVATION_UNAUTHORIZED"),
            dict(evidence("unadmitted"), sources=[2], expect="OBSERVATION_AUTHORITY"),
            dict(evidence("bad-source", source=9), expect="OBSERVATION_STRUCTURE"),
            dict(learn(), schedule=[[4, 1048576]], expect="COGNITION_FUEL_EXCEEDED"),
            {"op": "live"}])
        o = o[2:]
        self.check("N03", "unauthorized and unadmitted observations are refused", o[2]["met"] and o[3]["met"])
        self.check("N04", "an invalid source identity (outside structure's authority range) is refused", o[4]["met"])
        self.check("N10", "budget exhaustion is refused before any native work", o[5]["met"])
        self.check("N02b", "an unauthorized LEARN is refused", o[1]["met"])
        self.check("N00", "every refusal left the live state unchanged", o[0]["live"] == o[6]["live"])
        self.wrong_abi()
        self.pin_drift()
        self.warm_budget(d)
        results = {}
        for retired in (("init",), ("create",), ("publish",), ("candidates",), ("reconcile",), ("--home", "x")):
            rc, out = self.cli(*retired)
            results[retired[0]] = (rc, out[0].get("code"))
        self.check("N13", "every retired persistent command is refused and does nothing",
                   all(v == (2, "OPERATION_RETIRED") for v in results.values()), results=results)

    def wrong_abi(self):
        """A RuntimeCore library of another ABI version (a stub exporting only the version, 4) is refused."""
        from elpis.runtime.core import RuntimeLibrary
        from elpis.runtime.composition import CompositionError
        from elpis.substrate.authority import PinnedAuthority
        stub = self.work / "abi4"
        stub.mkdir()
        (stub / "abi4.c").write_text("unsigned elpis_runtime_abi_version(void) { return 4u; }\n")
        build = subprocess.run(["cc", "-shared", "-fPIC", "-o", str(stub / "libelpis_runtime.so"),
                                str(stub / "abi4.c")], capture_output=True, text=True)
        if build.returncode != 0:
            self.check("N07", "a wrong runtime ABI is refused", False, detail="no C compiler: " + build.stderr)
            return
        data = (stub / "libelpis_runtime.so").read_bytes()
        catalog = json.dumps({"schema": "elpis.inference-authority.v1", "provenance": "deployment",
                              "source": "acceptance: an ABI v4 stub", "assets": [],
                              "libraries": [{"library_id": "elpis_runtime", "size": len(data),
                                             "sha256": hashlib.sha256(data).hexdigest()}]}).encode()
        authority = PinnedAuthority(catalog, expected_sha256=hashlib.sha256(catalog).hexdigest())
        try:
            RuntimeLibrary.admit(stub, "libelpis_runtime.so", authority, "elpis_runtime")
            code = None
        except CompositionError as exc:
            code, detail = exc.code, str(exc)
        self.check("N07", "a wrong runtime ABI is refused (the version check)", code == "RUNTIME_INVALID",
                   code=code)

    def pin_drift(self):
        """A changed library, or a wrong pin, refuses every start (INSTALL_PIN): copies of the build's libraries, one
        byte of one library flipped (the harness's own copies; the build is never touched)."""
        def staged(name, flip=None):
            root = self.work / ("tamper-" + name)
            for rel in _LIBS:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.build / rel, root / rel)
            if flip:
                data = bytearray((root / flip).read_bytes())
                data[-1] ^= 1
                (root / flip).write_bytes(bytes(data))
            return root
        results = {}
        for name, flip in (("library", "native/runtime/composite/libelpis_ecs_composite.so"),
                           ("runtime", "native/runtime/libelpis_runtime.so")):
            rc, o = self.cli("status", source=staged(name, flip))
            results[name] = (rc, o[0].get("code"))
        rc, o = self.cli("status", source=staged("pin"), pin="0" * 64)
        results["pin"] = (rc, o[0].get("code"))
        self.check("N09", "pin drift refuses every start", results == {
            "library": (4, "INSTALL_PIN"), "runtime": (4, "INSTALL_PIN"), "pin": (4, "INSTALL_PIN")}, results=results)

    def warm_budget(self, d):
        """An FMS-resident K1 lineage on a RAM-only FMS context (no COLD tier: nothing can spill): at the WARM budget
        a new state is refused (FMS_E_LIMIT) with nothing demoted, spilled or written, and the bound lineage keeps
        answering from WARM (the HOT residency hold)."""
        from elpis.ECS.k1 import K1Error, K1FMSRuntime
        from elpis.runtime import Runtime, SessionConfig
        from elpis.runtime.cognition import LearnAuthority, QueryStimulus, Stimulus
        from elpis.substrate.residency import Context
        image = d.libraries.k1.envelope_bytes(6, 36) - 32
        ctx = Context(d.libraries.k1_fms, warm_bytes=2 * image + image // 2, max_objects=8)
        fms = K1FMSRuntime(d.libraries.k1, ctx, d.libraries.k1_fms, max_states=8)
        w = memoryview(array("d", [0.01 * ((i % 13) - 6) for i in range(216)]))
        states = []
        q = QueryStimulus(array("d", [v for row in X for v in row]), dim=6)
        stimulus = Stimulus(array("d", [v for row in X for v in row]), array("d", Y), array("Q", [4, 3]), dim=6)
        grant = LearnAuthority(0.002, "acceptance operator: residency fixture")
        try:
            states.append(fms.state(fms.register(bytes([1]) * 32, 6, 36, w, max_rows=8)))
            with Runtime(d.runtime_config(SessionConfig(capacity=4))) as runtime:
                runtime.anchor_cognition(states[0])
                runtime.run_numeric_learn(states[0], stimulus, grant)
                states.append(fms.state(fms.register(bytes([2]) * 32, 6, 36, w, max_rows=8)))
                try:
                    fms.register(bytes([3]) * 32, 6, 36, w, max_rows=8)
                    code = None
                except K1Error as exc:
                    code = exc.code
                r = fms.metrics()["residency"]
                answer = runtime.run_numeric_query(states[0], q)
            self.check("N11", "at the WARM budget a new FMS state is refused without spill; the HOT lineage answers",
                       code is not None and r["promotions"] == r["demotions"] == r["cold_reads"] == r["cold_writes"]
                       == 0 and len(answer.values) == 4 and runtime.fault is None,
                       code=code, residency={k: r[k] for k in ("promotions", "demotions", "cold_reads", "cold_writes")})
        finally:
            for s in states:
                try:
                    fms.close_state(s.id)
                except Exception:
                    pass

    # -- the whole run ----------------------------------------------------------------------------------------------
    def run(self) -> dict:
        self.work.mkdir(parents=True, exist_ok=True)
        if any(self.work.iterdir()):
            raise SystemExit(f"{self.work} is not empty")
        started = time.time()
        failure = None
        try:
            self.positive()
            self.negative()
        except Exception as exc:   # a crashed harness is a failed acceptance, reported, never hidden
            failure = f"{type(exc).__name__}: {exc}"
        import elpis
        return {"schema": REPORT_SCHEMA, "verdict": "PASS" if failure is None and all(c["ok"] for c in self.checks)
                else "FAIL", "failure": failure, "checks": self.checks,
                "passed": sum(c["ok"] for c in self.checks), "total": len(self.checks),
                "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                                "package": str(Path(elpis.__file__).parent), "native_build": str(self.build),
                                "source_commit": self.source_commit},
                "seconds": round(time.time() - started, 1),
                "nonclaims": ["mechanics only: fixture stimuli and evidence (TRAINING=NONE SEMANTICS=NONE)",
                              "no language, reasoning or learning capability claim",
                              "the harness's own writes (tamper copies, the ABI stub, the report) are not Elpis's"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m elpis.runtime.acceptance", description=__doc__.split("\n\n")[0])
    p.add_argument("--native-build", required=True, help="an operator CMake build tree (absolute)")
    p.add_argument("--work", required=True, help="an empty or new directory for the harness's own files")
    p.add_argument("--source-commit")
    p.add_argument("--report", help="write the JSON report here as well")
    args = p.parse_args(argv)
    report = Acceptance(Path(args.native_build).absolute(), Path(args.work).absolute(), args.source_commit).run()
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.report:
        Path(args.report).write_text(text + "\n")
    print(text)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
