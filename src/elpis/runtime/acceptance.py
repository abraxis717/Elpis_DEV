"""System acceptance of an installed Elpis: ``python -m elpis.runtime.acceptance --native-build BUILD --work DIR``.

It exercises the supported product, not the test suite: every operator step runs the installed CLI
(``python -m elpis.runtime.cli``) as a separate process over a fresh deployment in ``DIR`` (which must not exist or
be empty), and the adversarial checks use only the public runtime API and the deployment's own admitted libraries.
No test fixture or research module is imported. The result is one machine-readable JSON report (``--report``);
the exit status is 0 only when every check passed.

Positive sequence (one lineage, several processes): pinned identity of source, ABI and libraries; an admitted ECS
composite; QUERY; an authorized LEARN (a live transition, nothing published); validated HACF evidence recorded by the
observe-only turn (the K1 component unchanged) and LOOKUP; a contextual (WARM) view that changes no ECS state; a
recoverable publication with checkpoint and continuity in agreement; a restart in a new process that recovers the
geometric and memory components exactly; another transition and publication; then crashes at registered durability
boundaries (the testing RuntimeCore build parks its worker) with the prescribed dispositions: an unresolved
candidate that only the operator resolves, a synced record that resumes directly, volatile work that is lost.

Negative sequence: no codec authority, QUERY-to-LEARN escalation, unauthorized and unadmitted observations, an
invalid source identity, an unsupported memory capability, stale context, a wrong runtime ABI, a wrong state
envelope, pin drift, budget exhaustion, a COLD state on the HOT path, an incompatible recovery lineage, an existing
write-once destination and an unresolved candidate: each must be refused with its exact code, with nothing changed.

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

REPORT_SCHEMA = "elpis.acceptance-report.v1"
_LIBS = ("native/runtime/libelpis_runtime.so", "native/runtime/libelpis_runtime_testing.so",
         "native/runtime/libelpis_hacf_observation.so", "native/ECS_G/libelpis_ecsg_math.so",
         "native/ECS_G/libelpis_ecsg_k1.so", "native/ECS_G/libelpis_ecsg_k1_fms.so", "native/ECS_C/libelpis_ecs_memory.so",
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


LOOKUP = {"op": "lookup", "relation": "hyperedge:1", "target_role": "incidence:3",
          "cues": [{"role": "incidence:1", "node": node("alice")}, {"role": "incidence:2", "node": node("acme")}]}


class Acceptance:
    def __init__(self, native_build: Path, work: Path, source_commit: str | None):
        self.build, self.work, self.source_commit = native_build, work, source_commit
        self.checks: list[dict] = []

    # -- recording ------------------------------------------------------------------------------------------------
    def check(self, cid: str, what: str, ok: bool, **evidence) -> bool:
        self.checks.append({"id": cid, "check": what, "ok": bool(ok), "evidence": evidence})
        return bool(ok)

    # -- the installed CLI, as a separate process ------------------------------------------------------------------
    def cli(self, home: Path, *args, script=None) -> tuple[int, list]:
        cmd = [sys.executable, "-m", "elpis.runtime.cli", "--home", str(home), *args]
        if script is not None:
            cmd += ["session", "--script", "-"]
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        run = subprocess.run(cmd, input=json.dumps(script) if script is not None else None, capture_output=True,
                             text=True, env=env, cwd=str(self.work), timeout=600)
        out = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
        if run.returncode not in (0, 3, 4):
            raise RuntimeError(f"cli {args} crashed ({run.returncode}): {run.stderr[-2000:]}")
        return run.returncode, out

    def staged_build(self, name: str) -> Path:
        """Copy just the deployable libraries out of the build (removed after init: never used again)."""
        root = self.work / name
        for rel in _LIBS:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.build / rel, root / rel)
        return root

    def deploy(self, name: str, *, testing=False, seed=11) -> tuple[Path, dict]:
        staging = self.staged_build(name + "-build")
        home = self.work / name
        args = ["init", "--native-build", str(staging)] + (["--testing"] if testing else [])
        if self.source_commit:
            args += ["--source-commit", self.source_commit]
        rc, out = self.cli(home, *args)
        if rc != 0:
            raise RuntimeError(f"init refused: {out}")
        shutil.rmtree(staging)
        rc, created = self.cli(home, "create", "--seed", str(seed))
        if rc != 0:
            raise RuntimeError(f"create refused: {created}")
        return home, created[0]

    # -- the positive sequence --------------------------------------------------------------------------------------
    def positive(self):
        home, created = self.deploy("home")
        initial = created["identity"]
        rc, status = self.cli(home, "status")
        st = status[0]
        lib_digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (home / "lib").iterdir()}
        build_digests = {Path(r).name: hashlib.sha256((self.build / r).read_bytes()).hexdigest() for r in _LIBS
                         if not r.endswith("_testing.so")}
        self.check("P01", "a new lineage starts in an empty persistent directory", rc == 0 and st["ok"]
                   and created["state_format"] == "COMPOSITE", identity=initial, home=str(home))
        pinned = {i: v["sha256"] for i, v in st["libraries"].items()}
        self.check("P02", "source, ABI and every admitted library are the pinned ones (the build's bytes)",
                   st["abi"] == {"runtime": 6, "composite": 1} and st["deployment"]["source_commit"] == self.source_commit
                   and sorted(pinned.values()) == sorted(lib_digests.values()) == sorted(build_digests.values()),
                   abi=st["abi"], libraries=pinned, source_commit=st["deployment"]["source_commit"])
        self.check("P03", "an admitted ECS composite: anchored, durable first, RESUMABLE",
                   st["recovery"]["disposition"] == "RESUMABLE" and st["durable"]["identity"] == initial
                   and st["durable"]["state_format"] == "COMPOSITE", recovery=st["recovery"])
        s1 = [{"op": "live"}, {"op": "query", "x": X}, {"op": "counters", "reset": True}, learn(),
              {"op": "counters"}, {"op": "query", "x": X}, {"op": "live"},
              evidence("record-1"), {"op": "counters"}, {"op": "live"}, LOOKUP,
              {"op": "context_axiom", "key": node("ctx-1"),
               "references": [{"chunk": node("chunk"), "corpus": node("corpus"), "provenance": provenance("p")}]},
              {"op": "context_view", "wait": 1}, {"op": "live"}, {"op": "publish"}, {"op": "candidates"},
              {"op": "status"}]
        rc, o = self.cli(home, script=s1)
        (live0, q0, _, learned, after_learn, q1, live1, ev, after_ev, live2, found, axiom, view, live3, pub,
         cand, st1) = o
        self.check("P04", "QUERY answers from the initial live state", q0["ok"] and q0["live_generation"] == 0,
                   values_sha256=q0.get("values_sha256"))
        self.check("P05", "an authorized LEARN runs the K1 schedule natively, volatile",
                   learned["ok"] and learned["durability"] == "VOLATILE" and learned["epoch"] == [0, 8]
                   and after_learn["counters"]["k1_run_schedules"] == 1, counters=after_learn.get("counters"))
        self.check("P06", "the live state moved: new answers, generation + 1, new identity",
                   q1["values"] != q0["values"] and q1["live_generation"] == 1
                   and live1["live"]["identity"] != initial, values_sha256=q1["values_sha256"])
        self.check("P07", "no implicit durable publication", live1["durable"]["identity"] == initial,
                   durable=live1["durable"])
        entry = ev["entries"][0] if ev["ok"] else {}
        self.check("P08", "a structure hyperedge and assertion are validated and identified by structure",
                   ev["ok"] and len(entry.get("hyperedge", "")) == 64 and len(entry.get("assertion", "")) == 64,
                   entry=entry)
        self.check("P09", "the evidence is recorded by the observe-only managed turn (no K1 schedule)",
                   ev["ok"] and ev["k1_steps"] == 0 and after_ev["counters"]["k1_run_schedules"] == 1
                   and after_ev["counters"]["k1_observes"] == 1, counters=after_ev["counters"])
        rc_slot, slot = self.cli(home, script=[{"op": "slot", "key": entry.get("relation_key", "0" * 64)}])
        # (Session 1 published before it exited, so the next process reads the slot from the recovered state.)
        self.check("P10", "the memory slot keeps the exact identity and provenance; LOOKUP answers it",
                   found["found"] and found["answer"] == node("engineer") and slot[0]["found"]
                   and slot[0]["structure_ref"] == slot[0]["hyperedge"] == entry["hyperedge"]
                   and slot[0]["first_provenance"] == provenance("record-1") and slot[0]["last_authority"] == 1,
                   lookup=found, slot={k: slot[0].get(k) for k in ("structure_ref", "first_provenance")})
        self.check("P11", "the K1 component changed only with the authorized LEARN, never with the evidence",
                   live2["live"]["k1_identity"] == live1["live"]["k1_identity"]
                   and live2["live"]["memory_identity"] != live1["live"]["memory_identity"]
                   and live1["live"]["k1_identity"] != live0["live"]["k1_identity"],
                   k1=[live0["live"]["k1_identity"], live1["live"]["k1_identity"], live2["live"]["k1_identity"]])
        self.check("P12", "contextual WARM work publishes a complete versioned view",
                   axiom["ok"] and view["ok"] and view["waited"] == "APPLIED" and view["version"] == 1
                   and view["axioms"][0]["key"] == node("ctx-1"), view={k: view[k] for k in ("version", "digest")})
        self.check("P13", "context work changes no ECS state", live3["live"] == live2["live"])
        acked = pub["ticket"]
        self.check("P14", "a recoverable publication is acknowledged", acked["acknowledged"] and acked["recoverable"]
                   and acked["identity"] == live3["live"]["identity"], ticket=acked)
        self.check("P15", "checkpoint and continuity agree on the complete state",
                   cand["recovery"]["disposition"] == "RESUMABLE"
                   and cand["recovery"]["authorized"] == pub["durable"]["identity"] == acked["identity"]
                   and cand["recovery"]["envelope_bytes"] == st1["footprint"]["envelope_bytes"],
                   recovery=cand["recovery"])
        published = acked["identity"]
        # -- restart: a new process (the previous one closed and exited) --------------------------------------------
        rc, o2 = self.cli(home, script=[{"op": "live"}, {"op": "query", "x": X}, LOOKUP,
                                        {"op": "slot", "key": entry["relation_key"]}])
        live_r, q_r, found_r, slot_r = o2
        self.check("P16", "the runtime closes and a new process opens the same directory", rc == 0, rc=rc)
        self.check("P17", "restart recovers the acknowledged state", live_r["live"]["identity"] == published
                   and live_r["durable"]["identity"] == published)
        self.check("P18", "the geometric and memory components are exactly the published ones",
                   live_r["live"]["k1_identity"] == live3["live"]["k1_identity"]
                   and live_r["live"]["memory_identity"] == live3["live"]["memory_identity"])
        self.check("P19", "QUERY and LOOKUP operate after restart, byte-identically",
                   q_r["values_sha256"] == q1["values_sha256"] and found_r["answer"] == found["answer"]
                   and slot_r["structure_ref"] == entry["hyperedge"] and slot_r["assert_count"] == 1
                   and slot_r["first_provenance"] == provenance("record-1"))
        rc, o3 = self.cli(home, script=[learn(), {"op": "publish"}, {"op": "live"}])
        self.check("P20", "another authorized transition, re-published: lineage continuity",
                   rc == 0 and o3[1]["ticket"]["acknowledged"]
                   and o3[1]["durable"]["generation"] == pub["durable"]["generation"] + 1
                   and o3[2]["durable"]["identity"] == o3[2]["live"]["identity"] != published,
                   durable=o3[1]["durable"])
        # Context is never an ECS input: the same LEARN with and without a consumed view yields one identity.
        rc, with_ctx = self.cli(home, script=[
            {"op": "context_axiom", "key": node("ctx-2"),
             "references": [{"chunk": node("c2"), "corpus": node("corpus"), "provenance": provenance("p2")}]},
            {"op": "context_view", "wait": 1}, learn(consume_context=True)])
        rc2, without = self.cli(home, script=[learn()])
        self.check("P21", "a LEARN that consumed a context view equals its twin without one (no context-to-ECS path)",
                   with_ctx[2]["ok"] and with_ctx[2]["context"]["version"] == 1
                   and with_ctx[2]["identity"] == without[0]["identity"], identity=without[0].get("identity"))
        self.home = home
        return home

    # -- registered durability boundaries (the testing RuntimeCore build) -----------------------------------------
    _CRASH = r"""
import json, sys, time
from array import array
from pathlib import Path
from elpis.runtime.cognition import LearnAuthority, ObservationAuthority, Stimulus
from elpis.runtime.deployment import Deployment
from elpis.runtime.observation import HacfParticipant, StructuralEvidence, record_evidence
home, step, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
d = Deployment(home)
s = d.session()
x = array("d", [((i * 7 + j * 3) % 11 - 5) / 16.0 for i in range(4) for j in range(6)])
s.runtime.run_numeric_learn(s.state, Stimulus(x, array("d", [0.1, -0.2, 0.05, 0.3]), array("Q", [4, 3]), dim=6),
                            LearnAuthority(0.002, "acceptance operator: crash fixture"))
edge = d.libraries.hacf.hyperedge(0x20000001, tuple(HacfParticipant(bytes.fromhex(n), 0x30000000 | (i + 1))
                                                    for i, n in enumerate(sys.argv[4:7])))
a = d.libraries.hacf.assertion(edge, bytes.fromhex(sys.argv[7]), 2)
key = record_evidence(s.runtime, s.state, d.libraries.hacf, (StructuralEvidence(edge, a, "NEGATE", 5),),
                      ObservationAuthority("acceptance operator: crash fixture", frozenset({2}))).entries[0].relation_key
candidate = s.runtime.live_identity(s.state)[0]
if mode == "park":
    s.runtime._core.testing_durable_fault(2, step, 0)
    s.runtime.publish_recoverable(s.state, wait=False)
    while s.runtime.durability().parked != step:
        time.sleep(0.005)
print(json.dumps({"candidate": candidate.hex(), "key": key.hex()}), flush=True)
time.sleep(3600)
"""

    def crash(self, home: Path, step: int, mode: str) -> dict:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        child = subprocess.Popen([sys.executable, "-c", self._CRASH, str(home), str(step), mode, node("alice"),
                                  node("acme"), node("engineer"), provenance("crash")], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, env=env, cwd=str(self.work))
        try:
            line = child.stdout.readline()
            if not line:
                raise RuntimeError(child.stderr.read())
            report = json.loads(line)
        finally:
            child.send_signal(signal.SIGKILL)
            child.wait(timeout=60)
        report["returncode"] = child.returncode
        return report

    def crashes(self):
        home, created = self.deploy("home-testing", testing=True)
        rc, o = self.cli(home, script=[evidence("before-crash"), {"op": "publish"}])
        authorized = o[1]["ticket"]["identity"]
        evidence_key = o[0]["entries"][0]["relation_key"]
        # 1. Killed with the checkpoint slot synced and continuity not published: an unresolved candidate.
        c1 = self.crash(home, 5, "park")
        rc, cand = self.cli(home, "candidates")
        rc_q, refused = self.cli(home, "query", "--x", json.dumps(X))
        self.check("P22", "SIGKILL after the slot sync (registered boundary 5) leaves an unresolved candidate",
                   c1["returncode"] == -signal.SIGKILL and cand[0]["recovery"]["disposition"] == "CANDIDATE_UNRESOLVED"
                   and cand[0]["recovery"]["authorized"] == authorized
                   and cand[0]["recovery"]["candidate"] == c1["candidate"], recovery=cand[0]["recovery"])
        self.check("P23", "no implicit adoption: cognition is refused until the operator acts",
                   rc_q == 3 and refused[0]["code"] == "LINEAGE_CANDIDATE_UNRESOLVED")
        rc, adopted = self.cli(home, "reconcile", "--adopt", c1["candidate"])
        rc, after = self.cli(home, script=[{"op": "live"}, {"op": "slot", "key": c1["key"]},
                                           {"op": "slot", "key": evidence_key}])
        self.check("P24", "the operator's adoption resumes exactly the candidate, every observation once",
                   adopted[0]["ok"] and after[0]["live"]["identity"] == after[0]["durable"]["identity"] == c1["candidate"]
                   and after[1]["negate_count"] == 1 and after[2]["assert_count"] == 1
                   and after[1]["assert_count"] == 1,   # the same relation: one assert (before) and one negate
                   live=after[0]["live"])
        # 2. Killed with the continuity record synced but the acknowledgment not delivered: resumes directly.
        c2 = self.crash(home, 8, "park")
        rc, cand2 = self.cli(home, "candidates")
        self.check("P25", "SIGKILL after the record sync (boundary 8, never acknowledged to the caller) resumes it",
                   cand2[0]["recovery"]["disposition"] == "RESUMABLE"
                   and cand2[0]["recovery"]["authorized"] == c2["candidate"])
        # 3. Killed with only volatile work: the last acknowledged state, nothing lost or invented.
        c3 = self.crash(home, 0, "volatile")
        rc, live3 = self.cli(home, script=[{"op": "live"}])
        self.check("P26", "volatile work is lost by a crash; the acknowledged state is never lost",
                   live3[0]["live"]["identity"] == live3[0]["durable"]["identity"] == c2["candidate"] != c3["candidate"])
        return home

    # -- the negative sequence --------------------------------------------------------------------------------------
    def negative(self):
        home = self.home
        from elpis.ECS.composite import CompositeError, CompositeState
        from elpis.runtime.cognition import QueryRequest, QueryStimulus
        from elpis.runtime.composition import CompositionError
        from elpis.runtime.deployment import Deployment
        from elpis.runtime.context import ContextAxiom, ContextLane, ContextReference, Observation

        def refused(call):
            try:
                call()
            except (CompositionError, CompositeError) as exc:
                return exc.code
            return None

        d = Deployment(home)
        with d.session() as s:
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
        with CompositeState.create(d.libraries.composite, d.lineage().composite(),
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
        # CLI refusals over the live deployment (nothing may change).
        rc, o = self.cli(home, script=[
            {"op": "live"},
            learn(grant=None, expect="LEARN_UNAUTHORIZED"),
            evidence("unauthorized", grant=None, expect="OBSERVATION_UNAUTHORIZED"),
            dict(evidence("unadmitted"), sources=[2], expect="OBSERVATION_AUTHORITY"),
            dict(evidence("bad-source", source=9), expect="OBSERVATION_STRUCTURE"),
            dict(learn(), schedule=[[4, 1048576]], expect="COGNITION_FUEL_EXCEEDED"),
            {"op": "live"}])
        self.check("N03", "unauthorized and unadmitted observations are refused", o[2]["met"] and o[3]["met"])
        self.check("N04", "an invalid source identity (outside structure's authority range) is refused", o[4]["met"])
        self.check("N10", "budget exhaustion is refused before any native work", o[5]["met"])
        self.check("N02b", "an unauthorized LEARN is refused", o[1]["met"])
        self.check("N00", "every refusal left the live and durable state unchanged", o[0]["live"] == o[6]["live"]
                   and o[0]["durable"] == o[6]["durable"])
        self.wrong_abi()
        self.wrong_envelope(d)
        self.pin_drift(home)
        self.cold_from_hot(d)
        self.incompatible_lineage(home)
        rc, o = self.cli(home, "create")
        rc2, o2 = self.cli(home, "init", "--native-build", str(self.build))
        self.check("N13", "an existing write-once destination is never overwritten (lineage, deployment)",
                   o[0]["code"] == "LINEAGE_EXISTS" and o2[0]["code"] == "DEPLOYMENT_EXISTS")
        self.check("N14", "an unresolved candidate refuses cognition (P23)",
                   any(c["id"] == "P23" and c["ok"] for c in self.checks))

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
        rc, o = self._tampered_copy(self.home, "abi", lambda h: _rewrite_json(h / "elpis.json", abi={"runtime": 4,
                                                                                                     "composite": 1}))
        self.check("N07", "a wrong runtime ABI is refused (the version check; a deployment of another ABI)",
                   code == "RUNTIME_INVALID" and o[0]["code"] == "DEPLOYMENT_ABI", code=code,
                   deployment=o[0]["code"])

    def wrong_envelope(self, d):
        from elpis.ECS.composite import CompositeError, CompositeState
        from elpis.ECS.k1 import K1Error, K1State
        from elpis.runtime.composition import CompositionError
        config = d.lineage().composite()
        with d.session() as s:
            envelope = s.state.snapshot()
        codes = []
        for bad in (envelope[:-1] + bytes([envelope[-1] ^ 1]), b"ELPISGK1" + envelope[8:]):
            try:
                CompositeState.restore(d.libraries.composite, config, bad).close()
                codes.append(None)
            except CompositeError as exc:
                codes.append(exc.code)
        with K1State.create(d.libraries.k1, 6, 36, memoryview(array("d", [0.01] * 216))) as k1, d.session() as s:
            try:
                s.runtime.run_numeric_query(k1, __import__("elpis.runtime.cognition", fromlist=["x"]).QueryStimulus(
                    array("d", [0.1] * 6), dim=6))
                mismatch = None
            except (CompositionError, K1Error) as exc:
                mismatch = exc.code
        self.check("N08", "a wrong state envelope or format is refused (corrupt trailer, K1 magic, K1 for composite)",
                   codes == ["CORRUPT", "CORRUPT"] and mismatch == "CONTINUITY_STATE_MISMATCH",
                   restore=codes, bind=mismatch)

    def _tampered_copy(self, home: Path, name: str, tamper) -> tuple[int, list]:
        copy = self.work / ("tamper-" + name)
        shutil.copytree(home, copy, symlinks=True)
        for p in [copy, *copy.rglob("*")]:
            p.chmod(p.stat().st_mode | 0o200)
        tamper(copy)
        return self.cli(copy, "status")

    def pin_drift(self, home: Path):
        def flip(path):
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(bytes(data))
        results = {}
        for name, tamper in (("library", lambda h: flip(h / "lib" / "libelpis_ecs_composite.so")),
                             ("runtime", lambda h: flip(h / "lib" / "libelpis_runtime.so")),
                             ("catalog", lambda h: flip(h / "native-catalog.json")),
                             ("pin", lambda h: _rewrite_json(h / "elpis.json", catalog_sha256="0" * 64))):
            rc, o = self._tampered_copy(home, name, tamper)
            results[name] = (rc, o[0].get("code"))
        self.check("N09", "source/binary pin drift refuses every start", results == {
            "library": (4, "DEPLOYMENT_PIN"), "runtime": (4, "RUNTIME_UNPINNED"), "catalog": (4, "DEPLOYMENT_PIN"),
            "pin": (4, "DEPLOYMENT_PIN")}, results=results)

    def cold_from_hot(self, d):
        """A COLD FMS-resident K1 state on the HOT path (the HOT residency invariants B and C, on the deployment's own
        admitted libraries): the bound lineage is released, demoted exactly (every other WARM state held, then a
        newcomer), refused RUNTIME_NOT_RESIDENT by QUERY and LEARN with nothing promoted, read or fail-stopped, and
        resumed only by the owner's explicit materialization."""
        from elpis.ECS.k1 import K1FMSRuntime
        from elpis.runtime import Runtime
        from elpis.runtime.cognition import LearnAuthority, QueryStimulus, Stimulus
        from elpis.runtime.composition import CompositionError, RuntimeConfig
        from elpis.substrate.residency import Context
        image = d.libraries.k1.envelope_bytes(6, 36) - 32
        ctx = Context(d.libraries.k1_fms, warm_bytes=2 * image + image // 2, cold_bytes=10 ** 7, max_objects=8,
                      cold_root=self.work / "fms-cold")
        fms = K1FMSRuntime(d.libraries.k1, ctx, d.libraries.k1_fms, max_states=8)
        w = memoryview(array("d", [0.01 * ((i % 13) - 6) for i in range(216)]))
        states, held = [], []

        def register(key):
            states.append(fms.state(fms.register(bytes([key]) * 32, 6, 36, w, max_rows=8)))
            return states[-1]

        def residency():
            r = fms.metrics()["residency"]
            return {k: r[k] for k in ("promotions", "demotions", "cold_reads", "cold_writes")}

        def refused(call):
            try:
                call()
            except CompositionError as exc:
                return exc.code
            return None

        q = QueryStimulus(array("d", [v for row in X for v in row]), dim=6)
        stimulus = Stimulus(array("d", [v for row in X for v in row]), array("d", Y), array("Q", [4, 3]), dim=6)
        grant = LearnAuthority(0.002, "acceptance operator: residency fixture")
        config = RuntimeConfig(self.work / "fms-continuity", d.home / "lib" / d.config["runtime_library"],
                               native_authority=d.authority, runtime_library_id=d.config["runtime_library_id"])
        try:
            lineage = register(1)
            with Runtime(config) as runtime:
                runtime.anchor_cognition(lineage)
                runtime.run_numeric_learn(lineage, stimulus, grant)
                live = runtime.live_identity(lineage)
                runtime.release(lineage)
                other = register(2)
                fms.resident_hold(other.id)
                held.append(other)
                register(3)                                   # WARM must make room: only the lineage is demotable
                cold = fms.inspect(lineage.id)["tier"] == 2
                before = residency()
                codes = [refused(lambda: runtime.run_numeric_query(lineage, q)),
                         refused(lambda: runtime.run_numeric_learn(lineage, stimulus, grant))]
                untouched = residency() == before and runtime.fault is None
                fms.resident_release(other.id)
                held.remove(other)
                lineage.materialize()                         # the explicit WARM-lane restoration
                answer = runtime.run_numeric_query(lineage, q)
                resumed = runtime.live_identity(lineage)
            self.check("N11", "COLD access from HOT is refused without promotion or cold read; explicit "
                              "materialization resumes the same live lineage",
                       cold and codes == ["RUNTIME_NOT_RESIDENT"] * 2 and untouched and resumed[0] == live[0]
                       and len(answer.values) == 4, codes=codes, residency=before)
        finally:
            for s in held:
                fms.resident_release(s.id)
            for s in states:
                try:
                    fms.close_state(s.id)
                except Exception:
                    pass

    def incompatible_lineage(self, home: Path):
        rc, o = self._tampered_copy(home, "lineage", lambda h: _rewrite_lineage(h / "lineage" / "lineage.json"))
        code = o[0].get("code")
        self.check("N12", "an incompatible recovery lineage (another composite shape) is refused, nothing adopted",
                   o[0]["ok"] is False and code == "CORRUPT", code=code)

    # -- the whole run ----------------------------------------------------------------------------------------------
    def run(self) -> dict:
        self.work.mkdir(parents=True, exist_ok=True)
        if any(self.work.iterdir()):
            raise SystemExit(f"{self.work} is not empty")
        started = time.time()
        failure = None
        try:
            self.positive()
            self.crashes()
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
                              "process death, not power loss", "no language, reasoning or learning capability claim"]}


def _rewrite_json(path: Path, **fields):
    document = json.loads(path.read_text())
    document.update(fields)
    path.write_text(json.dumps(document))


def _rewrite_lineage(path: Path):
    document = json.loads(path.read_text())
    document["config"]["slot_capacity"] = document["config"]["slot_capacity"] // 2
    path.write_text(json.dumps(document))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m elpis.runtime.acceptance", description=__doc__.split("\n\n")[0])
    p.add_argument("--native-build", required=True, help="an operator CMake build tree (absolute)")
    p.add_argument("--work", required=True, help="an empty or new directory for the acceptance deployments")
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
