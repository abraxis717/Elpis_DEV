"""The offline reference operator interface: ``python -m elpis.runtime.cli --home DIR <command>``.

Not the production path. The production entry point is the native operator host ``elpis`` (docs/NATIVE_HOST.md),
which runs no interpreter. This interface operates the same composite lineage over the standalone continuity backend
and exists for differential tests (the native host must answer exactly as it does), the qualification instruments bound
to their own commits, and research (docs/OPERATING_ELPIS.md, "The offline reference interface").

One deployment (``elpis.runtime.deployment``) holds one ECS composite lineage. Every command prints JSON (one object
per operation) and exits 0 only when every operation succeeded or was refused exactly as its script expected:

    init --native-build BUILD [--testing] [--source-commit SHA]   pin an operator build's libraries into the home
    status                                         deployment, pins, ABI, lineage, live vs durable, footprint, limits
    create [--dim 6 --width 36 --seed 0 ...]       create and anchor the lineage (durable baseline)
    query --x ROWS                                 numeric QUERY of the live state (read-only)
    learn --x ROWS --y TARGETS --schedule PAIRS --rate R --grant TEXT [--publish]
    evidence --items JSON --grant TEXT --sources 1,2 [--publish]    typed HACF evidence, no K1 learning
    lookup --relation R --target-role R --cues JSON                 exact LOOKUP by net evidence
    slot --key HEX                                 one memory slot and the hyperedge it was observed from
    publish                                        recoverable publication of the live state
    candidates                                     the recovery disposition (an unresolved candidate, if any)
    reconcile (--discard ID | --adopt ID)          operator reconciliation of exactly one candidate
    session --script FILE|-                        many operations in one process (volatile work lives here)

Volatile work lives only inside one process: a one-shot ``learn`` or ``evidence`` reports ``durability: VOLATILE``
and is lost at exit unless ``--publish`` is given (or a ``session`` publishes it). Every restart resumes exactly the
last acknowledged state. There is no text interface: no qualified ECS<->DSV4 codec map exists
(``ECS_CODEC_UNQUALIFIED``); QUERY and LEARN take admitted numeric stimuli and return numbers.

The ``elpis`` console command is deliberately not claimed by this development distribution (ELPIS_SYSTEM.json,
``identity.release_authority = NONE_DEFERRED``); this module is the entry point.
"""
from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict
import json
from pathlib import Path
import sys

from elpis.ECS.k1 import K1Error

from .cognition import LearnAuthority, ObservationAuthority, QueryStimulus, Stimulus
from .context import (CANDIDATE, ContextAxiom, ContextLane, ContextReference, ContextRelation, Observation,
                      PROVISIONAL)
from .deployment import Deployment, LineageConfig, init_deployment, struct_digest
from .errors import CompositionError
from .fuel import CEILING, CognitiveBudget
from .observation import (HYPEREDGE_NAMESPACE, INCIDENCE_NAMESPACE, HacfParticipant, StructuralEvidence,
                          record_evidence)

CAPABILITIES = {
    "numeric_query": "WORKING, mechanically verified: f_W(x) of the live K1 state (read-only)",
    "numeric_learn": "WORKING, mechanically verified: the qualified K1 law (qualified only in its frozen regime) "
                     "under an explicit LearnAuthority; volatile until published",
    "typed_evidence": "WORKING, mechanically verified: validated HACF hyperedge + assertion recorded by an "
                      "observe-only turn (RuntimeCore ABI v5: no K1 learning)",
    "lookup": "WORKING: exact relation LOOKUP by net evidence; a conflict between sources is reported, never resolved",
    "recoverable_publication": "WORKING: checkpoint slot read back verified, then continuity (ACKNOWLEDGED)",
    "restart_recovery": "WORKING: exactly the last acknowledged state; an unresolved candidate needs reconcile",
    "context_lane": "WORKING, session-local: bounded immutable versioned views of admitted references; never "
                    "persisted; never authorizes ECS learning",
    "text": "UNAVAILABLE: ECS_CODEC_UNQUALIFIED (no qualified ECS<->DSV4 codec map)",
    "memory_learn": "UNQUALIFIED: never enabled (no memory capability is ever set)",
    "memory_slow_consolidation": "NOT PROVIDED",
    "contextual_dynamics": "NOT IMPLEMENTED: no qualified law (HCA research is research-only)",
    "power_loss_durability": "NOT CLAIMED: crash consistency is established for process death only",
}
NONCLAIMS = ("no language understanding", "no general intelligence", "no autonomous evolution",
             "no hierarchical superiority", "an observation is evidence storage, never truth or authority",
             "pins authenticate bytes, not authors")


class Expectation(Exception):
    """A script's declared expectation was not met."""


def _hex32(value, what) -> bytes:
    if type(value) is not str or len(value) != 64:
        raise CompositionError("CLI_INPUT", f"{what}: 64 lowercase hex characters")
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        raise CompositionError("CLI_INPUT", f"{what}: 64 lowercase hex characters") from None
    if raw.hex() != value:
        raise CompositionError("CLI_INPUT", f"{what}: lowercase hex")
    return raw


def _id(value, namespace, prefix, what) -> int:
    """A HACF type or role id: an int (namespace included) or ``"<prefix>:N"`` (the namespace added explicitly)."""
    if type(value) is int:
        return value
    if type(value) is str and value.startswith(prefix + ":") and value[len(prefix) + 1:].isdigit():
        return namespace | int(value[len(prefix) + 1:])
    raise CompositionError("CLI_INPUT", f"{what}: an int or '{prefix}:N'")


def _rows(value, dim, what) -> array:
    if type(value) is not list or not value or not all(type(r) is list and len(r) == dim for r in value):
        raise CompositionError("CLI_INPUT", f"{what}: a non-empty list of rows of {dim} numbers")
    out = array("d")
    for row in value:
        for v in row:
            if type(v) not in (int, float) or isinstance(v, bool):
                raise CompositionError("CLI_INPUT", f"{what}: numbers only")
            out.append(float(v))
    return out


def _ticket(t) -> dict:
    return {"sequence": t.sequence, "status": t.status, "kind": t.kind, "recoverable": t.recoverable,
            "acknowledged": t.acknowledged, "identity": t.identity.hex() if t.identity else None,
            "generation": t.generation, "code": t.code}


def _snapshot(s) -> dict | None:
    if s is None:
        return None
    return {"state_format": s.state_format, "identity": s.k1_state_digest.hex() if s.k1_state_digest else None,
            "generation": s.generation}


def _recovery(r) -> dict:
    return {"disposition": r.disposition, "authorized": r.authorized.hex() if r.authorized else None,
            "candidate": r.candidate.hex() if r.candidate else None,
            "envelope_bytes": len(r.envelope) if r.envelope else 0}


class Console:
    """The operations of one process over one deployment (``session`` runs many; one-shot commands run one)."""

    def __init__(self, deployment: Deployment):
        self.deployment = deployment
        self._session = None
        self._context = None

    # -- lifecycle ----------------------------------------------------------------------------------------------------
    @property
    def session(self):
        if self._session is None:
            self._session = self.deployment.session()
        return self._session

    def close(self):
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._session is not None:
            self._session.close()
            self._session = None

    # -- dispatch -----------------------------------------------------------------------------------------------------
    def run(self, op: dict) -> dict:
        if type(op) is not dict or type(op.get("op")) is not str:
            raise CompositionError("CLI_INPUT", "each operation is an object with an 'op' name")
        name = op["op"]
        handler = getattr(self, "op_" + name, None)
        if handler is None:
            raise CompositionError("CLI_INPUT", f"unknown operation {name!r}")
        return handler(op)

    # -- inspection ---------------------------------------------------------------------------------------------------
    def op_status(self, op) -> dict:
        d = self.deployment
        out = {"deployment": {"home": str(d.home), "schema": d.config["schema"], "testing_build": d.testing,
                              "source_commit": d.config["source_commit"],
                              "catalog_sha256": d.config["catalog_sha256"]},
               "abi": dict(d.config["abi"]), "libraries": d.libraries.identities,
               "budget_ceiling": {k: v for k, v in asdict(CEILING).items()},
               "capabilities": CAPABILITIES, "nonclaims": list(NONCLAIMS)}
        lineage = d.lineage()
        out["lineage"] = None if lineage is None else asdict(lineage)
        if lineage is None:
            return out
        s = self.session
        out["recovery"] = _recovery(s.recovery)
        out["durable"] = _snapshot(s.runtime.continuity.snapshot())
        if s.state is not None:
            identity, generation = s.runtime.live_identity(s.state)
            out["live"] = {"identity": identity.hex(), "generation": generation}
            out["footprint"] = s.state.inspect()
            out["residency"] = "RAM: an ECS composite substrate (FMS residency holds apply to K1 FMS lineages)"
        durability = s.runtime.durability()
        out["durability"] = {"live_generation": durability.live_generation, "pending": durability.pending,
                             "fault": durability.fault,
                             "last_acknowledged": _ticket(durability.last_acknowledged),
                             "checkpoint_capacity": durability.checkpoint_capacity}
        out["fault"] = s.runtime.fault
        return out

    def op_live(self, op) -> dict:
        s = self.session
        state = s.require_state()
        identity, generation = s.runtime.live_identity(state)
        composite, k1, memory = self.deployment.libraries.composite.envelope_identities(state.snapshot())
        return {"live": {"identity": identity.hex(), "generation": generation, "k1_identity": k1.hex(),
                         "memory_identity": memory.hex()},
                "durable": _snapshot(s.runtime.continuity.snapshot())}

    def op_counters(self, op) -> dict:
        """RuntimeCore's native crossings since this process opened the runtime (``reset`` restarts the count)."""
        return {"counters": self.session.runtime._core.counters(reset=bool(op.get("reset", False)))}

    def op_candidates(self, op) -> dict:
        s = self.session
        return {"recovery": _recovery(s.runtime.recover_k1())}

    # -- cognition: numeric QUERY and LEARN ---------------------------------------------------------------------------
    def op_query(self, op) -> dict:
        s = self.session
        state = s.require_state()
        stimulus = QueryStimulus(_rows(op.get("x"), state.dim, "x"), dim=state.dim)
        answer = s.runtime.run_numeric_query(state, stimulus, context=self._consumed(op))
        return {"values": list(answer.values), "values_sha256": struct_digest(answer.values),
                "live_generation": answer.live_generation, "rows": answer.rows,
                "context": asdict(answer.context) if answer.context else None}

    def op_learn(self, op) -> dict:
        s = self.session
        state = s.require_state()
        schedule = op.get("schedule")
        if type(schedule) is not list or not all(type(p) is list and len(p) == 2 and all(type(v) is int for v in p)
                                                 for p in schedule):
            raise CompositionError("CLI_INPUT", "schedule: a list of [rows, steps] pairs")
        y = op.get("y")
        if type(y) is not list or not all(type(v) in (int, float) and not isinstance(v, bool) for v in y):
            raise CompositionError("CLI_INPUT", "y: a list of numbers")
        stimulus = Stimulus(_rows(op.get("x"), state.dim, "x"), array("d", map(float, y)),
                            array("Q", [v for p in schedule for v in p]), dim=state.dim)
        authority = None
        if op.get("grant") is not None:
            rate = op.get("rate")
            authority = LearnAuthority(rate if type(rate) is float else float(rate or 0.0), op["grant"])
        result = s.runtime.run_numeric_learn(state, stimulus, authority, context=self._consumed(op))
        return {"durability": result.durability, "experiences": result.experiences, "rows": result.rows,
                "steps": result.steps, "epoch": [result.epoch_before, result.epoch_after],
                "generation": [result.generation_before, result.generation_after], "grant": result.grant,
                "context": asdict(result.context) if result.context else None,
                "identity": state.state_digest().hex()}

    # -- typed structural evidence ------------------------------------------------------------------------------------
    def op_evidence(self, op) -> dict:
        s = self.session
        state = s.require_state()
        hacf = self.deployment.libraries.hacf
        items = op.get("items")
        if type(items) is not list or not items:
            raise CompositionError("CLI_INPUT", "items: a non-empty list of evidence objects")
        evidence = []
        for item in items:
            if type(item) is not dict or type(item.get("participants")) is not list:
                raise CompositionError("CLI_INPUT", "evidence: relation, participants, provenance, authority, "
                                                    "polarity, tick")
            parts = tuple(HacfParticipant(_hex32(p.get("node"), "participant node"),
                                          _id(p.get("role"), INCIDENCE_NAMESPACE, "incidence", "participant role"),
                                          int(p.get("ordinal", 0)), bool(p.get("query_local", False)))
                          for p in item["participants"])
            edge = hacf.hyperedge(_id(item.get("relation"), HYPEREDGE_NAMESPACE, "hyperedge", "relation"), parts)
            assertion = hacf.assertion(edge, _hex32(item.get("provenance"), "provenance"), item.get("authority"))
            evidence.append(StructuralEvidence(edge, assertion, item.get("polarity"), item.get("tick")))
        authority = None
        if op.get("grant") is not None:
            sources = op.get("sources")
            if type(sources) is not list:
                raise CompositionError("CLI_INPUT", "sources: a list of source ids")
            authority = ObservationAuthority(op["grant"], frozenset(sources))
        receipt = record_evidence(s.runtime, state, hacf, tuple(evidence), authority)
        return {"durability": receipt.durability, "k1_steps": receipt.k1_steps,
                "generation": [receipt.generation_before, receipt.generation_after],
                "entries": [{"relation_key": e.relation_key.hex(), "hyperedge": e.hyperedge_identity.hex(),
                             "assertion": e.assertion_identity.hex(), "provenance": e.provenance_digest.hex(),
                             "source_authority": e.source_authority, "polarity": e.polarity, "tick": e.tick}
                            for e in receipt.entries],
                "identity": state.state_digest().hex()}

    def op_lookup(self, op) -> dict:
        state = self.session.require_state()
        cues = op.get("cues")
        if type(cues) is not list or not cues:
            raise CompositionError("CLI_INPUT", "cues: a non-empty list of {role, node, ordinal}")
        cue = tuple((_id(c.get("role"), INCIDENCE_NAMESPACE, "incidence", "cue role"), int(c.get("ordinal", 0)),
                     _hex32(c.get("node"), "cue node")) for c in cues)
        answer = state.lookup(_id(op.get("relation"), HYPEREDGE_NAMESPACE, "hyperedge", "relation"),
                              _id(op.get("target_role"), INCIDENCE_NAMESPACE, "incidence", "target role"), cue)
        return {"found": answer.found, "answer": answer.answer.hex() if answer.answer else None,
                "matches": answer.matches, "conflicts": answer.conflicts, "contradicted": answer.contradicted,
                "uncertain": answer.uncertain, "state_identity": answer.state_digest.hex()}

    def op_slot(self, op) -> dict:
        state = self.session.require_state()
        slot = state.slot(_hex32(op.get("key"), "key"))
        if slot is None:
            return {"found": False}
        edge = self.deployment.libraries.hacf.hyperedge_of(slot)
        return {"found": True, "relation_type": slot.relation_type, "assert_count": slot.assert_count,
                "negate_count": slot.negate_count, "uncertain_count": slot.uncertain_count,
                "last_polarity": slot.last_polarity, "last_authority": slot.last_authority,
                "first_provenance": slot.first_provenance.hex(), "last_provenance": slot.last_provenance.hex(),
                "structure_ref": slot.structure_ref.hex(), "hyperedge": edge.identity.hex(),
                "participants": [{"role": r, "ordinal": o, "node": n.hex()} for r, o, n in slot.participants]}

    # -- durability ---------------------------------------------------------------------------------------------------
    def op_publish(self, op) -> dict:
        s = self.session
        state = s.require_state()
        ticket = s.runtime.publish_recoverable(state, wait=op.get("wait", True) is not False)
        return {"ticket": _ticket(ticket), "durable": _snapshot(s.runtime.continuity.snapshot())}

    def op_reconcile(self, op) -> dict:
        s = self.session
        if s.state is not None:
            raise CompositionError("LINEAGE_BOUND", "nothing to reconcile: the lineage is resumable")
        action = op.get("action")
        candidate = _hex32(op.get("candidate"), "candidate")
        if action == "discard":
            s.runtime.discard_k1_candidate(candidate)
        elif action == "adopt":
            s.runtime.adopt_k1_candidate(candidate)
        else:
            raise CompositionError("CLI_INPUT", "action: discard or adopt")
        s.refresh()
        return {"action": action, "recovery": _recovery(s.recovery),
                "durable": _snapshot(s.runtime.continuity.snapshot())}

    # -- the contextual (WARM) lane: session-local ---------------------------------------------------------------------
    @property
    def context(self) -> ContextLane:
        if self._context is None:
            self._context = ContextLane()
        return self._context

    def _consumed(self, op):
        return self.context if op.get("consume_context") else None

    def op_context_axiom(self, op) -> dict:
        refs = tuple(sorted(ContextReference(r["chunk"], r["corpus"], r["provenance"]) for r in op.get("references", ())))
        axiom = ContextAxiom(op.get("key"), refs, op.get("status", PROVISIONAL), int(op.get("relevance", 0)))
        sequence, outcome = self.context.submit(Observation(axiom=axiom, required=bool(op.get("required", False))))
        return {"sequence": sequence, "outcome": outcome}

    def op_context_relation(self, op) -> dict:
        relation = ContextRelation(int(op.get("relation_type", 0)), tuple(op.get("members", ())),
                                   op.get("provenance"), CANDIDATE)   # ADMITTED only from structure's proof
        sequence, outcome = self.context.submit(Observation(relation=relation))
        return {"sequence": sequence, "outcome": outcome}

    def op_context_view(self, op) -> dict:
        if op.get("wait") is not None:
            outcome = self.context.wait(int(op["wait"]), 30.0)
        else:
            outcome = None
        view = self.context.view()
        return {"waited": outcome, "version": view.version, "digest": view.digest, "predecessor": view.predecessor,
                "through": view.through, "dropped": view.dropped,
                "axioms": [{"key": a.key, "status": a.status, "references": len(a.references)} for a in view.axioms],
                "relations": [{"type": r.relation_type, "members": list(r.members), "status": r.status}
                              for r in view.relations]}


def _result(name, op, call) -> tuple[dict, bool]:
    expected = op.get("expect") if type(op) is dict else None
    try:
        out = {"op": name, "ok": True, **call()}
    except (CompositionError, K1Error) as exc:
        out = {"op": name, "ok": False, "code": exc.code, "detail": str(exc)}
    if expected is not None:
        out["expected"] = expected
        out["met"] = (not out["ok"]) and out["code"] == expected
        return out, out["met"]
    return out, out["ok"]


def _emit(out: dict) -> None:
    sys.stdout.write(json.dumps(out, sort_keys=True) + "\n")
    sys.stdout.flush()


def _script(path: str) -> list:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    text = text.strip()
    if text.startswith("["):
        ops = json.loads(text)
    else:
        ops = [json.loads(line) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if type(ops) is not list:
        raise CompositionError("CLI_INPUT", "a script is a JSON array or JSON lines of operations")
    return ops


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m elpis.runtime.cli", description=__doc__.split("\n\n")[0])
    p.add_argument("--home", required=True, help="the deployment home (absolute path)")
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init")
    i.add_argument("--native-build", required=True)
    i.add_argument("--testing", action="store_true")
    i.add_argument("--source-commit")
    sub.add_parser("status")
    c = sub.add_parser("create")
    for name, default in asdict(LineageConfig()).items():
        c.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    q = sub.add_parser("query")
    q.add_argument("--x", required=True)
    learn = sub.add_parser("learn")
    for name in ("x", "y", "schedule"):
        learn.add_argument("--" + name, required=True)
    learn.add_argument("--rate", type=float, default=0.002)
    learn.add_argument("--grant")
    learn.add_argument("--publish", action="store_true")
    e = sub.add_parser("evidence")
    e.add_argument("--items", required=True)
    e.add_argument("--grant")
    e.add_argument("--sources", default="")
    e.add_argument("--publish", action="store_true")
    lk = sub.add_parser("lookup")
    lk.add_argument("--relation", required=True)
    lk.add_argument("--target-role", required=True)
    lk.add_argument("--cues", required=True)
    s = sub.add_parser("slot")
    s.add_argument("--key", required=True)
    sub.add_parser("publish")
    sub.add_parser("candidates")
    r = sub.add_parser("reconcile")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--discard")
    g.add_argument("--adopt")
    sc = sub.add_parser("session")
    sc.add_argument("--script", required=True, help="a JSON array or JSON lines of operations; '-' reads stdin")
    return p


def _role(text):
    return text if text.startswith(("hyperedge:", "incidence:")) else int(text, 0)


def _ops(args) -> list:
    """The operations a one-shot command runs (a session runs its script)."""
    cmd = args.command
    if cmd == "status":
        return [{"op": "status"}]
    if cmd == "query":
        return [{"op": "query", "x": json.loads(args.x)}]
    if cmd == "learn":
        ops = [{"op": "learn", "x": json.loads(args.x), "y": json.loads(args.y),
                "schedule": json.loads(args.schedule), "rate": args.rate, "grant": args.grant}]
        return ops + ([{"op": "publish"}] if args.publish else [])
    if cmd == "evidence":
        sources = [int(v) for v in args.sources.split(",") if v.strip()]
        ops = [{"op": "evidence", "items": json.loads(args.items), "grant": args.grant, "sources": sources}]
        return ops + ([{"op": "publish"}] if args.publish else [])
    if cmd == "lookup":
        return [{"op": "lookup", "relation": _role(args.relation), "target_role": _role(args.target_role),
                 "cues": json.loads(args.cues)}]
    if cmd == "slot":
        return [{"op": "slot", "key": args.key}]
    if cmd == "publish":
        return [{"op": "publish"}]
    if cmd == "candidates":
        return [{"op": "candidates"}]
    if cmd == "reconcile":
        action, candidate = ("discard", args.discard) if args.discard else ("adopt", args.adopt)
        return [{"op": "reconcile", "action": action, "candidate": candidate}]
    if cmd == "session":
        return _script(args.script)
    raise AssertionError(cmd)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    home = Path(args.home)
    if not home.is_absolute():
        home = Path.cwd() / home   # made absolute without resolving symlinks (the home is then checked as given)
    try:
        if args.command == "init":
            d = init_deployment(home, Path(args.native_build).absolute(), testing=args.testing,
                                source_commit=args.source_commit)
            _emit({"op": "init", "ok": True, "home": str(d.home), "catalog_sha256": d.config["catalog_sha256"],
                   "libraries": d.libraries.identities, "testing_build": d.testing})
            return 0
        deployment = Deployment(home)
        if args.command == "create":
            fields = {name: getattr(args, name) for name in asdict(LineageConfig())}
            out, ok = _result("create", {}, lambda: deployment.create_lineage(LineageConfig(**fields)))
            _emit(out)
            return 0 if ok else 3
        ops = _ops(args)
    except (CompositionError, K1Error) as exc:
        _emit({"op": args.command, "ok": False, "code": exc.code, "detail": str(exc)})
        return 4
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        _emit({"op": args.command, "ok": False, "code": "CLI_INPUT", "detail": str(exc)})
        return 2
    console = Console(deployment)
    all_ok = True
    try:
        for op in ops:
            name = op.get("op") if type(op) is dict else None
            out, ok = _result(name, op, lambda op=op: console.run(op))
            _emit(out)
            all_ok &= ok
    finally:
        console.close()
    return 0 if all_ok else 3


if __name__ == "__main__":
    sys.exit(main())
