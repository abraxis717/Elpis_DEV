"""The offline reference operator interface: ``python -m elpis.runtime.cli (--native-build BUILD | --libraries DIR)
[--catalog-sha256 HEX] [lineage options] <command>``.

Not the production path. The production entry point is the native operator host ``elpis`` (docs/NATIVE_HOST.md),
which runs no interpreter. This interface runs the same composite lineage in the same volatile ECS_C session
(RuntimeCore ABI v8) and exists for differential tests (the native host must answer exactly as it does), the
qualification instruments bound to their own commits, and research (docs/OPERATING_ELPIS.md, "The offline reference
interface").

RAM-only (docs/ELPIS_MISSION.md, Amendment 3). The libraries are read, read-only, from an operator build tree or an
installed library directory and admitted only against the operator's catalog pin; there is no home and nothing is
written. A process is one session at a time: the lineage's deterministic initial composite, built in memory and
anchored at a fresh genesis. Every command prints JSON (one object per operation) and exits 0 only when every operation
succeeded or was refused exactly as its script expected:

    catalog                                        the catalog of the libraries and its SHA-256 (the pin); no write
    status                                         installation, pins, ABI, lineage, session, live state, footprint
    query --x ROWS                                 numeric QUERY of the live state (read-only)
    learn --x ROWS --y TARGETS --schedule PAIRS --rate R --grant TEXT
    evidence --items JSON --grant TEXT --sources 1,2              typed HACF evidence, no K1 learning
                                                   (each source must be an ACTIVE source of the session)
    lookup --relation R --target-role R --cues JSON                 exact LOOKUP by net evidence
    slot --key HEX                                 one memory slot and the hyperedge it was observed from
    session --script FILE|-                        many operations in one process; a script founds the session's
                                                   sources ({"op": "source_found", "label", "source_authority",
                                                   "per_turn", "scope_label"[, "version", "activate"]}), and
                                                   "close" / "open" end the session and start a new one

Everything learned or recorded is VOLATILE: it lasts until the session ends (``close``, or the process). The
persistent commands of ABI v7 (init, create, publish, candidates, reconcile, --home, --publish) are refused
``OPERATION_RETIRED``. There is no text interface: no qualified ECS<->DSV4 codec map exists
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

from .authority import SourceGrant
from .cognition import LearnAuthority, ObservationAuthority, QueryStimulus, Stimulus
from .context import (CANDIDATE, ContextAxiom, ContextLane, ContextReference, ContextRelation, Observation,
                      PROVISIONAL)
from .deployment import Installation, LineageConfig, catalog_of, struct_digest
from .errors import CompositionError
from .fuel import CEILING, CognitiveBudget
from .observation import (HYPEREDGE_NAMESPACE, INCIDENCE_NAMESPACE, HacfParticipant, StructuralEvidence,
                          record_evidence)

CAPABILITIES = {
    "numeric_query": "WORKING, mechanically verified: f_W(x) of the live K1 state (read-only)",
    "numeric_learn": "WORKING, mechanically verified: the qualified K1 law (qualified only in its frozen regime) "
                     "under an explicit LearnAuthority; volatile: discarded when the session ends",
    "typed_evidence": "WORKING, mechanically verified: validated HACF hyperedge + assertion recorded by an "
                      "observe-only turn (RuntimeCore ABI v5: no K1 learning)",
    "lookup": "WORKING: exact relation LOOKUP by net evidence; a conflict between sources is reported, never resolved",
    "persistence": "NONE: nothing is written; every session starts at a fresh genesis from the lineage's initial state",
    "recoverable_publication": "RETIRED (OPERATION_RETIRED): no persistent writer exists",
    "restart_recovery": "RETIRED (OPERATION_RETIRED): a restart is a new session",
    "context_lane": "WORKING, session-local: bounded immutable versioned views of admitted references; never "
                    "persisted; never authorizes ECS learning",
    "text": "UNAVAILABLE: ECS_CODEC_UNQUALIFIED (no qualified ECS<->DSV4 codec map)",
    "memory_learn": "UNQUALIFIED: never enabled (no memory capability is ever set)",
    "memory_slow_consolidation": "NOT PROVIDED",
    "contextual_dynamics": "NOT IMPLEMENTED: no qualified law (HCA research is research-only)",
}
NONCLAIMS = ("no language understanding", "no general intelligence", "no autonomous evolution",
             "no hierarchical superiority", "an observation is evidence storage, never truth or authority",
             "pins authenticate bytes, not authors")


RETIRED = ("init", "create", "publish", "candidates", "reconcile", "authority_reconcile")


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


def _snapshot(s) -> dict | None:
    if s is None:
        return None
    return {"state_format": s.state_format, "identity": s.k1_state_digest.hex() if s.k1_state_digest else None,
            "generation": s.generation}


class Console:
    """The operations of one process over one installation: one session at a time (``session`` runs many operations;
    one-shot commands run one)."""

    def __init__(self, installation: Installation, lineage: LineageConfig, nonce_seed: bytes | None = None):
        self.installation, self.lineage, self.nonce_seed = installation, lineage, nonce_seed
        self._session = None
        self._context = None

    # -- lifecycle ----------------------------------------------------------------------------------------------------
    @property
    def session(self):
        if self._session is None:
            self._session = self.installation.session(self.lineage, nonce_seed=self.nonce_seed)
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
        if name in RETIRED:
            raise CompositionError("OPERATION_RETIRED", f"{name!r} belonged to the retired persistent runtime: Elpis "
                                                        "writes no state; a session ends with the process or `close`")
        handler = getattr(self, "op_" + name, None)
        if handler is None:
            raise CompositionError("CLI_INPUT", f"unknown operation {name!r}")
        return handler(op)

    # -- inspection ---------------------------------------------------------------------------------------------------
    def op_status(self, op) -> dict:
        i = self.installation
        out = {"installation": {"source": str(i.source), "build_tree": i.build_tree, "catalog_sha256": i.catalog_sha256,
                                "persistence": "NONE"},
               "abi": {"runtime": 8, "composite": 1}, "libraries": i.libraries.identities,
               "budget_ceiling": {k: v for k, v in asdict(CEILING).items()},
               "capabilities": CAPABILITIES, "nonclaims": list(NONCLAIMS), "lineage": asdict(self.lineage)}
        s = self._session
        out["session"] = None if s is None else s.runtime.continuity.status()
        if s is not None and s.state is not None:
            identity, generation = s.runtime.live_identity(s.state)
            out["live"] = {"identity": identity.hex(), "generation": generation}
            out["initial_identity"] = s.initial_identity.hex()
            out["footprint"] = s.state.inspect()
            out["residency"] = "RAM: an ECS composite substrate (FMS residency holds apply to K1 FMS lineages)"
            out["fault"] = s.runtime.fault
        return out

    def op_open(self, op) -> dict:
        s = self.session
        return {"session": s.runtime.continuity.status(), "anchored": _snapshot(s.anchored),
                "initial_identity": s.initial_identity.hex()}

    def op_close(self, op) -> dict:
        """End the session: everything it learned, recorded or founded is discarded (it was only ever in memory)."""
        s, discarded = self._session, None
        if s is not None and s.state is not None:
            identity, generation = s.runtime.live_identity(s.state)
            discarded = {"live_identity": identity.hex(), "live_generation": generation,
                         "learned": identity != s.initial_identity}
        self.close()
        return {"closed": s is not None, "discarded": discarded}

    def op_live(self, op) -> dict:
        s = self.session
        state = s.require_state()
        identity, generation = s.runtime.live_identity(state)
        composite, k1, memory = self.installation.libraries.composite.envelope_identities(state.snapshot())
        return {"live": {"identity": identity.hex(), "generation": generation, "k1_identity": k1.hex(),
                         "memory_identity": memory.hex()},
                "initial_identity": s.initial_identity.hex()}

    def op_counters(self, op) -> dict:
        """RuntimeCore's native crossings since this session opened (``reset`` restarts the count)."""
        return {"counters": self.session.runtime._core.counters(reset=bool(op.get("reset", False)))}

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
        hacf = self.installation.libraries.hacf
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
        edge = self.installation.libraries.hacf.hyperedge_of(slot)
        return {"found": True, "relation_type": slot.relation_type, "assert_count": slot.assert_count,
                "negate_count": slot.negate_count, "uncertain_count": slot.uncertain_count,
                "last_polarity": slot.last_polarity, "last_authority": slot.last_authority,
                "first_provenance": slot.first_provenance.hex(), "last_provenance": slot.last_provenance.hex(),
                "structure_ref": slot.structure_ref.hex(), "hyperedge": edge.identity.hex(),
                "participants": [{"role": r, "ordinal": o, "node": n.hex()} for r, o, n in slot.participants]}

    # -- the operator's sources ---------------------------------------------------------------------------------------
    def op_source_found(self, op) -> dict:
        """The operator founds (and by default activates) a source on the open session (the native host's op)."""
        text, scope = op.get("label"), op.get("scope_label")
        if type(text) is not str or not text or type(scope) is not str or not scope:
            raise CompositionError("CLI_INPUT", "label and scope_label: non-empty strings")
        activate = op.get("activate", True)
        if type(activate) is not bool:
            raise CompositionError("CLI_INPUT", "activate: a boolean")
        values = [op.get("source_authority"), op.get("per_turn"), op.get("version", 1)]
        if any(type(v) is not int for v in values):
            raise CompositionError("CLI_INPUT", "source_authority, per_turn and version: integers")
        grant = SourceGrant(text, values[0], values[1], scope, values[2], activate)
        return self.session.runtime.continuity.found_source(grant)

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
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--native-build", help="an operator build tree (absolute path; read-only)")
    where.add_argument("--libraries", help="an installed library directory (absolute path; read-only)")
    p.add_argument("--catalog-sha256", help="the pin of the libraries' catalog (`catalog` prints it)")
    p.add_argument("--nonce-seed", help="64 hex: reproducible sessions for differential runs")
    for name, default in asdict(LineageConfig()).items():
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    sub.add_parser("status")
    q = sub.add_parser("query")
    q.add_argument("--x", required=True)
    learn = sub.add_parser("learn")
    for name in ("x", "y", "schedule"):
        learn.add_argument("--" + name, required=True)
    learn.add_argument("--rate", type=float, default=0.002)
    learn.add_argument("--grant")
    e = sub.add_parser("evidence")
    e.add_argument("--items", required=True)
    e.add_argument("--grant")
    e.add_argument("--sources", default="")
    lk = sub.add_parser("lookup")
    lk.add_argument("--relation", required=True)
    lk.add_argument("--target-role", required=True)
    lk.add_argument("--cues", required=True)
    s = sub.add_parser("slot")
    s.add_argument("--key", required=True)
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
        return [{"op": "learn", "x": json.loads(args.x), "y": json.loads(args.y),
                 "schedule": json.loads(args.schedule), "rate": args.rate, "grant": args.grant}]
    if cmd == "evidence":
        sources = [int(v) for v in args.sources.split(",") if v.strip()]
        return [{"op": "evidence", "items": json.loads(args.items), "grant": args.grant, "sources": sources}]
    if cmd == "lookup":
        return [{"op": "lookup", "relation": _role(args.relation), "target_role": _role(args.target_role),
                 "cues": json.loads(args.cues)}]
    if cmd == "slot":
        return [{"op": "slot", "key": args.key}]
    if cmd == "session":
        return _script(args.script)
    raise AssertionError(cmd)


_RETIRED_COMMANDS = ("init", "create", "publish", "candidates", "reconcile", "authority-reconcile")
_RETIRED_OPTIONS = ("--home", "--publish", "--testing", "--source-commit")


def _retired(argv) -> str | None:
    """A retired option anywhere, or a retired command in the command position (the first token that is neither an
    option nor a global option's value)."""
    valued = {"--native-build", "--libraries", "--catalog-sha256", "--nonce-seed"} | {
        "--" + name.replace("_", "-") for name in asdict(LineageConfig())}
    retired = next((token for token in argv if token in _RETIRED_OPTIONS), None)
    if retired is not None:
        return retired
    skip = False
    for token in argv:
        if skip:
            skip = False
        elif token.startswith("--"):
            skip = token in valued
        else:
            return token if token in _RETIRED_COMMANDS else None
    return None


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    retired = _retired(argv)
    if retired is not None:
        _emit({"op": retired.lstrip("-"), "ok": False, "code": "OPERATION_RETIRED",
               "detail": "the persistent interface is retired: there is no home, no deployment state, no publication "
                         "and no recovery (docs/OPERATING_ELPIS.md)"})
        return 2
    args = _parser().parse_args(argv)
    source = Path(args.native_build or args.libraries)
    if not source.is_absolute():
        source = Path.cwd() / source   # made absolute without resolving symlinks
    build_tree = args.native_build is not None
    try:
        if args.command == "catalog":
            catalog, pin = catalog_of(source, build_tree=build_tree)
            _emit({"op": "catalog", "ok": True, "catalog": json.loads(catalog), "catalog_sha256": pin})
            return 0
        if args.catalog_sha256 is None:
            raise CompositionError("CLI_INPUT", "--catalog-sha256: the pin of the catalog (`catalog` prints it)")
        nonce = None if args.nonce_seed is None else _hex32(args.nonce_seed, "--nonce-seed")
        lineage = LineageConfig(**{name: getattr(args, name) for name in asdict(LineageConfig())})
        installation = Installation(source, args.catalog_sha256, build_tree=build_tree)
        ops = _ops(args)
    except CompositionError as exc:
        _emit({"op": args.command, "ok": False, "code": exc.code, "detail": str(exc)})
        return 2 if exc.code == "CLI_INPUT" else 4
    except K1Error as exc:
        _emit({"op": args.command, "ok": False, "code": exc.code, "detail": str(exc)})
        return 4
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        _emit({"op": args.command, "ok": False, "code": "CLI_INPUT", "detail": str(exc)})
        return 2
    console = Console(installation, lineage, nonce)
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
