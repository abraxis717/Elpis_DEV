"""TRACE COMPARISON AND FAILURE ATTRIBUTION.

The harness drives one contract-level trace through (a) the backend under
test and (b) the independent oracle, then attributes the FIRST semantic
divergence to an operation index and a state field.  A backend never
supplies its own expectations; only the oracle does.
"""
from __future__ import annotations
from dataclasses import dataclass

from . import oracle_r0 as O


def _pair(rec):
    if rec is None:
        return ("NO_RECEIPT", "NO_RECEIPT")
    return (rec.status, rec.code)


def evaluate(bank, oracle, trace, checkpoints=(), tol=1e-9):
    """Replay `trace` through both models; return an attribution record."""
    cps = set(checkpoints)
    cps.add(len(trace) - 1)
    bank_snaps, oracle_snaps = {}, {}
    divergences = []
    first = None
    for i, op in enumerate(trace):
        # Only the adapter-under-test is guarded: an adapter that raises or
        # returns a malformed receipt is a detected contract violation, and the
        # oracle is never guarded, so a genuine oracle fault still propagates.
        try:
            rb = O.replay_one(bank, op)
        except Exception as exc:                      # adapter boundary
            divergences.append((i, op[0], "adapter_raised",
                                type(exc).__name__ + ":" + str(exc)[:120], None))
            if first is None:
                first = i
            continue
        ro = O.replay_one(oracle, op)
        if O.compare_receipts(rb, ro):
            divergences.append((i, op[0], "receipt",
                                _pair(rb), _pair(ro)))
            if first is None:
                first = i
        if i in cps:
            try:
                sb = bank.snapshot()
            except Exception as exc:                  # adapter boundary
                divergences.append((i, op[0], "snapshot_raised",
                                    type(exc).__name__, None))
                if first is None:
                    first = i
                continue
            so = oracle.snapshot()
            fields = O.compare_snapshots(sb, so, tol)
            if fields:
                divergences.append((i, op[0], "snapshot",
                                    [f[0] for f in fields],
                                    [(f[1], f[2]) for f in fields]))
                if first is None:
                    first = i
    return {
        "trace_length": len(trace),
        "divergences": divergences,
        "first_divergence_index": first,
        "final_bank": bank.snapshot(),
        "final_oracle": oracle.snapshot(),
        "bank_occupancy": bank.occupancy(),
        "oracle_occupancy": oracle.occupancy(),
    }


replay_one = O.replay_one
compare_snapshots = O.compare_snapshots
compare_receipts = O.compare_receipts

from . import contract_r0 as _C
digest_state = _C.digest_state


def retention(bank, relation_ids):
    """Fraction of the given relation identities still resident."""
    snap = bank.snapshot()
    live = {r[0] for r in snap["relations"]}
    return sum(1 for k in relation_ids if k in live) / max(1, len(relation_ids))


def make_pair(spec, registry, transport, policy):
    """Fresh (backend, oracle) pair over the same contract inputs."""
    from . import reference_backend as RB
    return (RB.ReferenceBank(spec, registry, transport, policy=policy),
            O.OracleModel(spec, registry, transport, policy=policy))


def fresh_pair_factory(spec, registry, transport, policy):
    """Return a zero-arg callable producing a fresh pair (no state sharing)."""
    def make():
        from . import reference_backend as RB
        return RB.ReferenceBank(spec, registry, transport, policy=policy), O.OracleModel(
            spec, registry, transport, policy=policy)
    return make
