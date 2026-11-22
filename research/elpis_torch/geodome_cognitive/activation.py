"""R13 P2: activation schedules and the tensor executor. RESEARCH_ONLY.

Two schedules, never presented as numerically equivalent:

``synchronous`` (the trainable path)
    Fixed rounds. Every round, every admitted arc carries a message computed from its source's OLD row; every
    enrolled vertex with at least one admitted incoming arc updates from its old row and the sum of its messages
    (in arc-identity order); every other vertex keeps its row. No data-dependent control flow and no host
    extraction (``.item()``, ``.tolist()``, NumPy, callbacks, ctypes) anywhere on the path, so gradients reach every
    kernel through the whole bounded horizon. Information moves at most one physical hop per round.

``changed_destination`` (Geodome's ``SparseScheduler`` semantics, the reference mode)
    A frontier starts at the plan's ``start``. Each round, only arcs out of the frontier carry messages; only their
    destinations update; a destination whose row changed joins the next frontier; nothing else does. Budgets and
    statuses are Geodome's: ``ACTIVE_BUDGET`` (sources plus destinations above ``max_active``), ``ARC_BUDGET`` (visits
    above the budget), ``ROUND_BUDGET`` (rounds exhausted while running), ``QUIESCENT`` (nothing changed). Its
    frontier and status decisions read scalars from the rows (as Geodome must); it is the deterministic reference,
    not the training path. Rows still carry gradients through the updates that happened.

When the two coincide. If every enrolled vertex has an admitted incoming arc, the frontier starts with every
enrolled vertex, and every destination's row changes in every round, the changed-destination frontier stays the
whole enrolled set and both schedules perform the same computation round by round. ``reference.py`` checks that
class against the Geodome scheduler itself; outside it the schedules differ, by design.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SYNCHRONOUS, CHANGED_DESTINATION = "synchronous", "changed_destination"


@dataclass(frozen=True)
class RoundRecord:
    """One changed-destination round, the fields of Geodome's ``Round``."""
    status: str
    sources: int
    arcs: int
    destinations: int
    changed: int
    touched: int


@dataclass(frozen=True)
class Execution:
    rows: torch.Tensor                  # [N, D] the enrolled rows after the last round
    rounds: int
    visits: int                         # arc messages computed
    status: str
    records: tuple = ()                 # changed_destination only


def run_synchronous(rows: torch.Tensor, compiled, kernels) -> Execution:
    """``compiled.plan.rounds`` synchronous rounds over ``rows [N, D]``."""
    src, dst, ports, back = compiled.src, compiled.dst, compiled.ports, compiled.back_ports
    incoming = compiled.incoming[:, None]
    for _ in range(compiled.plan.rounds):
        messages = kernels.message(rows.index_select(0, src), ports, back)
        inbox = torch.zeros_like(rows).index_add(0, dst, messages)
        rows = torch.where(incoming, kernels.update(rows, inbox), rows)
    rounds = compiled.plan.rounds
    return Execution(rows=rows, rounds=rounds, visits=rounds * compiled.src.numel(), status="COMPLETED")


def run_changed_destination(rows: torch.Tensor, compiled, kernels) -> Execution:
    """Geodome's ``SparseScheduler`` loop (``start`` then ``step`` until it is no longer RUNNING), in tensors."""
    src, dst, ports, back = compiled.src, compiled.dst, compiled.ports, compiled.back_ports
    n = rows.shape[0]
    frontier = compiled.start.clone()
    status, rounds, visits, records = "RUNNING", 0, 0, []
    while status == "RUNNING":
        sources = int(frontier.sum())
        if rounds >= compiled.plan.rounds:
            status = "ROUND_BUDGET"
            records.append(RoundRecord(status, sources, 0, 0, 0, 0))
            break
        active = frontier.index_select(0, src)
        arcs = int(active.sum())
        dest = torch.zeros(n, dtype=torch.bool, device=rows.device)
        if arcs:
            dest[dst[active]] = True
        touched = int((frontier | dest).sum())
        if touched > compiled.max_active:
            status = "ACTIVE_BUDGET"
            records.append(RoundRecord(status, sources, 0, 0, 0, 0))
            break
        if visits + arcs > compiled.max_visits:
            status = "ARC_BUDGET"
            records.append(RoundRecord(status, sources, 0, 0, 0, 0))
            break
        if arcs:
            pick = active.nonzero().squeeze(1)
            messages = kernels.message(rows.index_select(0, src[pick]), ports[pick], back[pick])
            inbox = torch.zeros_like(rows).index_add(0, dst[pick], messages)
            candidate = kernels.update(rows, inbox)
            changed = dest & (candidate != rows).any(dim=1)
            rows = torch.where(changed[:, None], candidate, rows)
        else:
            changed = torch.zeros(n, dtype=torch.bool, device=rows.device)
        rounds += 1
        visits += arcs
        count = int(changed.sum())
        status = "RUNNING" if count else "QUIESCENT"
        records.append(RoundRecord(status, sources, arcs, int(dest.sum()), count, touched))
        frontier = changed
    return Execution(rows=rows, rounds=rounds, visits=visits, status=status, records=tuple(records))


def execute(rows: torch.Tensor, compiled, kernels) -> Execution:
    if compiled.plan.schedule == SYNCHRONOUS:
        return run_synchronous(rows, compiled, kernels)
    return run_changed_destination(rows, compiled, kernels)
