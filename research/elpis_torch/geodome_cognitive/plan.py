"""R13 P2: execution plans and their validation. RESEARCH_ONLY.

An ``ExecutionPlan`` is host-supplied authorization: the enrolled vertices, the ADMITTED arcs (directed arc
identities of the frozen index), the schedule, the number of rounds, and the bindings (which vertices receive the
query, which receive each admitted evidence item, which are pooled by the readout). ``compile_plan`` validates it
against the substrate and the session's working state, then turns it into integer tensors. Plans are not
differentiable; nothing in them is learned.

Validation mirrors Geodome's ``SparseScheduler``: an arc is admitted only if BOTH endpoints are enrolled (an arc
identity is not authority); duplicates are refused; the enrolment, arc, round and visit caps are Geodome's
(4096, 24576, 256, 6291456) intersected with the session's configured bounds. Every vertex must already be
resident in the working state (a valid Geodome handle of the current epoch). Arcs are sorted by identity, so the
aggregation order of every destination is the arc order Geodome's scheduler hands its update callback.

The touched-vertex budget (Geodome's ``max_active``, at most 1024 per round). A changed-destination plan meets it
round by round at run time (status ``ACTIVE_BUDGET``, as Geodome's scheduler). A synchronous round touches the same
vertices every round, every source and every destination of an admitted arc, so a synchronous plan whose touched
set exceeds the budget is refused here, before anything runs (R13 owner review, hardening 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SCHEDULES = ("synchronous", "changed_destination")
GEODOME_MAX_ENROLLED, GEODOME_MAX_ARCS, GEODOME_MAX_ROUNDS = 4096, 24576, 256
GEODOME_MAX_VISITS = GEODOME_MAX_ARCS * GEODOME_MAX_ROUNDS
GEODOME_MAX_ACTIVE = 1024


class PlanError(ValueError):
    """A refused execution plan. ``code`` names the rule; nothing was changed."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class ExecutionPlan:
    vertices: tuple                     # enrolled vertex ids (distinct)
    arcs: tuple                         # admitted directed arc identities (distinct)
    rounds: int                         # synchronous: exactly; changed_destination: the round budget
    schedule: str = "synchronous"
    start: tuple = ()                   # changed_destination only: the initial frontier
    query: tuple = ()                   # vertices written with the query ingress
    evidence: tuple = ()                # evidence[k] is written with admitted evidence item k
    readout: tuple = ()                 # vertices pooled into the readout latent
    max_visits: int | None = None       # arc visits allowed (default: rounds x arcs)
    max_active: int | None = None       # touched vertices per round (Geodome's max_active; default min(1024, N))


@dataclass(frozen=True)
class CompiledPlan:
    plan: ExecutionPlan
    handles: tuple                      # Geodome StateHandles, plan order
    epoch: int                          # the working state's epoch at compilation
    slots: torch.Tensor                 # [N] working-state slots of the enrolled vertices
    src: torch.Tensor                   # [E] local (plan-order) index of each arc's source
    dst: torch.Tensor                   # [E] local index of each arc's destination
    ports: torch.Tensor                 # [E] position of the destination in the source's neighbour list
    back_ports: torch.Tensor            # [E] position of the source in the destination's neighbour list
    arc_ids: torch.Tensor               # [E] sorted ascending
    incoming: torch.Tensor              # [N] bool: has at least one admitted incoming arc
    start: torch.Tensor                 # [N] bool: initial frontier (changed_destination)
    query: torch.Tensor                 # local indices
    evidence: torch.Tensor
    readout: torch.Tensor
    max_visits: int
    max_active: int
    touched: int | None = None          # synchronous: the vertices every round touches; changed_destination: None


def _ints(values, code: str) -> tuple:
    if type(values) not in (tuple, list) or any(type(v) is not int for v in values):
        raise PlanError(code, "expected a tuple of Python integers")
    return tuple(values)


def compile_plan(plan: ExecutionPlan, working, *, max_rounds: int, max_arcs: int, max_visits: int,
                 max_evidence: int, evidence_available: int = 0) -> CompiledPlan:
    """Validate ``plan`` against the substrate and ``working``; integer tensors on the working state's device."""
    if not isinstance(plan, ExecutionPlan):
        raise PlanError("PLAN_TYPE")
    substrate = working.substrate
    vertices = _ints(plan.vertices, "VERTEX_TYPE")
    arcs = _ints(plan.arcs, "ARC_TYPE")
    if plan.schedule not in SCHEDULES:
        raise PlanError("SCHEDULE", f"schedule must be one of {SCHEDULES}")
    if not 1 <= len(vertices) <= min(GEODOME_MAX_ENROLLED, working.capacity):
        raise PlanError("ENROLLMENT_CAP")
    if any(not 0 <= v < substrate.vertices for v in vertices):
        raise PlanError("VERTEX_RANGE")
    if len(set(vertices)) != len(vertices):
        raise PlanError("DUPLICATE_VERTEX")
    if len(arcs) > min(GEODOME_MAX_ARCS, max_arcs):
        raise PlanError("ARC_CAP")
    if len(set(arcs)) != len(arcs):
        raise PlanError("DUPLICATE_ARC")
    if type(plan.rounds) is not int or not 1 <= plan.rounds <= min(GEODOME_MAX_ROUNDS, max_rounds):
        raise PlanError("ROUND_RANGE")
    budget = plan.rounds * len(arcs) if plan.max_visits is None else plan.max_visits
    if type(budget) is not int or not 0 <= budget <= min(GEODOME_MAX_VISITS, max_visits):
        raise PlanError("VISIT_BUDGET")
    if plan.schedule == "synchronous" and plan.rounds * len(arcs) > budget:
        raise PlanError("VISIT_BUDGET", "a synchronous plan visits every admitted arc every round")
    local = {v: i for i, v in enumerate(vertices)}
    try:
        handles = tuple(working.handle(v) for v in vertices)
    except KeyError as error:
        raise PlanError("NONRESIDENT", "every enrolled vertex must be resident in the working state") from error
    working.validate(handles)
    ordered = sorted(arcs)
    src, dst, ports, back = [], [], [], []
    for arc in ordered:
        try:
            u, v = substrate.endpoints(arc)
        except ValueError as error:
            raise PlanError("ARC_RANGE") from error
        if u not in local or v not in local:
            raise PlanError("ARC_AUTHORITY", f"arc {arc} ({u} -> {v}) leaves the enrolled vertices")
        src.append(local[u])
        dst.append(local[v])
        ports.append(substrate.port(u, v))
        back.append(substrate.port(v, u))
    query = _bound(plan.query, local, "QUERY_BINDING")
    evidence = _bound(plan.evidence, local, "EVIDENCE_BINDING")
    readout = _bound(plan.readout, local, "READOUT_BINDING")
    if not readout:
        raise PlanError("READOUT_BINDING", "at least one readout vertex is required")
    if set(plan.query) & set(plan.evidence):
        raise PlanError("BINDING_OVERLAP", "a vertex receives either the query or evidence, not both")
    if len(evidence) > min(max_evidence, evidence_available):
        raise PlanError("EVIDENCE_BINDING", "more evidence bindings than admitted evidence items")
    start = _ints(plan.start, "START_TYPE")
    if plan.schedule == "changed_destination":
        if not start or len(set(start)) != len(start) or any(v not in local for v in start):
            raise PlanError("START", "the initial frontier must be distinct enrolled vertices")
    elif start:
        raise PlanError("START", "a synchronous plan has no frontier (every enrolled vertex is active)")
    active = plan.max_active if plan.max_active is not None else min(GEODOME_MAX_ACTIVE, len(vertices))
    if type(active) is not int or not 1 <= active <= GEODOME_MAX_ACTIVE:
        raise PlanError("ACTIVE_BUDGET")
    touched = len(set(src) | set(dst)) if plan.schedule == "synchronous" else None
    if touched is not None and touched > active:
        raise PlanError("ACTIVE_BUDGET", f"a synchronous round touches {touched} vertices (every source and "
                                         f"destination of an admitted arc); the budget is {active}")
    device = working.latent.device
    as_long = lambda values: torch.tensor(values, dtype=torch.long, device=device)   # noqa: E731
    dst_t = as_long(dst)
    incoming = torch.zeros(len(vertices), dtype=torch.bool, device=device)
    if dst:
        incoming[dst_t] = True
    start_mask = torch.zeros(len(vertices), dtype=torch.bool, device=device)
    if start:
        start_mask[as_long([local[v] for v in start])] = True
    return CompiledPlan(plan=plan, handles=handles, epoch=working.epoch, slots=working.slots(handles),
                        src=as_long(src), dst=dst_t, ports=as_long(ports), back_ports=as_long(back),
                        arc_ids=as_long(ordered), incoming=incoming, start=start_mask, query=as_long(query),
                        evidence=as_long(evidence), readout=as_long(readout), max_visits=budget, max_active=active,
                        touched=touched)


def _bound(values, local: dict, code: str) -> list:
    values = _ints(values, code)
    if len(set(values)) != len(values) or any(v not in local for v in values):
        raise PlanError(code, "bindings must be distinct enrolled vertices")
    return [local[v] for v in values]
