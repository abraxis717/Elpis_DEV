"""R13 P2: the published Geodome scheduler as the numerical reference. RESEARCH_ONLY; reference path only.

``scheduler_reference`` runs Geodome's own ``SparseScheduler`` over its own ``NodeState`` (float64, sparse or dense
residency) with PURE callbacks that compute the same kernel functions as the tensor executor:

    transport(arc, u, v, source_row) = kernels.message(source_row, port(u -> v), port(v -> u))
    update(v, old_row, inbox)        = kernels.update(old_row, sum of the inbox messages in Geodome's arc order)

The callbacks convert rows between Geodome's tuples and float64 tensors: this is the reference path, never the
differentiable one. Nothing here mutates state or scheduler inside a callback (Geodome's rule). The result is
compared with ``activation.run_changed_destination`` (matched semantics) and, on the workload class where the two
coincide, with ``activation.run_synchronous``.
"""
from __future__ import annotations

import torch

GEODOME_BATCH = 1024                     # NodeState.commit's batch cap


def scheduler_reference(substrate, kernels, vertices, arcs, initial: torch.Tensor, start, *, rounds: int,
                        max_active: int, max_visits: int, mode: str = "sparse") -> dict:
    """Run Geodome's scheduler until it stops. ``initial [N, D]`` float64 rows in ``vertices`` order."""
    g = substrate.geodome
    width = initial.shape[1]
    state = g.new_state(substrate.index, width=width, precision="f64", mode=mode,
                        capacity=len(vertices) if mode == "sparse" else None)
    handles = tuple(state.acquire_vertex(v) for v in vertices)
    rows = [tuple(r) for r in initial.tolist()]
    for at in range(0, len(handles), GEODOME_BATCH):
        state.commit(tuple(zip(handles[at:at + GEODOME_BATCH], rows[at:at + GEODOME_BATCH])))
    scheduler = g.SparseScheduler(state, handles, tuple(arcs), max_active=max_active, max_rounds=rounds,
                                  max_visits=max_visits)
    scheduler.start(tuple(start))
    dtype = torch.float64

    def transport(arc, u, v, row):
        with torch.no_grad():
            m = kernels.message(torch.tensor([row], dtype=dtype), torch.tensor([substrate.port(u, v)]),
                                torch.tensor([substrate.port(v, u)]))
        return tuple(m[0].tolist())

    def update(v, old, inbox):
        with torch.no_grad():
            total = torch.zeros(1, width, dtype=dtype)
            for _, message in inbox:                         # Geodome hands the inbox sorted by arc identity
                total = total + torch.tensor([message], dtype=dtype)
            new = kernels.update(torch.tensor([old], dtype=dtype), total)
        return tuple(new[0].tolist())

    records = []
    while scheduler.status == "RUNNING":
        r = scheduler.step(transport, update)
        records.append((r.status, r.sources, r.arcs, r.destinations, r.changed, r.touched))
    final = torch.tensor([state.read(h) for h in handles], dtype=dtype)
    return {"rows": final, "records": records, "status": scheduler.status, "rounds": scheduler.rounds,
            "visits": scheduler.visits, "payload": state.payload()}
