"""R13 P5: CONTROL graphs for the competence study. RESEARCH_ONLY. NOT Geodome topology.

The P5 controls run the SAME tensor executor (``activation.run_synchronous``) and the SAME bridge computation
(``CognitiveSession.compute`` / ``refine``) on graphs that are NOT Geodome's physical topology. They never pass
through ``compile_plan`` and never carry Geodome authorization. A ``ControlCompiled`` duck-types exactly the fields
the executor and ``compute`` read from a ``CompiledPlan`` (``src``, ``dst``, ``ports``, ``back_ports``,
``incoming``, ``query``, ``evidence``, ``readout``, ``plan.rounds``, ``plan.schedule``) and says what it is.

Builders (local vertex indices; every graph undirected, both arc directions admitted, as in a physical region):
- ``from_compiled``: a compiled physical plan's own arcs and ports (the reference for the others).
- ``relabeled`` (RELABEL_ISOMORPHIC): the same graph under a declared seeded permutation of the vertex indices.
  Arcs keep their order and their port ordinals, so the kernels receive the same inputs, permuted.
- ``rewired`` (REWIRED): degree-preserving double-edge swaps of the physical region (every vertex keeps its degree,
  the graph stays simple and connected); |E| / 2 successful swaps.
- ``randomized`` (COMPUTE_MATCHED): a fully mixed random graph with the region's degree sequence (a deterministic
  Havel-Hakimi graph, then 10 |E| successful degree-preserving swaps), simple and connected.
- ``star`` (FLAT_RECURRENT): one hub (query and readout) and ``evidence`` leaves, arcs leaf -> hub only, port 0.
- ``port_permuted`` (evaluation-only diagnostic): a compiled plan with its port ordinals remapped.

For ``rewired`` and ``randomized`` the bindings are placed to match the evidence-to-readout distance ``ring`` as
closely as possible: the readout is the vertex with the most vertices at that distance (lowest index on ties), and
the evidence vertices are taken evenly from that ring, then from the nearest distances. Ports of rebuilt graphs are
neighbour-list ordinals (a vertex's neighbours in ascending local index), as Geodome's are ordinals of its own
neighbour lists. Every mismatch against the physical region is recorded in ``stats``.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field, replace

import torch

from .regions import distances, even_pick

LABEL = "CONTROL GRAPH: not Geodome topology; never Geodome-authorized"


@dataclass(frozen=True)
class ControlPlan:
    rounds: int
    schedule: str = "synchronous"


@dataclass(frozen=True)
class ControlCompiled:
    plan: ControlPlan
    src: torch.Tensor
    dst: torch.Tensor
    ports: torch.Tensor
    back_ports: torch.Tensor
    incoming: torch.Tensor
    query: torch.Tensor
    evidence: torch.Tensor
    readout: torch.Tensor
    vertices: int
    kind: str
    stats: dict = field(default_factory=dict)
    label: str = LABEL


def _long(values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.long)


def build(n: int, arcs, *, query, evidence, readout, rounds: int, kind: str, stats: dict) -> ControlCompiled:
    """``arcs``: (src, dst, port, back_port) in aggregation order."""
    arcs = list(arcs)
    incoming = torch.zeros(n, dtype=torch.bool)
    if arcs:
        incoming[_long(a[1] for a in arcs)] = True
    return ControlCompiled(ControlPlan(rounds), _long(a[0] for a in arcs), _long(a[1] for a in arcs),
                           _long(a[2] for a in arcs), _long(a[3] for a in arcs), incoming, _long(query),
                           _long(evidence), _long(readout), n, kind, stats)


def with_evidence(compiled, count: int):
    """The same plan with only its first ``count`` evidence bindings (fewer admitted items)."""
    return replace(compiled, evidence=compiled.evidence[:count])


def from_compiled(compiled) -> tuple:
    """``(n, arcs, (query, evidence, readout))`` of a compiled plan, in local indices and its arc order."""
    arcs = list(zip(compiled.src.tolist(), compiled.dst.tolist(), compiled.ports.tolist(),
                    compiled.back_ports.tolist()))
    return (compiled.incoming.numel(), arcs,
            (compiled.query.tolist(), compiled.evidence.tolist(), compiled.readout.tolist()))


def edges_of(arcs) -> set:
    return {frozenset((u, v)) for u, v, *_ in arcs}


def adjacency(n: int, edges) -> dict:
    neighbors = {v: [] for v in range(n)}
    for e in edges:
        a, b = tuple(e)
        neighbors[a].append(b)
        neighbors[b].append(a)
    return {v: sorted(w) for v, w in neighbors.items()}


def graph_stats(n: int, edges, readout: int, evidence, ring: int) -> dict:
    neighbors = adjacency(n, edges)
    to_readout = distances(lambda v: neighbors[v], set(range(n)), readout)
    pairwise = [distances(lambda v: neighbors[v], set(range(n)), a).get(b) for i, a in enumerate(evidence)
                for b in evidence[i + 1:]]
    reach = [d for d in pairwise if d is not None]
    return {"vertices": n, "edges": len(edges), "arcs": 2 * len(edges),
            "degree_histogram": dict(sorted(Counter(len(x) for x in neighbors.values()).items())),
            "connected": len(to_readout) == n,
            "evidence_to_readout": [to_readout.get(v) for v in evidence],
            "evidence_distance_mismatch": sum(abs(to_readout.get(v, n) - ring) for v in evidence),
            "evidence_pairwise": {"min": min(reach), "mean": sum(reach) / len(reach), "max": max(reach)}
            if reach else {}}


def _neighbor_ports(n: int, edges) -> list:
    neighbors = adjacency(n, edges)
    arcs = []
    for u in range(n):
        for v in neighbors[u]:
            arcs.append((u, v, neighbors[u].index(v), neighbors[v].index(u)))
    return sorted(arcs, key=lambda a: (a[1], a[0]))           # aggregation in a fixed (destination, source) order


def _connected(n: int, edges) -> bool:
    neighbors = adjacency(n, edges)
    return len(distances(lambda v: neighbors[v], set(range(n)), 0)) == n


def _swap(n: int, edges: set, rng: random.Random, swaps: int, limit: int) -> set:
    """``swaps`` successful degree-preserving double-edge swaps keeping the graph simple and connected."""
    edges, done, tries = set(edges), 0, 0
    ordered = sorted(edges, key=sorted)
    while done < swaps:
        tries += 1
        if tries > limit:
            raise RuntimeError("degree-preserving swaps stalled")
        e1, e2 = rng.sample(ordered, 2)
        a, b = sorted(e1)
        c, d = sorted(e2)
        if rng.random() < 0.5:
            c, d = d, c
        new1, new2 = frozenset((a, d)), frozenset((c, b))
        if len(new1) < 2 or len(new2) < 2 or new1 in edges or new2 in edges or new1 == new2:
            continue
        trial = (edges - {e1, e2}) | {new1, new2}
        if not _connected(n, trial):
            continue
        edges, done = trial, done + 1
        ordered = sorted(edges, key=sorted)
    return edges


def _place(n: int, edges, count: int, ring: int) -> tuple:
    neighbors = adjacency(n, edges)
    best = max(range(n), key=lambda v: (sum(d == ring for d in distances(lambda x: neighbors[x], set(range(n)),
                                                                        v).values()), -v))
    dist = distances(lambda x: neighbors[x], set(range(n)), best)
    at_ring = [v for v, d in dist.items() if d == ring]
    if len(at_ring) >= count:
        return best, even_pick(at_ring, count)
    rest = sorted((v for v in dist if v != best and v not in at_ring), key=lambda v: (abs(dist[v] - ring), dist[v], v))
    return best, tuple(sorted(at_ring)) + tuple(rest[:count - len(at_ring)])


def relabeled(compiled, seed: int, rounds: int) -> ControlCompiled:
    n, arcs, (query, evidence, readout) = from_compiled(compiled)
    sigma = list(range(n))
    random.Random(f"r13-p5-relabel:{seed}").shuffle(sigma)
    mapped = [(sigma[u], sigma[v], p, b) for u, v, p, b in arcs]
    stats = {"permutation_seed": seed, "identical_ports": True, "arcs": len(mapped)}
    return build(n, mapped, query=[sigma[q] for q in query], evidence=[sigma[e] for e in evidence],
                 readout=[sigma[r] for r in readout], rounds=rounds, kind="RELABEL_ISOMORPHIC", stats=stats)


def rewired(compiled, seed: int, rounds: int, ring: int) -> ControlCompiled:
    n, arcs, _ = from_compiled(compiled)
    edges = edges_of(arcs)
    rng = random.Random(f"r13-p5-rewire:{seed}")
    new = _swap(n, edges, rng, len(edges) // 2, 200 * len(edges))
    readout, evidence = _place(n, new, compiled.evidence.numel(), ring)
    stats = graph_stats(n, new, readout, evidence, ring)
    stats["edges_kept_from_region"] = len(new & edges)
    return build(n, _neighbor_ports(n, new), query=[readout], evidence=list(evidence), readout=[readout],
                 rounds=rounds, kind="REWIRED", stats=stats)


def havel_hakimi(degrees) -> set:
    remaining = sorted(((d, v) for v, d in enumerate(degrees)), reverse=True)
    edges = set()
    while remaining and remaining[0][0] > 0:
        d, v = remaining.pop(0)
        if d > len(remaining):
            raise ValueError("not graphical")
        for i in range(d):
            dv, w = remaining[i]
            edges.add(frozenset((v, w)))
            remaining[i] = (dv - 1, w)
        remaining.sort(reverse=True)
    return edges


def randomized(compiled, seed: int, rounds: int, ring: int) -> ControlCompiled:
    n, arcs, _ = from_compiled(compiled)
    edges = edges_of(arcs)
    degrees = [sum(v in e for e in edges) for v in range(n)]
    start = havel_hakimi(degrees)
    rng = random.Random(f"r13-p5-randomize:{seed}")
    if not _connected(n, start):                       # connect first (swaps from a connected graph stay connected)
        start = _connect(n, start, rng)
    new = _swap(n, start, rng, 10 * len(edges), 400 * len(edges))
    readout, evidence = _place(n, new, compiled.evidence.numel(), ring)
    stats = graph_stats(n, new, readout, evidence, ring)
    stats["edges_shared_with_region"] = len(new & edges)
    stats["degree_sequence_equal"] = sorted(degrees) == sorted(sum(v in e for e in new) for v in range(n))
    return build(n, _neighbor_ports(n, new), query=[readout], evidence=list(evidence), readout=[readout],
                 rounds=rounds, kind="COMPUTE_MATCHED", stats=stats)


def _connect(n: int, edges: set, rng: random.Random) -> set:
    """Join components by degree-preserving cross swaps (one edge from each of two components)."""
    edges = set(edges)
    for _ in range(10 * n):
        neighbors = adjacency(n, edges)
        comp = distances(lambda x: neighbors[x], set(range(n)), 0)
        if len(comp) == n:
            return edges
        inside = [e for e in edges if next(iter(e)) in comp]
        outside = [e for e in edges if next(iter(e)) not in comp]
        a, b = sorted(rng.choice(inside))
        c, d = sorted(rng.choice(outside))
        new1, new2 = frozenset((a, c)), frozenset((b, d))
        if new1 in edges or new2 in edges:
            continue
        edges = (edges - {frozenset((a, b)), frozenset((c, d))}) | {new1, new2}
    raise RuntimeError("could not connect the degree sequence")


def star(evidence: int, rounds: int) -> ControlCompiled:
    arcs = [(leaf, 0, 0, 0) for leaf in range(1, evidence + 1)]
    stats = {"vertices": evidence + 1, "arcs": evidence, "ports": "all 0 (no port information)",
             "evidence_to_readout": [1] * evidence}
    return build(evidence + 1, arcs, query=[0], evidence=list(range(1, evidence + 1)), readout=[0], rounds=rounds,
                 kind="FLAT_RECURRENT", stats=stats)


def port_permuted(compiled, permutation) -> ControlCompiled:
    n, arcs, (query, evidence, readout) = from_compiled(compiled)
    mapped = [(u, v, permutation[p], permutation[b]) for u, v, p, b in arcs]
    return build(n, mapped, query=query, evidence=evidence, readout=readout, rounds=compiled.plan.rounds,
                 kind="PORT_PERMUTED", stats={"permutation": list(permutation)})
