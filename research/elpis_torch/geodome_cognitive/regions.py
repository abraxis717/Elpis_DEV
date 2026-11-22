"""R13 P5: generic physical regions of the Geodome substrate and their fixed bindings. RESEARCH_ONLY.

A region is the first ``size`` vertices of a breadth-first search from a center (neighbours in the index's own
order), with EVERY physical arc among them admitted (``GeodomeSubstrate.physical_arcs``: real edges only, never an
edge inferred from address proximity). Its bindings depend on the region alone, never on an episode, a route or an
answer:
- the query and the readout at the center;
- evidence item k at the k-th of ``evidence`` vertices taken evenly, by vertex id, from the ring at graph distance
  ``ring`` from the center inside the region.

Centers are drawn per study seed, uniformly and independently from all vertices; a draw is accepted when its region
is disjoint from the earlier ones and its ring holds enough vertices. Every topology, port and distance statistic is
recorded with the region.
"""
from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    center: int
    vertices: tuple                 # Geodome vertex ids, ascending
    arcs: tuple                     # every physical arc among them, ascending arc ids
    query: int
    readout: int
    evidence: tuple                 # evidence binding k -> vertex
    stats: dict


def ball(sub, center: int, size: int) -> tuple:
    order, seen, at = [center], {center}, 0
    while len(order) < size:
        for v in sub.neighbors(order[at]):
            if v not in seen and len(order) < size:
                seen.add(v)
                order.append(v)
        at += 1
    return tuple(sorted(order))


def distances(neighbors, inside, source) -> dict:
    """Breadth-first graph distances from ``source`` within the vertex set ``inside`` (``neighbors(v)`` -> iterable)."""
    out, queue = {source: 0}, deque([source])
    while queue:
        u = queue.popleft()
        for v in neighbors(u):
            if v in inside and v not in out:
                out[v] = out[u] + 1
                queue.append(v)
    return out


def even_pick(candidates, count: int) -> tuple:
    """``count`` of the sorted ``candidates``, evenly spaced (deterministic)."""
    ordered = sorted(candidates)
    return tuple(ordered[(i * len(ordered)) // count] for i in range(count))


def make_region(sub, center: int, *, size: int = 48, evidence: int = 8, ring: int = 2) -> Region:
    vertices = ball(sub, center, size)
    inside = set(vertices)
    dist = distances(sub.neighbors, inside, center)
    on_ring = [v for v, d in dist.items() if d == ring]
    if len(on_ring) < evidence:
        raise ValueError(f"only {len(on_ring)} vertices at distance {ring}")
    bound = even_pick(on_ring, evidence)
    arcs = sub.physical_arcs(vertices)
    region = Region(center, vertices, arcs, center, center, bound, {})
    return Region(center, vertices, arcs, center, center, bound, region_stats(sub, region))


def region_stats(sub, region: Region) -> dict:
    inside = set(region.vertices)
    induced = {v: [w for w in sub.neighbors(v) if w in inside] for v in region.vertices}
    to_readout = distances(lambda v: induced[v], inside, region.readout)
    pairwise = [distances(lambda v: induced[v], inside, a)[b] for i, a in enumerate(region.evidence)
                for b in region.evidence[i + 1:]]
    ports = Counter(sub.port(*sub.endpoints(a)) for a in region.arcs)
    eccentric = max(max(distances(lambda v: induced[v], inside, v).values()) for v in region.vertices)
    return {"vertices": len(region.vertices), "arcs": len(region.arcs), "edges": len(region.arcs) // 2,
            "degree_histogram_induced": dict(sorted(Counter(len(n) for n in induced.values()).items())),
            "degree_five_vertices": sum(len(sub.neighbors(v)) == 5 for v in region.vertices),
            "connected": len(to_readout) == len(region.vertices),
            "evidence_to_readout": [to_readout[v] for v in region.evidence],
            "evidence_pairwise": {"min": min(pairwise), "mean": sum(pairwise) / len(pairwise), "max": max(pairwise)},
            "readout_eccentricity": max(to_readout.values()), "diameter": eccentric,
            "port_histogram": dict(sorted(ports.items()))}


def draw_regions(sub, seed: int, count: int, *, size: int = 48, evidence: int = 8, ring: int = 2,
                 limit: int = 1000) -> tuple:
    """``count`` disjoint regions with independent uniform centers, from a generator fixed by ``seed``."""
    rng = random.Random(f"r13-p5-regions:{seed}")
    regions, used, draws = [], set(), 0
    while len(regions) < count:
        draws += 1
        if draws > limit:
            raise RuntimeError("no admissible region")
        center = rng.randrange(sub.vertices)
        try:
            region = make_region(sub, center, size=size, evidence=evidence, ring=ring)
        except ValueError:
            continue
        if used & set(region.vertices):
            continue
        used |= set(region.vertices)
        regions.append(region)
    return tuple(regions)
