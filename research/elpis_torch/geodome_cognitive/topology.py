"""R13 P1: identity and topology of the frozen Geodome substrate. RESEARCH_ONLY.

``substrate()`` is the process-wide ``GeodomeSubstrate``: the qualified frequency-120 ``GeoIndex`` (built ONCE by
``geodome.build_index``, which rejects identity drift) plus read-only helpers. Every helper answers from the
index; none invents adjacency:

* Hexaflag identity: ``address(v)``, ``resolve(address)``, ``coordinate(v)``, ``resolve_coordinate(xy)``,
  ``prefix_count`` / ``prefix_page`` (LOGICAL groups: a prefix is not necessarily a connected region; Geodome's C1
  found 236 of 343 occupied depth-three groups disconnected).
* Physical adjacency: ``neighbors(v)``, ``arc(u, v)`` (``KeyError`` for a non-edge, exactly as ``GeoIndex.arc_id``),
  ``endpoints(arc)``, ``port(u, v)`` (the position of ``v`` in ``u``'s neighbour list, 0..5: an ordinal of the
  index, not a geometric direction), ``physical_arcs(vertices)`` (every arc between members, both directions,
  never one that is not an edge), ``components(vertices)``, ``shortest_path(u, v)``.

An arc identity is identity, not authorization: an execution plan admits arcs explicitly (``plan.py``).
"""
from __future__ import annotations

import time
from collections import deque

from .source import IDENTITY, load_geodome


class GeodomeSubstrate:
    """The frozen index and its invariants. One instance per process (``substrate()``)."""

    def __init__(self, geodome):
        self.geodome = geodome
        started = time.perf_counter()
        self.index = geodome.build_index()
        self.build_seconds = time.perf_counter() - started
        index = self.index
        found = {"vertices": index.vertices, "edges": index.edges, "arcs": 2 * index.edges,
                 "mapping_sha256": index.mapping_sha256(), "edge_sha256": index.edge_sha256(),
                 "index_payload_bytes": index.payload_bytes()}
        expected = {k: IDENTITY[k] for k in found}
        if found != expected:
            raise RuntimeError(f"Geodome identity drift: {found} != {expected}")
        import geodome_index                                       # noqa: PLC0415 (loaded by load_geodome)
        if (geodome_index.CAPACITY, geodome_index.DEPTH) != (IDENTITY["capacity"], IDENTITY["depth"]):
            raise RuntimeError("Geodome address capacity drift")
        self.vertices, self.edges, self.arcs = index.vertices, index.edges, 2 * index.edges

    # -- Hexaflag identity ----------------------------------------------------------------------------------
    def address(self, v: int) -> tuple:
        return self.index.address(v)

    def resolve(self, address: tuple) -> int:
        return self.index.resolve(address)

    def coordinate(self, v: int) -> tuple:
        return self.index.coordinate(v)

    def resolve_coordinate(self, xy: tuple) -> int:
        return self.index.resolve_coordinate(xy)

    def prefix_count(self, prefix: tuple) -> int:
        return self.index.prefix_count(prefix)

    def prefix_page(self, prefix: tuple, offset: int = 0, limit: int = 1024) -> tuple:
        return self.index.prefix_page(prefix, offset, limit)

    # -- physical adjacency ---------------------------------------------------------------------------------
    def neighbors(self, v: int) -> tuple:
        return self.index.neighbors(v)

    def arc(self, u: int, v: int) -> int:
        """The directed arc identity ``u -> v``. ``KeyError('NONEDGE')`` unless ``u`` and ``v`` are adjacent."""
        return self.index.arc_id(u, v)

    def endpoints(self, arc: int) -> tuple:
        return self.index.arc_endpoints(arc)

    def port(self, u: int, v: int) -> int:
        """The position of ``v`` in ``u``'s neighbour list (0..5). ``KeyError`` for a non-edge."""
        row = self.index.neighbors(u)
        if v not in row:
            raise KeyError("NONEDGE")
        return row.index(v)

    def physical_arcs(self, vertices) -> tuple:
        """Every directed arc between two members of ``vertices``, sorted by arc identity. Only real edges."""
        members = set(_vertices(vertices, self.vertices))
        return tuple(sorted(self.index.arc_id(u, v) for u in members for v in self.index.neighbors(u)
                            if v in members))

    def components(self, vertices) -> list:
        """Connected components of ``vertices`` under physical adjacency (each sorted; ordered by smallest id)."""
        members = set(_vertices(vertices, self.vertices))
        seen, out = set(), []
        for start in sorted(members):
            if start in seen:
                continue
            part, queue = [], deque([start])
            seen.add(start)
            while queue:
                u = queue.popleft()
                part.append(u)
                for v in self.index.neighbors(u):
                    if v in members and v not in seen:
                        seen.add(v)
                        queue.append(v)
            out.append(tuple(sorted(part)))
        return out

    def shortest_path(self, u: int, v: int, max_length: int = 4096) -> tuple:
        """A shortest physical path ``u .. v`` (vertices), breadth first with neighbours in index order."""
        _vertices((u, v), self.vertices)
        parent, queue = {u: None}, deque([u])
        while queue:
            x = queue.popleft()
            if x == v:
                path = [x]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return tuple(reversed(path))
            for y in self.index.neighbors(x):
                if y not in parent:
                    parent[y] = x
                    queue.append(y)
            if len(parent) > max_length * 7:
                break
        raise ValueError("no path within the search bound")

    def identity_record(self) -> dict:
        index = self.index
        return {"vertices": self.vertices, "edges": self.edges, "arcs": self.arcs, "capacity": IDENTITY["capacity"],
                "depth": IDENTITY["depth"], "mapping_sha256": index.mapping_sha256(),
                "edge_sha256": index.edge_sha256(), "index_payload_bytes": index.payload_bytes(),
                "degree_five_vertices": sum(len(index.neighbors(v)) == 5 for v in range(self.vertices)),
                "build_seconds": round(self.build_seconds, 3)}


def _vertices(vertices, limit: int) -> tuple:
    out = tuple(vertices)
    if any(type(v) is not int or not 0 <= v < limit for v in out):
        raise ValueError("VERTEX_RANGE")
    return out


_SUBSTRATE: dict = {}


def substrate() -> GeodomeSubstrate:
    """The process-wide substrate: the index is built once, on first use, and never per token or round."""
    if "instance" not in _SUBSTRATE:
        _SUBSTRATE["instance"] = GeodomeSubstrate(load_geodome())
    return _SUBSTRATE["instance"]
