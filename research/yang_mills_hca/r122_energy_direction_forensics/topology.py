"""R122 incidence-topology audit: isomorphism signatures, historical inventory, generation.

A fixture graph is four relations (rows) over participant labels; slot 0 is the source
relation and slot 3 the readout relation of the historical paired response.

* ``signature``   complete invariant of the incidence structure up to participant
  relabeling and relation permutation: for every participant the set of relations that
  contain it is a 4-bit mask; the histogram of masks 1..15, minimized over the 24
  relation permutations. Two graphs are incidence-isomorphic iff the signatures agree.
* ``rooted_signature`` the same with the source and readout slots fixed (only the two
  intermediate relations may be exchanged). It separates fixtures that share an
  incidence class but place source and readout differently.
* ``canonical_form`` an independently coded complete invariant (sorted per-participant
  masks instead of a histogram) used to cross-check ``signature``.

``historical_graphs`` inventories *every* four-relation graph in the frozen host
protocols (R61-R78), the R92 helper protocol and the R121 protocol. R121's own novelty
gate compared against 12 of these classes only (R61 ``r55_structural``, R76, R77).
"""
from __future__ import annotations

import itertools
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
YM_ROOT = HERE.parent
SNAPSHOT = YM_ROOT / "frozen_host" / "source" / "frozen_snapshot"
R92_PROTOCOL = YM_ROOT / "frozen_host" / "source" / "research_helpers" / "r92" / "protocol.json"
R121_PROTOCOL = YM_ROOT / "r121_temporal_topology" / "PROTOCOL_R121.json"


def _check(graph):
    if len(graph) != 4 or any(len(row) != 4 or len(set(row)) != 4 for row in graph):
        raise ValueError("INCIDENCE_SHAPE")
    if any(type(p) is not int or p < 0 for row in graph for p in row):
        raise ValueError("PARTICIPANT_LABEL")


def _masks(graph, perm):
    members = sorted({p for row in graph for p in row})
    return [sum(1 << i for i, r in enumerate(perm) if p in graph[r]) for p in members]


def signature(graph) -> tuple:
    """Histogram over membership masks 1..15, minimized over relation permutations."""
    _check(graph)
    best = None
    for perm in itertools.permutations(range(4)):
        masks = _masks(graph, perm)
        hist = tuple(sum(m == k for m in masks) for k in range(1, 16))
        best = hist if best is None or hist < best else best
    return best


def rooted_signature(graph) -> tuple:
    """As ``signature`` but with slot 0 (source) and slot 3 (readout) fixed."""
    _check(graph)
    return min(tuple(sum(m == k for m in _masks(graph, perm)) for k in range(1, 16))
               for perm in ((0, 1, 2, 3), (0, 2, 1, 3)))


def canonical_form(graph) -> tuple:
    """Independent complete invariant: sorted per-participant masks, minimized over permutations."""
    _check(graph)
    return min(tuple(sorted(_masks(graph, perm))) for perm in itertools.permutations(range(4)))


def participants(graph) -> int:
    return len({p for row in graph for p in row})


def max_degree(graph) -> int:
    members = [p for row in graph for p in row]
    return max(members.count(p) for p in set(members))


def connected(graph) -> bool:
    reached, frontier = {0}, set(graph[0])
    for _ in range(4):
        for i, row in enumerate(graph):
            if frontier.intersection(row):
                frontier.update(row)
                reached.add(i)
    return len(reached) == 4


def hop_distance(graph, src: int = 0, dst: int = 3) -> int:
    """Relation-overlap hops from the source slot to the readout slot (-1 if unreachable)."""
    dist = {src: 0}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        for nxt in range(4):
            if nxt not in dist and set(graph[cur]).intersection(graph[nxt]):
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist.get(dst, -1)


def _is_graph(obj) -> bool:
    return (isinstance(obj, list) and len(obj) == 4 and all(
        isinstance(row, list) and len(row) == 4 and len(set(map(str, row))) == 4
        and all(type(x) is int and x >= 0 for x in row) for row in obj))


def _walk(obj, path, source, out):
    if isinstance(obj, dict):
        for key, val in obj.items():
            _walk(val, path + [str(key)], source, out)
    elif _is_graph(obj):
        out[source + ":" + "/".join(path)] = [list(row) for row in obj]
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            _walk(val, path + [str(i)], source, out)


def historical_graphs() -> dict:
    """Every 4x4 relation graph found in the frozen protocols, the R92 helper and R121."""
    out = {}
    for path in sorted(SNAPSHOT.glob("FROZEN_PROTOCOL_R*.json")):
        _walk(json.loads(path.read_bytes()), [], path.name.replace("FROZEN_PROTOCOL_", "").replace(".json", ""), out)
    _walk(json.loads(R92_PROTOCOL.read_bytes()), [], "R92helper", out)
    _walk(json.loads(R121_PROTOCOL.read_bytes()), [], "R121", out)
    return out


def historical_classes() -> dict:
    """{signature: sorted list of 'source:name' labels} over the complete historical inventory."""
    classes = {}
    for label, graph in historical_graphs().items():
        source, path = label.split(":", 1)
        classes.setdefault(signature(graph), set()).add(source + ":" + path.rsplit("/", 1)[-1])
    return {sig: sorted(names) for sig, names in classes.items()}


def generate(seed: int, degree_schedule, used: set, attempts: int = 300000) -> list:
    """Deterministic scan for new connected 10-participant fixture graphs.

    Row order of a returned graph is its slot order (source 0, readout 3). ``used`` is
    extended, so the returned graphs are mutually non-isomorphic and absent from it.
    """
    rng = random.Random(seed)
    used = set(used)
    out = []
    for degree in degree_schedule:
        for _ in range(attempts):
            graph = [sorted(rng.sample(range(10), 4)) for _ in range(4)]
            if len({tuple(row) for row in graph}) != 4 or participants(graph) != 10 or not connected(graph):
                continue
            if max_degree(graph) != degree:
                continue
            sig = signature(graph)
            if sig in used:
                continue
            used.add(sig)
            out.append(graph)
            break
        else:
            raise RuntimeError("GRAPH_SCAN_EXHAUSTED:degree=" + str(degree))
    return out


def audit(graphs: dict) -> dict:
    """Novelty audit of candidate graphs against the complete historical inventory."""
    hist = historical_classes()
    hist_canon = {canonical_form(g) for g in historical_graphs().values()}
    rows = {}
    seen, seen_canon = set(), set()
    ok = True
    for name, graph in graphs.items():
        sig, canon = signature(graph), canonical_form(graph)
        collides = sorted(hist.get(sig, []))
        duplicate = sig in seen
        # the two complete invariants must agree on every equality decision
        consistent = ((sig in hist) == (canon in hist_canon)) and ((sig in seen) == (canon in seen_canon))
        seen.add(sig)
        seen_canon.add(canon)
        novel = (not collides) and (not duplicate) and consistent and participants(graph) == 10 and connected(graph)
        ok = ok and novel
        rows[name] = {"signature": list(sig), "rooted_signature": list(rooted_signature(graph)),
                      "max_degree": max_degree(graph), "source_readout_hops": hop_distance(graph),
                      "historical_collisions": collides, "duplicate_within_set": duplicate,
                      "invariants_consistent": consistent, "novel": novel}
    return {"historical_graph_definitions": len(historical_graphs()), "historical_classes": len(hist),
            "graphs": rows, "all_novel": ok}
