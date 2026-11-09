"""R118 fixtures: historical windows imported losslessly, and synthetic sparse graphs.

Historical import (needs the frozen oracle / torch): the frames are produced by the
frozen pipeline itself (``portable_mechanics.rotation`` -> ``r62.transport_matrices``)
and stored as exact 2x2 blocks; participant labels map to IDs order-preservingly so
the canonical link order equals ``sorted(...)`` in ``link_builder``.

Synthetic graphs (numpy only) cover disconnected, connected sparse, heterogeneous
(Zipf) degree, overlap-heavy and adversarial constructions. Windows are formed by
``extract_window``: a bounded, deterministic neighborhood extraction.
"""
from __future__ import annotations

import numpy as np

from .registry import RegistryError, RegistryLimits, build_registry, matrix_to_blocks, rotation_blocks

HIST_PARTICIPANT_BASE = 7_000_000_000_000
HIST_RELATION_BASE = 9_000_000_000_000


def historical_frames(o, graph, mode: str):
    """{(rid, p): 4x4 torch frame} exactly as portable_mechanics.graph_adjs builds them."""
    T, pm = o.torch, o.pm
    base = {(rid, p): pm.rotation(.19 + .17 * rid - .067 * p, -.41 + .103 * rid + .049 * p)
            for rid, members in enumerate(graph) for p in members}
    return o.r62.transport_matrices(None, base, mode, o.r61, pm.R3Adapter(), T)


def historical_registry(o, graph, mode: str = "PARTICIPANT_NONFLAT", copies: int = 1, limits=RegistryLimits()):
    """``copies`` disjoint relation copies of one 4-relation fixture over shared participants."""
    mats = historical_frames(o, graph, mode)
    labels = sorted({p for row in graph for p in row})
    pids = [HIST_PARTICIPANT_BASE + p for p in labels]
    rel_ids, offsets, mids, frames = [], [0], [], []
    for c in range(copies):
        for rid, members in enumerate(graph):
            rel_ids.append(HIST_RELATION_BASE + 4 * c + rid)
            for p in sorted(members):
                mids.append(HIST_PARTICIPANT_BASE + p)
                frames.append(matrix_to_blocks(mats[(rid, p)].numpy()))
            offsets.append(len(mids))
    reg = build_registry(pids, rel_ids, offsets, mids, np.array(frames), limits)
    return reg, [[HIST_RELATION_BASE + 4 * c + rid for rid in range(4)] for c in range(copies)]


# ---------------------------------------------------------------------------------
def synthetic_registry(num_participants: int, num_relations: int, seed: int, arity=(2, 6), zipf: float = 0.0,
                       components: int = 1, limits=RegistryLimits(), hub_fraction: float = 0.0):
    """Random sparse hypergraph with stable random 64-bit IDs (small graphs: per-relation
    sampling is O(P) with ``zipf``; use ``bulk_synthetic_registry`` for large ones).

    * arity uniform in [arity[0], arity[1]];
    * ``zipf > 0`` draws members with probability ~ rank^-zipf (heterogeneous degree);
    * ``components > 1`` splits participants and relations into disconnected blocks;
    * ``hub_fraction`` forces participant 0 of each component into that fraction of relations.
    """
    rng = np.random.default_rng(seed)
    pid = rng.choice(np.iinfo(np.int64).max, size=num_participants, replace=False).astype(np.uint64)
    pid |= np.uint64(1) << np.uint64(63)  # exercise the full unsigned 64-bit range
    rid = rng.choice(np.iinfo(np.int64).max, size=num_relations, replace=False).astype(np.uint64)
    lo, hi = arity
    ar = rng.integers(lo, hi + 1, size=num_relations)
    comp_of_rel = np.arange(num_relations) % components
    pcomp = np.array_split(np.arange(num_participants), components)
    offsets = np.zeros(num_relations + 1, dtype=np.int64)
    np.cumsum(ar, out=offsets[1:])
    members = np.empty(int(offsets[-1]), dtype=np.int64)
    for c in range(components):
        pool = pcomp[c]
        rels = np.nonzero(comp_of_rel == c)[0]
        if zipf > 0:
            w = 1.0 / np.arange(1, pool.size + 1) ** zipf
            w = w / w.sum()
        else:
            w = None
        for r in rels:
            k = int(ar[r])
            if hub_fraction > 0 and rng.random() < hub_fraction:
                rest = rng.choice(pool[1:], size=k - 1, replace=False, p=None if w is None else w[1:] / w[1:].sum())
                chosen = np.concatenate(([pool[0]], rest))
            else:
                chosen = rng.choice(pool, size=k, replace=False, p=w)
            members[offsets[r]:offsets[r + 1]] = chosen
    frames = rotation_blocks(rng.uniform(-np.pi, np.pi, members.size), rng.uniform(-np.pi, np.pi, members.size),
                             exact_libm=False)
    return build_registry(pid, rid, offsets, pid[members], frames, limits)


def bulk_synthetic_registry(num_participants: int, num_relations: int, seed: int, arity=(2, 6), zipf: float = 1.0,
                            limits=RegistryLimits()):
    """Fully vectorized generator for the million-relation stress registry.

    Members are drawn with replacement then de-duplicated per relation by resampling
    collisions; heterogeneous (Zipf) participant degree.
    """
    rng = np.random.default_rng(seed)
    pid = rng.choice(np.iinfo(np.int64).max, size=num_participants, replace=False).astype(np.uint64)
    pid |= np.uint64(1) << np.uint64(63)
    rid = rng.choice(np.iinfo(np.int64).max, size=num_relations, replace=False).astype(np.uint64)
    lo, hi = arity
    ar = rng.integers(lo, hi + 1, size=num_relations)
    offsets = np.zeros(num_relations + 1, dtype=np.int64)
    np.cumsum(ar, out=offsets[1:])
    M = int(offsets[-1])
    w = 1.0 / np.arange(1, num_participants + 1) ** zipf
    cdf = np.cumsum(w / w.sum())
    rel_of = np.repeat(np.arange(num_relations), ar)
    members = np.searchsorted(cdf, rng.random(M))
    members = np.minimum(members, num_participants - 1)
    for _ in range(64):
        order = np.lexsort((members, rel_of))
        dup = np.zeros(M, dtype=bool)
        s_rel, s_mem = rel_of[order], members[order]
        dup[order[1:]] = (s_rel[1:] == s_rel[:-1]) & (s_mem[1:] == s_mem[:-1])
        if not dup.any():
            break
        members[dup] = rng.integers(0, num_participants, size=int(dup.sum()))
    else:
        raise RegistryError("BULK_GENERATOR_DEDUP")
    frames = rotation_blocks(rng.uniform(-np.pi, np.pi, M), rng.uniform(-np.pi, np.pi, M), exact_libm=False)
    return build_registry(pid, rid, offsets, pid[members], frames, limits)


def extract_window(registry, seed_relation: int, max_incidences: int = 4096, exclude=None):
    """Bounded deterministic neighborhood: the seed plus up to 3 relations sharing the
    most participants with it (ties: lowest relation index). Raises if the participant
    expansion exceeds ``max_incidences`` (never silently truncated)."""
    mem = registry.members_of(seed_relation)
    lo, hi = registry.part_offsets[mem], registry.part_offsets[mem + 1]
    total = int((hi - lo).sum())
    if total > max_incidences:
        raise RegistryError("WINDOW_EXPANSION_BUDGET", f"{total}>{max_incidences}")
    if total == 0:
        return [int(seed_relation)]
    idx = np.repeat(lo - np.cumsum(np.concatenate(([0], (hi - lo)[:-1]))), hi - lo) + np.arange(total)
    rels = registry.inc_relation[registry.part_incidences[idx]]
    rels = rels[rels != seed_relation]
    if exclude is not None:
        rels = rels[~np.isin(rels, exclude)]
    if rels.size == 0:
        return [int(seed_relation)]
    uniq, counts = np.unique(rels, return_counts=True)
    order = np.lexsort((uniq, -counts))[:3]
    return [int(seed_relation)] + [int(r) for r in uniq[order]]
