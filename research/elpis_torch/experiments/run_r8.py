#!/usr/bin/env python3
"""Elpis Torch R8: what the R5 evidence read costs, and whether bounded reuse of query-independent work reduces it.
Records go only where --out says.

    baseline    P0. One stream per (geometry, PLE setting), prefilled once to each occupancy without reads. Every
                configuration then starts from an identical clone of that snapshot:
                  off           the R5 read disabled (mode "off"; the index is still kept current)
                  uncached      R5 on; every candidate is reread and re-encoded
                  cache_<n>     R5 on with the existing R6 BlockCache bounded to n encoded blocks
                Decode, interleaved step by step (the order rotates):
                  natural       the cache starts empty at the snapshot and warms by itself
                  replay        the same steps again from the snapshot, each cache as the natural pass left it
                                (repeated queries: the warm bound)
                  novel         the cache emptied before every step (no reuse: the cold bound with the cache's cost)
                Prefill of 512 tokens: logits at every position (two chunks: cold cache, then warm) and at the last
                position only. Stage times come from the model's own laps plus wrappers that split the reread
                into validation, token reads, re-encoding and the rest (measurement only).
    represent   P1. The query-independent tensors of the read path: shapes, bytes per block and per stream, when
                they exist, and the 1M context budget of the R6 scale configurations (formulas; nothing allocated)
    admission   P4. The R5 reference read (``fine_keys="reread"``) against R8's stored fine keys (``"stored"``),
                each without and with the R6 cache, from the same snapshots: decode (natural, replay, novel),
                prefill (every position, last position), stage times, counters, bytes, and the logit gap between
                the two paths at every measured step. A warm-up pass precedes the first measurement.
    smoke       every command once, tiny

Wall-clock numbers are PERFORMANCE_ONLY: one shared CPU, untrained weights (the candidate sets of an untrained
model say nothing about a trained model's; the natural decode pass is one point between the replay and novel
bounds). Nothing here writes outside --out.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from .. import evidence_model, evidence_ring
from ..context_runtime import BlockCache
from ..evidence_model import EvidenceConfig, EvidenceLM

PHASE = "ELPIS_TORCH_R8_EVIDENCE_READ_PATH"
REPO = Path(__file__).resolve().parents[3]
SEED = 8
# R5's default evidence head (the R7 latency model's) and the R6 scale configurations' head (a 256-wide encoder)
GEOMETRIES = {
    "enc64": {},
    "enc256": dict(encoder_dim=256, encoder_heads=4, read_heads=8, read_head_dim=128, fine_dim=64, query_tile=32),
}
OCCUPANCIES = (4096, 65536, 262144, 1 << 20)
CACHE_BLOCKS = (64, 512, 4096)          # one query's candidates, eight queries', sixty-four queries'
COLD_STEPS = 8                          # decode steps reported as "cold" (the cache starts empty at the snapshot)


# -- models and state -------------------------------------------------------------------------------------------

def timing_model(geometry: str, *, capacity: int = 1 << 20, fine_keys: str = "reread") -> EvidenceLM:
    """The R7 latency core (d 256, 4 blocks, 8 experts top-2 in blocks 1 and 3, a PLE before block 1) with an R5
    head of ``geometry``. UNTRAINED, frozen, float32. ``model.ple.enabled`` switches the PLE (exactly). The weights
    are seeded by name: models built with other ``fine_keys`` are identical."""
    from .run_r7 import latency_model
    ersa = latency_model(True, False).ersa
    model = EvidenceLM(ersa, EvidenceConfig(capacity=capacity, fine_keys=fine_keys, **GEOMETRIES[geometry])).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def clone_state(state, *, tier: bool = True):
    """An independent copy of an ``EvidenceState`` (measurement only): tensors copied, a new ring identity, no
    cache, no timing. The cache is volatile and never copied (it refuses serialization). ``tier=False`` drops an R8
    fine-key tier: the copy is then exactly an R5 state (nothing is maintained that the reference does not need)."""
    cache, timing, fine = state.cache, state.timing, state.ring.fine
    state.cache = state.timing = None
    if not tier:
        state.ring.fine = None
    try:
        twin = copy.deepcopy(state)
    finally:
        state.cache, state.timing, state.ring.fine = cache, timing, fine
    twin.ring.session = next(evidence_ring._SESSIONS)
    twin.counters = {}
    if not tier:
        twin.ring.stats.pop("fine_reads", None)
        twin.fine_revision, twin.fine_stale = None, False
    return twin


def entry_bytes(model, batch: int = 1) -> int:
    """Bytes of one BlockCache entry: an encoded block ``[B, s + 1, width]`` in the model's dtype."""
    e = model.evidence
    return batch * (e.block + 1) * model.encoder.width * model.embed.weight.element_size()


# -- measurement ------------------------------------------------------------------------------------------------

class StageProbe:
    """Measurement only. Wraps (as instance attributes; no class is changed) the model's ``_encode`` and
    ``_reread_encoded`` and each measured ring's ``_check`` and ``reread``, accumulating EXCLUSIVE seconds:

        validate        ring liveness / identity checks
        token_reads     exact token rereads from the ring (validation excluded)
        reencode        encoder calls made by the reread (candidates not served by the cache)
        reread_rest     the rest of the reread: cache lookups and inserts, bookkeeping, stacking
        chunk_encode    encoder calls for the chunk's own touched blocks (the local block at decode)
        topk            the deterministic top-K selections (coarse candidates, then fine evidence)

    and the blocks each reread asked for (first-time versus repeated reads)."""

    def __init__(self, model):
        self.sink, self.stack, self.in_reread, self.asked = {}, [], False, []
        encode, reread = model._encode, model._reread_encoded

        def timed_encode(windows, valid):
            return self._timed("reencode" if self.in_reread else "chunk_encode", encode, windows, valid)

        def timed_reread(state, blocks, device):
            self.asked.append(blocks.tolist())
            self.in_reread = True
            try:
                return self._timed("reread_rest", reread, state, blocks, device)
            finally:
                self.in_reread = False

        model._encode, model._reread_encoded = timed_encode, timed_reread
        if not _TOPK:                                           # a module global of evidence_model: wrapped once
            topk = evidence_model.exact_topk
            _TOPK.append(topk)
            evidence_model.exact_topk = lambda scores, ids, k: (
                _ACTIVE[0]._timed("topk", topk, scores, ids, k) if _ACTIVE[0] is not None else topk(scores, ids, k))
        _ACTIVE[0] = self

    def activate(self) -> None:
        _ACTIVE[0] = self

    def attach(self, ring) -> None:
        check, reread = ring._check, ring.reread
        ring._check = lambda blocks, at: self._timed("validate", check, blocks, at)
        ring.reread = lambda blocks, overlap, at=None: self._timed("token_reads", reread, blocks, overlap, at)

    def _timed(self, name, fn, *args):
        started = time.perf_counter()
        self.stack.append(0.0)
        try:
            return fn(*args)
        finally:
            elapsed = time.perf_counter() - started
            nested = self.stack.pop()
            self.sink[name] = self.sink.get(name, 0.0) + elapsed - nested
            if self.stack:
                self.stack[-1] += elapsed

    def take(self) -> tuple[dict, list]:
        out, asked = self.sink, self.asked
        self.sink, self.asked = {}, []
        return out, asked


_TOPK: list = []
_ACTIVE: list = [None]
LAPS = ("coarse_scan", "exact_reread", "rerank", "readout")
SUBSTAGES = ("validate", "token_reads", "reencode", "reread_rest", "chunk_encode", "topk")


class Run:
    """One configuration on its own clone of a snapshot."""

    def __init__(self, model, probe, snapshot, name: str, *, cache_blocks: int = 0, cache=None, tier: bool = True,
                 keep_logits: bool = False):
        self.model, self.probe, self.name = model, probe, name
        self.mode = "off" if name == "off" else "learned"
        self.state = clone_state(snapshot, tier=tier)
        self.keep_logits, self.logits = keep_logits, []
        if cache is not None:
            self.state.cache = cache
        elif cache_blocks:
            self.state.cache = BlockCache(cache_blocks * entry_bytes(model))
        self.state.timing = {}
        probe.attach(self.state.ring)
        self.ever: set = set()
        self.records: list = []
        self.previous: set | None = None

    def call(self, tokens: torch.Tensor, positions: torch.Tensor):
        return self.model(tokens, self.state, logit_positions=positions, mode=self.mode)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> dict:
        state, e = self.state, self.model.evidence
        cache = state.cache
        laps0 = dict(state.timing)
        counters0, ring0 = dict(state.counters), dict(state.ring.stats)
        cache0 = dict(cache.stats) if cache is not None else {}
        cached0 = set(cache.entries) if cache is not None else set()
        self.probe.activate()
        self.probe.take()
        started = time.perf_counter()
        logits = self.call(tokens, positions)
        seconds = time.perf_counter() - started
        if self.keep_logits:
            self.logits.append(logits)
        sub, asked = self.probe.take()
        row = {"seconds": seconds}
        for name in LAPS:
            row[name] = state.timing.get(name, 0.0) - laps0.get(name, 0.0)
        for name in SUBSTAGES:
            row[name] = sub.get(name, 0.0)
        delta = lambda new, old, key: new.get(key, 0) - old.get(key, 0)         # noqa: E731
        row["reencoded"] = delta(state.counters, counters0, "blocks_reencoded")
        row["queries"] = delta(state.counters, counters0, "queries")
        row["coarse_records"] = delta(state.counters, counters0, "coarse_records_scored")
        row["token_rereads"] = delta(state.ring.stats, ring0, "rereads")
        row["fine_tier_reads"] = delta(state.counters, counters0, "fine_tier_reads")
        if cache is not None:
            for key in ("hits", "misses", "evictions", "purged", "inserts"):
                row[f"cache_{key}"] = delta(cache.stats, cache0, key)
            row["cache_management_s"] = delta(cache.stats, cache0, "management_ns") * 1e-9
            row["cache_bytes"] = cache.bytes
        requested = [b for blocks in asked for b in blocks]
        fresh = [b for b in requested if b not in self.ever]
        row["requested"] = len(requested)
        row["first_time"] = len(fresh)
        row["repeat_served_by_cache"] = sum(1 for b in requested if b in cached0)
        if requested and positions.numel() == 1:           # decode: the age of what was asked (query's block - b)
            local = (state.position - 1) // e.block
            row["ages"] = sorted(local - b for b in requested)
            current = set(requested)
            if self.previous is not None:
                row["overlap_previous"] = len(current & self.previous) / len(current)
            self.previous = current
        self.ever.update(requested)
        # logical bytes: token ids reread (int32), coarse descriptors scanned, encodings served by the cache
        row["token_bytes"] = row["token_rereads"] * e.span * state.ring.tokens.element_size() * state.batch
        row["descriptor_bytes"] = row["coarse_records"] * e.per_block * e.key_dim * state.ring.keys.element_size()
        row["cache_bytes_served"] = row.get("cache_hits", 0) * entry_bytes(self.model, state.batch)
        row["fine_key_bytes"] = row["fine_tier_reads"] * e.block * e.fine_dim * \
            (state.ring.fine.element_size() if state.ring.fine is not None else 0) * state.batch
        self.records.append(row)
        return row


class SessionRun(Run):
    """The same cached configuration driven through ``ContextSession(cache_bytes=...)``: its revision check and its
    decode step. The session's own fresh state is replaced by the snapshot clone (measurement only)."""

    def __init__(self, model, probe, snapshot, name: str, *, cache_blocks: int):
        from ..context_runtime import ContextSession
        super().__init__(model, probe, snapshot, name, cache_blocks=cache_blocks)
        self.session = ContextSession(model, batch=1, cache_bytes=cache_blocks * entry_bytes(model))
        self.session.state = self.state

    def call(self, tokens: torch.Tensor, positions: torch.Tensor):
        if tokens.shape[1] != 1 or positions.tolist() != [0]:
            raise ValueError("the session run measures decode steps only")
        return self.session.step(tokens.reshape(1))[:, None]


def percentiles(samples: list, scale: float = 1000.0) -> dict:
    if not samples:
        return {"samples": 0}
    ordered = sorted(samples)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]   # noqa: E731
    return {"p50": round(scale * pick(0.5), 4), "p95": round(scale * pick(0.95), 4),
            "mean": round(scale * statistics.fmean(ordered), 4), "samples": len(ordered)}


def summarize(records: list, *, paired: list | None = None) -> dict:
    """Milliseconds (p50 / p95 / mean) of the step and of each stage, totals of the counters, and the per-step
    difference against a paired configuration (``off``) when given."""
    out = {"step_ms": percentiles([r["seconds"] for r in records])}
    if not records:
        return out
    out["stage_ms_mean"] = {name: round(1000 * statistics.fmean([r[name] for r in records]), 4)
                            for name in (*LAPS, *SUBSTAGES)}
    out["stage_ms_mean"]["outside_laps"] = round(1000 * statistics.fmean(
        [r["seconds"] - sum(r[n] for n in LAPS) - r["chunk_encode"] for r in records]), 4)
    out["reencoded_per_step"] = percentiles([r["reencoded"] for r in records], 1.0)
    for key in ("queries", "reencoded", "requested", "first_time", "repeat_served_by_cache", "token_rereads",
                "coarse_records", "cache_hits", "cache_misses", "cache_evictions", "cache_purged", "cache_inserts",
                "token_bytes", "descriptor_bytes", "cache_bytes_served", "fine_tier_reads", "fine_key_bytes"):
        if any(key in r for r in records):
            out[key] = sum(r.get(key, 0) for r in records)
    if any("cache_management_s" in r for r in records):
        out["cache_management_ms_mean"] = round(1000 * statistics.fmean([r["cache_management_s"] for r in records]), 4)
        out["cache_bytes_end"] = records[-1]["cache_bytes"]
    if out.get("cache_hits") is not None and out.get("requested"):
        out["hit_rate"] = round(out["cache_hits"] / out["requested"], 4)
    if out.get("requested"):
        out["first_time_fraction"] = round(out["first_time"] / out["requested"], 4)
    ages = [a for r in records for a in r.get("ages", ())]
    if ages:
        ages.sort()
        out["age_blocks_quantiles"] = {str(q): ages[int(q * (len(ages) - 1))] for q in (0.0, 0.1, 0.5, 0.9, 1.0)}
    overlaps = [r["overlap_previous"] for r in records if "overlap_previous" in r]
    if overlaps:
        out["overlap_with_previous_step_mean"] = round(statistics.fmean(overlaps), 4)
    if paired is not None:
        out["minus_off_ms"] = percentiles([a["seconds"] - b["seconds"] for a, b in zip(records, paired)])
    return out


def cost_fit(records: list) -> dict:
    """Least squares over decode steps: seconds = fixed + per_block x blocks re-encoded (every R5-on step of an
    occupancy: natural, replay and novel passes pooled)."""
    xs, ys = [r["reencoded"] for r in records], [r["seconds"] for r in records]
    if len(xs) < 3 or len(set(xs)) < 2:
        return {"samples": len(xs)}
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    fixed = my - slope * mx
    residual = sum((y - fixed - slope * x) ** 2 for x, y in zip(xs, ys))
    total = sum((y - my) ** 2 for y in ys)
    return {"fixed_ms": round(1000 * fixed, 4), "per_reencoded_block_ms": round(1000 * slope, 4),
            "r2": round(1 - residual / total, 4) if total else None, "samples": len(xs)}


# -- P0: the baseline -------------------------------------------------------------------------------------------

def _configs(args) -> list:
    return ["off", "uncached", *[f"cache_{n}" for n in args.cache_blocks]]


def _blocks(name: str) -> int:
    return int(name.split("_", 1)[1]) if name.startswith("cache_") else 0


@torch.no_grad()
def measure_occupancy(model, probe, snapshot, args, feed, chunks) -> dict:
    names = _configs(args)
    prefill_names = ["off", "uncached", *[f"cache_{n}" for n in args.prefill_cache_blocks]]
    lo, hi = snapshot.ring.live()
    out = {"tokens": snapshot.position, "live_blocks": hi - lo, "entry_bytes": entry_bytes(model),
           "configs": {n: {"cache_blocks": _blocks(n), "cache_capacity_bytes": _blocks(n) * entry_bytes(model)}
                       for n in dict.fromkeys(names + prefill_names)}}
    one = torch.zeros(1, dtype=torch.long)
    # decode: natural, interleaved
    runs = {n: Run(model, probe, snapshot, n, cache_blocks=_blocks(n)) for n in names}
    cached = [n for n in names if _blocks(n)]
    if cached:                    # C: the existing mechanism as a caller uses it, ContextSession(cache_bytes=...)
        through = cached[min(1, len(cached) - 1)]
        runs["session_" + through] = SessionRun(model, probe, snapshot, through, cache_blocks=_blocks(through))
        out["configs"]["session_" + through] = {"cache_blocks": _blocks(through),
                                                 "cache_capacity_bytes": _blocks(through) * entry_bytes(model)}
    order = list(runs)
    for i in range(feed.shape[0]):
        for k in range(len(order)):
            runs[order[(i + k) % len(order)]].forward(feed[i], one)
    off = runs["off"].records
    for n in order:
        records = runs[n].records
        row = {"all": summarize(records, paired=off if n != "off" else None)}
        if n != "off":
            row["cold"] = summarize(records[:COLD_STEPS], paired=off[:COLD_STEPS])
            row["warm"] = summarize(records[COLD_STEPS:], paired=off[COLD_STEPS:])
        out["configs"][n]["decode_natural"] = row
    out["state_bytes"] = runs["uncached"].state.nbytes()
    # decode: replay (repeated queries; each cache as the natural pass left it) and novel (no reuse)
    replay = {n: Run(model, probe, snapshot, n, cache=runs[n].state.cache) for n in cached}
    novel_name = cached[min(1, len(cached) - 1)] if cached else None
    novel = {"novel": Run(model, probe, snapshot, novel_name, cache_blocks=_blocks(novel_name))} if cached else {}
    both = {**{f"replay:{n}": r for n, r in replay.items()}, **novel}
    order = list(both)
    for i in range(feed.shape[0]):
        for k in range(len(order)):
            name = order[(i + k) % len(order)]
            run = both[name]
            if name == "novel":
                run.state.cache.clear()
            run.forward(feed[i], one)
    for n, run in replay.items():
        out["configs"][n]["decode_replay"] = summarize(run.records, paired=off)
    if novel:
        out["decode_novel"] = {"cache_blocks": _blocks(novel_name),
                               **summarize(novel["novel"].records, paired=off)}
    learned = [r for run in (*runs.values(), *replay.values(), *novel.values()) if run.mode == "learned"
               for r in run.records]
    out["decode_cost_fit"] = cost_fit(learned)
    del runs, replay, novel, both
    # prefill: logits at every position (cold chunk, then warm chunk) and at the last position only
    every = torch.arange(chunks.shape[-1])
    last = torch.tensor([chunks.shape[-1] - 1])
    for label, positions, pieces in (("prefill_all", every, chunks.shape[0]), ("prefill_last", last, 1)):
        timings = {n: [[] for _ in range(pieces)] for n in prefill_names}
        rows = {n: [[] for _ in range(pieces)] for n in prefill_names}
        for repeat in range(args.prefill_runs):
            for k in range(len(prefill_names)):
                n = prefill_names[(repeat + k) % len(prefill_names)]
                run = Run(model, probe, snapshot, n, cache_blocks=_blocks(n))
                for piece in range(pieces):
                    row = run.forward(chunks[piece], positions)
                    timings[n][piece].append(row["seconds"])
                    rows[n][piece].append(row)
                del run
        for n in prefill_names:
            out["configs"][n][label] = [summarize(rows[n][piece], paired=rows["off"][piece] if n != "off" else None)
                                        for piece in range(pieces)]
    scratch = model.context_interface().scratch_bytes
    out["projected_scratch_bytes"] = {
        "decode": scratch(batch=1, chunk=1, options={}, queries=1)["total"],
        "prefill_512_every_position": scratch(batch=1, chunk=512, options={}, queries=512)["total"]}
    return out


@torch.no_grad()
def command_baseline(args) -> dict:
    occupancies = tuple(int(x) for x in args.occupancies.split(",")) if args.occupancies else \
        ((2048, 4096) if args.smoke else OCCUPANCIES)
    geometries = args.geometries.split(",") if args.geometries else list(GEOMETRIES)
    ples = [p == "on" for p in args.ple.split(",")]
    steps, chunk = (COLD_STEPS + 4, 512) if args.smoke else (args.decode_steps, 512)
    generator = torch.Generator().manual_seed(SEED)
    stream = torch.randint(0, 256, (1, max(occupancies)), generator=generator)
    feed = torch.randint(0, 256, (steps, 1, 1), generator=generator)
    chunks = torch.randint(0, 256, (2, 1, chunk), generator=generator)
    out = {"phase": PHASE, "command": "baseline", "seed": SEED, "decode_steps": steps, "cold_steps": COLD_STEPS,
           "prefill_runs": args.prefill_runs, "cache_blocks": list(args.cache_blocks),
           "prefill_cache_blocks": list(args.prefill_cache_blocks), "threads": torch.get_num_threads(),
           "model": "R7 latency core (d 256, 4 blocks, 8 experts top-2 in blocks 1 and 3, PLE 2 x 4 x 32771 x 32 "
                    "before block 1), R5 head per geometry, capacity 1,048,576; UNTRAINED (timing only)",
           "stream": "uniform random byte ids (seeded); the decode feed and the prefill chunks are the same for "
                     "every configuration", "rows": []}
    for geometry in geometries:
        model = timing_model(geometry, capacity=args.capacity)
        probe = StageProbe(model)
        for ple in ples:
            model.ple.enabled = ple
            snapshot = model.new_state(1)
            ingest, at = [], 0
            for target in occupancies:
                while at < target:
                    piece = stream[:, at:min(target, at + chunk)]
                    started = time.perf_counter()
                    model(piece, snapshot, logit_positions=torch.zeros(0, dtype=torch.long))
                    ingest.append((time.perf_counter() - started) / piece.shape[1])
                    at += piece.shape[1]
                row = {"geometry": geometry, "evidence": dataclass_dict(model.evidence), "ple": ple,
                       "occupancy": target, "ingest_us_per_token_no_reads": percentiles(ingest, 1e6)}
                row.update(measure_occupancy(model, probe, snapshot, args, feed, chunks))
                out["rows"].append(row)
                decode = {n: c["decode_natural"]["all"]["step_ms"]["p50"] for n, c in row["configs"].items()
                          if "decode_natural" in c}
                print(f"[r8 baseline] {geometry} ple={ple} {target}: decode p50 ms {decode}", flush=True)
                if args.out:
                    _write(args.out, out)
        del model, probe
    return out


# -- P4: admission -------------------------------------------------------------------------------------------------

# name: (path, cache blocks). "off" is the reference model with the read disabled. Reference configurations run on a
# copy without the fine-key tier (an R5 state); stored configurations on the tiered snapshot.
ADMISSION = {"off": ("reread", 0), "reread": ("reread", 0), "reread_cache": ("reread", None),
             "stored": ("stored", 0), "stored_cache": ("stored", None)}


def _logit_gap(a: list, b: list) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b)) if a else 0.0


@torch.no_grad()
def warm_up(models, probes, snapshot, feed, chunks) -> None:
    """Throwaway calls before anything is measured, so allocator growth and first-use costs land here (P0 found the
    first occupancy of a process measured slow: EVIDENCE_R8)."""
    one = torch.zeros(1, dtype=torch.long)
    for path, model in models.items():
        run = Run(model, probes[path], snapshot, path, tier=path == "stored")
        for i in range(min(8, feed.shape[0])):
            run.forward(feed[i], one)
        run.forward(chunks[0], torch.arange(chunks.shape[-1]))


@torch.no_grad()
def measure_admission(models, probes, snapshot, args, feed, chunks) -> dict:
    blocks = args.admission_cache_blocks
    configs = {n: (path, blocks if cache is None else cache) for n, (path, cache) in ADMISSION.items()}
    lo, hi = snapshot.ring.live()
    out = {"tokens": snapshot.position, "live_blocks": hi - lo, "configs": {}}
    one = torch.zeros(1, dtype=torch.long)

    def make(name, **kw):
        path, cache = configs[name]
        model = models[path]
        return Run(model, probes[path], snapshot, name, cache_blocks=cache, tier=path == "stored", **kw)

    for name, (path, cache) in configs.items():
        model = models[path]
        face = model.context_interface()
        out["configs"][name] = {
            "path": path, "cache_blocks": cache, "cache_capacity_bytes": cache * entry_bytes(model),
            "projected_scratch_bytes": {
                "decode": face.scratch_bytes(batch=1, chunk=1, options={}, queries=1)["total"],
                "prefill_512_every_position": face.scratch_bytes(batch=1, chunk=512, options={}, queries=512)["total"]}}
    # decode: natural, interleaved; logits kept to measure the gap between the paths
    runs = {n: make(n, keep_logits=True) for n in configs}
    order = list(runs)
    for i in range(feed.shape[0]):
        for k in range(len(order)):
            runs[order[(i + k) % len(order)]].forward(feed[i], one)
    off = runs["off"].records
    for n in order:
        records = runs[n].records
        row = {"all": summarize(records, paired=off if n != "off" else None)}
        if n != "off":
            row["cold"] = summarize(records[:COLD_STEPS], paired=off[:COLD_STEPS])
            row["warm"] = summarize(records[COLD_STEPS:], paired=off[COLD_STEPS:])
        out["configs"][n]["decode_natural"] = row
        out["configs"][n]["state_bytes"] = runs[n].state.nbytes()
        if runs[n].state.cache is not None:
            out["configs"][n]["cache_bytes_end"] = runs[n].state.cache.bytes
    out["decode_logit_gap"] = {"stored_vs_reread": _logit_gap(runs["stored"].logits, runs["reread"].logits),
                               "stored_cache_vs_reread": _logit_gap(runs["stored_cache"].logits,
                                                                    runs["reread"].logits),
                               "reread_cache_vs_reread": _logit_gap(runs["reread_cache"].logits,
                                                                    runs["reread"].logits)}
    # decode: replay (repeated queries, each cache as the natural pass left it) and novel (cache emptied every step)
    cached = [n for n in order if configs[n][1]]
    both = {}
    for n in cached:
        both[f"replay:{n}"] = make(n, cache=runs[n].state.cache)
        both[f"novel:{n}"] = make(n)
    names = list(both)
    for i in range(feed.shape[0]):
        for k in range(len(names)):
            name = names[(i + k) % len(names)]
            if name.startswith("novel:"):
                both[name].state.cache.clear()
            both[name].forward(feed[i], one)
    for name, run in both.items():
        kind, n = name.split(":")
        out["configs"][n][f"decode_{kind}"] = summarize(run.records, paired=off)
    learned = {path: [r for run in (*runs.values(), *both.values()) if run.mode == "learned"
                      and configs[run.name][0] == path for r in run.records] for path in models}
    out["decode_cost_fit"] = {path: cost_fit(records) for path, records in learned.items()}
    del runs, both
    # prefill: logits at every position (a cold chunk, then a warm one) and at the last position only
    every = torch.arange(chunks.shape[-1])
    last = torch.tensor([chunks.shape[-1] - 1])
    kinds = (("prefill_all", every, chunks.shape[0]), ("prefill_last", last, 1))
    for label, positions, pieces in kinds[1:] if args.no_prefill_all else kinds:
        rows = {n: [[] for _ in range(pieces)] for n in configs}
        logits = {n: [] for n in configs}
        for repeat in range(args.prefill_runs):
            for k in range(len(order)):
                n = order[(repeat + k) % len(order)]
                run = make(n, keep_logits=True)
                for piece in range(pieces):
                    rows[n][piece].append(run.forward(chunks[piece], positions))
                logits[n].extend(run.logits)
                del run
        for n in configs:
            out["configs"][n][label] = [summarize(rows[n][piece], paired=rows["off"][piece] if n != "off" else None)
                                        for piece in range(pieces)]
        out[f"{label}_logit_gap"] = {"stored_vs_reread": _logit_gap(logits["stored"], logits["reread"]),
                                     "stored_cache_vs_reread": _logit_gap(logits["stored_cache"], logits["reread"])}
    # ingest without reads: the reference (no tier) against stored keys (the tier written at every completion)
    ingest = {"reread": [], "stored": []}
    none = torch.zeros(0, dtype=torch.long)
    for repeat in range(args.prefill_runs):
        for path in (("reread", "stored") if repeat % 2 == 0 else ("stored", "reread")):
            run = Run(models[path], probes[path], snapshot, path, tier=path == "stored")
            for piece in range(chunks.shape[0]):
                ingest[path].append(run.forward(chunks[piece], none)["seconds"] / chunks.shape[-1])
            del run
    out["ingest_us_per_token_no_reads"] = {path: percentiles(v, 1e6) for path, v in ingest.items()}
    import resource
    out["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return out


@torch.no_grad()
def command_admission(args) -> dict:
    occupancies = tuple(int(x) for x in args.occupancies.split(",")) if args.occupancies else \
        ((2048, 4096) if args.smoke else OCCUPANCIES)
    geometries = args.geometries.split(",") if args.geometries else list(GEOMETRIES)
    ples = [p == "on" for p in args.ple.split(",")]
    steps, chunk = (COLD_STEPS + 4, 512) if args.smoke else (args.decode_steps, 512)
    generator = torch.Generator().manual_seed(SEED)
    stream = torch.randint(0, 256, (1, max(occupancies)), generator=generator)
    feed = torch.randint(0, 256, (steps, 1, 1), generator=generator)
    chunks = torch.randint(0, 256, (2, 1, chunk), generator=generator)
    out = {"phase": PHASE, "command": "admission", "seed": SEED, "decode_steps": steps, "cold_steps": COLD_STEPS,
           "prefill_runs": args.prefill_runs, "cache_blocks": args.admission_cache_blocks,
           "threads": torch.get_num_threads(), "configs": {n: {"path": p, "cache": c} for n, (p, c) in ADMISSION.items()},
           "model": "R7 latency core (d 256, 4 blocks, 8 experts top-2 in blocks 1 and 3, PLE 2 x 4 x 32771 x 32 "
                    "before block 1), R5 head per geometry, capacity 1,048,576; UNTRAINED (timing only)",
           "stream": "the same seeded byte stream, decode feed and prefill chunks as P0 (run_r8 baseline)",
           "rows": []}
    warmed = False
    for geometry in geometries:
        reference = timing_model(geometry, capacity=args.capacity)
        stored = timing_model(geometry, capacity=args.capacity, fine_keys="stored")
        stored.load_state_dict(reference.state_dict())
        models = {"reread": reference, "stored": stored}
        probes = {path: StageProbe(model) for path, model in models.items()}
        for ple in ples:
            for model in models.values():
                model.ple.enabled = ple
            snapshot = stored.new_state(1)                     # tiered: the reference runs on copies without it
            ingest, at = [], 0
            for target in occupancies:
                while at < target:
                    piece = stream[:, at:min(target, at + chunk)]
                    started = time.perf_counter()
                    stored(piece, snapshot, logit_positions=torch.zeros(0, dtype=torch.long))
                    ingest.append((time.perf_counter() - started) / piece.shape[1])
                    at += piece.shape[1]
                if not warmed:
                    warm_up(models, probes, snapshot, feed, chunks)
                    warmed = True
                row = {"geometry": geometry, "evidence": dataclass_dict(stored.evidence), "ple": ple,
                       "occupancy": target, "ingest_us_per_token_no_reads_with_tier": percentiles(ingest, 1e6)}
                row.update(measure_admission(models, probes, snapshot, args, feed, chunks))
                out["rows"].append(row)
                decode = {n: c["decode_natural"]["all"]["step_ms"]["p50"] for n, c in row["configs"].items()}
                print(f"[r8 admission] {geometry} ple={ple} {target}: decode p50 ms {decode} gap "
                      f"{row['decode_logit_gap']['stored_vs_reread']:.2e}", flush=True)
                if args.out:
                    _write(args.out, out)
        del models, probes, reference, stored
    return out


def dataclass_dict(config) -> dict:
    import dataclasses
    return dataclasses.asdict(config)


# -- P1: query-independent representations ------------------------------------------------------------------------

def tensors_per_block(e: EvidenceConfig, width: int) -> dict:
    """Per stored block, the tensors the read path derives from the block's exact tokens alone (no query)."""
    s = e.block
    return {
        "encoding": {"shape": [s + 1, width], "values": (s + 1) * width,
                     "used_by": "fine keys, read keys and values, descriptors (and the training replay)"},
        "descriptors": {"shape": [e.per_block, e.key_dim], "values": e.per_block * e.key_dim,
                        "used_by": "coarse scan (already stored at completion, in the ring's index)"},
        "fine_keys": {"shape": [s, e.fine_dim], "values": s * e.fine_dim,
                      "used_by": "fine reranking of EVERY coarse candidate"},
        "read_keys_values": {"shape": [2, s, e.read_head_dim], "values": 2 * s * e.read_head_dim,
                             "used_by": "readout of the SELECTED blocks only"},
    }


def command_represent(args) -> dict:
    """Formulas only. Bytes per block, per stream at each occupancy, and the R6 scale configurations' 1M budget."""
    from .run_r6 import SCALES, scale_model
    out = {"phase": PHASE, "command": "represent", "status": "EXACT shapes and byte counts; PROJECTED budgets",
           "geometries": {}, "scale_budget_1m": {}}
    for geometry in GEOMETRIES:
        e = EvidenceConfig(capacity=1 << 20, **GEOMETRIES[geometry])
        tensors = tensors_per_block(e, e.encoder_dim)
        rows = {}
        for name, t in tensors.items():
            rows[name] = dict(t, bytes_fp32=4 * t["values"], bytes_bf16=2 * t["values"],
                              per_stream_mib_fp32={str(n): round(4 * t["values"] * (n // e.block) / 2 ** 20, 2)
                                                   for n in OCCUPANCIES},
                              per_stream_mib_bf16={str(n): round(2 * t["values"] * (n // e.block) / 2 ** 20, 2)
                                                   for n in OCCUPANCIES})
        out["geometries"][geometry] = {"evidence": dataclass_dict(e), "per_block": rows}
    budget = 1 << 30
    for name in SCALES:
        with torch.device("meta"):
            model = scale_model(name)
        interface = model.context_interface()
        e = model.evidence
        state = {"bf16_index": interface.fixed_bytes(1, {"history_device": "cuda"})["total"],
                 "fp32_index": interface.fixed_bytes(1)["total"]}
        kept = model.evidence
        scratch = {}
        try:
            for tile in (32, 16):
                model.evidence = kept.replace(query_tile=tile)
                scratch[str(tile)] = {
                    "decode": model.context_interface().scratch_bytes(batch=1, chunk=1, options={}, queries=1)["total"],
                    "prefill512": model.context_interface().scratch_bytes(batch=1, chunk=512, options={},
                                                                          queries=512)["total"]}
        finally:
            model.evidence = kept
        tensors = tensors_per_block(e, e.encoder_dim)
        tiers = {k: 4 * t["values"] * e.slots for k, t in tensors.items() if k != "descriptors"}
        headroom = {f"{index}_tile{tile}_{mode}": budget - state[index] - scratch[tile][mode]
                    for index in state for tile in scratch for mode in ("decode", "prefill512")}
        stored = {}
        try:                                    # R8: the same configuration with the fine-key tier (stored keys)
            for tile in (32, 16):
                model.evidence = kept.replace(query_tile=tile, fine_keys="stored")
                face = model.context_interface()
                stored[str(tile)] = {
                    "state_bf16_index": face.fixed_bytes(1, {"history_device": "cuda"})["total"],
                    "state_fp32_index": face.fixed_bytes(1)["total"],
                    "decode": face.scratch_bytes(batch=1, chunk=1, options={}, queries=1)["total"],
                    "prefill512": face.scratch_bytes(batch=1, chunk=512, options={}, queries=512)["total"]}
        finally:
            model.evidence = kept
        out["scale_budget_1m"][name] = {
            "budget_bytes": budget, "state_bytes": state, "scratch_bytes_by_query_tile": scratch,
            "headroom_bytes": headroom,
            "full_tier_bytes_fp32": tiers,
            "cache_blocks_that_fit": {k: max(0, v // (4 * (e.block + 1) * e.encoder_dim)) for k, v in headroom.items()},
            "stored_fine_keys_by_query_tile": stored,
            "live_blocks": e.slots}
    return out


def command_smoke(args) -> dict:
    ns = argparse.Namespace(**{**vars(args), "smoke": True, "out": None, "occupancies": "2048,4096", "no_prefill_all": False,
                               "geometries": "enc64", "ple": "off,on", "cache_blocks": (64,),
                               "prefill_cache_blocks": (64,), "prefill_runs": 1, "capacity": 1 << 12})
    base = command_baseline(ns)
    admit = command_admission(argparse.Namespace(**{**vars(ns), "ple": "on", "admission_cache_blocks": 64}))
    return {"baseline_rows": len(base["rows"]), "admission_rows": len(admit["rows"]),
            "admission_gap": max(r["decode_logit_gap"]["stored_vs_reread"] for r in admit["rows"]),
            "represent": sorted(command_represent(ns)["geometries"])}


# -- records ------------------------------------------------------------------------------------------------------

def _finite(value):
    if isinstance(value, float):          # six significant digits (never decimal places: they erase small values)
        return float(f"{value:.6g}") if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _revision() -> str:
    """The source revision measured: HEAD, marked "+dirty" if a tracked research source or test (records excluded)
    differs from it. Read once, when the process starts (P0's records read it at every write, so a record written
    while results were being committed names a later commit: EVIDENCE_R8 section 1)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "research", "tests",
                                ":(exclude)research/elpis_torch/results"],
                               cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


_STARTED_AT: list = []


def _write(path: str, record: dict) -> None:
    environment = {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
                   "cuda_available": torch.cuda.is_available(),
                   "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                   "aten_cpu_capability": os.environ.get("ATEN_CPU_CAPABILITY"),
                   "mkl_enable_instructions": os.environ.get("MKL_ENABLE_INSTRUCTIONS"),
                   "source_revision": _STARTED_AT[0] if _STARTED_AT else _revision()}
    text = json.dumps(_finite(dict(record, environment=environment)), indent=1, sort_keys=True, allow_nan=False)
    with open(path, "w", encoding="utf-8") as handle:           # explicit offline artifact, requested by --out
        handle.write(text + "\n")


def _ints(text: str) -> tuple:
    return tuple(int(x) for x in text.split(",") if x)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["baseline", "represent", "admission", "smoke"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--geometries", default=None, help="comma-separated: " + ",".join(GEOMETRIES))
    parser.add_argument("--occupancies", default=None)
    parser.add_argument("--ple", default="off,on")
    parser.add_argument("--capacity", type=int, default=1 << 20)
    parser.add_argument("--cache-blocks", type=_ints, default=CACHE_BLOCKS)
    parser.add_argument("--prefill-cache-blocks", type=_ints, default=(512,))
    parser.add_argument("--admission-cache-blocks", type=int, default=512)
    parser.add_argument("--no-prefill-all", action="store_true", help="admission: skip the every-position prefill")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--prefill-runs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    args.smoke = args.command == "smoke"
    torch.set_num_threads(args.threads)
    _STARTED_AT[:] = [_revision()]
    result = {"baseline": command_baseline, "represent": command_represent, "admission": command_admission,
              "smoke": command_smoke}[args.command](args)
    if args.out:
        _write(args.out, result)
    else:
        print(json.dumps(_finite(result), indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
