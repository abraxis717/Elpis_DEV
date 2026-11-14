"""Generated long-stream retrieval task for the scrolling-context experiments. SEMANTICS=NONE.

A stream is filler bytes with sparse events. Records write facts, queries ask for them later:

    [K:v]      bind      base key K now has value v. Binding K again CONTRADICTS the earlier fact: the latest wins.
    [d>K]      alias     alias symbol d now points at base key K.
    ?K=v       direct    v is the latest value of K.
    ~c=v       assoc     c is NOT a key symbol. A fixed permutation pairs each c with one base key; the model has
                         to have learned the pairing, so the stored and the asked symbols never match lexically.
    ^d>K=v     two-hop   K is what d points at (hop 1, emitted), then v is the latest value of K (hop 2).

Only the answer tokens are determined by the past. Each one is a probe with a kind, a distance (tokens back to
the fact it depends on) and a flag saying whether an older, contradicted fact for the same key exists.
``span`` probes bind reserved keys in the first 2% of a stream and ask for them in the last 2%: their
distance is the whole stream, which is how eviction beyond the horizon is observed.

Streams are drawn fresh from a seed; training and evaluation use disjoint seed domains.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .tokenizer import ByteTokenizer

BASE = [ord("A") + i for i in range(20)]            # ordinary keys
SPAN = [ord("U") + i for i in range(4)]             # reserved for whole-stream probes
ALIAS = [ord("0") + i for i in range(8)]
VALUES = [ord("a") + i for i in range(16)]
ASSOC = [0xC0 + i for i in range(20)]               # one per base key, through a fixed permutation
FILLER = [0xE0 + i for i in range(16)]
_PAIRING = random.Random("elpis-torch-r1:assoc").sample(range(20), 20)
ASSOC_OF = {BASE[k]: ASSOC[_PAIRING[k]] for k in range(20)}      # key symbol -> the symbol that asks for it

DIRECT, ASSOC_Q, HOP1, HOP2, SPAN_Q = 1, 2, 3, 4, 5
KINDS = {DIRECT: "direct", ASSOC_Q: "assoc", HOP1: "hop1", HOP2: "hop2", SPAN_Q: "span"}
CHANCE = {"direct": 1 / 16, "assoc": 1 / 16, "hop2": 1 / 16, "span": 1 / 16, "hop1": 1 / 20}


@dataclass(frozen=True)
class Streams:
    """``tokens`` ``[n, T]``; per position the probe kind of the token there (0: not a probe), the distance
    to the fact it depends on, and whether that key had been bound to something else before."""
    tokens: torch.Tensor
    kind: torch.Tensor
    distance: torch.Tensor
    contradicted: torch.Tensor


def _events(length: int, events: int, rng: random.Random, span: bool):
    """Event start positions with room for the longest event (6 tokens) and a gap, in order."""
    lo, hi = (int(0.02 * length), int(0.98 * length)) if span else (2, length - 8)
    step = (hi - lo) // events
    if step < 8:
        raise ValueError("the stream is too short for this many events")
    return [lo + i * step + rng.randrange(step - 7) for i in range(events)]


def generate(count: int, length: int, *, events: int, seed: int, span: bool = False) -> Streams:
    """``count`` streams of ``length`` tokens with about ``events`` records and queries each."""
    tokens = torch.tensor(FILLER)[torch.randint(len(FILLER), (count, length),
                                                generator=torch.Generator().manual_seed(seed))]
    kind = torch.zeros(count, length, dtype=torch.int8)
    distance = torch.zeros(count, length, dtype=torch.int64)
    contradicted = torch.zeros(count, length, dtype=torch.bool)
    tokens[:, 0] = ByteTokenizer.BOS
    for row in range(count):
        rng = random.Random(f"elpis-torch-r1:{seed}:{row}")
        bound: dict = {}          # key -> (value, position of the value token, rebound?)
        alias: dict = {}          # alias symbol -> (key, position of the key token)

        def write(at, symbols, probes=()):
            tokens[row, at:at + len(symbols)] = torch.tensor(symbols)
            for offset, what, source, flag in probes:
                kind[row, at + offset], distance[row, at + offset] = what, at + offset - source
                contradicted[row, at + offset] = flag

        if span:
            cursor = 2
            for key in SPAN:
                value = rng.choice(VALUES)
                write(cursor, [ord("["), key, ord(":"), value, ord("]")])
                bound[key] = (value, cursor + 3, False)
                cursor += 8
        for at in _events(length, events, rng, span):
            roll = rng.random()
            keys = [k for k in bound if k in BASE]
            if roll < 0.40 or not keys:
                key, value = rng.choice(BASE), rng.choice(VALUES)
                changed = key in bound and (bound[key][2] or bound[key][0] != value)
                write(at, [ord("["), key, ord(":"), value, ord("]")])
                bound[key] = (value, at + 3, changed)
            elif roll < 0.50:
                symbol, key = rng.choice(ALIAS), rng.choice(keys)
                write(at, [ord("["), symbol, ord(">"), key, ord("]")])
                alias[symbol] = (key, at + 3)
            elif roll < 0.75 or (roll >= 0.90 and not alias):
                key = rng.choice(keys)
                value, source, flag = bound[key]
                write(at, [ord("?"), key, ord("="), value], [(3, DIRECT, source, flag)])
            elif roll < 0.90:
                key = rng.choice(keys)
                value, source, flag = bound[key]
                write(at, [ord("~"), ASSOC_OF[key], ord("="), value], [(3, ASSOC_Q, source, flag)])
            else:
                symbol = rng.choice(sorted(alias))
                key, where = alias[symbol]
                value, source, flag = bound[key]
                write(at, [ord("^"), symbol, ord(">"), key, ord("="), value],
                      [(3, HOP1, where, False), (5, HOP2, source, flag)])
        if span:
            cursor = length - 6 * len(SPAN) - 2
            for key in SPAN:
                value, source, _ = bound[key]
                write(cursor, [ord("?"), key, ord("="), value], [(3, SPAN_Q, source, False)])
                cursor += 6
    return Streams(tokens, kind, distance, contradicted)


def probe_loss(logits: torch.Tensor, streams_tokens: torch.Tensor, kind: torch.Tensor, *,
               background: float = 0.1) -> torch.Tensor:
    """Teacher-forced next-token loss: the mean over probe targets plus a small weight on every other target.

    ``logits`` ``[n, T-1, vocab]`` for ``tokens`` ``[n, T]``. The filler is noise; the probes are the task.
    """
    targets = streams_tokens[:, 1:]
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                               reduction="none").view(targets.shape)
    probes = kind[:, 1:] != 0
    return losses[probes].mean() + background * losses.mean()


def score(predicted: torch.Tensor, nll: torch.Tensor, streams: Streams, *, edges=()) -> dict:
    """Accuracy and loss per probe kind, split by distance bucket and by contradiction.

    ``predicted[n, t]`` and ``nll[n, t]`` are the model's argmax and loss for token ``t + 1``.
    """
    out: dict = {}
    right = predicted[:, :-1] == streams.tokens[:, 1:]
    for code, name in KINDS.items():
        mask = streams.kind[:, 1:] == code
        if not int(mask.sum()):
            continue
        entry = {"n": int(mask.sum()), "accuracy": float(right[mask].double().mean()),
                 "loss": float(nll[:, :-1][mask].double().mean()), "chance": CHANCE[name]}
        dist, flag = streams.distance[:, 1:], streams.contradicted[:, 1:]
        for label, sub in (("contradicted", mask & flag), ("uncontradicted", mask & ~flag)):
            if int(sub.sum()) and name in ("direct", "assoc", "hop2"):
                entry[label] = {"n": int(sub.sum()), "accuracy": float(right[sub].double().mean())}
        bounds = [0, *edges, 1 << 62]
        buckets = {}
        for low, high in zip(bounds[:-1], bounds[1:]):
            sub = mask & (dist >= low) & (dist < high)
            if int(sub.sum()):
                buckets[f"{low}..{'inf' if high == 1 << 62 else high}"] = {
                    "n": int(sub.sum()), "accuracy": float(right[sub].double().mean())}
        entry["by_distance"] = buckets
        out[name] = entry
    return out
