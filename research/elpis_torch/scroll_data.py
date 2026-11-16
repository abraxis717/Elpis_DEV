"""Generated long-stream retrieval task for the scrolling-context experiments. SEMANTICS=NONE.

A stream is filler tokens with sparse events. A fact is ONE token, so reading it back is one attention hop:

    pair(K, v)             bind      key K now has value v. A later pair for K CONTRADICTS it: the latest wins.
    alias(d, K)            alias     alias symbol d now points at key K.
    ask(K) v               direct    v is the latest value of K.
    assoc(c) v             assoc     a second family of question tokens, paired with the keys by a fixed
                                     permutation: nothing in the question shares a token with the fact.
    hop(d) ask(K) v        two-hop   ask(K) is what d points at (hop 1, emitted as a token), then v is the latest
                                     value of K (hop 2, scored with the true ask(K) in the input).

No question token equals a fact token, so every retrieval goes through learned associations, never through
lexical identity. Only the answers are determined by the past. Each answer is a probe with a kind, a distance
(tokens back to the fact it depends on) and a flag saying whether an older, contradicted fact for the same
key exists. ``span`` probes bind reserved keys at the start of a stream and ask for them at its end: their
distance is the whole stream, which is how eviction beyond the horizon is observed.

Version 1 of this task spelled facts as byte strings (``[K:v]`` ... ``?K=v``), which needs a two-layer
induction circuit. No topology learned it in the budget, the dense reference included
(docs/research/elpis_torch/EVIDENCE_R1.md, "Failures"); it was replaced by this single-token form.

Streams are drawn fresh from a seed; training and evaluation use disjoint seed domains.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

KEYS, SPAN_KEYS, VALUE_COUNT, ALIASES, FILLERS = 8, 2, 8, 4, 8        # SPAN_KEYS are the keys after KEYS
FILLER0, VALUE0 = 0, FILLERS
PAIR0 = VALUE0 + VALUE_COUNT                 # pair(K, v) = PAIR0 + K * VALUE_COUNT + v
ASK0 = PAIR0 + (KEYS + SPAN_KEYS) * VALUE_COUNT    # ask(K)
ASSOC0 = ASK0 + KEYS + SPAN_KEYS                   # assoc(c), c < KEYS
ALIAS0 = ASSOC0 + KEYS                             # alias(d, K) = ALIAS0 + d * KEYS + K
HOP0 = ALIAS0 + ALIASES * KEYS                     # hop(d)
BOS = HOP0 + ALIASES
VOCAB = BOS + 1
_PAIRING = random.Random("elpis-torch-r1:assoc").sample(range(KEYS), KEYS)
ASSOC_OF = {key: ASSOC0 + _PAIRING[key] for key in range(KEYS)}        # key -> the assoc token that asks for it

DIRECT, ASSOC_Q, HOP1, HOP2, SPAN_Q = 1, 2, 3, 4, 5
KINDS = {DIRECT: "direct", ASSOC_Q: "assoc", HOP1: "hop1", HOP2: "hop2", SPAN_Q: "span"}
CHANCE = {"direct": 1 / VALUE_COUNT, "assoc": 1 / VALUE_COUNT, "hop2": 1 / VALUE_COUNT, "span": 1 / VALUE_COUNT,
          "hop1": 1 / KEYS}


def pair(key: int, value: int) -> int:
    return PAIR0 + key * VALUE_COUNT + value


@dataclass(frozen=True)
class Streams:
    """``tokens`` ``[n, T]``; per position the probe kind of the token there (0: not a probe), the distance
    to the fact it depends on, and whether that key had been bound to something else before."""
    tokens: torch.Tensor
    kind: torch.Tensor
    distance: torch.Tensor
    contradicted: torch.Tensor


def _events(length: int, events: int, rng: random.Random, span: bool):
    """Event start positions with room for the longest event (3 tokens) and a gap, in order."""
    lo, hi = (int(0.02 * length), int(0.98 * length)) if span else (2, length - 4)
    step = (hi - lo) // events
    if step < 4:
        raise ValueError("the stream is too short for this many events")
    return [lo + i * step + rng.randrange(step - 3) for i in range(events)]


def generate(count: int, length: int, *, events: int, seed: int, span: bool = False) -> Streams:
    """``count`` streams of ``length`` tokens with about ``events`` facts and questions each."""
    tokens = torch.randint(FILLERS, (count, length), generator=torch.Generator().manual_seed(seed))
    kind = torch.zeros(count, length, dtype=torch.int8)
    distance = torch.zeros(count, length, dtype=torch.int64)
    contradicted = torch.zeros(count, length, dtype=torch.bool)
    tokens[:, 0] = BOS
    for row in range(count):
        rng = random.Random(f"elpis-torch-r1:{seed}:{row}")
        bound: dict = {}          # key -> (value, position of the pair token, contradicts an earlier fact?)
        alias: dict = {}          # alias symbol -> (key, position of the alias token)

        def write(at, symbols, probes=()):
            tokens[row, at:at + len(symbols)] = torch.tensor(symbols)
            for offset, what, source, flag in probes:
                kind[row, at + offset], distance[row, at + offset] = what, at + offset - source
                contradicted[row, at + offset] = flag

        if span:
            for index in range(SPAN_KEYS):
                value = rng.randrange(VALUE_COUNT)
                write(2 + 3 * index, [pair(KEYS + index, value)])
                bound[KEYS + index] = (value, 2 + 3 * index, False)
        for at in _events(length, events, rng, span):
            roll = rng.random()
            keys = [k for k in bound if k < KEYS]
            if roll < 0.40 or not keys:
                key, value = rng.randrange(KEYS), rng.randrange(VALUE_COUNT)
                changed = key in bound and (bound[key][2] or bound[key][0] != value)
                write(at, [pair(key, value)])
                bound[key] = (value, at, changed)
            elif roll < 0.50:
                symbol, key = rng.randrange(ALIASES), rng.choice(keys)
                write(at, [ALIAS0 + symbol * KEYS + key])
                alias[symbol] = (key, at)
            elif roll < 0.75 or (roll >= 0.90 and not alias):
                key = rng.choice(keys)
                value, source, flag = bound[key]
                write(at, [ASK0 + key, VALUE0 + value], [(1, DIRECT, source, flag)])
            elif roll < 0.90:
                key = rng.choice(keys)
                value, source, flag = bound[key]
                write(at, [ASSOC_OF[key], VALUE0 + value], [(1, ASSOC_Q, source, flag)])
            else:
                symbol = rng.choice(sorted(alias))
                key, where = alias[symbol]
                value, source, flag = bound[key]
                write(at, [HOP0 + symbol, ASK0 + key, VALUE0 + value],
                      [(1, HOP1, where, False), (2, HOP2, source, flag)])
        if span:
            for index in range(SPAN_KEYS):
                value, source, _ = bound[KEYS + index]
                at = length - 3 * SPAN_KEYS - 1 + 3 * index
                write(at, [ASK0 + KEYS + index, VALUE0 + value], [(1, SPAN_Q, source, False)])
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
