"""R3 synthetic long-range tasks. SEMANTICS=NONE. Generated from seeds; train/validation/test use disjoint seeds.

Facts are two tokens, the way a binding is spelled in text, and every question is answered by the token that
followed some earlier occurrence:

    KEY_k VAL_v           bind      key k now has value v. A later binding of k CONTRADICTS it: the latest wins.
    ALIAS_d KEY_k         alias     alias d now points at key k.
    LINK_e ALIASOBJ_d     link      second-level alias e now points at alias d.
    KEY_x VAL_v (x >= keys)          distractor binding: a key that is never asked about.

    ASK_k            -> VAL_v      direct       (question family 1; no token in common with the fact)
    ASSOC_c          -> VAL_v      associative  (question family 2, paired with keys by a fixed permutation)
    HOP_d            -> KEY_k -> VAL_v                     two hops, each answer emitted as a token
    HOP3_e           -> ALIASOBJ_d -> KEY_k -> VAL_v       three hops

Every answer token is a PROBE with a kind, the absolute position of the token that holds its evidence
(``source``: the answer token of the latest binding it depends on), and whether an older, contradicted binding
of the same key exists. ``planted`` keys are bound once at fixed fractions of the stream (begin / middle /
end) and asked for in the final 2 % of the stream: their distance is set by where they were planted.

Labels (``kind``, ``source``, ``contradicted``, ``region``) are for scoring and for the TRAINING-ONLY
addressing loss. They are never inputs of a model at evaluation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

MAX_KEYS, VALUES, ALIASES, LINKS, FILLERS = 128, 16, 32, 32, 32
BOS = 0
FILLER0 = 1
KEY0 = FILLER0 + FILLERS
VAL0 = KEY0 + MAX_KEYS
ASK0 = VAL0 + VALUES
ASSOC0 = ASK0 + MAX_KEYS
ALIAS0 = ASSOC0 + MAX_KEYS
ALIASOBJ0 = ALIAS0 + ALIASES
LINK0 = ALIASOBJ0 + ALIASES
HOP0 = LINK0 + LINKS
HOP30 = HOP0 + ALIASES
VOCAB = HOP30 + LINKS
_PAIRING = random.Random("elpis-torch-r3:assoc").sample(range(MAX_KEYS), MAX_KEYS)

DIRECT, ASSOC, HOP2_KEY, HOP2_VALUE, HOP3_ALIAS, HOP3_KEY, HOP3_VALUE, PLANTED = 1, 2, 3, 4, 5, 6, 7, 8
KINDS = {DIRECT: "direct", ASSOC: "assoc", HOP2_KEY: "hop2_key", HOP2_VALUE: "hop2_value",
         HOP3_ALIAS: "hop3_alias", HOP3_KEY: "hop3_key", HOP3_VALUE: "hop3_value", PLANTED: "planted"}
REGIONS = {1: "begin", 2: "middle", 3: "end"}


@dataclass(frozen=True)
class FactTask:
    """One distribution of streams. ``keys`` is the fact density: how many keys may be bound at once."""
    keys: int = 32
    events: int = 96                   # facts and questions per stream (spread uniformly)
    p_bind: float = 0.40
    p_alias: float = 0.06
    p_link: float = 0.04
    p_distract: float = 0.0            # distractor bindings (keys that are never asked)
    p_assoc: float = 0.12
    p_hop2: float = 0.08
    p_hop3: float = 0.04
    planted: tuple = ()                # fractions in (0, 0.95) where a reserved key is bound once
    distractor_keys: int = 16

    def __post_init__(self):
        if not 1 <= self.keys or self.keys + len(self.planted) + (self.distractor_keys if self.p_distract else 0) \
                > MAX_KEYS:
            raise ValueError("keys + planted + distractor keys must fit MAX_KEYS")
        if any(not 0.0 < f < 0.95 for f in self.planted):
            raise ValueError("planted fractions must be in (0, 0.95)")


@dataclass(frozen=True)
class FactStreams:
    tokens: torch.Tensor        # [n, T] int64
    kind: torch.Tensor          # [n, T] int8: probe kind of the token at that position (0: none)
    source: torch.Tensor        # [n, T] int64: absolute position of the evidence token (-1: none)
    contradicted: torch.Tensor  # [n, T] bool
    region: torch.Tensor        # [n, T] int8: planted probes only (1 begin, 2 middle, 3 end)

    def query_source(self) -> torch.Tensor:
        """Aligned to INPUT positions: the token at ``q`` predicts ``q + 1``, so its evidence is
        ``source[q + 1]``. ``[n, T]``, -1 where the next token is not a probe."""
        out = torch.full_like(self.source, -1)
        out[:, :-1] = self.source[:, 1:]
        return out

    def query_block(self, block: int) -> torch.Tensor:
        """The source BLOCK of each query (training-only addressing label; -1: none)."""
        src = self.query_source()
        return torch.where(src >= 0, src // block, src)


def _slots(length: int, count: int, rng: random.Random, lo: int, hi: int) -> list[int]:
    step = (hi - lo) // max(count, 1)
    if step < 6:
        raise ValueError("the stream is too short for this many events")
    return [lo + i * step + rng.randrange(step - 5) for i in range(count)]


def generate(task: FactTask, count: int, length: int, *, seed: int) -> FactStreams:
    tokens = torch.randint(FILLER0, FILLER0 + FILLERS, (count, length),
                           generator=torch.Generator().manual_seed(seed))
    kind = torch.zeros(count, length, dtype=torch.int8)
    source = torch.full((count, length), -1, dtype=torch.int64)
    contradicted = torch.zeros(count, length, dtype=torch.bool)
    region = torch.zeros(count, length, dtype=torch.int8)
    tokens[:, 0] = BOS
    planted_keys = list(range(task.keys, task.keys + len(task.planted)))
    distract0 = task.keys + len(task.planted)
    for row in range(count):
        rng = random.Random(f"elpis-torch-r3:{seed}:{row}")
        bound: dict = {}       # key -> (value, position of the VAL token, an older different binding existed?)
        alias: dict = {}       # alias d -> (key, position of the KEY token)
        link: dict = {}        # link e -> (alias d, position of the ALIASOBJ token)

        def put(at, symbols, probes=()):
            tokens[row, at:at + len(symbols)] = torch.tensor(symbols)
            for offset, what, src, flag in probes:
                kind[row, at + offset], source[row, at + offset] = what, src
                contradicted[row, at + offset] = flag

        tail = int(0.98 * length) - 4 * len(planted_keys) - 2
        plants = {int(f * length) + 1: key for f, key in zip(task.planted, planted_keys)}
        busy = set()
        for at, key in plants.items():
            value = rng.randrange(VALUES)
            put(at, [KEY0 + key, VAL0 + value])
            bound[key] = (value, at + 1, False)
            busy.update(range(at - 4, at + 3))
        for at in _slots(length, task.events, rng, 2, tail):
            if any(a in busy for a in range(at, at + 4)):
                continue
            roll = rng.random()
            keys = [k for k in bound if k < task.keys]
            cuts = [task.p_bind, task.p_alias, task.p_link, task.p_distract, task.p_assoc, task.p_hop2, task.p_hop3]
            edges = [sum(cuts[:i + 1]) for i in range(len(cuts))]
            if roll < edges[0] or not keys:
                key, value = rng.randrange(task.keys), rng.randrange(VALUES)
                changed = key in bound and (bound[key][2] or bound[key][0] != value)
                put(at, [KEY0 + key, VAL0 + value])
                bound[key] = (value, at + 1, changed)
            elif roll < edges[1]:
                d, key = rng.randrange(ALIASES), rng.choice(keys)
                put(at, [ALIAS0 + d, KEY0 + key])
                alias[d] = (key, at + 1)
            elif roll < edges[2] and alias:
                e, d = rng.randrange(LINKS), rng.choice(sorted(alias))
                put(at, [LINK0 + e, ALIASOBJ0 + d])
                link[e] = (d, at + 1)
            elif roll < edges[3]:
                put(at, [KEY0 + distract0 + rng.randrange(task.distractor_keys), VAL0 + rng.randrange(VALUES)])
            elif roll < edges[4]:
                key = rng.choice(keys)
                value, src, flag = bound[key]
                put(at, [ASSOC0 + _PAIRING[key], VAL0 + value], [(1, ASSOC, src, flag)])
            elif roll < edges[5] and alias:
                d = rng.choice(sorted(alias))
                key, where = alias[d]
                value, src, flag = bound[key]
                put(at, [HOP0 + d, KEY0 + key, VAL0 + value], [(1, HOP2_KEY, where, False), (2, HOP2_VALUE, src, flag)])
            elif roll < edges[6] and link:
                e = rng.choice(sorted(link))
                d, where_d = link[e]
                key, where_k = alias[d]
                value, src, flag = bound[key]
                put(at, [HOP30 + e, ALIASOBJ0 + d, KEY0 + key, VAL0 + value],
                    [(1, HOP3_ALIAS, where_d, False), (2, HOP3_KEY, where_k, False), (3, HOP3_VALUE, src, flag)])
            else:
                key = rng.choice(keys)
                value, src, flag = bound[key]
                put(at, [ASK0 + key, VAL0 + value], [(1, DIRECT, src, flag)])
        for index, (fraction, key) in enumerate(zip(task.planted, planted_keys)):
            at = tail + 2 + 3 * index
            value, src, _ = bound[key]
            put(at, [ASK0 + key, VAL0 + value], [(1, PLANTED, src, False)])
            region[row, at + 1] = 1 if fraction < 1 / 3 else 2 if fraction < 2 / 3 else 3
    return FactStreams(tokens, kind, source, contradicted, region)


def probe_loss(logits: torch.Tensor, tokens: torch.Tensor, kind: torch.Tensor, *, background: float = 0.05):
    """Teacher-forced loss: mean over probe targets plus a small weight on every other target."""
    targets = tokens[:, 1:]
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                                               reduction="none").view(targets.shape)
    probes = kind[:, 1:] != 0
    return losses[probes].mean() + background * losses.mean()


DISTANCE_EDGES = (64, 256, 1024, 4096, 16384, 65536, 262144, 1048576)


def score(predicted: torch.Tensor, streams: FactStreams, *, edges=DISTANCE_EDGES, extra: dict | None = None) -> dict:
    """Accuracy per probe kind, by distance bucket, contradiction and planted region.

    ``predicted[n, t]`` is the argmax for token ``t + 1``. ``extra`` maps a name to a boolean ``[n, T]`` tensor
    aligned like ``predicted`` (per query); each is averaged over the probes of each kind (diagnostics such as
    'source block selected').
    """
    out: dict = {}
    right = predicted[:, :-1] == streams.tokens[:, 1:]
    target_pos = torch.arange(streams.tokens.shape[1])[None, 1:]
    distance = target_pos - streams.source[:, 1:]
    for code, name in KINDS.items():
        mask = streams.kind[:, 1:] == code
        n = int(mask.sum())
        if not n:
            continue
        entry = {"n": n, "accuracy": float(right[mask].double().mean())}
        for label, sub in (("contradicted", mask & streams.contradicted[:, 1:]),
                           ("uncontradicted", mask & ~streams.contradicted[:, 1:])):
            if int(sub.sum()):
                entry[label] = {"n": int(sub.sum()), "accuracy": float(right[sub].double().mean())}
        buckets = {}
        bounds = [0, *edges, 1 << 62]
        for low, high in zip(bounds[:-1], bounds[1:]):
            sub = mask & (distance >= low) & (distance < high)
            if int(sub.sum()):
                buckets[f"{low}..{'inf' if high == 1 << 62 else high}"] = {
                    "n": int(sub.sum()), "accuracy": float(right[sub].double().mean())}
        entry["by_distance"] = buckets
        if code == PLANTED:
            entry["by_region"] = {label: {"n": int(sub.sum()), "accuracy": float(right[sub].double().mean())}
                                  for value, label in REGIONS.items()
                                  for sub in [mask & (streams.region[:, 1:] == value)] if int(sub.sum())}
        for label, flags in (extra or {}).items():
            flags = flags[:, :-1]
            entry[label] = float(flags[mask].double().mean())
            both = mask & flags
            entry[f"accuracy_given_{label}"] = float(right[both].double().mean()) if int(both.sum()) else None
            entry[f"accuracy_given_not_{label}"] = (float(right[mask & ~flags].double().mean())
                                                    if int((mask & ~flags).sum()) else None)
        out[name] = entry
    return out
