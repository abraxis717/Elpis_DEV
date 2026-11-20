"""R10: deterministic tasks for in-context adaptation and for retention under interference. RESEARCH_ONLY.

Pure Python (no PyTorch). Every episode is a byte stream for the Elpis byte tokenizer (one token per byte; the
stream starts with BOS = 256) made of examples ``R K V <space>``: a regime symbol, a key symbol, the value symbol
and a space. The value is the answer. A model predicts it from the regime and key tokens and from whatever the
episode has shown it before. Mappings and rules are drawn fresh for every episode, so no global static mapping
predicts an answer (``static_baseline`` measures it): an answer is knowable only from examples earlier in the
same episode. Nothing here learns, and nothing writes.

Families
  A  REGIME-CONDITIONED MAPPINGS. ``regimes`` regimes over one shared key set; each regime is an injective random
     map keys -> values, and the same key maps to a different value in every regime (conflicts by construction).
     Examples are drawn from the bindings with episode-random Zipf frequencies, so some bindings recur often and
     some rarely. k = the earlier exposures of the queried (regime, key) binding.
  B  SWITCHING TRANSFORMATIONS. Each regime is a cyclic shift ``value_index = (key_index + c) mod 20`` with its own
     offset ``c`` (distinct per regime); regimes alternate in segments and re-enter. A regime's inputs are drawn
     without replacement, so every answer is on an input that regime has not shown: it must come from the rule
     inferred from the regime's earlier examples. k = the earlier examples of the queried regime.
  C  RETENTION UNDER INTERFERENCE. Phase 1 shows regime P's bindings; phases 2.. show regimes Q, S (, T) that rebind
     the same keys to other values; an optional distractor gap follows; then P's bindings are probed with P's
     symbol (and Q's), then P is re-entered. Pre-interference accuracy (P's last exposures in phase 1) is set
     against post-interference accuracy (the probes). Recency errors (the latest conflicting value) are counted.

Variants place the evidence inside a 64-token attention window (``C_short``), outside it (``C_long``) or behind a
distractor gap (``C_gap``). Held-out variants never appear in training: ``B_heldout`` (offsets 15-19),
``C_gap128`` (a 128-token gap) and ``C_4reg`` (three interfering regimes).

Every answer position is a ``Probe``: ``pos`` is the INPUT position whose next-token prediction is scored (the
key token; the target there is the value), with the facts every metric and every baseline needs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass, field

BOS = 256
REGIMES = "ABCDEFGH"
KEYS = "abcdefghijklmnopqrst"                 # 20
VALUES = "0123456789!#$%&*@^_~"               # 20
SEP = " "
FILLER = ".,;:-+/="
NO_REGIME = "?"
CYCLE = len(KEYS)
assert len(VALUES) == CYCLE and not set(REGIMES + KEYS + SEP + FILLER + NO_REGIME) & set(VALUES)

TRAIN_OFFSETS = tuple(range(0, 15))           # family B rules seen in training
HELDOUT_OFFSETS = tuple(range(15, 20))        # family B rules never seen in training

TRAIN_VARIANTS = ("A", "B", "C_short", "C_long", "C_gap")
HELDOUT_VARIANTS = ("B_heldout", "C_gap128", "C_4reg")
FAMILY = {"A": "A", "B": "B", "B_heldout": "B", "C_short": "C", "C_long": "C", "C_gap": "C", "C_gap128": "C",
          "C_4reg": "C"}


def tok(ch: str) -> int:
    return ord(ch)


@dataclass(frozen=True)
class Probe:
    pos: int                       # input position whose prediction is scored (the key token)
    answer: int                    # the value token
    family: str
    variant: str
    kind: str                      # query | pre | post | mid | reentry
    regime: int                    # index of the regime within the episode
    key: int                       # key symbol index (0..19)
    k: int                         # A: earlier exposures of the binding; B: earlier examples of the regime; C: as A
    distance: int | None           # tokens since the binding's (A, C) or the regime's (B) last exposure
    novel: bool                    # B: the input was not shown before in this regime
    conflicts: tuple = ()          # values the same key had in OTHER regimes, shown earlier in the episode
    latest: int | None = None      # the most recent value shown for this key in any regime (recency baseline)
    candidates: int = CYCLE        # how many answers the generator could have produced here, given the context
    segment: int = 0               # B: segment index; C: phase index
    since_switch: int = 0          # B: earlier examples since the last regime switch


@dataclass
class Episode:
    variant: str
    tokens: list                   # BOS + content; inputs = tokens[:-1], targets = tokens[1:]
    probes: list
    signature: str                 # digest of the episode's mappings/rules (train/test disjointness)
    regime_symbols: list = field(default_factory=list)

    @property
    def family(self) -> str:
        return FAMILY[self.variant]


@dataclass(frozen=True)
class TaskSpec:
    """The frozen generator parameters (recorded with every result)."""
    length: int = 256              # input tokens per training episode
    a_regimes: int = 3
    a_keys: int = 4
    a_examples: int = 63
    a_zipf: float = 1.0
    b_regimes: int = 3
    b_examples: int = 63
    b_segment: tuple = (3, 12)     # segment lengths, inclusive
    c_short: tuple = (3, 2, 2)     # regimes, keys, exposures per key per phase
    c_long: tuple = (3, 4, 3)
    c_gap: int = 64
    c_gap_heldout: int = 128
    c_reentry: int = 1             # re-entry exposures per key

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        text = json.dumps({"spec": self.to_dict(), "regimes": REGIMES, "keys": KEYS, "values": VALUES, "sep": SEP,
                           "filler": FILLER, "train_offsets": TRAIN_OFFSETS, "heldout_offsets": HELDOUT_OFFSETS},
                          sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


def episode_seed(*parts) -> int:
    """A deterministic 63-bit seed from integers/strings (stable across processes and Python versions)."""
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


class _Writer:
    """Accumulates tokens and probes; tracks exposures for k, distances, conflicts and recency."""

    def __init__(self, variant: str):
        self.variant = variant
        self.tokens = [BOS]
        self.probes = []
        self.exposures = {}            # (regime, key) -> count
        self.last_binding = {}         # (regime, key) -> position of the last value
        self.regime_count = {}         # regime -> examples
        self.last_regime = {}          # regime -> position of the last value
        self.seen_value = {}           # (regime, key) -> value token shown
        self.latest = {}               # key -> latest value token (any regime)

    def filler(self, rng: random.Random, n: int) -> None:
        self.tokens.extend(tok(rng.choice(FILLER)) for _ in range(n))

    def example(self, regime: int, regime_symbol: str, key: int, value: int, *, kind: str, novel: bool = False,
                candidates: int = CYCLE, segment: int = 0, since_switch: int = 0, family_k: str = "binding",
                regime_token: str | None = None) -> None:
        start = len(self.tokens)
        self.tokens += [tok(regime_token or regime_symbol), tok(KEYS[key]), tok(VALUES[value]), tok(SEP)]
        binding = (regime, key)
        if family_k == "binding":
            k = self.exposures.get(binding, 0)
            last = self.last_binding.get(binding)
            if k:                                  # shown before: determined by the context
                candidates = 1
            else:                                  # injective per regime, conflicting across regimes
                excluded = {v for (r, kk), v in self.seen_value.items() if (r == regime) != (kk == key)}
                candidates = CYCLE - len(excluded)
        else:
            k = self.regime_count.get(regime, 0)
            last = self.last_regime.get(regime)
        conflicts = tuple(sorted({v for (r, kk), v in self.seen_value.items() if kk == key and r != regime}))
        value_pos = start + 2
        self.probes.append(Probe(pos=start + 1, answer=tok(VALUES[value]), family=FAMILY[self.variant],
                                 variant=self.variant, kind=kind, regime=regime, key=key, k=k,
                                 distance=None if last is None else value_pos - last, novel=novel,
                                 conflicts=conflicts, latest=self.latest.get(key), candidates=candidates,
                                 segment=segment, since_switch=since_switch))
        self.exposures[binding] = self.exposures.get(binding, 0) + 1
        self.last_binding[binding] = value_pos
        self.regime_count[regime] = self.regime_count.get(regime, 0) + 1
        self.last_regime[regime] = value_pos
        self.seen_value[binding] = tok(VALUES[value])
        self.latest[key] = tok(VALUES[value])

    def finish(self, rng: random.Random, length: int) -> list:
        if len(self.tokens) > length + 1:
            raise ValueError(f"{self.variant}: {len(self.tokens) - 1} tokens exceed the episode length {length}")
        self.filler(rng, length + 1 - len(self.tokens))
        return self.tokens


def _conflicting_tables(rng: random.Random, regimes: int, keys: int) -> list:
    """``table[r][i]``: value index of key i in regime r; injective per regime, distinct across regimes per key."""
    while True:
        table = [rng.sample(range(CYCLE), keys) for _ in range(regimes)]
        if all(len({table[r][i] for r in range(regimes)}) == regimes for i in range(keys)):
            return table


def _signature(kind: str, symbols, keys, table) -> str:
    return hashlib.sha256(json.dumps([kind, list(symbols), list(keys), table]).encode()).hexdigest()[:24]


def family_a(rng: random.Random, spec: TaskSpec, variant: str = "A", length: int | None = None) -> Episode:
    length = length or spec.length
    symbols = rng.sample(REGIMES, spec.a_regimes)
    keys = rng.sample(range(CYCLE), spec.a_keys)
    table = _conflicting_tables(rng, spec.a_regimes, spec.a_keys)
    bindings = [(r, i) for r in range(spec.a_regimes) for i in range(spec.a_keys)]
    ranks = list(range(1, len(bindings) + 1))
    rng.shuffle(ranks)
    weights = [rank ** -spec.a_zipf for rank in ranks]
    w = _Writer(variant)
    for _ in range(spec.a_examples):
        r, i = rng.choices(bindings, weights)[0]
        w.example(r, symbols[r], keys[i], table[r][i], kind="query")
    return Episode(variant, w.finish(rng, length), w.probes, _signature("A", symbols, keys, table), symbols)


def family_b(rng: random.Random, spec: TaskSpec, variant: str = "B", length: int | None = None,
             offsets=TRAIN_OFFSETS) -> Episode:
    length = length or spec.length
    symbols = rng.sample(REGIMES, spec.b_regimes)
    shifts = rng.sample(list(offsets), spec.b_regimes)
    pools = [rng.sample(range(CYCLE), CYCLE) for _ in range(spec.b_regimes)]
    w = _Writer(variant)
    done, segment, previous = 0, 0, None
    while done < spec.b_examples:
        choices = [r for r in range(spec.b_regimes) if r != previous and pools[r]]
        if not choices:
            break
        r = rng.choice(choices)
        run = min(rng.randint(*spec.b_segment), spec.b_examples - done, len(pools[r]))
        for step in range(run):
            x = pools[r].pop()
            known = w.regime_count.get(r, 0) > 0
            taken = {shifts[rr] for rr in range(spec.b_regimes) if rr != r and w.regime_count.get(rr, 0)}
            w.example(r, symbols[r], x, (x + shifts[r]) % CYCLE, kind="query", novel=True,
                      candidates=1 if known else len(offsets) - len(taken), segment=segment, since_switch=step,
                      family_k="regime")
            done += 1
        previous, segment = r, segment + 1
    return Episode(variant, w.finish(rng, length), w.probes, _signature("B", symbols, [], shifts), symbols)


def family_c(rng: random.Random, spec: TaskSpec, variant: str, length: int | None = None) -> Episode:
    regimes, keys_n, reps = {"C_short": spec.c_short, "C_long": spec.c_long, "C_gap": spec.c_long,
                             "C_gap128": spec.c_long, "C_4reg": (4,) + spec.c_long[1:]}[variant]
    gap = {"C_gap": spec.c_gap, "C_gap128": spec.c_gap_heldout}.get(variant, 0)
    length = length or (spec.length if variant != "C_gap128" else spec.length + spec.c_gap_heldout)
    symbols = rng.sample(REGIMES, regimes)
    keys = rng.sample(range(CYCLE), keys_n)
    table = _conflicting_tables(rng, regimes, keys_n)
    w = _Writer(variant)
    for phase in range(regimes):
        order = [i for i in range(keys_n) for _ in range(reps)]
        rng.shuffle(order)
        last = {i: max(n for n, j in enumerate(order) if j == i) for i in range(keys_n)}
        for n, i in enumerate(order):
            kind = "pre" if phase == 0 and n == last[i] else "query"
            w.example(phase, symbols[phase], keys[i], table[phase][i], kind=kind, segment=phase)
    w.filler(rng, gap)
    for target, kind in ((0, "post"), (1, "mid")):
        order = list(range(keys_n))
        rng.shuffle(order)
        for i in order:
            w.example(target, symbols[target], keys[i], table[target][i], kind=kind, segment=regimes)
    order = [i for i in range(keys_n) for _ in range(spec.c_reentry)]
    rng.shuffle(order)
    for i in order:
        w.example(0, symbols[0], keys[i], table[0][i], kind="reentry", segment=regimes + 1)
    return Episode(variant, w.finish(rng, length), w.probes, _signature(variant[:1], symbols, keys, table), symbols)


def make_episode(variant: str, rng: random.Random, spec: TaskSpec = TaskSpec()) -> Episode:
    if variant == "A":
        return family_a(rng, spec)
    if variant == "B":
        return family_b(rng, spec)
    if variant == "B_heldout":
        return family_b(rng, spec, "B_heldout", offsets=HELDOUT_OFFSETS)
    if variant.startswith("C_"):
        return family_c(rng, spec, variant)
    raise ValueError(f"unknown variant {variant!r}")


# -- training batches and frozen evaluation sets ------------------------------------------------------------------

TRAIN_MIX = ("A",) * 6 + ("B",) * 5 + ("C_short", "C_long", "C_gap", "C_long", "C_gap")   # 16 rows per batch


def train_episode(run_seed: int, step: int, row: int, spec: TaskSpec = TaskSpec()) -> Episode:
    variant = TRAIN_MIX[row % len(TRAIN_MIX)]
    return make_episode(variant, random.Random(episode_seed("train", run_seed, step, row)), spec)


def eval_set(variant: str, count: int, *, split_seed: int, spec: TaskSpec = TaskSpec()) -> list:
    """A frozen evaluation set: ``count`` episodes of ``variant`` from ``split_seed`` (never a training seed)."""
    return [make_episode(variant, random.Random(episode_seed("eval", split_seed, variant, n)), spec)
            for n in range(count)]


def set_digest(episodes) -> str:
    h = hashlib.sha256()
    for e in episodes:
        h.update(bytes(str(e.tokens), "ascii"))
    return h.hexdigest()


# -- diagnostics: transformed copies of an episode (the probes keep their true answers) ----------------------------

def without_regime_identity(e: Episode) -> Episode:
    """Every regime symbol replaced by ``?``: conflicting bindings can no longer be told apart."""
    regime_tokens = {tok(s) for s in REGIMES}
    tokens = [tok(NO_REGIME) if t in regime_tokens else t for t in e.tokens]
    return dataclasses.replace(e, tokens=tokens)


def with_shuffled_targets(e: Episode, rng: random.Random) -> Episode:
    """Every revealed value replaced by a random value: the support no longer carries the mapping or the rule, so
    an answer scored against the truth is unknowable (expected: chance)."""
    tokens = list(e.tokens)
    for p in e.probes:
        tokens[p.pos + 1] = tok(rng.choice(VALUES))
    return dataclasses.replace(e, tokens=tokens)


def _insert(e: Episode, at: int, block: list, *, shift_from: int) -> Episode:
    """Insert ``block`` before token index ``at``; probes at or after ``shift_from`` move with it; the episode keeps
    its length by trimming filler from the end (refused if the tail is not filler)."""
    tokens = e.tokens[:at] + block + e.tokens[at:]
    tail = tokens[len(e.tokens):]
    if any(t not in {tok(c) for c in FILLER} for t in tail):
        raise ValueError("no room: the insertion would push answers past the episode end")
    tokens = tokens[:len(e.tokens)]
    probes = [dataclasses.replace(p, pos=p.pos + len(block)) if p.pos >= shift_from else p for p in e.probes]
    return dataclasses.replace(e, tokens=tokens, probes=probes)


def first_probe_index(e: Episode, kind: str = "post") -> int:
    """Token index of the first example (its regime token) of the given kind."""
    return min(p.pos for p in e.probes if p.kind == kind) - 1


def with_irrelevant_demonstrations(e: Episode, rng: random.Random, count: int = 4) -> Episode:
    """``count`` examples of an unrelated regime with unrelated keys, inserted right before the probes."""
    used_symbols = set(e.regime_symbols)
    symbol = rng.choice([s for s in REGIMES if s not in used_symbols])
    used_keys = {p.key for p in e.probes}
    keys = rng.sample([i for i in range(CYCLE) if i not in used_keys], count)
    block = []
    for i in keys:
        block += [tok(symbol), tok(KEYS[i]), tok(VALUES[rng.randrange(CYCLE)]), tok(SEP)]
    at = first_probe_index(e)
    return _insert(e, at, block, shift_from=at)


def with_conflicting_recent(e: Episode) -> Episode:
    """Before each post-interference probe, the LATEST conflicting binding of the probed key (the last interfering
    regime's value) is shown again: the recency trap at its strongest. The episode keeps its length."""
    posts = sorted((p for p in e.probes if p.kind == "post"), key=lambda p: p.pos)
    if not posts:
        return e
    last_regime = max(p.regime for p in e.probes if p.kind in ("query", "pre"))
    symbol = tok(e.regime_symbols[last_regime])
    inserts = {p.pos - 1: [symbol, tok(KEYS[p.key]), p.latest, tok(SEP)] for p in posts}
    tokens, cursor = [], 0
    for at in sorted(inserts):
        tokens += e.tokens[cursor:at] + inserts[at]
        cursor = at
    tokens += e.tokens[cursor:]
    if any(t not in {tok(c) for c in FILLER} for t in tokens[len(e.tokens):]):
        raise ValueError("no room for the conflicting examples")
    probes = [dataclasses.replace(p, pos=p.pos + 4 * sum(1 for at in inserts if at <= p.pos - 1)) for p in e.probes]
    return dataclasses.replace(e, tokens=tokens[:len(e.tokens)], probes=probes)


# -- baselines that use no model ------------------------------------------------------------------------------------

def oracle_correct(p: Probe) -> float:
    """Expected accuracy of the Bayes-optimal predictor that knows the generator and remembers the whole episode:
    1 when the answer is determined by the context, else 1 / (answers still possible)."""
    return 1.0 / max(1, p.candidates)


def recency_prediction(p: Probe) -> int | None:
    """The trivial recency strategy: the latest value shown for this key, whatever the regime."""
    return p.latest


def static_baseline(train_episodes, test_episodes) -> dict:
    """The best GLOBAL static mapping (key -> its most frequent training value; (regime symbol, key) likewise)
    scored on the test answers: what a model that ignores the episode could reach."""
    from collections import Counter
    by_key, by_pair = {}, {}
    for e in train_episodes:
        for p in e.probes:
            regime_token = e.tokens[p.pos - 1]
            by_key.setdefault(p.key, Counter())[p.answer] += 1
            by_pair.setdefault((regime_token, p.key), Counter())[p.answer] += 1
    hits_key = hits_pair = total = 0
    for e in test_episodes:
        for p in e.probes:
            total += 1
            regime_token = e.tokens[p.pos - 1]
            if p.key in by_key and by_key[p.key].most_common(1)[0][0] == p.answer:
                hits_key += 1
            pair = (regime_token, p.key)
            if pair in by_pair and by_pair[pair].most_common(1)[0][0] == p.answer:
                hits_pair += 1
    return {"key_only": hits_key / max(1, total), "regime_and_key": hits_pair / max(1, total), "answers": total,
            "chance": 1.0 / CYCLE}
