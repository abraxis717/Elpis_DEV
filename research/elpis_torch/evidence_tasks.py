"""R5 synthetic evidence-retrieval tasks: revisions, hard negatives and scoped program symbols. SEMANTICS=NONE.

The vocabulary EXTENDS the R3 facts vocabulary (``longrange_tasks``): every R3 stream is a valid R5 input, so the
R3 benchmark is scored by an R5 model unchanged. Streams are filler with statements placed at chosen positions;
nothing here is natural language.

Families
    records   ``KEY_k VAL_v`` binds key k (a later bind REVISES it: the latest wins). Questions: ``ASK KEY_k``
              (direct: the question names the key) or ``ASSOC_c`` (associative: c is paired with k by R3's fixed
              permutation, a learned association), answered by ``VAL_v``.
    scoped    ``DEF SCOPE_s SYM_x VAL_v`` declares symbol x in scope s (a later DEF of (s, x) redefines it).
              Question: ``GET SCOPE_s SYM_x`` answered by ``VAL_v``. A DEF of x in another scope is NOT evidence.
    r3facts   the R3 generator, converted (aliases, links, 2/3-hop chains, planted facts).

Hard negatives, each generated explicitly and recorded per probe (``NEGATIVES``):
    superseded      an older bind of the same key / an older DEF of the same scoped symbol (older definition)
    negated         ``NOT KEY_k VAL_x``: a verbatim bind pattern of the same key with a wrong value, negated
    reference       ``USE KEY_k`` / ``CALL SCOPE_s SYM_x``: the key or symbol mentioned without a value
    overlap         ``USE KEY_k KEY_j VAL_x``: the key, then another key's bind with a wrong value (lexical overlap)
    same_value      the right value under another key (records) or another symbol of the same scope (scoped)
    stale_question  ``ASK KEY_k VAL_old``: an earlier question answered with the then-current value; it contains
                    the bind pattern ``KEY_k VAL_old`` verbatim (duplicate-like)
    stale_assoc     ``ASSOC_c VAL_old``: the same through the learned associate of k (semantic distractor)
    wrong_scope     ``DEF SCOPE_t SYM_x VAL_y``, t != s, placed before or AFTER the gold declaration

Labels (``kind``, ``source``, the probe table) serve scoring and the TRAINING-ONLY retrieval losses. They are
never inputs of a model in learned evaluation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from . import longrange_tasks as r3

NOT, USE, DEF, GET, CALL, ASK = range(r3.VOCAB, r3.VOCAB + 6)
SCOPES, SYMBOLS = 16, 32
SCOPE0 = CALL + 1
SYM0 = SCOPE0 + SCOPES
VOCAB = SYM0 + SYMBOLS
VALUES = r3.VALUES
SCOPED = 9                                    # probe kind of a GET answer (R3 kinds are 1 .. 8)
KINDS = {**r3.KINDS, SCOPED: "scoped"}
FAMILIES = {1: "records", 2: "scoped", 3: "r3facts"}
NEGATIVES = ("superseded", "negated", "reference", "overlap", "same_value", "stale_question", "stale_assoc",
             "wrong_scope")
NEG_BIT = {name: 1 << i for i, name in enumerate(NEGATIVES)}
SLOTS = 8                                     # stale / wrong-scope evidence positions kept per probe


@dataclass(frozen=True)
class RecordTask:
    """``queries`` final questions per stream, each on its own key, with ``revisions`` superseded binds and the
    listed hard negatives per queried key. ``background`` unrelated binds per 1,024 tokens set the fact density
    independently of the length. Gold distances are log-uniform in ``[min_distance, length]``."""
    keys: int = 128
    queries: int = 16
    revisions: tuple = (0, 2)
    negated: tuple = (0, 1)
    reference: tuple = (0, 1)
    overlap: tuple = (0, 1)
    p_stale_question: float = 0.5             # per superseded bind: an earlier question answered with its value
    p_assoc: float = 0.25                     # final (and stale) questions asked through ASSOC_c
    background: float = 8.0
    min_distance: int = 4
    far: float = 0.0                          # fraction of gold distances drawn uniformly (else log-uniform)


@dataclass(frozen=True)
class ScopeTask:
    queries: int = 16
    revisions: tuple = (0, 2)
    wrong_scope: tuple = (1, 2)
    reference: tuple = (0, 1)
    same_value: tuple = (0, 1)
    p_stale_question: float = 0.5
    background: float = 6.0
    min_distance: int = 4
    far: float = 0.0


@dataclass
class EvidenceStreams:
    tokens: torch.Tensor                      # [n, T] int64
    kind: torch.Tensor                        # [n, T] int8: probe kind of the ANSWER token at that position
    source: torch.Tensor                      # [n, T] int64: absolute position of its evidence token (-1: none)
    family: str = "records"
    probes: dict = field(default_factory=dict)   # one row per probe: row, answer, kind, source, revised,
    #                                              negatives (bit mask), stale [P, SLOTS], wrong_scope [P, SLOTS]

    def query_source(self) -> torch.Tensor:
        """Aligned to INPUT positions: the token at q predicts q + 1, so its evidence is ``source[q + 1]``."""
        out = torch.full_like(self.source, -1)
        out[:, :-1] = self.source[:, 1:]
        return out

    def query_block(self, block: int) -> torch.Tensor:
        src = self.query_source()
        return torch.where(src >= 0, src // block, src)


class _Stream:
    """One row under construction: filler, an occupancy map, and the probe records."""

    def __init__(self, length: int, rng: random.Random, filler: torch.Tensor):
        self.length, self.rng = length, rng
        self.tokens = filler
        self.tokens[0] = r3.BOS
        self.busy = bytearray(length)
        self.busy[0] = 1
        self.kind, self.source = {}, {}
        self.probes = []

    def free(self, at: int, size: int) -> bool:
        return 1 <= at and at + size <= self.length and not any(self.busy[at:at + size])

    def put(self, at: int, symbols) -> None:
        self.tokens[at:at + len(symbols)] = torch.tensor(symbols, dtype=torch.long)
        self.reserve(at, len(symbols))

    def reserve(self, at: int, size: int) -> None:
        for i in range(size):
            self.busy[at + i] = 1

    def release(self, at: int, size: int) -> None:
        for i in range(size):
            self.busy[at + i] = 0

    def place(self, size: int, lo: int, hi: int, tries: int = 64):
        """A free start in ``[lo, hi]`` (inclusive), or None."""
        lo, hi = max(lo, 1), min(hi, self.length - size)
        if hi < lo:
            return None
        for _ in range(tries):
            at = self.rng.randint(lo, hi)
            if self.free(at, size):
                return at
        for at in range(lo, hi + 1):            # dense corner: scan (bounded by the interval)
            if self.free(at, size):
                return at
        return None

    def probe(self, answer: int, kind: int, source: int, *, revised=False, negatives=0, stale=(), wrong=()):
        self.kind[answer], self.source[answer] = kind, source
        self.probes.append((answer, kind, source, revised, negatives, list(stale)[-SLOTS:], list(wrong)[-SLOTS:]))


def _log_uniform(rng: random.Random, lo: int, hi: int, far: float = 0.0) -> int:
    """A gold distance: uniform in ``[lo, hi]`` with probability ``far``, else log-uniform."""
    if hi <= lo:
        return lo
    if far and rng.random() < far:
        return rng.randint(lo, hi)
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def _assemble(rows, length: int, family: str) -> EvidenceStreams:
    count = len(rows)
    tokens = torch.stack([row.tokens for row in rows])
    kind = torch.zeros(count, length, dtype=torch.int8)
    source = torch.full((count, length), -1, dtype=torch.int64)
    table = {"row": [], "answer": [], "kind": [], "source": [], "revised": [], "negatives": [], "stale": [],
             "wrong_scope": []}
    for index, row in enumerate(rows):
        for at, what in row.kind.items():
            kind[index, at], source[index, at] = what, row.source[at]
        for answer, what, src, revised, negatives, stale, wrong in sorted(row.probes):
            table["row"].append(index)
            table["answer"].append(answer)
            table["kind"].append(what)
            table["source"].append(src)
            table["revised"].append(bool(revised))
            table["negatives"].append(negatives)
            table["stale"].append(stale + [-1] * (SLOTS - len(stale)))
            table["wrong_scope"].append(wrong + [-1] * (SLOTS - len(wrong)))
    probes = {k: torch.tensor(v, dtype=torch.bool if k == "revised" else torch.long) for k, v in table.items()}
    if not table["row"]:
        probes["stale"] = probes["stale"].view(0, SLOTS)
        probes["wrong_scope"] = probes["wrong_scope"].view(0, SLOTS)
    return EvidenceStreams(tokens, kind, source, family, probes)


def _filler(count: int, length: int, seed: int) -> torch.Tensor:
    return torch.randint(r3.FILLER0, r3.FILLER0 + r3.FILLERS, (count, length),
                         generator=torch.Generator().manual_seed(seed))


def generate_records(task: RecordTask, count: int, length: int, *, seed: int) -> EvidenceStreams:
    filler = _filler(count, length, seed)
    rows = []
    for index in range(count):
        rng = random.Random(f"elpis-torch-r5-records:{seed}:{index}")
        row = _Stream(length, rng, filler[index])
        queried = rng.sample(range(task.keys), min(task.queries, task.keys))
        others = [k for k in range(r3.MAX_KEYS) if k not in set(queried)]
        plans = []
        for key in queried:
            answer_q = rng.randint(int(0.05 * length) + 5, length - 1)        # the final answer token
            ask_at = row.place(3, answer_q - 2, answer_q - 2, tries=1)
            if ask_at is None:
                ask_at = row.place(3, int(0.05 * length) + 3, length - 3)
            if ask_at is None:
                continue
            row.reserve(ask_at, 3)                                            # the question is written last
            distance = _log_uniform(rng, task.min_distance, max(task.min_distance, ask_at - 3), task.far)
            bind = row.place(2, ask_at - distance - 1, ask_at - distance - 1, tries=1)
            if bind is None:
                bind = row.place(2, max(1, ask_at - 2 * distance), ask_at - 3)
            if bind is None:
                row.release(ask_at, 3)
                continue
            value = rng.randrange(VALUES)
            row.put(bind, [r3.KEY0 + key, r3.VAL0 + value])
            plans.append((key, ask_at, bind, value))
        for key, ask_at, bind, value in plans:
            negatives, stale, history = 0, [], []
            revisions = rng.randint(*task.revisions)
            older = []
            for _ in range(revisions):                                        # superseded binds, before the gold
                at = row.place(2, 1, bind - 3)
                if at is None:
                    break
                old = rng.choice([v for v in range(VALUES) if v != value])
                row.put(at, [r3.KEY0 + key, r3.VAL0 + old])
                older.append((at, old))
            older.sort()
            if older:
                negatives |= NEG_BIT["superseded"]
            chain = older + [(bind, value)]
            for (at, old), (nxt, _) in zip(chain[:-1], chain[1:]):           # earlier questions, then revised
                stale.append(at + 1)
                if rng.random() < task.p_stale_question:
                    assoc = rng.random() < task.p_assoc
                    qa = row.place(3, at + 2, nxt - 3)
                    if qa is not None:
                        _question(row, qa, key, old, assoc)
                        row.probe(qa + 2, r3.ASSOC if assoc else r3.DIRECT, at + 1,
                                  revised=bool(history), negatives=0, stale=[h + 1 for h, _ in history])
                        negatives |= NEG_BIT["stale_assoc" if assoc else "stale_question"]
                history.append((at, old))
            for _ in range(rng.randint(*task.negated)):
                at = row.place(3, 1, ask_at - 3)
                if at is not None:
                    row.put(at, [NOT, r3.KEY0 + key, r3.VAL0 + rng.choice([v for v in range(VALUES) if v != value])])
                    negatives |= NEG_BIT["negated"]
            for _ in range(rng.randint(*task.reference)):
                at = row.place(2, 1, ask_at - 2)
                if at is not None:
                    row.put(at, [USE, r3.KEY0 + key])
                    negatives |= NEG_BIT["reference"]
            for _ in range(rng.randint(*task.overlap)):
                at = row.place(4, 1, ask_at - 4)
                if at is not None:
                    wrong = rng.choice([v for v in range(VALUES) if v != value])
                    row.put(at, [USE, r3.KEY0 + key, r3.KEY0 + rng.choice(others), r3.VAL0 + wrong])
                    negatives |= NEG_BIT["overlap"]
            assoc = rng.random() < task.p_assoc
            _question(row, ask_at, key, value, assoc)
            row.probe(ask_at + 2, r3.ASSOC if assoc else r3.DIRECT, bind + 1, revised=bool(older),
                      negatives=negatives, stale=stale)
        for _ in range(int(task.background * length / 1024)):                # unrelated binds (fact density)
            at = row.place(2, 1, length - 2, tries=8)
            if at is not None:
                row.put(at, [r3.KEY0 + rng.choice(others), r3.VAL0 + rng.randrange(VALUES)])
        rows.append(row)
    streams = _assemble(rows, length, "records")
    _mark_same_value(streams)
    return streams


def _question(row: _Stream, at: int, key: int, value: int, assoc: bool) -> None:
    """Three reserved positions: ``ASK KEY_k VAL_v`` or ``(filler) ASSOC_c VAL_v``; the answer is at ``at + 2``."""
    if assoc:
        row.reserve(at, 1)
        row.put(at + 1, [r3.ASSOC0 + r3._PAIRING[key], r3.VAL0 + value])
    else:
        row.put(at, [ASK, r3.KEY0 + key, r3.VAL0 + value])


def _mark_same_value(streams: EvidenceStreams) -> None:
    """Set the ``same_value`` bit of each records probe whose answer value is also bound to another key between
    its gold bind and its question (a bind pattern ``KEY_j VAL_v`` with j != k)."""
    tokens, p = streams.tokens, streams.probes
    for i in range(p["row"].numel()):
        row, answer, src = int(p["row"][i]), int(p["answer"][i]), int(p["source"][i])
        if src < 1:
            continue
        value, key = int(tokens[row, src]), int(tokens[row, src - 1])
        span = tokens[row, src + 1:answer - 1]
        if span.numel() < 2:
            continue
        hit = (span[1:] == value) & (span[:-1] >= r3.KEY0) & (span[:-1] < r3.KEY0 + r3.MAX_KEYS) & (span[:-1] != key)
        if bool(hit.any()):
            p["negatives"][i] |= NEG_BIT["same_value"]


def generate_scoped(task: ScopeTask, count: int, length: int, *, seed: int) -> EvidenceStreams:
    filler = _filler(count, length, seed + 1)
    rows = []
    for index in range(count):
        rng = random.Random(f"elpis-torch-r5-scoped:{seed}:{index}")
        row = _Stream(length, rng, filler[index])
        pairs = rng.sample([(s, x) for s in range(SCOPES) for x in range(SYMBOLS)], task.queries)
        queried = set(pairs)
        plans = []
        for scope, symbol in pairs:
            answer_q = rng.randint(int(0.05 * length) + 5, length - 2)
            ask_at = row.place(4, answer_q - 3, answer_q - 3, tries=1)
            if ask_at is None:
                ask_at = row.place(4, int(0.05 * length) + 3, length - 4)
            if ask_at is None:
                continue
            row.reserve(ask_at, 4)
            distance = _log_uniform(rng, task.min_distance, max(task.min_distance, ask_at - 4), task.far)
            bind = row.place(4, ask_at - distance - 3, ask_at - distance - 3, tries=1)
            if bind is None:
                bind = row.place(4, max(1, ask_at - 2 * distance), ask_at - 5)
            if bind is None:
                row.release(ask_at, 4)
                continue
            value = rng.randrange(VALUES)
            row.put(bind, [DEF, SCOPE0 + scope, SYM0 + symbol, r3.VAL0 + value])
            plans.append((scope, symbol, ask_at, bind, value))
        for scope, symbol, ask_at, bind, value in plans:
            negatives, stale, wrong, older, history = 0, [], [], [], []
            for _ in range(rng.randint(*task.revisions)):
                at = row.place(4, 1, bind - 5)
                if at is None:
                    break
                old = rng.choice([v for v in range(VALUES) if v != value])
                row.put(at, [DEF, SCOPE0 + scope, SYM0 + symbol, r3.VAL0 + old])
                older.append((at, old))
            older.sort()
            if older:
                negatives |= NEG_BIT["superseded"]
            chain = older + [(bind, value)]
            for (at, old), (nxt, _) in zip(chain[:-1], chain[1:]):
                stale.append(at + 3)
                if rng.random() < task.p_stale_question:
                    qa = row.place(4, at + 4, nxt - 4)
                    if qa is not None:
                        row.put(qa, [GET, SCOPE0 + scope, SYM0 + symbol, r3.VAL0 + old])
                        row.probe(qa + 3, SCOPED, at + 3, revised=bool(history), stale=[h + 3 for h, _ in history])
                        negatives |= NEG_BIT["stale_question"]
                history.append((at, old))
            others = [s for s in range(SCOPES) if s != scope and (s, symbol) not in queried]
            for _ in range(rng.randint(*task.wrong_scope)):
                if not others:
                    break
                at = row.place(4, 1, ask_at - 4)                              # before OR after the gold
                if at is not None:
                    row.put(at, [DEF, SCOPE0 + rng.choice(others), SYM0 + symbol,
                                 r3.VAL0 + rng.choice([v for v in range(VALUES) if v != value])])
                    wrong.append(at + 3)
                    negatives |= NEG_BIT["wrong_scope"]
            for _ in range(rng.randint(*task.reference)):
                at = row.place(3, 1, ask_at - 3)
                if at is not None:
                    row.put(at, [CALL, SCOPE0 + scope, SYM0 + symbol])
                    negatives |= NEG_BIT["reference"]
            siblings = [y for y in range(SYMBOLS) if y != symbol and (scope, y) not in queried]
            for _ in range(rng.randint(*task.same_value)):
                at = row.place(4, bind + 4, ask_at - 4)
                if at is not None and siblings:
                    row.put(at, [DEF, SCOPE0 + scope, SYM0 + rng.choice(siblings), r3.VAL0 + value])
                    negatives |= NEG_BIT["same_value"]
            row.put(ask_at, [GET, SCOPE0 + scope, SYM0 + symbol, r3.VAL0 + value])
            row.probe(ask_at + 3, SCOPED, bind + 3, revised=bool(older), negatives=negatives, stale=stale,
                      wrong=sorted(wrong))
        free_pairs = [(s, x) for s in range(SCOPES) for x in range(SYMBOLS) if (s, x) not in queried]
        for _ in range(int(task.background * length / 1024)):
            at = row.place(4, 1, length - 4, tries=8)
            if at is not None:
                s, x = rng.choice(free_pairs)
                row.put(at, [DEF, SCOPE0 + s, SYM0 + x, r3.VAL0 + rng.randrange(VALUES)])
        rows.append(row)
    streams = _assemble(rows, length, "scoped")
    _mark_wrong_scope(streams)
    return streams


def _mark_wrong_scope(streams: EvidenceStreams) -> None:
    """Every DEF of the probe's symbol in ANOTHER scope before the question (planted or background) is a
    wrong-scope declaration: the ``SLOTS`` most recent value positions are recorded and the bit is set."""
    tokens, p = streams.tokens, streams.probes
    for i in range(p["row"].numel()):
        row, answer = int(p["row"][i]), int(p["answer"][i])
        scope, symbol = int(tokens[row, answer - 2]), int(tokens[row, answer - 1])
        head = tokens[row, :max(0, answer - 3)]
        if head.numel() < 4:
            continue
        hit = (head[:-3] == DEF) & (head[2:-1] == symbol) & (head[1:-2] != scope)
        where = (hit.nonzero(as_tuple=True)[0] + 3)[-SLOTS:].tolist()
        p["wrong_scope"][i] = torch.tensor(where + [-1] * (SLOTS - len(where)), dtype=torch.long)
        if where:
            p["negatives"][i] |= NEG_BIT["wrong_scope"]


def from_r3(streams: r3.FactStreams) -> EvidenceStreams:
    """The R3 generator's streams as an R5 record: the same tokens and labels, a probe table with
    ``revised`` = the R3 ``contradicted`` flag (stale positions are not tracked by R3)."""
    kind, source = streams.kind, streams.source
    rows, answers = (kind != 0).nonzero(as_tuple=True)
    count = rows.numel()
    probes = {"row": rows, "answer": answers, "kind": kind[rows, answers].long(), "source": source[rows, answers],
              "revised": streams.contradicted[rows, answers],
              "negatives": torch.where(streams.contradicted[rows, answers], NEG_BIT["superseded"], 0).long(),
              "stale": torch.full((count, SLOTS), -1, dtype=torch.long),
              "wrong_scope": torch.full((count, SLOTS), -1, dtype=torch.long)}
    return EvidenceStreams(streams.tokens, kind, source, "r3facts", probes)


def generate_r3(count: int, length: int, *, seed: int, keys: int = 32, events: int | None = None,
                planted: tuple = ()) -> EvidenceStreams:
    task = r3.FactTask(keys=keys, events=events if events is not None else length * 7 // 64, planted=planted)
    return from_r3(r3.generate(task, count, length, seed=seed))


def concat(parts: list) -> EvidenceStreams:
    """Rows of several families of the same length in one batch (probe rows renumbered)."""
    tokens = torch.cat([p.tokens for p in parts])
    kind = torch.cat([p.kind for p in parts])
    source = torch.cat([p.source for p in parts])
    probes, offset = {}, 0
    for p in parts:
        for name, value in p.probes.items():
            value = value + offset if name == "row" else value
            probes.setdefault(name, []).append(value)
        offset += p.tokens.shape[0]
    probes = {k: torch.cat(v) for k, v in probes.items()}
    family = torch.cat([torch.full((p.probes["row"].numel(),), code, dtype=torch.long)
                        for p in parts for code in [next(c for c, n in FAMILIES.items() if n == p.family)]])
    probes["family"] = family
    return EvidenceStreams(tokens, kind, source, "+".join(p.family for p in parts), probes)


def with_family(streams: EvidenceStreams) -> EvidenceStreams:
    code = next(c for c, n in FAMILIES.items() if n == streams.family)
    streams.probes["family"] = torch.full((streams.probes["row"].numel(),), code, dtype=torch.long)
    return streams
