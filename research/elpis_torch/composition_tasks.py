"""R13 P5: synthetic multi-hop record composition (1-, 2- and 3-hop strata). RESEARCH_ONLY.

An episode is one token stream of 18 blocks of 16 tokens (288 tokens):

    block 0        header: BOS, then PAD
    blocks 1..16   8 FACT blocks and 8 NOTE blocks, in a seeded order
                   FACT  x REL y PAD ...     one directed edge x -> y
                   NOTE  a b c PAD ...       three entity mentions, no edge (a lexical distractor)
    block 17       the query: PAD ..., QUERY s HOPk ASK   (ASK is the stream's last token)

The answer is the entity reached from the start entity ``s`` by following ``k`` edges. The query gives the start
symbol and the hop count, never the answer; the answer is the next token after ``ASK``.

Every required edge (the path ``s -> e1 -> ... -> ek``) sits in its own, earlier FACT block. The other FACT blocks
are hard negatives:
- an overshoot edge out of the answer (``ek -> z``), so following ``k + 1`` edges gives a wrong answer;
- converging edges into path entities (``w -> ei``), so following an edge backwards misleads;
- unrelated edges.

Ambiguous-path rejection and no shortcut edges: every entity has at most one outgoing edge in an episode and the
path visits ``k + 1`` distinct entities. The ``k``-hop answer is therefore unique, the only edge leaving ``s`` is
the path's own, and no path entity has an edge to any path entity but its successor.

Held-out combinations: a fixed fraction of the ordered entity pairs never forms an edge (required or distractor) in
a training episode; a held-out combination episode puts at least one such pair on the path. The vocabulary is fixed
and every entity occurs in training with other partners, so its embedding is trained.

Held-out structures: the stream order of the required facts (a permutation of the hop indices). Training uses some
orders; evaluation holds out the others.

``solve`` re-derives the answer from the rendered tokens alone and checks every invariant (the tests use it).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

SPECIALS = ("PAD", "BOS", "FACT", "REL", "NOTE", "QUERY", "ASK", "HOP1", "HOP2", "HOP3")
TOKEN = {name: i for i, name in enumerate(SPECIALS)}
PAD, BOS, FACT, REL, NOTE, QUERY, ASK = (TOKEN[n] for n in ("PAD", "BOS", "FACT", "REL", "NOTE", "QUERY", "ASK"))
HOP = {1: TOKEN["HOP1"], 2: TOKEN["HOP2"], 3: TOKEN["HOP3"]}
ENTITIES = 48
FIRST_ENTITY = len(SPECIALS)
VOCAB = FIRST_ENTITY + ENTITIES
BLOCK = 16
BLOCKS = 18
LENGTH = BLOCK * BLOCKS
CONTENT = tuple(range(1, 17))
QUERY_BLOCK = 17
FACT_BLOCKS = 8
NOTE_BLOCKS = 8
HELDOUT_FRACTION = 0.2
HELDOUT_SEED = 1300
TRAIN_STRUCTURES = {1: ((0,),), 2: ((0, 1),), 3: ((0, 1, 2), (2, 1, 0), (1, 0, 2), (0, 2, 1))}
HELDOUT_STRUCTURES = {2: ((1, 0),), 3: ((1, 2, 0), (2, 0, 1))}
SPLITS = ("train", "iid", "heldout_combination", "heldout_structure")


class TaskError(ValueError):
    """An episode that violates the task's invariants (never produced by ``make_episode``)."""


def entity(index: int) -> int:
    return FIRST_ENTITY + index


def heldout_pairs() -> frozenset:
    """The fixed held-out ordered entity pairs (seed ``HELDOUT_SEED``; the same for every study seed)."""
    pairs = [(a, b) for a in range(ENTITIES) for b in range(ENTITIES) if a != b]
    random.Random(HELDOUT_SEED).shuffle(pairs)
    return frozenset(pairs[:round(HELDOUT_FRACTION * len(pairs))])


HELDOUT_PAIRS = heldout_pairs()


@dataclass(frozen=True)
class Episode:
    split: str
    hops: int
    structure: tuple            # stream order of the required facts, as hop indices
    path: tuple                 # entity indices s, e1, ..., ek
    tokens: tuple               # LENGTH token ids
    answer: int                 # token id of ek
    required_blocks: tuple      # the FACT block of hop j, j = 0 .. k - 1
    fact_blocks: tuple          # every FACT block, ascending
    note_blocks: tuple
    distractors: tuple          # (kind, a, b) for every distractor FACT


def make_episode(split: str, hops: int, rng: random.Random) -> Episode:
    if split not in SPLITS or hops not in (1, 2, 3):
        raise TaskError("unknown split or hop count")
    if split == "heldout_structure" and hops == 1:
        raise TaskError("a 1-hop query has a single structure; nothing is held out")
    structures = HELDOUT_STRUCTURES[hops] if split == "heldout_structure" else TRAIN_STRUCTURES[hops]
    for _ in range(10000):
        path = rng.sample(range(ENTITIES), hops + 1)
        pairs = tuple(zip(path, path[1:]))
        novel = [p in HELDOUT_PAIRS for p in pairs]
        if (split == "heldout_combination") == any(novel):
            break
    else:
        raise TaskError("no admissible path")
    structure = structures[rng.randrange(len(structures))]
    out = dict(pairs)                                      # entity -> its one successor
    outside = [e for e in range(ENTITIES) if e not in path]
    distractors = []
    while len(distractors) < FACT_BLOCKS - hops:
        if not distractors and path[-1] not in out and rng.random() < 0.5:
            kind, a, b = "overshoot", path[-1], rng.choice(outside)
        elif rng.random() < 0.5:
            kind, a, b = "converging", rng.choice(outside), path[rng.randint(1, hops)]
        else:
            kind, a, b = "unrelated", *rng.sample(outside, 2)
        if a in out or a == b or (a, b) in HELDOUT_PAIRS:   # one outgoing edge each; no held-out distractor
            continue
        out[a] = b
        distractors.append((kind, a, b))
    slots = list(CONTENT)
    rng.shuffle(slots)
    earliest = sorted(slots[:hops])
    required = [0] * hops
    for position, hop in enumerate(structure):
        required[hop] = earliest[position]
    distractor_slots, note_slots = slots[hops:FACT_BLOCKS], slots[FACT_BLOCKS:]
    tokens = [PAD] * LENGTH
    tokens[0] = BOS

    def put(block, items):
        tokens[block * BLOCK:block * BLOCK + len(items)] = items
    for (a, b), block in zip(pairs, required):
        put(block, [FACT, entity(a), REL, entity(b)])
    for (_, a, b), block in zip(distractors, distractor_slots):
        put(block, [FACT, entity(a), REL, entity(b)])
    for block in note_slots:
        put(block, [NOTE, *(entity(e) for e in rng.sample(range(ENTITIES), 3))])
    put(QUERY_BLOCK, [PAD] * (BLOCK - 4) + [QUERY, entity(path[0]), HOP[hops], ASK])
    episode = Episode(split, hops, tuple(structure), tuple(path), tuple(tokens), entity(path[-1]), tuple(required),
                      tuple(sorted(required + distractor_slots)), tuple(sorted(note_slots)), tuple(distractors))
    solve(episode)                                          # every invariant, from the tokens alone
    return episode


def episode_rng(seed: int, split: str, index: int) -> random.Random:
    """The generator of one episode: a pure function of (study seed, split, index); string seeding is stable."""
    return random.Random(f"r13-p5:{seed}:{split}:{index}")


def episodes(seed: int, split: str, count: int, hops=(1, 2, 3), *, offset: int = 0) -> list:
    """``count`` episodes of ``split``, hop counts cycling through ``hops`` (strata of equal size)."""
    return [make_episode(split, hops[i % len(hops)], episode_rng(seed, split, offset + i)) for i in range(count)]


def solve(episode: Episode) -> tuple:
    """Re-derive ``(answer token, required blocks)`` from the tokens, checking every invariant (``TaskError``)."""
    tokens = episode.tokens
    if len(tokens) != LENGTH or tokens[0] != BOS:
        raise TaskError("stream shape")
    blocks = [tokens[b * BLOCK:(b + 1) * BLOCK] for b in range(BLOCKS)]
    query = blocks[QUERY_BLOCK]
    if tuple(query[-4:-3]) != (QUERY,) or query[-1] != ASK or any(t != PAD for t in query[:-4]):
        raise TaskError("query block")
    start, hops = query[-3], {v: k for k, v in HOP.items()}.get(query[-2])
    if hops is None:
        raise TaskError("hop token")
    edges, where = {}, {}
    facts, notes = [], []
    for b in CONTENT:
        kind = blocks[b][0]
        if kind == FACT:
            a, rel, c = blocks[b][1:4]
            if rel != REL or any(t != PAD for t in blocks[b][4:]):
                raise TaskError("fact block")
            if a in edges:
                raise TaskError("an entity with two outgoing edges (ambiguous path)")
            edges[a], where[a] = c, b
            facts.append(b)
        elif kind == NOTE:
            if any(t != PAD for t in blocks[b][4:]):
                raise TaskError("note block")
            notes.append(b)
        else:
            raise TaskError("content block kind")
    if len(facts) != FACT_BLOCKS or len(notes) != NOTE_BLOCKS:
        raise TaskError("block counts")
    walk, used = [start], []
    for _ in range(hops):
        if walk[-1] not in edges:
            raise TaskError("the path breaks")
        used.append(where[walk[-1]])
        walk.append(edges[walk[-1]])
    if len(set(walk)) != hops + 1:
        raise TaskError("the path revisits an entity")
    if walk[-1] in edges and edges[walk[-1]] in walk:        # one edge per entity: the only possible shortcut
        raise TaskError("an edge from the answer back into the path")   # is a cycle out of the answer
    if len(used) != len(set(used)) or max(used) >= QUERY_BLOCK:
        raise TaskError("required facts must sit in separate, earlier blocks")
    answer, required = walk[-1], tuple(used)
    if (answer, required) != (episode.answer, episode.required_blocks) or start != entity(episode.path[0]):
        raise TaskError("the rendered stream disagrees with the episode record")
    pairs = list(zip(episode.path, episode.path[1:]))
    novel = any(p in HELDOUT_PAIRS for p in pairs)
    if (episode.split == "heldout_combination") != novel:
        raise TaskError("held-out pairs on the wrong split")
    if any((a - FIRST_ENTITY, c - FIRST_ENTITY) in HELDOUT_PAIRS
           for a, c in edges.items() if (a, c) not in {(entity(x), entity(y)) for x, y in pairs}):
        raise TaskError("a distractor uses a held-out pair")
    order = tuple(sorted(range(hops), key=lambda j: required[j]))
    allowed = HELDOUT_STRUCTURES if episode.split == "heldout_structure" else TRAIN_STRUCTURES
    if order != episode.structure or order not in allowed[hops]:
        raise TaskError("structure")
    return answer, required


def training_row(episode: Episode) -> list:
    """The backbone's self-supervised row: the stream and its answer (next-token targets over every position)."""
    return list(episode.tokens) + [episode.answer]
