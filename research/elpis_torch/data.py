"""Generated token tasks with disjoint training and held-out splits. No file is read or written.

The task is a finite-state transduction. A world is a table ``delta[op][state] -> state`` (one permutation of
the states per operation). A sequence is a trajectory rendered as bytes:

    <BOS> K a=F bc=D a=A ...          (no spaces; states are 'A'.., operations 'a'.., '=' announces a state)

``a=F`` is a single step ``F = delta_a(K)``. ``bc=D`` is a double step ``D = delta_c(delta_b(F))`` whose
intermediate state is never written: the model has to compose two transitions to predict ``D``.

Only the token after ``=`` is determined by the past; the operations are uniform noise with a known entropy.
The scientific quantities are therefore measured at the state targets:

    single      state after a single step
    double      state after a double step whose (state, op, op) triple occurs in training
    heldout     state after a double step whose triple NEVER occurs in training (compositional generalization)

Splits (``build_splits``): a finite training set; a validation set from another seed with every sequence that
also occurs in training removed; a compositional set in which every double step is a held-out triple. Training
and validation never contain a held-out triple, which ``build_splits`` verifies sequence by sequence.

SEMANTICS=NONE: the symbols mean nothing. This is a mechanics and feasibility task, not language.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .tokenizer import ByteTokenizer

EQUALS = ord("=")
NONE, SINGLE, DOUBLE, HELDOUT = 0, 1, 2, 3
KIND_NAMES = {SINGLE: "single", DOUBLE: "double", HELDOUT: "heldout"}


@dataclass(frozen=True)
class World:
    """``table[op][state]`` is the next state; ``heldout`` are the (state, op1, op2) triples kept out of training."""
    states: int
    ops: int
    table: tuple
    heldout: frozenset = frozenset()

    def step(self, state: int, op: int) -> int:
        return self.table[op][state]

    def state_token(self, state: int) -> int:
        return ord("A") + state

    def op_token(self, op: int) -> int:
        return ord("a") + op

    def changed_from(self, other: "World") -> frozenset:
        """The (state, op) transitions on which this world differs from ``other``."""
        return frozenset((s, o) for o in range(self.ops) for s in range(self.states)
                         if self.table[o][s] != other.table[o][s])


def make_world(states: int = 10, ops: int = 5, *, heldout_fraction: float = 0.2, seed: int = 0) -> World:
    if not (2 <= states <= 26 and 2 <= ops <= 26):
        raise ValueError("states and ops must be in [2, 26]")
    rng = random.Random(f"elpis-torch-world:{seed}")
    table = tuple(tuple(rng.sample(range(states), states)) for _ in range(ops))
    triples = [(s, a, b) for s in range(states) for a in range(ops) for b in range(ops)]
    rng.shuffle(triples)
    return World(states, ops, table, frozenset(triples[:round(heldout_fraction * len(triples))]))


def rewire(world: World, *, op: int, seed: int = 0) -> World:
    """The same world with one operation replaced by a derangement of its old outputs (every state changes)."""
    rng = random.Random(f"elpis-torch-rewire:{seed}:{op}")
    old = world.table[op]
    while True:
        new = tuple(rng.sample(range(world.states), world.states))
        if all(new[s] != old[s] for s in range(world.states)):
            break
    table = tuple(new if o == op else world.table[o] for o in range(world.ops))
    return World(world.states, world.ops, table, frozenset())


@dataclass(frozen=True)
class Batch:
    """``tokens`` ``[n, T]`` and, per position, the kind of state target it holds (``NONE`` elsewhere).

    ``state`` and ``op`` give, for every state target, the (state before, last operation) of its step: the
    transition that produced it.
    """
    tokens: torch.Tensor
    kind: torch.Tensor
    state: torch.Tensor
    op: torch.Tensor

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def target_mask(self, *kinds: int) -> torch.Tensor:
        """``[n, T-1]`` mask over next-token targets (aligned with ``next_token_loss``)."""
        kinds = kinds or (SINGLE, DOUBLE, HELDOUT)
        return torch.isin(self.kind[:, 1:], torch.tensor(kinds))

    def select(self, index) -> "Batch":
        return Batch(self.tokens[index], self.kind[index], self.state[index], self.op[index])


def generate(world: World, count: int, length: int, *, seed: int, p_double: float = 0.3,
             mode: str = "seen") -> Batch:
    """``count`` trajectories of exactly ``length`` tokens.

    ``mode="seen"``     double steps avoid the held-out triples (training and validation).
    ``mode="heldout"``  a double step is taken only where a held-out triple exists, and is always one.
    """
    if mode not in ("seen", "heldout"):
        raise ValueError("mode must be 'seen' or 'heldout'")
    rng = random.Random(f"elpis-torch-data:{seed}:{mode}")
    tokens = torch.empty(count, length, dtype=torch.long)
    kind = torch.zeros(count, length, dtype=torch.long)
    before = torch.full((count, length), -1, dtype=torch.long)
    last_op = torch.full((count, length), -1, dtype=torch.long)
    pairs = [(a, b) for a in range(world.ops) for b in range(world.ops)]
    for row in range(count):
        state = rng.randrange(world.states)
        seq, kinds, states, ops = [ByteTokenizer.BOS, world.state_token(state)], [NONE, NONE], [-1, -1], [-1, -1]
        while len(seq) < length:
            held = [p for p in pairs if (state, *p) in world.heldout]
            seen = [p for p in pairs if (state, *p) not in world.heldout]
            if mode == "heldout":
                choice = rng.choice(held) if held and rng.random() < max(p_double, 0.5) else None
            else:
                choice = rng.choice(seen) if seen and rng.random() < p_double else None
            if choice is None:
                op = rng.randrange(world.ops)
                nxt = world.step(state, op)
                seq += [world.op_token(op), EQUALS, world.state_token(nxt)]
                kinds += [NONE, NONE, SINGLE]
                states += [-1, -1, state]
                ops += [-1, -1, op]
            else:
                middle = world.step(state, choice[0])
                nxt = world.step(middle, choice[1])
                seq += [world.op_token(choice[0]), world.op_token(choice[1]), EQUALS, world.state_token(nxt)]
                kinds += [NONE, NONE, NONE, HELDOUT if mode == "heldout" else DOUBLE]
                states += [-1, -1, -1, middle]
                ops += [-1, -1, -1, choice[1]]
            state = nxt
        tokens[row] = torch.tensor(seq[:length])
        kind[row] = torch.tensor(kinds[:length])
        before[row] = torch.tensor(states[:length])
        last_op[row] = torch.tensor(ops[:length])
    return Batch(tokens, kind, before, last_op)


def _contains_heldout(world: World, row: torch.Tensor) -> bool:
    """Whether a rendered trajectory contains a double step on a held-out triple (parsed from the bytes)."""
    seq = row.tolist()
    state, i = seq[1] - ord("A"), 2
    while i < len(seq):
        ops = []
        while i < len(seq) and seq[i] != EQUALS:
            ops.append(seq[i] - ord("a"))
            i += 1
        if i + 1 >= len(seq):
            break
        if len(ops) == 2 and (state, ops[0], ops[1]) in world.heldout:
            return True
        state, i = seq[i + 1] - ord("A"), i + 2
    return False


@dataclass(frozen=True)
class Splits:
    train: Batch
    val: Batch
    comp: Batch
    report: dict


def build_splits(world: World, *, train: int, val: int, comp: int, length: int, seed: int = 0,
                 p_double: float = 0.3) -> Splits:
    """Training, validation and compositional sets, with the split's disjointness verified, not assumed."""
    train_set = generate(world, train, length, seed=seed * 3 + 1, p_double=p_double)
    candidates = generate(world, val * 2, length, seed=seed * 3 + 2, p_double=p_double)
    comp_set = generate(world, comp, length, seed=seed * 3 + 3, p_double=p_double, mode="heldout")
    seen = {tuple(row.tolist()) for row in train_set.tokens}
    keep, unique = [], set()
    for index, row in enumerate(candidates.tokens):
        key = tuple(row.tolist())
        if key not in seen and key not in unique:
            unique.add(key)
            keep.append(index)
    if len(keep) < val:
        raise RuntimeError("could not draw enough validation sequences that are absent from training")
    val_set = candidates.select(torch.tensor(keep[:val]))
    leaked = sum(_contains_heldout(world, row) for row in (*train_set.tokens, *val_set.tokens))
    if leaked:
        raise RuntimeError(f"{leaked} training/validation sequences contain a held-out triple")
    if int((comp_set.kind == HELDOUT).sum()) == 0:
        raise RuntimeError("the compositional set holds no held-out target")
    report = {
        "train_sequences": len(train_set), "val_sequences": len(val_set), "comp_sequences": len(comp_set),
        "val_candidates_removed_as_duplicates_of_training": len(candidates) - len(keep),
        "train_val_overlap": len(seen & {tuple(r.tolist()) for r in val_set.tokens}),
        "heldout_triples": len(world.heldout), "all_double_triples": world.states * world.ops ** 2,
        "train_val_sequences_containing_a_heldout_triple": leaked,
        "comp_heldout_targets": int((comp_set.kind == HELDOUT).sum()),
        "length": length, "p_double": p_double,
    }
    return Splits(train_set, val_set, comp_set, report)
