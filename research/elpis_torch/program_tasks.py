"""R3 long-form generation benchmark: a token-level structured program language with an exact checker.
SEMANTICS=NONE. Nothing here runs Python's ``exec`` or any external interpreter: ``check`` is a small state
machine over token ids.

Statements (``;`` is END):

    DEF FN_i AR_n ;               declare (or REDECLARE: the newest arity is current) function i with n args
    LET VAR_j NUM_v ;             assign (or reassign) variable j
    CALL FN_i VAR_a ... VAR_z ;   exactly the current arity of i, every argument a defined variable
    PRINT VAR_j = NUM_v ;         v must be the CURRENT value of j  (execution-based check)
    NOTE w w ... w ;              a comment of filler words

A program is a HEADER (every function and variable declared, revisions and comments in between: thousands of
tokens) and a BODY. Free-running evaluation feeds the header and lets the model write the body itself; the
checker then parses what it wrote and counts violations against the state the model's OWN statements produce
(its LET and DEF revisions are honoured). The determined decisions (where a CALL ends, what a PRINT prints)
depend on facts that may be thousands of tokens back.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

FUNCTIONS, VARIABLES, ARITIES, NUMBERS, WORDS = 32, 32, 4, 16, 32
BOS, DEF, LET, CALL, PRINT, EQ, END, NOTE = range(8)
FN0 = 8
VAR0 = FN0 + FUNCTIONS
AR0 = VAR0 + VARIABLES                 # AR0 + (n - 1) is arity n
NUM0 = AR0 + ARITIES
WORD0 = NUM0 + NUMBERS
VOCAB = WORD0 + WORDS

# Per-token roles of a reference program (for teacher-forced scoring):
FREE, STRUCTURE, ARITY_END, ARITY_ARG, PRINT_VALUE, COMMENT = 0, 1, 2, 3, 4, 5
ROLES = {STRUCTURE: "structure", ARITY_END: "call_end", ARITY_ARG: "call_continues", PRINT_VALUE: "print_value"}


@dataclass(frozen=True)
class ProgramTask:
    functions: int = 24
    variables: int = 24
    header_tokens: int = 1024          # approximate header length (comments pad it)
    body_statements: int = 160
    p_call: float = 0.45
    p_print: float = 0.35
    p_let: float = 0.15                # the rest (0.05) redeclares a function
    comment_words: tuple = (4, 24)
    header_revisions: float = 0.25     # fraction of declarations that are later revised inside the header


@dataclass
class Programs:
    tokens: torch.Tensor               # [n, T]
    role: torch.Tensor                 # [n, T]
    header: list = field(default_factory=list)   # header length per row (tokens, BOS included)
    source: torch.Tensor | None = None  # [n, T]: evidence position of each determined token (-1: none)

    def query_source(self) -> torch.Tensor:
        """Aligned to INPUT positions (the token at q predicts q + 1). TRAINING-ONLY label."""
        out = torch.full_like(self.source, -1)
        out[:, :-1] = self.source[:, 1:]
        return out

    def query_block(self, block: int) -> torch.Tensor:
        src = self.query_source()
        return torch.where(src >= 0, src // block, src)


def evidence(tokens: list[int], roles: list[int]) -> list[int]:
    """The position whose token decides each determined token: the arity token of the latest DEF of the called
    function (where a CALL continues or ends), the value token of the latest LET of a printed variable."""
    fn_at, var_at, out, at = {}, {}, [-1] * len(tokens), 0
    current = None
    for at, t in enumerate(tokens):
        if t == DEF and at + 3 < len(tokens) and tokens[at + 3] == END:
            fn_at[tokens[at + 1] - FN0] = at + 2
        elif t == LET and at + 3 < len(tokens) and tokens[at + 3] == END:
            var_at[tokens[at + 1] - VAR0] = at + 2
        if t == CALL and at + 1 < len(tokens):
            current = fn_at.get(tokens[at + 1] - FN0, -1)
        if roles[at] in (ARITY_ARG, ARITY_END):
            out[at] = current if current is not None else -1
        elif roles[at] == PRINT_VALUE:
            out[at] = var_at.get(tokens[at - 2] - VAR0, -1)
    return out


def _statement(rng, kind, state, task):
    fns, vrs = state
    if kind == "call":
        f = rng.choice(sorted(fns))
        args = [VAR0 + rng.choice(sorted(vrs)) for _ in range(fns[f])]
        roles = [STRUCTURE, FREE] + [FREE if i == 0 else ARITY_ARG for i in range(len(args))] + [ARITY_END]
        # roles: CALL (structure), FN (free), first arg (free choice of variable, but its presence is forced
        # because every arity >= 1), later args continue (ARITY_ARG), END (ARITY_END)
        return [CALL, FN0 + f, *args, END], roles
    if kind == "print":
        j = rng.choice(sorted(vrs))
        return [PRINT, VAR0 + j, EQ, NUM0 + vrs[j], END], [STRUCTURE, FREE, STRUCTURE, PRINT_VALUE, STRUCTURE]
    if kind == "let":
        j, v = rng.randrange(task.variables), rng.randrange(NUMBERS)
        vrs[j] = v
        return [LET, VAR0 + j, NUM0 + v, END], [STRUCTURE, FREE, FREE, STRUCTURE]
    if kind == "def":
        f, n = rng.randrange(task.functions), rng.randrange(1, ARITIES + 1)
        fns[f] = n
        return [DEF, FN0 + f, AR0 + n - 1, END], [STRUCTURE, FREE, FREE, STRUCTURE]
    words = rng.randrange(*task.comment_words)
    return [NOTE, *[WORD0 + rng.randrange(WORDS) for _ in range(words)], END], \
        [STRUCTURE] + [COMMENT] * words + [STRUCTURE]


def _program(task: ProgramTask, rng: random.Random):
    fns, vrs, out, roles = {}, {}, [BOS], [STRUCTURE]
    pending = [("def", f) for f in range(task.functions)] + [("let", j) for j in range(task.variables)]
    rng.shuffle(pending)
    revisions = int(task.header_revisions * len(pending))
    pending += [(rng.choice(("def", "let")), None) for _ in range(revisions)]
    per = max(0, task.header_tokens - 4 * len(pending)) / max(1, len(pending))     # comment tokens per gap
    for index, (kind, which) in enumerate(pending):
        while len(out) < 1 + (per + 4) * index + per:
            symbols, role = _statement(rng, "note", (fns, vrs), task)
            out += symbols
            roles += role
        if which is None:
            symbols, role = _statement(rng, kind, (fns, vrs), task)
        elif kind == "def":
            n = rng.randrange(1, ARITIES + 1)
            fns[which] = n
            symbols, role = [DEF, FN0 + which, AR0 + n - 1, END], [STRUCTURE, FREE, FREE, STRUCTURE]
        else:
            v = rng.randrange(NUMBERS)
            vrs[which] = v
            symbols, role = [LET, VAR0 + which, NUM0 + v, END], [STRUCTURE, FREE, FREE, STRUCTURE]
        out += symbols
        roles += role
    header = len(out)
    cuts = (task.p_call, task.p_call + task.p_print, task.p_call + task.p_print + task.p_let)
    for _ in range(task.body_statements):
        roll = rng.random()
        kind = "call" if roll < cuts[0] else "print" if roll < cuts[1] else "let" if roll < cuts[2] else "def"
        symbols, role = _statement(rng, kind, (fns, vrs), task)
        out += symbols
        roles += role
    return out, roles, header


def generate(task: ProgramTask, count: int, *, seed: int, length: int | None = None) -> Programs:
    """``count`` programs, cut or padded (with comments) to a common ``length``."""
    rows, roles, headers = [], [], []
    for row in range(count):
        tokens, role, header = _program(task, random.Random(f"elpis-torch-r3-program:{seed}:{row}"))
        rows.append(tokens)
        roles.append(role)
        headers.append(header)
    size = length or min(len(r) for r in rows)
    for index, row in enumerate(rows):
        rng = random.Random(f"elpis-torch-r3-pad:{seed}:{index}")
        while len(row) < size:
            symbols, role = _statement(rng, "note", ({}, {}), task)
            row += symbols
            roles[index] += role
    tokens = torch.tensor([r[:size] for r in rows], dtype=torch.long)
    role = torch.tensor([r[:size] for r in roles], dtype=torch.int8)
    source = torch.tensor([evidence(r[:size], q[:size]) for r, q in zip(rows, roles)], dtype=torch.long)
    return Programs(tokens, role, headers, source)


def program_loss(logits, tokens, role, *, comment_weight: float = 0.02):
    """Teacher-forced loss; comment words (unpredictable filler) are down-weighted."""
    targets, roles = tokens[:, 1:], role[:, 1:]
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                                               reduction="none").view(targets.shape)
    weight = torch.where(roles == COMMENT, comment_weight, 1.0)
    return (losses * weight).sum() / weight.sum()


def teacher_forced(predicted: torch.Tensor, programs: Programs) -> dict:
    """Accuracy at the determined positions of the reference body, by role (header positions excluded)."""
    right = predicted[:, :-1] == programs.tokens[:, 1:]
    roles = programs.role[:, 1:]
    body = torch.zeros_like(right)
    for row, header in enumerate(programs.header):
        body[row, header:] = True
    # Where a CALL continues, the variable is a free choice: the determined decision is "not END yet".
    continues = (predicted[:, :-1] >= VAR0) & (predicted[:, :-1] < VAR0 + VARIABLES)
    right = torch.where(roles == ARITY_ARG, continues, right)
    return {name: {"n": int(m.sum()), "accuracy": float(right[m].double().mean())}
            for code, name in ROLES.items() for m in [body & (roles == code)] if int(m.sum())}


def _defined_state(tokens: list[int]):
    """The function arities and variable values a token sequence declares (well-formed statements only)."""
    fns, vrs, at = {}, {}, 0
    while at < len(tokens):
        t = tokens[at]
        if t == DEF and at + 3 < len(tokens) and tokens[at + 3] == END:
            fns[tokens[at + 1] - FN0] = tokens[at + 2] - AR0 + 1
            at += 4
        elif t == LET and at + 3 < len(tokens) and tokens[at + 3] == END:
            vrs[tokens[at + 1] - VAR0] = tokens[at + 2] - NUM0
            at += 4
        else:
            at += 1
    return fns, vrs


def check(prompt: list[int], generated: list[int]) -> dict:
    """Parse a free-running body against the state the prompt (and the body's own revisions) defines.

    Counts per statement kind; a malformed statement is a grammar error and the parser resynchronizes at the
    next END. Execution-based: a PRINT is correct only if it prints the current value.
    """
    fns, vrs = _defined_state(prompt)
    stats = {"statements": 0, "grammar_errors": 0, "calls": 0, "call_unknown_function": 0, "call_arity_wrong": 0,
             "call_undefined_variable": 0, "prints": 0, "print_undefined_variable": 0, "print_wrong_value": 0,
             "lets": 0, "defs": 0, "notes": 0, "first_error_token": None, "tokens": len(generated),
             "distinct_functions_called": 0}
    called = set()
    at = 0

    def error(where):
        if stats["first_error_token"] is None:
            stats["first_error_token"] = where

    is_fn = lambda t: FN0 <= t < FN0 + FUNCTIONS     # noqa: E731
    is_var = lambda t: VAR0 <= t < VAR0 + VARIABLES  # noqa: E731
    is_num = lambda t: NUM0 <= t < NUM0 + NUMBERS    # noqa: E731
    while at < len(generated):
        try:
            end = generated.index(END, at)
        except ValueError:
            break                                       # an unfinished last statement is not counted
        s = generated[at:end + 1]
        head = s[0]
        stats["statements"] += 1
        if head == CALL and len(s) >= 3 and is_fn(s[1]) and all(is_var(t) for t in s[2:-1]):
            stats["calls"] += 1
            f, args = s[1] - FN0, [t - VAR0 for t in s[2:-1]]
            called.add(f)
            if f not in fns:
                stats["call_unknown_function"] += 1
                error(at)
            elif len(args) != fns[f]:
                stats["call_arity_wrong"] += 1
                error(at)
            if any(a not in vrs for a in args):
                stats["call_undefined_variable"] += 1
                error(at)
        elif head == PRINT and len(s) == 5 and is_var(s[1]) and s[2] == EQ and is_num(s[3]):
            stats["prints"] += 1
            j = s[1] - VAR0
            if j not in vrs:
                stats["print_undefined_variable"] += 1
                error(at)
            elif s[3] - NUM0 != vrs[j]:
                stats["print_wrong_value"] += 1
                error(at)
        elif head == LET and len(s) == 4 and is_var(s[1]) and is_num(s[2]):
            stats["lets"] += 1
            vrs[s[1] - VAR0] = s[2] - NUM0
        elif head == DEF and len(s) == 4 and is_fn(s[1]) and AR0 <= s[2] < AR0 + ARITIES:
            stats["defs"] += 1
            fns[s[1] - FN0] = s[2] - AR0 + 1
        elif head == NOTE and all(WORD0 <= t < WORD0 + WORDS for t in s[1:-1]):
            stats["notes"] += 1
        else:
            stats["grammar_errors"] += 1
            error(at)
        at = end + 1
    stats["distinct_functions_called"] = len(called)
    bad = (stats["grammar_errors"] + stats["call_unknown_function"] + stats["call_arity_wrong"]
           + stats["call_undefined_variable"] + stats["print_undefined_variable"] + stats["print_wrong_value"])
    # A fraction over zero statements is None (no evidence), never a perfect score.
    stats.update(valid_statement_fraction=1 - min(1.0, bad / stats["statements"]) if stats["statements"] else None,
                 arity_accuracy=1 - stats["call_arity_wrong"] / stats["calls"] if stats["calls"] else None,
                 print_accuracy=(1 - (stats["print_wrong_value"] + stats["print_undefined_variable"]) / stats["prints"]
                                 if stats["prints"] else None),
                 violations_per_1k_tokens=1000 * bad / max(1, len(generated)))
    return stats
