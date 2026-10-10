"""Total cognitive fuel: deterministic integer bounds admitted before any ECS reserve, transaction or mutation.

Per-field stimulus bounds (64 experiences, 256 rows and 2^20 K1 steps each) still admit a schedule of tens of millions
of learning steps. A cognitive operation is therefore admitted against a :class:`CognitiveBudget` of *totals*:
experiences, rows, rows of one experience (also the row capacity a LEARN may reserve: the memory ceiling), learning
steps, query rows, output tokens and **ECS work units**.

An ECS work unit is one multiply-accumulate-class operation of the native K1 kernels as counted by a fixed integer
formula (not measured, not wall-clock, not machine-dependent). For a state of input dimension ``d``, width ``w`` and
``F = features(d)``::

    K1 learning step on r rows   2*r*d*w + 2*d*w*F + F*F
    consolidation of r rows      r*F*F + d*w*F
    experience (r rows, k steps) k * step(r) + consolidation(r)
    QUERY of r rows              r*d*w

This module is the language boundary's copy of RuntimeCore's ``fuel`` (native/runtime/src/fuel.rs), which is the
authority for managed operations and for every C caller; the two are compared by test, and the ceiling here must
equal the compiled one. A budget can only narrow :data:`CEILING`. No wall-clock deadline is claimed: the K1 ABI is
synchronous and has no cooperative cancellation point, so fuel bounds an operation before it starts and nothing
preempts it once running.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from .errors import CompositionError

__all__ = ("CEILING", "FUEL_EXCEEDED", "CognitiveBudget", "admit_learn", "admit_query", "features", "query_units",
           "schedule_units")

FUEL_EXCEEDED = "COGNITION_FUEL_EXCEEDED"
_U64 = (1 << 64) - 1


# The canonical ceiling, equal to the compiled native one (native/runtime/src/fuel.rs ``CEILING``; tested).
_CEILING = {"max_experiences": 64, "max_rows": 16_384, "max_experience_rows": 256, "max_steps": 1 << 20,
            "max_work_units": 1 << 30, "max_query_rows": 4096, "max_output_tokens": 4096}


@dataclass(frozen=True)
class CognitiveBudget:
    """One cognitive operation's totals. Every field is an int >= 1 and at most :data:`CEILING`'s."""

    max_experiences: int = _CEILING["max_experiences"]
    max_rows: int = _CEILING["max_rows"]
    max_experience_rows: int = _CEILING["max_experience_rows"]
    max_steps: int = _CEILING["max_steps"]
    max_work_units: int = _CEILING["max_work_units"]
    max_query_rows: int = _CEILING["max_query_rows"]
    max_output_tokens: int = _CEILING["max_output_tokens"]

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if type(value) is not int or not 1 <= value <= _CEILING[f.name]:
                raise CompositionError(FUEL_EXCEEDED, f"budget {f.name}: an int in 1..{_CEILING[f.name]}")

    def native_fields(self) -> tuple:
        """The fields RuntimeCore admits (``elpis_runtime_budget``), in ABI order."""
        return (self.max_experiences, self.max_rows, self.max_experience_rows, self.max_steps, self.max_work_units,
                self.max_query_rows)


CEILING = CognitiveBudget()


def features(dim: int) -> int:
    """The S3 length of input dimension ``dim`` (``elpis_ecsg_k1_features``); 0 outside 1..64."""
    if type(dim) is not int or not 1 <= dim <= 64:
        return 0
    return dim + dim * (dim + 1) // 2 + dim * (dim + 1) * (dim + 2) // 6


def schedule_units(dim: int, width: int, pairs) -> int | None:
    """ECS work units of ``(rows, steps)`` pairs on a ``dim x width`` state; None if invalid or beyond uint64."""
    f = features(dim)
    if f == 0 or type(width) is not int or width < 1 or not pairs:
        return None
    dw = dim * width
    total = 0
    for rows, steps in pairs:
        if rows < 1 or steps < 1:
            return None
        total += steps * (2 * rows * dw + 2 * dw * f + f * f) + (rows * f * f + dw * f)
    return total if total <= _U64 else None


def query_units(dim: int, width: int, rows: int) -> int | None:
    if features(dim) == 0 or type(width) is not int or width < 1 or rows < 1:
        return None
    units = rows * dim * width
    return units if units <= _U64 else None


def _refuse(detail):
    raise CompositionError(FUEL_EXCEEDED, detail)


def _budget(budget):
    if type(budget) is not CognitiveBudget:
        raise CompositionError(FUEL_EXCEEDED, "a CognitiveBudget is required")
    return budget


def admit_learn(stimulus, dim: int, width: int, budget: CognitiveBudget) -> int:
    """Admit a LEARN schedule under ``budget``; returns its work units. Before any reserve or transaction."""
    budget = _budget(budget)
    schedule = stimulus.schedule
    pairs = tuple(zip(schedule[0::2], schedule[1::2]))   # bounded metadata: at most 64 pairs
    if len(pairs) > budget.max_experiences:
        _refuse("experiences")
    if max(rows for rows, _ in pairs) > budget.max_experience_rows:
        _refuse("rows of one experience (the reserve ceiling)")
    if sum(rows for rows, _ in pairs) > budget.max_rows:
        _refuse("total rows")
    if sum(steps for _, steps in pairs) > budget.max_steps:
        _refuse("total K1 learning steps")
    units = schedule_units(dim, width, pairs)
    if units is None or units > budget.max_work_units:
        _refuse("ECS work units")
    return units


def admit_query(stimulus, dim: int, width: int, budget: CognitiveBudget, max_output_tokens: int) -> int:
    """Admit a QUERY under ``budget``; returns its work units. Before the native query."""
    budget = _budget(budget)
    if stimulus.rows > budget.max_query_rows:
        _refuse("query rows")
    if max_output_tokens > budget.max_output_tokens:
        _refuse("output tokens")
    units = query_units(dim, width, stimulus.rows)
    if units is None or units > budget.max_work_units:
        _refuse("ECS work units")
    return units
