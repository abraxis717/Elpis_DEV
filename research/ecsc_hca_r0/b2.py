"""DEV-only unlearned asynchronous recurrence. No knowledge/admission updates or I/O.

Counter-derived schedule, replacing each fired cell by its seed clamp or strongest
local conjunction. This is not a neural NCA, trained HCA or certified stable law.
"""
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from sys import getsizeof
from types import MappingProxyType

from .baselines import Baselines, Refusal, bounded_digest, integer, require
from .task import Task, message, prepare, readout

SCHEDULE_DOMAIN = b"ecsc-hca-r1.B2.schedule\0"


@dataclass(frozen=True)
class Activation:
    values: bytes  # unsigned 16-bit big-endian cells in canonical key order
    step: int
    operations: tuple  # this explicit call only; no growing history
    owner: str  # evidence + exact query identity; blocks cross-session state transplant

    @property
    def work(self):
        return sum(n for _, n in self.operations)


class Packed:
    """Transient local accessor; only its bytearray can change, never graph objects."""
    def __init__(self, values, offsets):
        self.values, self.offsets = values, offsets

    def __getitem__(self, key):
        i = self.offsets[key]
        return 256 * self.values[i] + self.values[i + 1]

    def __setitem__(self, key, value):
        i = self.offsets[key]
        self.values[i], self.values[i + 1] = divmod(value, 256)


def footprint(value, excluded=()):
    """Python-owned object bytes, excluding shared immutable graph/task objects.

    Used as a runtime backstop, not an allocator/RSS claim. The larger reservation
    below covers transient schedules, meters, result buffers and frame overhead.
    """
    seen = {id(v) for v in excluded}
    def visit(v):
        if id(v) in seen:
            return 0
        seen.add(id(v))
        size = getsizeof(v)
        if is_dataclass(v):
            size += getsizeof(v.__dict__)
            size += sum(visit(getattr(v, f.name)) for f in fields(v))
        elif isinstance(v, (dict, MappingProxyType)):
            size += sum(visit(k) + visit(x) for k, x in v.items())
            if isinstance(v, MappingProxyType):
                # Account for the hidden backing table conservatively.
                size += 256 + 64 * len(v)
        elif isinstance(v, (tuple, list, frozenset)):
            size += sum(visit(x) for x in v)
        return size
    return visit(value)


@dataclass(frozen=True)
class B2:
    graph: Baselines
    task: Task
    cells: tuple
    offsets: object
    factors: object
    seed_set: frozenset
    auxiliary_reservation: int
    owner: str

    @classmethod
    def _start(cls, graph, task, meter):
        # Conservative object/scratch reservation, separate from exact 2*N payload.
        # Count before prepare allocates cell state. Count all scoped factors,
        # including predicate-filtered factors, to keep this admission conservative.
        count = factor_count = 0
        for hit in graph.axioms.values():
            meter.tick("state_preflight")
            count += int(hit.grant.scope == task.scope)
        for hit in graph.relations.values():
            meter.tick("state_preflight")
            factor_count += int(hit.scope == task.scope)
        reservation = (8192 + 1024 * count + 256 * factor_count
                       + 256 * len(task.predicates) + 256 * graph.limits.degree)
        require(reservation <= graph.limits.state_bytes, "STATE_LIMIT")
        scores, factors = prepare(graph, task, meter)
        owner = bounded_digest({"evidence": graph.evidence_digest, "task": task}, graph.limits, meter,
                               b"ecsc-hca-r1.B2.state-authority\0")
        meter.tick("state_index", len(scores))
        cells = tuple(sorted(scores))
        offsets = MappingProxyType({key: 2 * i for i, key in enumerate(cells)})
        meter.tick("state_allocate_byte", 2 * len(cells))
        values = bytearray(2 * len(cells))
        packed = Packed(values, offsets)
        for key in cells:
            meter.tick("state_init")
            packed[key] = scores[key]
        session = cls(graph, task, cells, offsets, MappingProxyType(factors), frozenset(task.seeds), reservation, owner)
        meter.tick("state_seal_byte", len(values))
        meter.tick("state_size_check", len(cells) + len(factors) + 1)
        state = Activation(bytes(values), 0, meter.result(), owner)
        session._memory(state)
        return session, state

    def _memory(self, state):
        # The graph and task are borrowed immutable evidence, counted in common
        # capacity. Include all auxiliary objects, conservatively even shared keys.
        require(footprint((self, state), (self.graph, self.task)) <= self.auxiliary_reservation,
                "STATE_LIMIT")

    @classmethod
    def start(cls, graph, task, *, budget=None):
        require(type(graph) is Baselines and type(task) is Task, "INVALID_TASK")
        require(budget is None or integer(budget, 1, task.work_budget), "INVALID_BUDGET")
        meter = graph._meter(task.work_budget if budget is None else budget)
        try:
            return cls._start(graph, task, meter)
        except Refusal as exc:
            exc.used = meter.used
            raise

    def _validate(self, state, meter):
        require(type(state) is Activation and type(state.values) is bytes
                and len(state.values) == 2 * len(self.cells)
                and integer(state.step, 0, 2**64 - 1) and state.owner == self.owner
                and type(state.operations) is tuple and len(state.operations) <= 32, "INVALID_STATE")
        for item in state.operations:
            meter.tick("state_metadata_check")
            require(type(item) is tuple and len(item) == 2 and type(item[0]) is str and len(item[0]) <= 64
                    and integer(item[1], 0, self.graph.limits.work), "INVALID_STATE")
        scores = Packed(state.values, self.offsets)
        for key in self.cells:
            meter.tick("state_validate")
            require(scores[key] <= 1000, "INVALID_STATE")
        return scores

    def _schedule(self, step, meter):
        entries = []
        for key in self.cells:
            meter.tick("schedule_hash_byte", len(SCHEDULE_DOMAIN) + 48)
            priority = sha256(SCHEDULE_DOMAIN + self.task.schedule_seed.to_bytes(8, "big")
                              + step.to_bytes(8, "big") + bytes.fromhex(key)).digest()
            entries.append((priority, key))
        # Comparison/allocation overhead is wall time, not instruction-counted.
        meter.tick("schedule_order", len(entries))
        return tuple(key for _, key in sorted(entries))

    def _warm(self, state, meter):
        self._validate(state, meter)
        require(state.step < 2**64 - 1, "STEP_LIMIT")
        order = self._schedule(state.step, meter)
        meter.tick("state_copy_byte", len(state.values))
        values = bytearray(state.values)
        scores = Packed(values, self.offsets)
        for key in order[:self.task.updates_per_tick]:
            meter.tick("cell_fire")
            value = 1000 if key in self.seed_set else 0
            for rkey, position in self.graph.incidence[key]:
                meter.tick("incidence_visit")
                hit = self.factors.get(rkey)
                if hit is not None and position == len(hit.roles) - 1:
                    value = max(value, message(self.graph, hit, scores, meter))
            meter.tick("cell_update")
            scores[key] = value if self.graph.axioms[key].source_valid else 0
        meter.tick("state_seal_byte", len(values))
        meter.tick("state_size_check", len(self.cells) + len(self.factors) + 1)
        result = Activation(bytes(values), state.step + 1, meter.result(), self.owner)
        self._memory(result)
        return result

    def _call(self, action, budget):
        require(budget is None or integer(budget, 1, self.task.work_budget), "INVALID_BUDGET")
        meter = self.graph._meter(self.task.work_budget if budget is None else budget)
        try:
            return action(meter)
        except Refusal as exc:
            exc.used = meter.used
            raise

    def warm(self, state, *, budget=None):
        """One explicit bounded WARM tick; failure cannot mutate the old snapshot."""
        return self._call(lambda m: self._warm(state, m), budget)

    def _observe(self, state, meter):
        scores = self._validate(state, meter)
        return readout(self.graph, self.task, scores, self.factors, meter)

    def observe(self, state, *, budget=None):
        return self._call(lambda m: self._observe(state, m), budget)

    def damage(self, state, changes, *, budget=None):
        """Explicit activation-only intervention. No source, binding or label edits.

        Damaged positives can transiently affect salience. They never certify facts;
        unsupported cycles need not clear. Fresh exact recomputation is the control.
        """
        def apply(meter):
            self._validate(state, meter)
            require(type(changes) is tuple and len(changes) <= len(self.cells), "INVALID_DAMAGE")
            seen = set()
            for change in changes:
                meter.tick("damage_validate")
                require(type(change) is tuple and len(change) == 2, "INVALID_DAMAGE")
                key, value = change
                require(type(key) is str and key in self.offsets and key not in seen
                        and integer(value, 0, 1000), "INVALID_DAMAGE")
                seen.add(key)
            meter.tick("state_copy_byte", len(state.values))
            values = bytearray(state.values)
            packed = Packed(values, self.offsets)
            for key, value in changes:
                meter.tick("damage_write")
                packed[key] = value
            meter.tick("state_seal_byte", len(values))
            meter.tick("state_size_check", len(self.cells) + len(self.factors) + 1)
            result = Activation(bytes(values), state.step, meter.result(), self.owner)
            self._memory(result)
            return result
        return self._call(apply, budget)

    def reset(self, *, budget=None):
        """Reinitialize from the immutable graph/query, including seed validation."""
        return self._call(lambda m: self._start(self.graph, self.task, m)[1], budget)


def predict_b2(graph, task):
    """Cold setup + all requested ticks + common readout under ONE work cap."""
    require(type(graph) is Baselines and type(task) is Task, "INVALID_TASK")
    meter = graph._meter(task.work_budget)
    try:
        session, state = B2._start(graph, task, meter)
        for _ in range(task.steps):
            state = session._warm(state, meter)
        return session._observe(state, meter)
    except Refusal as exc:
        exc.used = meter.used
        raise
