"""R1 task channel. Inspectors are never evaluated predictions; no target/oracle input."""
from dataclasses import dataclass
from heapq import heappop, heappush

from elpis.runtime.context import BINDING, ADMITTED, CANDIDATE
from .baselines import Baselines, Refusal, digest, integer, label, require


@dataclass(frozen=True)
class Predicate:
    relation_type: int
    roles: tuple[str, ...]
    kinds: tuple[str, ...]
    statuses: tuple[str, ...] = (CANDIDATE, ADMITTED)

    def __post_init__(self):
        require(integer(self.relation_type, 1, 2**32 - 1)
                and type(self.roles) is tuple and type(self.kinds) is tuple
                and 2 <= len(self.roles) == len(self.kinds) <= 16
                and all(label(s) for s in self.roles + self.kinds)
                and len(set(self.roles)) == len(self.roles)
                and type(self.statuses) is tuple and 1 <= len(self.statuses) <= 2
                and len(set(self.statuses)) == len(self.statuses)
                and all(s in (CANDIDATE, ADMITTED) for s in self.statuses), "INVALID_PREDICATE")


@dataclass(frozen=True)
class Task:
    scope: str
    seeds: tuple[str, ...]
    predicates: tuple[Predicate, ...]
    output_kinds: tuple[str, ...]
    readout_budget: int = 32  # records: constraints + predictions + their direct relation evidence
    work_budget: int = 100_000
    steps: int = 8  # B1 rounds / B2 explicit WARM ticks; B0 computes exact closure
    schedule_seed: int = 1729
    updates_per_tick: int = 32

    def __post_init__(self):
        require(label(self.scope) and type(self.seeds) is tuple and len(self.seeds) <= 256
                and all(digest(k) for k in self.seeds) and len(set(self.seeds)) == len(self.seeds)
                and type(self.predicates) is tuple and len(self.predicates) <= 512
                and all(type(p) is Predicate for p in self.predicates)
                and len(set(self.predicates)) == len(self.predicates)
                and type(self.output_kinds) is tuple and 1 <= len(self.output_kinds) <= 256
                and all(label(k) for k in self.output_kinds)
                and integer(self.readout_budget, 0, 1024) and integer(self.work_budget, 1, 10_000_000)
                and integer(self.steps, 1, 32) and integer(self.schedule_seed, 0, 2**64 - 1)
                and integer(self.updates_per_tick, 1, 256), "INVALID_TASK")


@dataclass(frozen=True)
class Prediction:
    hit: object
    salience: int
    relations: tuple  # direct active incoming evidence, preserving original labels/provenance
    claim: str = "SALIENCE_ONLY"  # neither semantic truth nor native proof, even at 1000


@dataclass(frozen=True)
class PredictionResult:
    query: Task
    coordinate: object
    policy_digest: str
    evidence_digest: str
    predictions: tuple[Prediction, ...]
    constraints: tuple  # mandatory BINDING visibility, never auto-seeded or scored as answers
    omitted: int
    operations: tuple

    @property
    def work(self):
        return sum(n for _, n in self.operations)


def prepare(graph, task, meter):
    require(type(graph) is Baselines and type(task) is Task, "INVALID_TASK")
    require(len(task.seeds) <= graph.limits.axioms and len(task.predicates) <= graph.limits.relations
            and len(task.output_kinds) <= graph.limits.axioms, "CAPACITY")
    require(task.steps <= graph.limits.rounds, "ROUND_LIMIT")
    require(task.updates_per_tick <= graph.limits.updates_per_tick, "UPDATE_LIMIT")
    meter.tick("predicate_index", len(task.predicates))
    predicates = set(task.predicates)
    scores, factors = {}, {}
    for key, hit in graph.axioms.items():
        meter.tick("cell_init")
        if hit.grant.scope == task.scope:
            scores[key] = 0
    for key, hit in graph.relations.items():
        meter.tick("factor_filter")
        schema = graph.schemas[hit.relation.relation_type]
        # At most two status checks, independent of the number of query predicates.
        allowed = any(Predicate(schema.relation_type, schema.roles, schema.kinds, statuses) in predicates
                      for statuses in ((hit.relation.status,), (CANDIDATE, ADMITTED), (ADMITTED, CANDIDATE)))
        if hit.scope == task.scope and allowed:
            factors[key] = hit
    for key in task.seeds:
        meter.tick("seed_read")
        require(key in graph.axioms, "NOT_FOUND")
        require(key in scores, "SCOPE_MISMATCH")
        require(graph.axioms[key].source_valid, "SOURCE_INVALID")
        scores[key] = 1000
    return scores, factors


def message(graph, hit, scores, meter):
    meter.tick("factor_update")
    value = 1000
    for member in hit.relation.members[:-1]:
        meter.tick("input_read")
        value = min(value, scores[member])
    meter.tick("output_source_check")
    return graph.schemas[hit.relation.relation_type].weight * value // 1000 if hit.source_valid else 0


def readout(graph, task, scores, factors, meter):
    constraints, ranked = [], []
    for key, hit in graph.axioms.items():
        meter.tick("readout_scan")
        if hit.grant.scope != task.scope:
            continue
        if hit.grant.axiom.status == BINDING:
            constraints.append(hit)
        if scores[key] > 0 and hit.source_valid and hit.grant.kind in task.output_kinds:
            ranked.append(key)
    require(len(constraints) <= task.readout_budget, "BINDING_READOUT_LIMIT")
    remaining = task.readout_budget - len(constraints)
    meter.tick("constraint_read", len(constraints))
    predictions = []
    for key in sorted(ranked, key=lambda k: (-scores[k], k)):
        evidence = []
        for rkey, position in graph.incidence[key]:
            meter.tick("readout_incidence")
            hit = factors.get(rkey)
            if hit is not None and position == len(hit.roles) - 1 and message(graph, hit, scores, meter) > 0:
                evidence.append(hit)
        cost = 1 + len(evidence)
        if cost > remaining:
            break  # deterministic ranked prefix, with explicit loss count
        meter.tick("prediction_read", cost)
        predictions.append(Prediction(graph.axioms[key], scores[key], tuple(evidence)))
        remaining -= cost
    return PredictionResult(task, graph.view.coordinate, graph.policy_digest, graph.evidence_digest,
                            tuple(predictions), tuple(constraints), len(ranked) - len(predictions), meter.result())


def predict(graph, task, method):
    """Common task API: B0 exact worklist closure; B1 weighted synchronous rounds.

    Identical available immutable evidence and predicates. B0 is a strong exact
    recompute control, not restricted to caller-supplied answer keys or B1 rounds.
    Neither method receives the scorer's withheld keys.
    """
    require(type(graph) is Baselines and type(task) is Task, "INVALID_TASK")
    require(method in ("B0", "B1"), "INVALID_METHOD")
    meter = graph._meter(task.work_budget)
    try:
        scores, factors = prepare(graph, task, meter)
        if method == "B0":
            queue = sorted(task.seeds)
            queued = set(queue)
            meter.tick("queue_init", len(queue))
            while queue:
                meter.tick("queue_pop")
                key = heappop(queue)
                queued.remove(key)
                for rkey, position in graph.incidence[key]:
                    meter.tick("incidence_visit")
                    hit = factors.get(rkey)
                    if hit is None or position == len(hit.roles) - 1:
                        continue
                    value = message(graph, hit, scores, meter)
                    output = hit.relation.members[-1]
                    meter.tick("cell_update")
                    if value > scores[output]:
                        scores[output] = value
                        if output not in queued:
                            meter.tick("queue_push")
                            heappush(queue, output)
                            queued.add(output)
        else:
            for _ in range(task.steps):
                meter.tick("cell_copy", len(scores))
                updated = dict(scores)
                for hit in factors.values():
                    value = message(graph, hit, scores, meter)
                    output = hit.relation.members[-1]
                    meter.tick("cell_update")
                    updated[output] = max(updated[output], value)
                scores = updated
        return readout(graph, task, scores, factors, meter)
    except Refusal as exc:
        exc.used = meter.used
        raise
