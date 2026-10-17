"""Withheld answer authority lives ONLY here/the harness, never in a solver Task."""
from dataclasses import dataclass
from .baselines import digest, require
from .task import PredictionResult


@dataclass(frozen=True)
class Score:
    true_positive: int
    false_positive: int
    false_negative: int
    invalid_predictions: int
    source_states: tuple
    axiom_statuses: tuple
    relation_statuses: tuple
    omitted: int
    metric: str = "SYNTHETIC_KEY_RETRIEVAL_NOT_SEMANTIC_TRUTH"


def score(result, withheld_keys):
    require(type(result) is PredictionResult and type(withheld_keys) is frozenset
            and len(withheld_keys) <= 256 and all(digest(k) for k in withheld_keys), "INVALID_EVALUATION")
    valid, invalid = set(), 0
    sources, axioms, relations = {}, {}, {}
    for prediction in result.predictions:
        hit = prediction.hit
        sources[hit.source_state] = sources.get(hit.source_state, 0) + 1
        status = hit.grant.axiom.status
        axioms[status] = axioms.get(status, 0) + 1
        for relation in prediction.relations:
            status = relation.relation.status
            relations[status] = relations.get(status, 0) + 1
        eligible = hit.source_valid and hit.source_state in ("VALID", "AUTHORITY_ONLY")
        if eligible:
            valid.add(hit.grant.axiom.key)
        else:
            invalid += 1
    return Score(len(valid & withheld_keys), len(valid - withheld_keys), len(withheld_keys - valid),
                 invalid, tuple(sorted(sources.items())), tuple(sorted(axioms.items())),
                 tuple(sorted(relations.items())), result.omitted)
