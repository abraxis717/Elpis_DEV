"""Deterministic truncation selection verifies everything it is given."""
from __future__ import annotations

from dataclasses import replace

import pytest

from elpis.evolution import (
    FitnessObservation,
    FitnessPolicyV1,
    Genotype,
    IntegerGene,
    LifecycleState,
    LineageIdentity,
    OrganismFitnessRecord,
    OrganismState,
    PopulationState,
    SelectionRejectionCode,
    TruncationSelectionPolicyV1,
    commit_selection,
)

WORLD = "a" * 64
FITNESS = FitnessPolicyV1()


def organism(*, value: int, ordinal: int, lifecycle: LifecycleState = LifecycleState.ALIVE) -> OrganismState:
    genome = Genotype(genes=(IntegerGene("trait", value=value, minimum=0, maximum=4),))
    lineage = LineageIdentity.founder(birth_tick=0, birth_ordinal=ordinal, genotype_digest=genome.digest())
    return OrganismState(lineage=lineage, genotype=genome, energy=100,
                         age_ticks=0 if lifecycle is LifecycleState.EMBRYO else 10, lifecycle=lifecycle)


def population() -> PopulationState:
    return PopulationState(
        source_world_state_digest=WORLD,
        revision=7,
        organisms=(
            organism(value=0, ordinal=0),
            organism(value=1, ordinal=1),
            organism(value=4, ordinal=2, lifecycle=LifecycleState.REPRODUCTIVE),
            organism(value=2, ordinal=3, lifecycle=LifecycleState.EMBRYO),
        ),
    )


def observation(energy_delta: int) -> FitnessObservation:
    return FitnessObservation(window_start_tick=0, window_end_tick=10, energy_delta=energy_delta,
                              survival_ticks=10, viable_offspring=0, resource_efficiency_ppm=0,
                              ecological_damage=0)


def eligible(state: PopulationState):
    return [o for o in state.organisms if o.lifecycle in (LifecycleState.ALIVE, LifecycleState.REPRODUCTIVE)]


def records(state: PopulationState, deltas=None):
    out = []
    for index, o in enumerate(eligible(state)):
        delta = (deltas or {}).get(o.organism_id, 10 * index)
        out.append(OrganismFitnessRecord.evaluate(organism=o, observation=observation(delta), policy=FITNESS))
    return out


def commit(state=None, recs=None, survivors=2, expected=None, fitness=FITNESS):
    state = state or population()
    return commit_selection(
        population=state,
        expected_population_digest=expected or state.digest(),
        fitness_records=recs if recs is not None else records(state),
        fitness_policy=fitness,
        policy=TruncationSelectionPolicyV1(survivor_count=survivors),
    )


def test_population_order_is_canonical():
    state = population()
    reordered = PopulationState(source_world_state_digest=WORLD, revision=7,
                                organisms=tuple(reversed(state.organisms)))
    assert reordered.organism_ids == tuple(sorted(state.organism_ids))
    assert reordered.digest() == state.digest()


def test_population_rejects_duplicate_identity():
    o = organism(value=0, ordinal=0)
    with pytest.raises(ValueError):
        PopulationState(source_world_state_digest=WORLD, revision=0, organisms=(o, o))


def test_fittest_survive_and_the_rest_are_dying():
    state = population()
    result = commit(state)
    assert result.accepted and result.rejection_code is None
    ranked = [organism_id for organism_id, _ in result.ranking]
    assert [s for _, s in result.ranking] == sorted((s for _, s in result.ranking), reverse=True)
    assert set(result.selected_ids) == set(ranked[:2])
    assert result.transitioned_to_dying == tuple(sorted(ranked[2:]))
    after = result.population_after
    for organism_id in ranked[2:]:
        assert after.organism(organism_id).lifecycle is LifecycleState.DYING
    for organism_id in ranked[:2]:
        assert after.organism(organism_id) == state.organism(organism_id)


def test_noncandidates_are_unchanged_and_revision_advances_once():
    state = population()
    result = commit(state)
    embryo = next(o for o in state.organisms if o.lifecycle is LifecycleState.EMBRYO)
    assert result.population_after.organism(embryo.organism_id) == embryo
    assert result.population_after.revision == state.revision + 1


def test_ties_break_by_organism_id_and_result_is_deterministic():
    state = population()
    flat = {o.organism_id: 5 for o in eligible(state)}
    first = commit(state, records(state, flat), survivors=1)
    assert first.selected_ids == (min(flat),)
    assert commit(state, records(state, flat), survivors=1).digest() == first.digest()
    assert commit(state, list(reversed(records(state, flat))), survivors=1).digest() == first.digest()


def test_survivor_count_at_or_above_candidates_kills_nobody():
    result = commit(survivors=10)
    assert result.accepted and result.transitioned_to_dying == ()


@pytest.mark.parametrize("mutate,code", [
    (lambda s, r: dict(expected="0" * 64), SelectionRejectionCode.STALE_POPULATION),
    (lambda s, r: dict(recs=r[:-1]), SelectionRejectionCode.MISSING_FITNESS_RECORD),
    (lambda s, r: dict(recs=r + r[:1]), SelectionRejectionCode.DUPLICATE_FITNESS_RECORD),
    (lambda s, r: dict(recs=[replace(r[0], scalar_fitness=r[0].scalar_fitness + 1)] + r[1:]),
     SelectionRejectionCode.FITNESS_SCORE_MISMATCH),
    (lambda s, r: dict(fitness=FitnessPolicyV1(survival_tick_weight=2)),
     SelectionRejectionCode.FITNESS_POLICY_MISMATCH),
])
def test_defects_reject_atomically(mutate, code):
    state = population()
    result = commit(state, **mutate(state, records(state)))
    assert not result.accepted and result.rejection_code is code
    assert result.population_after == result.population_before == state
    assert result.transitioned_to_dying == () and result.selected_ids == ()


def test_record_for_a_non_candidate_is_rejected():
    state = population()
    embryo = next(o for o in state.organisms if o.lifecycle is LifecycleState.EMBRYO)
    extra = OrganismFitnessRecord.evaluate(organism=embryo, observation=observation(1), policy=FITNESS)
    result = commit(state, records(state) + [extra])
    assert result.rejection_code is SelectionRejectionCode.UNEXPECTED_FITNESS_RECORD


def test_record_of_an_older_revision_is_stale():
    state = population()
    target = eligible(state)[0]
    older = replace(target, energy=target.energy + 1)
    stale = OrganismFitnessRecord.evaluate(organism=older, observation=observation(1), policy=FITNESS)
    recs = [stale if r.organism_id == target.organism_id else r for r in records(state)]
    result = commit(state, recs)
    assert result.rejection_code is SelectionRejectionCode.STALE_FITNESS_RECORD


def test_population_without_eligible_candidates_is_rejected():
    state = PopulationState(source_world_state_digest=WORLD, revision=0,
                            organisms=(organism(value=2, ordinal=3, lifecycle=LifecycleState.EMBRYO),))
    result = commit(state, [])
    assert result.rejection_code is SelectionRejectionCode.NO_ELIGIBLE_CANDIDATE


def test_policy_requires_a_positive_survivor_count():
    for bad in (0, -1, True, 1.0):
        with pytest.raises((TypeError, ValueError)):
            TruncationSelectionPolicyV1(survivor_count=bad)
