# HACF-ECS Scientific Bridge R0: qualification specification

Status: **NOT RUN** · Subject: **UNQUALIFIED** · Classification: `RESEARCH_ONLY`

## Question

Can an identified observation map turn validated HACF structural evidence into ECS observations whose read-only
QUERY answers depend on the evidence in the way the map's hypothesis predicts, and abstain or degrade on withheld,
permuted, stale and misleading evidence, without any LEARN, writeback or laundering of evidence into authority?

## Subject

`research/hacf_ecs_bridge_r0` (bridge and harness; README there). The only map present is `StructuralFixtureMap`
(`TEST_ONLY TRAINING=NONE SEMANTICS=NONE`); it is **not** a candidate for this qualification. A qualifying map must
be proposed, pinned (`ObservationMapPin`) and frozen before QUAL.

## Blockers (why it cannot run now)

* No real structural evidence is frozen: a qualification needs a pinned corpus, a pinned context graph snapshot and
  pinned queries with recorded bundles, all write-once.
* No candidate observation map with a stated hypothesis exists.
* No K1 state with a meaningful W for this domain exists; Cognition R0's synthetic regime does not transfer.

## Inputs (once unblocked)

* DEV and QUAL query sets, disjoint; recorded bundles for each, write-once, digest-bound.
* The pinned map (measured identity) and its hypothesis, written before DEV.
* One K1 state per world, pinned by envelope digest; nothing learns during the run.
* Bounded scale: at most 64 queries per partition, one shape.

## Procedure

For every query, under each evidence condition (VALID, WITHHELD, PERMUTED, PERMUTED_RERANKED, STALE, MISLEADING,
MISLEADING_WELL_FORMED, as `harness.py` constructs them from the recorded bundle), record the bridge outcome, the
observation and answer digests, and the state-unchanged proof.

## Gates (pre-registered)

* G1 (mechanics): every non-VALID refused condition refused with its code; every run's state unchanged; zero LEARN,
  writeback or continuity calls (structural and dynamic checks).
* G2 (sensitivity): the hypothesis-specific gate, written with the map before DEV (for example: answers under VALID
  differ from MISLEADING_WELL_FORMED beyond a calibrated envelope on at least a stated fraction of QUAL queries).
* G3 (no laundering): no observation or answer appears in any canonical store or as a codec admission.

## Would establish / would not establish

Passing would establish, for one pinned map and evidence set, that the bridge is a valid instrument and that the
map's stated hypothesis held on QUAL. It would **not** qualify an ECS<->DSV codec, make HACF evidence authoritative,
make any answer true, or make the bridge part of the canonical runtime.

## Evidence

`research/hacf_ecs_bridge_r0/evidence/` (write-once; not present: the run has not happened).
