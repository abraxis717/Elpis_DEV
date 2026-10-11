# HACF -> ECS Scientific Bridge R0: infrastructure

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `UNQUALIFIED` · `NO_CLAIM`

The missing piece between Elpis's structural memory (HACF) and its computation (ECS) is a scientifically identified
way to turn retrieved structure into something an ECS state can be asked about. This package is the instrument for
studying that, not an answer: it makes no claim that any observation map carries meaning.

```
RetrievalBundle --(canonical bundle gate + frozen ObservationContext)--> StructuralObservationPacket
StructuralObservationPacket --(identified ObservationMap: measured == ObservationMapPin)--> ECSObservation
ECSObservation --(one read-only K1 query_identity; state proven unchanged)--> ObservationAnswer
```

| Law | Code |
|---|---|
| Only evidence the canonical bundle gate accepts against the protocol's frozen query and corpus | `EVIDENCE_REFUSED` |
| The packet binds the protocol's graph snapshot and both epochs | `STALE_EVIDENCE` |
| The bundle's self-declared digest is a claim; the packet digest is recomputed from structure | (property) |
| Structure only: no chunk text crosses into ECS | (property) |
| No evidence, no observation (absence is never a default observation) | `NO_EVIDENCE` |
| A map is identified by measurement (source SHA-256, parameter SHA-256, shape), never by its own claims | `MAP_UNIDENTIFIED` |
| Rows are finite and of the pinned shape | `OBSERVATION_SHAPE` |
| QUERY only: no LEARN, no continuity, no runtime authority, no HACF writeback (asserted structurally) | (property) |
| An observation is not a codec; the managed runtime refuses it | (property) |

`harness.py` runs one tiny synthetic bundle under the evidence conditions VALID, WITHHELD, PERMUTED,
PERMUTED_RERANKED, STALE, MISLEADING and MISLEADING_WELL_FORMED and records refusals, observations, answers and the
state-unchanged proof. It has no thresholds. MISLEADING_WELL_FORMED is observed: structural validation cannot detect
semantically misleading evidence, and the harness records that rather than hiding it.

Fixtures: `StructuralFixtureMap` is `TEST_ONLY TRAINING=NONE SEMANTICS=NONE`; the bundles are synthetic. Tests:
`tests/research/hacf_ecs_bridge_r0` (scientific lane). No real evidence has been frozen and no qualification has
been run; the specification is `docs/qualification/HACF_ECS_BRIDGE_R0.md` (NOT RUN).
