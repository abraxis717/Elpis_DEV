# ECS_C HCA: the native research substrate (gated)

`native/ECS_C/hca` (`libelpis_ecs_c_hca`, C ABI v1 `include/elpis/ecs_c_hca.h`) is the native port of the DEV
typed-hypergraph cellular memory reference `research/ecsc_hca_r0` (`baselines.py`, `task.py`, `b2.py`). It exists so
that the reference's three comparators can be studied natively, under one resource accounting, beside the exact
control. It has no other purpose.

**Status: research only, gated.**
- Nothing canonical links or loads it: not RuntimeCore, the composite, the memory, the deployment's library set or
  `src/` (`tests/boundary/test_ecs_c_hca.py`).
- It implements no admitted law and qualifies nothing. Every prediction is `SALIENCE_ONLY`: neither truth nor proof,
  even at salience 1000.
- The reference's DEV record stands: no useful repair advantage is demonstrated (`DEV/ecsc_hca/README.md`).
- Nothing here is a scientific result.

## The comparators

| Method | Rule |
|---|---|
| B0 | Exact worklist closure of weighted conjunctions: the least key is popped first, and an output is pushed when its score rises. It is the exact-storage control, not limited to B1's rounds. |
| B1 | `steps` synchronous rounds. Every admitted factor's message is computed from the previous round's scores, and each output keeps its maximum. |
| B2 | An unlearned asynchronous recurrence over explicit WARM ticks. Each tick orders the scoped cells by SHA-256(`ecsc-hca-r1.B2.schedule\0` ‖ seed (8 bytes, big-endian) ‖ step (8 bytes, big-endian) ‖ key), ties by key. It fires at most `updates_per_tick` of them. A fired cell becomes the maximum of its seed clamp and its incoming messages, read from the current (in-place) values; an invalid source makes it 0. There is no fairness or convergence guarantee. |

A factor's message is `weight * min(input scores) / 1000` (integers). It is 0 unless every member's source admits
(VALID or AUTHORITY_ONLY; UNKNOWN and STALE never do), and only the output role (the last) receives it.

A task names:
- a scope;
- seed keys (each must exist, be in scope and have a valid source);
- predicates: a factor is admitted only by a predicate naming its exact schema (relation type, roles, kinds) with
  its status in the mask;
- output kinds, a readout budget, a work budget, the steps, the schedule seed and the updates per tick.

The readout lists every BINDING constraint of the scope, then the ranked prefix (descending score, then key) that
the budget admits. Each prediction costs 1 plus its direct incoming evidence, and the rest is counted as omitted.
There is no answer key.

## Resources and boundedness

- **Graph.** Immutable once built: at most 256 cells, 512 factors, arity 16 and degree 32. It is one heap block,
  sized at create.
- **Sessions.** A B2 session is one block. `elpis_hca_predict` runs B2 on a stack session, allocating nothing.
- **Activations.** A caller-owned fixed-size value (560 bytes): the scoped cells' values, the tick and the owner. The
  owner is SHA-256 over the graph's evidence identity and the exact task, which blocks cross-session transplant.
- **Persistence and history.** Nothing is persisted and no history is kept.
- **Work.** Every call charges logical work (records, visits, copies, hashed bytes; not instructions) against its
  budget, and refuses `WORK_LIMIT` atomically. A refused call writes nothing.
- **B2 reservation.** B2 refuses `STATE_LIMIT` when the reference's reservation exceeds the graph's state bytes:
  8192 + 1024 × scoped cells + 256 × scoped factors + 256 × predicates + 256 × degree.

The allocation test (`ECS_C.test_ecs_c_hca_alloc`) proves one block per graph and per session, and zero heap
operations over 2,000 rounds of every call.

## What is proven

**The differential** (`tests/research/ecsc_hca_native`, SCIENTIFIC lane). The native substrate equals the reference
on every prediction, salience, evidence set, constraint, omitted count and refusal:
- the reference's coding fixture at arity 4..8, with every omitted premise, both scopes, foreign and absent seeds,
  readout limits, predicate statuses and mismatched roles;
- the reference's chain fixture;
- source invalidation (stale, missing and absent checks);
- delayed distractor streams of 8, 32 and 128 turns with dropout;
- 60 seeded random graphs × 16 tasks × B0/B1/B2.

It also equals the reference on every B2 activation, byte for byte, through 6 ticks, a random damage, 3 more ticks
and a reset, on the fixture and 40 random graphs × 4 tasks.

The comparisons run under budgets neither side reaches, because work accounting differs by design. A guard asserts
the random corpus is not vacuous: messages propagate to unseeded cells, carry direct evidence and sub-maximal
salience, and every shared refusal occurs.

**The native mechanics** (`ECS_C.test_ecs_c_hca`):

| Area | What is checked |
|---|---|
| n-ary relations | arity 4..16: all premises give the output; any omitted premise gives nothing |
| Rounds | a chain deeper than B1's rounds stays partial; B0 closes it |
| Interference and conflicting scope | two identical contradictory scopes never leak; a seed or a factor across scopes is refused |
| Source invalidation | stale and unknown sources disable their cells and every factor over them; AUTHORITY_ONLY admits; an invalid output never reads out |
| Delayed memory | 512 turns of a replaced fixed-width distractor, with dropout: the same answer, readout and work, and nothing grows |
| Activation damage | local repair by ticks; a seed restored by its clamp; reset restores the initial activation exactly |
| The unsupported-cycle limitation | a 2-cycle with injected positives persists under B2's ticks, while B0 and a fresh B2 return nothing. It is a known limitation, reported and not repaired |
| Provenance misbinding | an activation of another task or graph, a tampered owner, out-of-range or trailing values, or a wrong cell count is refused `INVALID_STATE`; damage naming a foreign, repeated or out-of-range cell is refused |
| Matched resources | B0, B1 and B2 under one budget give the same answer; each call's reported work suffices, and one unit less refuses atomically |
| Refusals and determinism | every refusal, and determinism across two graphs and two sessions |

The descriptive figure on the arity-6 fixture is B0 159, B1 231, B2 4,799 logical units; B2 pays for its schedule.

**Mutation sensitivity.** 20 targeted mutants of the laws were compiled outside the repository and judged by the C
suite and the differential:

| Mutant | Outcome |
|---|---|
| min input → max; the output read as an input; the weight ignored; invalid sources admitted or UNKNOWN admitting | killed |
| rank ties reversed; the readout budget off by one; evidence from an input role | killed |
| B1 asynchronous | killed, after the chain tests were added for it |
| the B2 domain altered; the step encoded little-endian; the seed clamp dropped; one cell too many fired | killed |
| damage advancing the tick; the owner check dropped | killed |
| predicate statuses or roles ignored; the B2 reservation or the degree limit unchecked | killed |
| an invalid B2 cell keeping its value | equivalent (survives both suites, explained below) |

The surviving mutant is equivalent: an invalid cell is never a seed, and every factor over it has a zero message, so
a fired invalid cell computes 0 either way.

## Not claimed

- **Science.** No QUAL, training, B3, learned or trained HCA, neural cellular automaton, stability, fairness or
  convergence claim.
- **Successor builds.** The reference's successor checks (predecessor view, binding persistence, policy continuity)
  are not ported: a native graph is always a fresh build.
- **Admission.** The reference's synthetic allowlists and source checks are applied before translation. The native
  graph takes already-authorized records and validates their structure.
- **Canonical status.** The substrate is not an admission authority, not memory and not canonical.
