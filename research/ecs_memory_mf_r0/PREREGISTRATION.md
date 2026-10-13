# ECS Memory MF-R0 laboratory: preregistration

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `TRAINING=NONE` · `SEMANTICS=NONE` · `NO_LANGUAGE_CLAIM`

Registered before the laboratory exists and before any run. Hypotheses, thresholds, arms, budgets, seeds and
scoring below are fixed here and are not changed after any result is seen. During DEV only defects of the
harness (it does not do what this document says) may be repaired; every repair is recorded in the DEV
evidence notes. After DEV the laboratory and core sources are frozen by digest; QUAL runs once on held-out
seeds. A failed hypothesis is preserved as NONPASS; a repair is a new experiment version, never a re-run.

## Question

Under explicit, byte-matched resource budgets, what does the MF-R0 memory's recurrent state (FAST: gated
delta law; SLOW: replay consolidation law C1) actually retain and answer, compared with ordinary lookup,
byte-matched keyed storage, the PR #47 lattice and negative controls? Specifically: acquisition from
observations, updating, partial-cue completion, retention under interference, and consolidation of rare but
recurrent relations.

This is a mechanism study on synthetic relational worlds with random identities. Nothing here measures
meaning, language, semantic similarity or learned generalization across entities; codes of distinct
identities are quasi-orthogonal by construction, so any completion from a partial cue is a property of the
fixed encoding geometry (superposition), classified as **algebraic completion**, not learning to generalize.

## Worlds (synthetic, deterministic from a seed)

Entity identities: `SHA-256("elpis.ecs-mem-lab.entity" | seed | k)`. Relation types and roles are small
integers. Each scenario draws its own entities; every query offers the arm the same candidate set (the
scenario's target pool, at least 64 entities); chance = 1 / |candidates|.

| Scenario | Stream | Queries |
|---|---|---|
| S1 ACQUIRE | 64 binary relations `R(s_i -> o_i)`, each observed and learned once | `o_i` from `s_i` (full cue) |
| S2 CHANGE | 32 subjects; the object of each follows AB, ABA or ABAB (three conditions), changes interleaved across subjects | the latest object of every subject |
| S3 SHARED | 48 ternary relations over 24 entities so that every entity fills several roles across relations | every role of every relation from the other two |
| S4 CONFLICT | 32 subjects each asserted with two objects by two sources at the same tick | object from subject; conflict reporting |
| S5 PARTIAL | 48 ternary `give(agent, recipient, object)` learned with the full cue; condition UNIQUE (each agent in one relation) and SHARED (each agent in two relations) | object from agent only, from recipient only, from both |
| S6 INTERFERENCE | 32 target relations, then K one-off interfering relations, K ∈ {0, 256, 1024, 4096} | target objects |
| S7 RECURRENT | 4000 steps of one-off relations; 16 targets observed at the start and, in condition RECUR, re-observed every 250 steps (last at step 3750); consolidation every 25 steps (16 items); condition NORECUR has no re-observation | target objects at step 4000 |
| S8 COMPOSITION | 32 chains `R1(a -> b)`, `R2(b -> c)`; the composite `a -> c` is never observed | `c` from `a` by chaining two queries |

## Arms

All arms receive the same typed observations (relation type, roles, entity identities) and the same typed
queries. Information access is therefore **not** architecture-neutral: every arm is handed explicit typed
store/query boundaries that a learning system would otherwise have to infer. KEYED additionally receives
exact keys; it cannot answer partial cues by construction (disclosed, not a finding about it).

| Arm | What it is |
|---|---|
| FAST | MF-R0, D = 128, B = 4, DELTA law, learn rate 1, retain 1 (S7: retain 64880/65536), every observation OBSERVEd then LEARNed; FAST channel |
| SLOW | the same handle, SLOW channel (consolidated only in S7; elsewhere expected empty) |
| HEBB | MF-R0 as FAST with the HEBBIAN law (recurrence/residual ablation) |
| LOOKUP | MF-R0 resident evidence only (LOOKUP channel), slot capacity byte-matched |
| KEYED | byte-matched exact keyed store: key = (type, target role, ordinal, canonical cue), value = target entity, last write wins, LRU eviction, 80 bytes per entry including index |
| D31C | the PR #47 lattice through its adapter (`ecs_mf_*`): context = relation type, entity = hash of the canonical cue, role = target role, value = 64-float code of the target; cells byte-matched |
| FIXED | MF-R0 FAST channel over fixed random banks (never learns) |
| NONE | MF-R0 FAST channel over empty banks (no memory) |
| PERM | the FAST handle queried with role identifiers permuted by a fixed derangement (address permutation) |

Byte budget: the total ELPISEM1 state bytes of the FAST handle (both bank sets, its 64 evidence slots and
64-entry codebook). KEYED, LOOKUP and D31C receive that same budget for their state. Bytes, write work,
read work and query latency are reported for every arm. Latency is descriptive and never scored.

## Hypotheses (thresholds fixed here)

Accuracy is pooled over all seeds of a phase: correct answers / queries.

* **H1 acquisition and state causality (S1).** FAST ≥ 0.95; NONE, FIXED and PERM each ≤ chance + 0.05;
  FAST − max(NONE, FIXED, PERM) ≥ 0.80. Falsified if any fails.
* **H2 updating (S2).** FAST reports the latest object with accuracy ≥ 0.90 in each of AB, ABA and ABAB;
  and HEBB ≤ FAST − 0.20 on ABAB. Falsified if any fails.
* **H3 partial-cue completion (S5, UNIQUE).** FAST answers the object from the agent alone with accuracy
  ≥ 0.80. Classified as algebraic completion by fixed geometry if it holds. Falsified otherwise.
* **H4 byte-matched retention (S6).** FAST accuracy on targets ≥ KEYED accuracy at every K. Falsified if
  FAST is below KEYED at any K.
* **H5 consolidation of rare recurrent relations (S7).** SLOW(RECUR) ≥ FAST(RECUR) + 0.25 and
  SLOW(RECUR) ≥ SLOW(NORECUR) + 0.25. Falsified if either fails.

S3, S4 and S8 are descriptive (no threshold). Every arm's accuracy is reported in every scenario whatever the
verdicts. A PASS of H1–H5 is a mechanism finding in this synthetic regime only.

## Phases and seeds

* DEV: seeds 1, 2, 3, 4. Purpose: show that the harness implements this document. Evidence recorded,
  verdicts computed and reported, nothing tuned.
* Freeze: after DEV, a manifest binds the SHA-256 of this document, the laboratory, the runner, the scorer
  and the MF-R0 / lattice sources.
* QUAL: seeds 1001–1008, run once from the frozen sources; the evidence records source digests, binary
  digest, compiler, build type, host and the frozen manifest digest.

## Scoring

`research/ecs_memory_mf_r0/score.py` applies exactly the thresholds above to the pooled counts. Verdicts are
PASS or NONPASS per hypothesis; a scenario an arm cannot run is NOT_RUN for that arm and never PASS.
