# DRAFT_NOT_FROZEN — prospective comparative protocol

Written after R1A mechanics and before B2 implementation or outcomes. This is a
proposal, not a scientific preregistration or authorization to execute QUAL.
The user's R1 instruction authorizes a mechanical freeze and B2 DEV only, overriding
the supplied R0 task's blanket B2 prohibition. Supplied contracts are unchanged.

Proposed DEV generator seeds: 1729, 1733, 1741. Proposed withheld QUAL seeds:
880301, 880307, 880313. They are disjoint suggestions, not frozen science authority.
Only DEV 1729 is used by comparative demonstrations here; other synthetic seeds in
unit tests remain mechanical test inputs. No QUAL generator is invoked.

Proposed withheld-target construction: harness generates source-labelled scoped
hypergraphs first; selects a task family and public kind/role/type predicates;
exposes input premise keys, scope, predicates and resource budgets only. Harness
keeps answer key sets in a separate scorer argument. Include multiple same-kind
reachable targets, unreachable same-kind distractors, cycles, empty answers and
renamed/isomorphic graphs, so kind alone cannot identify answers. Target construction
must not consult B2 outputs. Target labels/answer arrays never enter Task. All
methods receive the same entire bounded immutable graph, authority, observations,
query and available context capacity; this is bounded retrieval/activation, not
hidden graph completion. Inspector outputs and caller-keyed B0 lookup are explicitly
excluded from scored predictions. An all-keys B0 inspector may be reported as an
unscored visibility upper bound. B0's task solver performs exact closure itself.

Proposed metrics: exact synthetic key retrieval TP/FP/FN, omissions, constraint
retention, scope confusion, invalid-source emission, epistemic/provenance changes,
and repair distance to exact recomputation. Always separate VALID/STALE/UNKNOWN and
AUTHORITY_ONLY source state from PROVISIONAL/BINDING and CANDIDATE/ADMITTED labels.
Score salience as synthetic retrieval only; never score it as semantic entailment.
Native source attestation and n-ary admission are absent. A candidate-dependent
ADMITTED-labelled downstream factor is still no proof of any prediction.

Work/capacity proposal: freeze the task envelope, source checks, limits, logical
operation categories and scorer before any future QUAL; graph capacity and readout
budgets identical for B0/B1/B2. Include initialization, schedule generation, state
copying, update, evidence readout and mandatory binding visibility; report shared
build cost separately, plus cold and warm totals. Match a grid of operation caps
and show actual work, refusals/omissions and wall time, rather than matching only
rounds. Logical ops are not instructions/FLOPs and are not calibrated compute.
Wall time includes Python overhead, measured outside the solvers by perf_counter_ns.
Future timing proposal: isolated sequential runs, 5 untimed warmups, 30 measurements,
median and interquartile range; report interpreter/host versions and batch ordering.
No such statistical timing exercise is performed in this DEV mechanics phase.

Proposed controls: exact B0 recompute, full-incidence B1, empty/all/one-missing
premises at each arity, contradictory scopes, type/role/status filters, invalid
output sources, delayed/missing observations, mandatory bindings, saturation,
cycles and distractors. Damage only activation state (deletion and spurious positive
values); compare local recurrence with fresh B0/B1 and B2 reinitialization, counting
all repair work. Unsupported cycles may retain spurious activation until reset;
that is a failure control, not evidence to change the frozen baseline. Refused
updates preserve the prior snapshot. Hash-randomization replay is a mechanics gate.

Prospective threshold suggestions (DRAFT, not chosen from B2 results): zero source,
scope, provenance, binding or admission violations; exact deterministic replay;
no budget overshoot. A possible future utility threshold is at least 5 percentage
points higher constraint-correct retrieval at a common work cap, with a paired 95%
interval excluding zero, or at least 20% fewer operations at equal exact recall,
and no wall-time regression beyond a predeclared tolerance. Sample sizes, tolerance,
endpoint hierarchy, task mix and statistical tests remain unspecified and require
separate authorization before science. Neither these numbers nor current mechanics
results establish scientific PASS/NONPASS. If B0/B1 reconstruct more efficiently,
B2 repair has no demonstrated utility advantage. No post-hoc baseline edits after
B2: a baseline defect requires STOP and a newly authorized mechanical freeze.
