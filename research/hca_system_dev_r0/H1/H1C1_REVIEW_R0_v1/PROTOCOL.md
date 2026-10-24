# H1C1 observer protocol — frozen DEV audit, no promotion

Schema: `elpis.hca.h1c1.independent_observer.dev.v1`.

**Scope.** Independently verify the closed H1C0 event-prefix corpus, exact predictions, pair semantics and accounting. Treat H1A/H1B1/H1B2A/H1B2B/H1C0 as immutable predecessors, with the last result and source SHA pinned. No post-hoc corrections or sealed-set generation. A new evidence directory is created only after all mandatory checks pass.

**Auditor independence.** H1C1 implements an exhaustive triplet-existence oracle and separate input/pair validator. It does not import the H1C0 reference oracle, contract checker or generator for host scoring. The optional synthetic self-test uses those solely to supply a fixture, and is expressly NOT science. The separate engine benchmark does import only the byte-verified historical H1C0 engines and then checks the reported category counters against the closed recorded census.

**Coverage and adversaries.** Every TRAIN/TUNE case, prefix, query, independently recomputed label, saved exact prediction and reported counter is checked. 18 deliberate mutations must be rejected (no catch-all swallowing). Outcome polarity must invert on each pair, with declared nuisance symmetry. Source valid/stale/unknown epoch matching is enforced in an independently organized exhaustive triplet oracle.

**Resource measurement caveats.** One subprocess for each split × exact engine; 3 whole-dataset wall-time replicates inside each subprocess, measuring construction + all queries and updates but excluding JSON loading, evidence I/O, oracle and audit. A separate allocation-tracing pass reports traced peak Python allocations, and process peak resident set is recorded. Python environment/interpreter and dataset allocations may dominate RSS; traced peaks exclude input parsing and do not measure all native allocations. H1C0's replay serialized footprint excludes static input, while incremental counts stored index and graph, so their reported footprint ratios are NOT direct memory-efficiency ratios. Work counters have different semantics and cannot be compared as FLOPs.

**Success disposition**: `PASS_H1C1_INDEPENDENT_OBSERVER_DEV_ONLY`, retain `HOLD_B3_NO_HCA_EFFECT_MEASURED_EXACT_SOLUTION`. The comparator gap to a learned HCA is unknown because HCA has not run. Exact solvability plus an efficient exact incremental method provide no justification for speculative B3 advantage. Next work may be a prospectively versioned uncertain/recovery task with identifiability, full-cost, strong exact, symbolic and typed factor-node alternatives, or close this research track. No QUAL, production, Git or native ABI changes.

**Failure disposition**: `MECHANICS_HOLD`, preserve all originals, create a separately versioned mechanical repair rather than declaring a scientific NONPASS.
