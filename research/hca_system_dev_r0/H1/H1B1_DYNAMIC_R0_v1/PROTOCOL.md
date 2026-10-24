# H1B1 DYNAMIC SOURCE-CONDITIONED CONTRACT v1 — DEV DESIGN, UNFROZEN

## Objective

Instrument T1 missing-target support-conditioned relational prediction under dynamic T2-style source-state updates, with paired causal counterfactuals. H1B1 is a **one-shot scene snapshot** prototype, not yet an online retained-state T2 stress test. No HCA-specific architecture or scientific claim is tested here.

## Visible grammar

* Four relation tokens: A, B, C, D. Each n-ary typed hyperedge is `factor(fid, rel, roles{from,to,scope}, source, source_version)`. A factor asserts candidate relational evidence, **not** an admitted target.
* `query{from,to,scope}` asks whether an admissible length-two composition exists.
* Source events `SOURCE(tick, ordinal, source, status VALID|STALE|UNKNOWN, epoch)` advance a single source's current status/version. Events are ordered canonically by (tick,ordinal); last event for a source wins. An edge is eligible iff source currently VALID and its `source_version` matches current epoch.
* An edge only participates when its scope matches query scope. Two active edges `(from→middle, middle→to)` suffice only if their relation pair matches the **world-specific** law, one of `(A,C),(A,D),(B,C),(B,D)`.
* Rule hidden from probes; visible within-world **two labelled support queries** identify it uniquely (support 0 contains relation candidates 0,1; support 1 contains candidates 0,2). The independent scorer enumerates eligible edges and joins them, importing no generator function.
* Candidate target edges and target labels are **never inserted** into public probes. TRAIN and TUNE gold maps are stored separately from public files. The TUNE labels are *consumed exploratory DEV*, explicitly unsuitable as sealed heldout.

## Causal counterfactuals (4 pairs per type per world)

1. `source_swap`: identical n-ary factors and query; one causal source switches VALID↔STALE against a decoy source, preserving final number of valid sources; label must flip.
2. `event_order`: identical multiset of source/status events and graph, but their timestamp assignment reverses final causal state; label must flip.
3. `epoch_rebind`: source epochs advance; exactly one replacement factor is bound at the current epoch—either causal or decoy. Both variants have same source state and factor count.
4. `structural`: role-preserving, type-preserving marginal rewiring of n-ary edges breaks/makes a compositional route while retaining the same source events. Both variants have equal role-occurrence marginals by relation.

In every pair: two cases, one true and one false, same query, same number of factors/events. Presentation polarity randomized per pair. World IDs and case IDs are independently sampled opaque identifiers. Names, order and IDs are not evidence of ground-truth class. Source-state transitions and relation topology are admissible causal evidence.

## Guardrails / nonclaims

* Rule identifiability is established from *visible* support; a solver without support must not be used as a fair competitor. Baseline `support-conditioned exact join` has **legitimate** access to within-world support and event log, and should be expected to saturate both TRAIN and TUNE accuracy. This does NOT mean a learned HCA advantage is demonstrated.
* All score comparisons must report exact rows, per-type paired both/one/neither, invalid-source false positives, unsupported-cycle errors (future extension), and *total* costs including per-world support/rule induction, graph construction, updates, memory and inference.
* Current v1 cases are tiny; no cost advantage can be claimed from them. Incremental retained-state benchmark, B0/B0I exact controls, native B1/B2 fidelity, trained typed factor-node matching, and full-cost measurement are **H1B2+**, not secretly performed here.
* The hidden rule's generator-side selection is NOT a permitted oracle for any predictor. The symbolic baseline infers law only from visible support. A ceiling solver with hidden generator access would be a distinct, unfair upper bound.
* TUNE gold has been opened; sealed DEV seeds/worlds have not been generated. No model selection, freeze or release authority is implied by successful contract tests.
* H1A source bytes were not given to the assistant in this turn. The runner hashes actual host-local H1A source and creates an optional source-review capsule; no claim of completed human semantic review until it is inspected.

## Independent audit / deliberate mutations

`checker.py` implements scene schema, source versions, unique identifiers, chronological events, opposite-label pair checks, per-pair nuisance checks, cross-split identities, and gold parity with independently implemented `oracle.py`. `tests.py` rejects 12 mutants: gold flip, public label leak, duplicate case ID, dropped pair row, query drift, event index collision, duplicate factor ID, missing gold, duplicate world ID, support label flip, SEALED split forgery, and invalid source status. Tests require an actual `ContractError` and never catch their own `assert` failure. The mutation suite tests instrument adequacy only, *not model learning or HCA utility*.

## H1B2 entry criteria

Record H1B1 evidence SHA and actual source review. Challenge symbolic exact recomputation with exact incremental B0I-style temporal update, including dynamic status invalidation and epoch rebinding; independently measure cost with all setup and training amortization declared, and compare a real n-ary typed factor-node learner at matched information and budget. Either produce a prospective cost/accuracy claim worth testing or record `HOLD_TASK_SOLVED_NO_COST_CASE` before any B3 implementation or sealed DEV adjudication. B3 learning and formal QUAL remain prohibited absent separate freeze.
