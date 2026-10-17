# ECS_C HCA R0 — independent research contract (DEV-only, research-only)

Status: DESIGN / NOT IMPLEMENTED / NOT QUALIFIED. This document is an independent research design; it is not part of any Git checkout and does not confer permission to alter canonical ECS, the publication repo, or frozen scientific evidence.

Date: 2026-10-15. Audited development branch: `abraxis717/Elpis_DEV`, `claude/affectionate-euler-e46ng8`, commit `cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3` (subject to change while other work proceeds).

## Motivation and paper constraints

- Etcheverry et al., *Reasoning with Neural Cellular Automata*, arXiv:2609.36126v1 (2026-09-28): trained grid NCAs solve visual tasks with local shared recurrent updates, stochastic firing and sample-replay/perturbation training. Adaptive activity concentrates on uncertain/damaged cells. **Not proof of semantic memory or typed-hypergraph reasoning.** https://arxiv.org/html/2609.36126v1
- Heilig et al., *Port-Hamiltonian Neural Networks for Systems with Multiple Asymptotically Stable Equilibria*, arXiv:2610.01356v2 (2026-10-04): product-of-Bregman Hamiltonian supports multiple stable equilibria under stated regularity and dissipation assumptions. **Not a general guarantee for arbitrary discrete asynchronous HCA, nor a mechanism that preserves several simultaneous independent axioms.** https://arxiv.org/html/2610.01356v2
- Herrera-Marín, *Fractional Laplace Neural Operators*, arXiv:2610.00515v1 (2026-09-30): structured hereditary operators, stability under specified model classes, finite-rational versus fractional criticality distinctions. **Commuting/spectral assumptions and exact physical memory kernels do not automatically hold for HACF graphs.** https://arxiv.org/html/2610.00515v1

## The object to be studied

Existing production-facing context substrate (`src/elpis/runtime/context.py`, reviewed commit) has `ContextReference`, `ContextAxiom`, `ContextRelation`, bounded `ContextLane`, immutable digest-sealed `ContextView`, `PROVISIONAL`/`BINDING` axiom states and `CANDIDATE`/`ADMITTED` relation states. It publishes a session-local context coordinate and explicitly does not implement HCA dynamics or K1 input encoding. The next experiment is **not** a replacement or extension of its epistemic authority.

Keep two separate layers:

1. **Discrete, authoritative evidence layer:** immutable source coordinates, verified claims, relation type/role/provenance, externally admitted bindings. It defines what assertions are permitted, not what cells feel salient.
2. **Continuous, research-only dynamical layer:** each contextual axiom has a bounded vector state `z_i` (activation/latent state), an optional local update clock, and optional compressed timescale channels. Dynamics can rank or activate already admitted candidates but can never upgrade `CANDIDATE` to `ADMITTED`, fabricate a source, modify HACF files, or rewrite a `BINDING` axiom.

A typed hyperedge `e` has membership/roles and optionally a message state; neighborhoods use **incidence**, not geometric adjacency. Each transition may use only the local cell, incident edges, and their participants. Bound degree and arity explicitly; an unbounded hub destroys local-work guarantees. Locality is with respect to the hypergraph and must not be confused with O(1) total work if degree grows.

Generic candidate law (NOT frozen / NOT trained):

`m_e^k = Phi_theta(type_e, {(role_ei, z_i^k) : i in e})`

`z_i^(k+1) = Project_B[ z_i^k + fire_i^k * F_theta(z_i^k, {m_e^k : e incident on i}, verified_input_i) ]`

Use deterministic counter-based seeded schedules for reproducible asynchronous trials; never let a sampled schedule be treated as ground-truth evidence. Candidate perturbation experiments must not perturb source identities, provenance, or binding constraints. Preserve the previous immutable published context while staging dynamical updates; only publish a bounded stable successor view at an explicit WARM publication boundary. Do not hash or publish every cell tick. Existing HOT inference neither invokes nor waits for this research law.

## Critical adversarial distinctions

- A memory that *remembers* an axiom is different from a dynamical system that *converges* to an attractor. Stable attractors can coexist without simultaneous representation of several required facts. Test both retention and simultaneous constraint satisfaction.
- HCA does not gain an automatic advantage over pairwise GNNs: a typed **factor-node/incidence graph** can represent every n-ary hyperedge without throwing away its semantics. That is a mandatory comparator; a naive clique-projection-only baseline is insufficient.
- Stochastic firing, damage repair, sample replay and confidence-based sleep are **empirical candidates**, not universal improvements. Train and test ablations separately.
- A finite time horizon does not justify a fractional-memory operator. First compare exact bounded event access, exponential traces, multiscale rational traces and finite contextual recurrence. Only study fractional atoms after evidence of non-exponential long-tail effects and horizon-transfer failures.
- Port-Hamiltonian certificates are model-specific: any claim for a discrete HCA needs its **own** discrete contraction/Lyapunov argument or measured, explicitly non-certified stability. Never borrow the PHNN theorem by analogy.

## R0 baselines (all with identical context and source evidence)

B0. Exact bounded typed-axiom index with direct provenance lookup, no dynamics.
B1. Deterministic weighted incidence/factor-graph propagation with explicit typed n-ary constraints.
B2. Matched-compute asynchronous local recurrence on typed hyperedges, no learned weights.
B3. Learned/shared local HCA candidate (only if B2 shows a mechanical reason to add trainable dynamics); ablate stochastic firing, local clocks, perturbations and sample replay.
B4. Optional time-memory experiment: bounded exponential/multirate traces versus fractional/rational kernels, withheld pending horizon-dependent data.

All conditions must use the same authorized resources, number of visible axioms, relation arities, source labels, evaluation inputs, observation budget and hard task constraints. Report wall time and total cell/edge updates; compare Pareto frontiers rather than matching only iteration count.

## Required synthetic test domains (research workloads, not evidence of semantic understanding)

1. **Cross-turn coding constraints:** Arch Linux environment, dependency requirement, and prohibition on mutating system-managed packages appear at separated turns. Later answer must cite the correct original resource, preserve all mandatory constraints and not invent a package command. Include stale/altered source re-verification.
2. **N-ary collision:** three or more simultaneously necessary typed premises determine a relation; compare HCA with exact index and full-incidence factor graph. Include partial premises, adversarial distractors and contradictory claims. A candidate relation cannot become admitted without the explicit proof channel.
3. **Interference and project switching:** interleave conflicting Debian/Arch, two software projects and irrelevant distractors. Test simultaneous maintenance of distinct active scopes and task-conditioned activation without cross-contamination.
4. **Damage and recovery:** corrupt *only the dynamical activation state* (not sources or bindings) and quantify steps and operations to restore readout. Compare exact refresh from verified source/graph against CA repair; a repair claim requires added value beyond exact restoration.
5. **Delay and horizon:** delayed references across 8/32/128/512 turns with fixed capacity and dropout; measure recall and false activation. Heterogeneous timescale variants only if this identifies a real deficit.
6. **Width/topology transfer:** vary axiom count and incidence degree, separately (e.g. 32, 64, 128, 256 axioms), with degree/arity bounded. Same rule parameters do not prove graphon convergence.

## Pre-registration and pass/fail rules before running science

Freeze generator seeds, split, subject versions, hypergraph construction, authority maps, exact baseline and factor-graph baseline, budgets, evaluation scripts, scoring criteria, and all hypothesis-specific thresholds BEFORE any QUAL outcomes. Use disjoint DEV and QUAL. Do not tune on QUAL, resume a failed scientific run under altered authority, or recast a mechanical failure as scientific NONPASS.

Nonnegotiable correctness gates for every configuration:
- Zero unsupported `ADMITTED` relation promotions; zero implicit source mutation; zero loss of required `BINDING` constraints; zero invalid source references represented as valid.
- No persistent file creation or filesystem synchronization on the HOT path; WARM stalls never make a HOT caller wait.
- Bounded cells, relations, degree, queue, memory, compute per WARM step, and observation-loss accounting.
- Prior `ContextView` bytes/digest immutable throughout successor construction; views published atomically with clear predecessor/version coordinates.
- Exact reproducibility for seeded schedules and exact source/provenance contracts.

Scientific primary endpoints: (a) constraint-correct delayed recall with citations; (b) interference/misbinding rate; (c) robust recovery after dynamical-state damage; (d) extra task value over best exact/factor-graph baseline at matched work and capacity; (e) accuracy/work Pareto frontier. If B0 or B1 equals or beats HCA, record NONPASS for claims of added utility. A mechanically correct NCA alone is not cognitive qualification.

## Implementation path that does not collide with Claude's ongoing runtime fixes

- Keep this protocol outside canonical `native/` and `src/`. Do not modify the active Claude branch while runtime repairs are underway.
- After Claude's runtime work is reviewed, introduce a separate DEV `research/ecsc_hca_r0/` subject only, with independent pinned test fixtures, and READ-ONLY imports from the qualified contextual interfaces.
- First freeze and execute B0/B1 plus mechanical contracts; only then create an independently isolated candidate B2/B3 implementation. Do not reuse frozen scientific milestones or introduce a new release surface.
- No new caches, persistent indexes, build directories, venvs or growing evidence generated automatically. Use existing shared dependencies and explicit bounded evidence artifacts under established DEV research paths.

## Nonclaims

No ECS_C/HCA inference or long-term-memory utility is established by this design. The existing ContextLane has a versioned working-context contract, not a trained cell law. No ECS_C-to-K1 qualified observation encoding exists. Neither the NCA's visual task performance, ms-PHNN's continuous stability theorem, nor fLNO's analytic fractional-memory guarantee transfers to ECS_C without new experiments and, where required, separate proofs.
