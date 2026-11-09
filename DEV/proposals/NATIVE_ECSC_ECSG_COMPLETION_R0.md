# PROPOSAL R0 — Native ECS_G / ECS_C architecture and completion charter

**Status:** proposed architectural direction and implementation work order. This document does not declare any subsystem qualified, alter a scientific result, or authorize a merge. All implementation and evidence must be obtained within this repository, under the repository's existing qualification and CI procedures.

## 1. Mandatory target

The runtime must execute **entirely in native code** (C, C++ and/or Rust). Python is permitted as an offline reference, a scientific research instrument, a differential-test oracle, or build/test tooling; **no Python interpreter, Python facade, Python callback, Python-owned state, ctypes bridge, or Python command-line process participates in runtime startup, services, mutation, scheduling, recovery, admission, persistence, communication, or operational control**.

Partition the canonical ECS as two distinct authorities:

- **ECS_G**: geometric cognition and dynamics; native K1, its state `(W, epoch, H, a)`, exact geometry/forward/S3, deterministic learning/consolidation, snapshot/transaction identities, and geometric FMS residency interfaces. Its numerical law and existing qualified ABI/serialization identities remain protected. It does not depend on ECS_C, the codec, or RuntimeCore.
- **ECS_C**: the native continuity and memory subsystem: durable lineage and evolution authority, entity and evidence identity, canonical event/receipt representation, deterministic semantic replay/recovery, bounded typed memory, contextual source/provenance and interaction topology, state roots, controlled compaction, and bounded memory dynamics when separately admitted. It owns its durable memory/history authority, without becoming the geometric K1 state owner. It never upgrades evidence or a candidate relation to authority by itself.
- **RuntimeCore**: the sole native composition/orchestration boundary. It binds ECS_G and ECS_C through typed, explicitly owned native handles and coordinates. Preserve the current HOT/WARM/durability separation, native managed-turn transaction model, explicit persistence and recovery disposition, fuel and admission, and fail-stop behaviors. Only RuntimeCore may compose the two ECS domains.
- **FMS / substrate / structure**: retain narrowly scoped native contracts that the runtime actually consumes. Avoid duplicating their authority or importing unrelated mechanics.

**Retirement objective:** remove the old `native/continuity` standalone runtime dependency once an independently tested ECS_C successor serves every admitted consumer contract. A mere file rename or private relocation is insufficient. Do not retire any current behavior before compatibility, recovery and refusal checks pass against the accepted native runtime law.

## 2. Repository starting point and conflicts to resolve deliberately

The baseline for reconciliation is `main` (pin its exact head before any work). At the charter's creation it was `9dc2c495bf1a4fbedc6fd84058d329c4f7bf00e3`. Inspect the actual current head and merged changes before editing.

The code on `main` currently expresses an earlier one-ECS/standalone-continuity architecture, including `ELPIS_SYSTEM.json`, `docs/ELPIS_MISSION.md`, `docs/ARCHITECTURE.md`, `docs/RUNTIME_CORE.md`, `docs/ECS_MEMORY.md`, `tests/boundary/test_one_ecs.py`, and other mission, naming, build and CI gates. In the implementation branch, **replace these incompatible assumptions with the new reviewed target**, not by weakening or skipping the gates. Keep the real anti-drift tests, but re-specify and test them for the new boundaries. This charter proposes the new boundary; it is not a statement that the current code already implements it.

Historical in-repository work for selective inspection:

- `integration/native-ecsc-turn-continuity-r1`, including `native/ECS_C/`, `src/elpis/ECS_C/`, and `tests/ECS_C/` at `7d00038b7199cf991d0d5fbbe8d967a7cab44648`; merge history around PR #34; and the subsequent `CONT-4`, `CONT-5` and `CONT-6` changes. Treat older implementation as a **donor**, not as a branch to merge wholesale. Its Python kernel/replay is reference code only.
- `native/ECS/` on current `main`, including K1, `ecs_memory`, composite state, the native headers/tests and their qualified ABI/persistence forms. Partition responsibly into ECS_G geometric ownership and ECS_C memory ownership, preserving exact established identities and proof obligations. A legacy symbol with a geometric name may remain as a compatibility protocol identifier if ABI evidence requires it; namespace ownership and canonical dependency direction must nevertheless be unambiguous.
- `native/runtime/`, `native/runtime/include/elpis/runtime.h`, `src/elpis/runtime/`, `docs/RUNTIME_CORE.md`, `docs/K1_RECOVERY_R0.md`, `docs/OPERATING_ELPIS.md`, `docs/PYTHON_AUTHORITY_CENSUS.md`, `docs/qualification/COGNITIVE_BOUNDARY_R2_V5.md`, `research/qualification/k1_recovery_r0_v4/`.
- `DEV/ecsc_hca/RESEARCH_CONTRACT.md`, `DEV/ecsc_hca/MIGRATION_PROVENANCE.md`, `DEV/ecsc_hca/R19_PROTOCOL.md`, `research/ecsc_hca_r0/` and associated tests. They describe bounded typed-incidence/hypergraph cellular memory and its scientific limits. These are not a shortcut to production admission.
- `docs/CI_POLICY.md`, `docs/NONCLAIMS.md`, `docs/ECS_COMPOSITE_STATE.md`, `docs/ECS_MEMORY.md`, `docs/qualification/README.md`, and the corresponding native and Python reference regression tests.

**Existing qualification is commit- and regime-specific.** The in-repository K1 Recovery R0 v4 result is RUN PASS for its recorded commit and fault scope; it does not automatically qualify a newly factored ECS_C or ABI v5 migration. ABI v5 Cognitive Boundary R2 is NOT RUN / UNQUALIFIED. ECS codec is UNQUALIFIED. Memory's exact storage mechanics are admitted; its learned FAST behavior and SLOW consolidation are not. HCA R0-R19 evidence is research evidence only and is not a validated general-purpose semantic-memory mechanism. Honor the exact in-repository statuses and never broaden them by renaming or aggregation.

## 3. Native ECS_C completion requirements

Build a **single native owner** for durable identity/history/memory. It must include, under explicit finite limits:

1. Exact content identity, canonical serialization, genesis/lineage binding, entity lifecycle, bounded evidence and typed relation records, sender attribution, clocks, event ordering, and transparent refusal codes.
2. A canonical, durable append/checkpoint or other explicitly specified bounded layout whose effects and crash semantics are defined before implementation. Structural scan of frames is distinct from semantic validation. **Complete frames must be semantically verified before recovery can install state or discard bytes**. Incomplete trailing frames are treated according to the frozen recovery law, with uncertain outcomes fail-stopped, not silently resolved.
3. Native replay of every admitted event type, state-root verification, sequence/watermark reconstruction, ownership and collision controls, mailbox/scheduler state only if actually consumed, compaction generation/base lineage/floor if actually used, and explicit reconciliation. No hidden duplicate index or independent mutable state authority.
4. A self-contained native bootstrap and control path: provisioning, open, replay, inspect, query/lookup, mutation, bounded compaction, close, restart, inspect and explicit reconcile must be usable without Python. Provide a native operational CLI with machine-readable output and clearly classified exit/refusal codes.
5. A trustworthy **deployment authority boundary**. Neither a bare unkeyed digest nor a generation value kept in the same writable history can authenticate a maliciously rewritten lineage. Specify the independent anchor/pin and threat model; prove forged higher generations, corrupted newest, stale bases, conflicting checkpoint candidates and history rewind cannot become acknowledged states silently. Retain crash-recoverability without conflating ordinary crash fallback with adversarial rollback.
6. Integration with the native composite's K1 and memory envelope identities without recomputing, reinterpreting or silently migrating established serialized states. Preserve acknowledged-versus-volatile semantics, live generation, K1 and memory ownership, observe-only turn constraints, fuel admission, HOT zero-synchronous-I/O law, residency holds, partial-commit fail-stop and explicit operator reconciliation. Cross-domain atomicity must be specified precisely; never claim it when not provided.
7. The required memory subset must be classified item-by-item as **admitted and production**, **implemented but gated**, **research-only**, or **retired**; then only the admitted subset participates in the default runtime. No gratuitous duplication of existing native FMS, structure, or geometric kernels.

## 4. Cellular / hypergraph contextual memory

The design's typed-hypergraph cellular memory belongs to ECS_C, but its research nonclaims remain binding. Differentiate:

- **Evidence authority:** exact source coordinates, provenance, scope, typed roles and hyperedges, immutable BINDING constraints, `CANDIDATE` versus `ADMITTED` statuses, staleness and source-version validation. These are not determined by recurrence.
- **Derived context and activation:** bounded non-authoritative cell states over typed incidence relationships; deterministic seeded, locally scheduled updates; exact resource/work accounting; WARM publication of a complete successor view while the predecessor remains immutable; no per-cell HOT persistence; bounded failure/refusal behavior. Changes of activation never mutate sources or elevate relation status.
- **Comparators and proof obligations:** B0 exact lookup, B1 typed factor-node propagation, B2 unlearned asynchronous recurrence; learned/shared HCA only under separate preregistration and demonstrated benefit over equal-budget exact/factor-graph controls. Test n-ary role constraints, conflicting scopes, delayed references, provenance misbinding, stale sources, source damage versus activation damage, transfer, shortcut controls, and adversarial distractors. The R19 formal Boolean work is not an open-language semantic proof.

Implement native substrate and interfaces in DEV as appropriate; **do not enable an unqualified dynamical memory law in the canonical runtime**. Full implementation means a real native system plus explicit gates and tests, not a fabricated scientific qualification.

## 5. Execution order and delivery

Work from exact source census and ownership graph, then proceed in separately reviewable, dependency-ordered units:

A. **Inventory and plan:** record exact current baseline, existing ABI owners, imported historical donor pieces, tracked runtime Python reachability, format and symbol compatibility, qualification pins, root gates, and all consumers of the standalone continuity implementation. Propose source/ABI/state transition and rollback matrices *before* touching active owner code. No broad blind rename.

B. **Ownership split:** establish `native/ECS_G` geometry and `native/ECS_C` continuity/memory, with matching build, headers, tests and derived documents. Minimize changes to qualified math, preserved ABI identifiers, fixed evidence and stored formats. Requalify only changed subjects; preserve byte-identity where a pure relocation suffices.

C. **Native ECS_C authority:** implement semantic validation/replay, durable memory state, recovery and the needed history/identity contracts against deterministic fixtures, fault matrices and adversarial inputs. Begin with closed mechanics gates and numeric, text-free inputs. Reuse existing native building blocks when they meet the target law.

D. **RuntimeCore integration:** use native-only control and operators; remove Python from every production execution path, including CLI and bootstrap. Keep native existing critical invariants and explicitly bound cross-domain commit/ack/recovery ordering. No implicit authority handoff.

E. **Cellular memory:** separately implement/bind the bounded native typed-hypergraph research mechanism, with comparator and qualification gates. Enable only the independently admitted operations; keep experiments and unfrozen hypotheses isolated.

F. **Retirement and end-to-end:** remove old dependency only when all consumers have passed the replacement contract tests. Add native bootstrap/restart, synthetic crash/kill, torn-write, lost-sync, misuse, concurrent ownership and rollback tests. Demonstrate the native process's actual shared-library/import closure does not launch or dynamically load Python. Update code authority, tests, CI, docs and user-facing native operations together. Keep a precise qualification and nonclaim ledger.

Each unit: a scoped PR, exact changed-file list and digests, tests with commands/exit codes, mechanical versus scientific classification, failure boundaries, and no unreviewed merge. Preserve immutable scientific evidence, implementation performance constraints, all source and qualification pins, and CI policy. Never rewrite completed trials. When a test fails, diagnose its smallest mechanical layer; never alter a scientific object to produce a passing result. Never manufacture a PASS for a successor merely because a predecessor was qualified.

## 6. Acceptance gates

- No Python runtime process, import, callback or mutable runtime authority; offline Python tests remain permissible.
- Exact current native K1 math/ABI/serialized identity behavior preserved or explicitly versioned and requalified.
- Native ECS_C semantically recovers and validates its admitted full state; no duplicate mutable history or competing lineage authority.
- Bounded memory, file counts, work, size, fault envelopes; deterministic replay and exact refusal atomicity.
- Admitted evidence never gains authority through topology, dynamic activation, codec, or model outputs.
- No unprotected lineage forgery/rollback within the declared threat model; independent deployment pin semantics tested.
- No Python or persistence/sync activity on HOT; explicit WARM/durability operations control durable acknowledgement.
- Native startup/CLI and end-to-end acceptance exercises the actual installed/runtime binaries with faults.
- Compiler/sanitizer, platform CI and architecture checks pass at the exact proposed PR heads; release claims remain limited to actual frozen evidence.
- Every remaining deferred capability is explicitly unqualified and gated, not omitted from the engineering report.

**Important:** this proposal is a prospective implementation charter. It contains no code, no new qualified source and no permission to merge itself. A red legacy architecture gate is evidence of work remaining, not permission to disable it without an equivalent successor gate.
