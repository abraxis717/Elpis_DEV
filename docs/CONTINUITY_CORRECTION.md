# Continuity/evolution correction: findings and DEV review

Base: `233cf82533e0ee1f0fe97c435eb768e2163ef6fa` (main).

The correction reserves one assertion identity durably before Runtime executes
an evolution attempt, then finalizes that exact pending authority. A failed
reservation executes nothing. Failed final publication leaves pending authority
or, when an uncertain write became durable, the next idle revision. Both refuse
the old assertion. Explicit finalization requires an externally established
result and never executes the attempt.

The path gate checks the assertion's predecessor claim against the trusted
continuity head and derives the receipt predecessor from that head. Assertion
v1 and receipt v0 payloads and identities are unchanged; permanent tests pin
representative identities computed with the base revision's implementation.
Assertion v0 remains retired.

The register is explicitly version 2: two 176-byte slots, 352 logical durable
bytes total. Pending identity is exactly 32 bytes plus a state byte. The previous
136-byte format fails closed; no automatic migration, reset, history or
external-side-effect reconciliation was added.

## Permanent regression coverage

The focused suite covers reservation refusal, torn writes, both uncertain
durability outcomes, process death at every publication step, repeated pending
restart, exceptions and invalid attempt results, one-time explicit finalization,
stale authority, forged predecessors, unchanged persisted path identities,
counter exhaustion, and 1,000 reserve/finalize cycles with constant disk and
resident state and two fixed restart reads. Cognition publications preserve
pending evolution identity.

Final focused command (with native libraries required) runs
`tests/continuity`, `tests/evolution`,
`tests/integration/test_evolution.py`,
`tests/integration/test_runtime_hot_path.py`,
`tests/integration/test_codec_ecs_turn.py`, and
`tests/boundary/test_one_ecs.py`.

Result: **214 passed, 1 existing skip**. The skip requires the absent external
real V4.1 tokenizer asset. No skips, xfails or weakened tests were introduced.

AST comparison against the base confirms `Runtime.run_turn`,
`Runtime.anchor_cognition`, and `Runtime._reconcile_cognition_substrate`
are unchanged. Native sources, ECS mathematics, runtime cognition and frozen
research files are unchanged. Hot-path tests pass: the record-size consequence
is the only change to the canonical turn's publication; no native crossing was
added.

## Repository-equivalent qualification

Commands follow `.github/workflows/ci.yml`, using repository-local build and
temporary directories. Local raw outputs are under ignored `build-corrective/`.
These results are **not a fully green CI qualification**.

| Check | Observed result at handoff | Local output |
|---|---|---|
| Focused requested suites | **214 passed, 1 existing asset skip** | `focused-complete.log` |
| Base-package neutrality | **Passed**; isolated base import contains only elpis-dev distribution, no NumPy/model stack | `base-neutrality.log` |
| Boundary | **113 passed, 1 failed, 10 existing donor skips**; failure is socket creation denied by host | `boundary-final.log` |
| Python 3.11 full suite | **Still running**; known socket-guard failure already observed; no final result | `python311.log` |
| Python 3.12 full suite | **Blocked**; pytest absent; repository-local environment installation cannot fetch dependencies | `python312.log`, `python312-install.log` |
| Python without network | Namespace and negative connection probe succeeded; **full suite still running**, same socket-guard failure observed | `airgap.log`, `airgap-probe.log` |
| Scientific authority, Python | **Still running**; Retention R2 current-runtime regression passed, Retention R3 current-runtime regression in progress; available NumPy does not match the declared pin | `scientific.log` |
| Scientific authority, native | **16/16 passed** | `scientific-native.log` |
| Donor parity, required mode | **10 setup errors**; pinned donor checkout absent | `parity.log` |
| Native gcc Debug | **119/119 passed** | `gcc-debug-test.log` |
| Native gcc Release | **119/119 passed** | `gcc-release-test.log` |
| Native clang Debug | **119/119 passed** | `clang-debug-test.log` |
| Native clang Release | **119/119 passed** | `clang-release-test.log` |
| ASan+UBSan, leak detection enabled | **119/119 aborted** due to LeakSanitizer/ptrace incompatibility; no clean sanitizer claim | `asan-ubsan-test.log` |
| TSan | **119/119 passed** | `tsan-test.log` |
| K1 method/source preservation | **Passed**; AST and git diff checks | `unchanged-authority.log` |
| Persisted path identity pins | **21 gate tests passed**, also included in the final focused run | `schema-identities.log` |

These are snapshots, not a claim that the unfinished jobs eventually passed.
Existing reproduction/profile and external-asset skips are left intact.
No surviving test was disabled, weakened, deleted, or marked xfail for this work.
The full Python runs started before the final two additional test cases were
added; the final focused run includes both additions. Final exact-head
qualification remains outstanding.

The available Python environment is 3.11.9, pytest 9.0.2 and NumPy 2.4.2.
NumPy is outside the declared `>=1.26,<2` range and the scientific job's
`1.26.4` pin. A Python 3.12.5 environment was created inside the repository,
but installation could not fetch build/test dependencies because network/DNS
was unavailable. No dependencies or data were taken from another checkout.

The separate strict donor run requires its pinned checkout. It is absent
inside this repository; no substitute donor or relaxed parity check was used.
The network-isolation probe confirmed that connection creation is refused.

ASan+UBSan used the requested leak detection and halt settings. All 119 tests
aborted because LeakSanitizer reports that it cannot work under ptrace.
Leak detection was not disabled to obtain a green run.

## Limits

At-most-once is a Runtime composition guarantee under the register's documented
filesystem durability model. The stateless gate relies on its composing owner
for trusted authority and durable reservation. A trusted reconciler must verify
the completed result's assertion and predecessor before supplying its digest.
Unresolved effects stay pending. No automatic old-format migration, corruption
recovery beyond the existing two-slot model, new latency claim, scientific
qualification claim, or external-side-effect recovery is made.

## Findings and review decisions

### Defect 1: an executed assertion could become admissible again

Previously, Runtime executed the attempt before publishing the next evolution
authority. A publication failure could leave the previous idle revision on
disk. Restart could then admit the same assertion again.

The new ordering is validation -> durable reservation -> execution -> durable
finalization. A pending record contains only the authoritative revision/head
and the exact 32-byte assertion identity. The runtime also leaves authority
pending when the attempt raises, is interrupted, returns an invalid result, or
fails a second admission check. None of those outcomes proves that an external
side effect did not occur.

Reservation publication failures execute zero attempts and fail-stop Runtime.
After successful reservation, a failed final write leaves the reservation.
A sync error may instead leave a complete next-idle record on disk. The user
explicitly approved this clarified uncertainty law: restart may resolve to
pending **or next idle**, provided both forbid execution of the old assertion.
The implementation and fault-injection tests exercise both outcomes.

### Defect 2: assertion authority and predecessor could disagree

Previously, a caller could supply a legitimate evolution authority binding
while independently choosing the predecessor copied into the receipt.
The trusted runtime binding now also carries the current head. The gate rejects
a conflicting assertion predecessor before reservation and derives the
receipt predecessor from that trusted head.

This narrows admission without changing the existing persisted assertion or
receipt payload meaning. No new assertion/receipt schema was needed. Their
digest computations and representative persisted identities are unchanged.

### Review-sensitive API changes

- `EvolutionAuthority` gains `pending_assertion: str | None`.
- `EvolutionAuthorityBinding` now requires `head`.
- `reserve_evolution_assertion(expected_idle, assertion_digest)` is the only
  new public store method.
- `commit_evolution_transition(expected_pending, receipt_digest)` now refuses
  idle authority and compares the entire pending authority, including identity.
- `Runtime.evolution_authority()` and `Runtime.evolve()` refuse pending
  authority. Inspect it through `Runtime.continuity.snapshot().evolution`.
- Explicit reconciliation uses the existing commit method with the exact
  pending authority and an externally verified result digest. It advances the
  revision once; it does not clear pending back to the old idle state.
- Register format/checksum domain and evolution-authority digest domain are
  version 2. A v1 register is refused before mutation. Old-format authority
  cannot establish whether an earlier unpublished attempt already ran.

## Every Git change to review

All paths are relative to this repository. No other source area was changed.

| Path | Change and reason |
|---|---|
| `src/elpis/continuity/record.py` | Explicit format v2, pending assertion field and validation, 176-byte serialization, authority digest includes reservation state |
| `src/elpis/continuity/store.py` | Durable reservation; finalization requires exact pending authority; reservation refuses counter exhaustion before execution |
| `src/elpis/continuity/__init__.py` | Package documentation describes the fixed-size pending identity |
| `src/elpis/evolution/path_gate.py` | Trusted binding exposes head; predecessor mismatch refusal; receipt predecessor derives from authority |
| `src/elpis/runtime/composition.py` | Validate/reserve/execute/finalize ordering; pending refusal and failure handling; K1 methods unchanged |
| `tests/continuity/test_store.py` | Reservation/finalization laws, fixed layout, old-format refusal, bounded memory/storage/restart, cognition preserves pending |
| `tests/evolution/test_path_gate.py` | Predecessor forgery refusal, correct chain authority, unchanged schema identity pins |
| `tests/integration/test_evolution.py` | Refusal/uncertainty/torn-write/process-death matrices, pending restart stability, explicit finalization, invalid attempt results, predecessor integrity |
| `tests/integration/test_runtime_hot_path.py` | Exact expected record size changes from 136 to 176; existing syscall/native-crossing assertions preserved |
| `tests/boundary/test_one_ecs.py` | Public API allowlist adds only the specific reservation method; residue exemptions unchanged |
| `ELPIS_SYSTEM.json` | Current authority describes reservation, exact footprint and format-v2 domains |
| `docs/CONTINUITY.md` | Normative layout, failure law, reconciliation contract, predecessor rule, old-format refusal; old measured performance observations preserved and labeled |
| `docs/ARCHITECTURE.md` | Current fixed authority/API and evolution admission description |
| `docs/ELPIS_MISSION.md` | Correct exact continuity write size in the existing hot-path table |
| `docs/NONCLAIMS.md` | Correct footprint and limits of pending/reconciliation behavior |
| `docs/CONTINUITY_CORRECTION.md` | This findings, qualification and DEV handoff review |

Generated build trees, temporary test directories, local environments and raw
logs are ignored outputs, not proposed source changes. No frozen scientific
payload, native source, ECS numerical code, residue exemption, inference,
pipeline, substrate or structure implementation was changed. There was no
merge, rebase, squash or history rewrite.

## Recommendations for the next DEV session

### Priority 0: finish qualification before claiming closure

1. **Read the existing logs before rerunning anything.** Three long-running
   jobs were active when this handoff was written: Python 3.11 full suite
   (tool session `69582`), network-isolated full suite (`79507`), and
   Scientific authority (`6832`). Session handles may not survive a new
   session; their output files remain the source of observed results.
   Collect their final summaries if they finish. They were not intentionally
   cancelled or silently counted as passes.
2. **Use the repository-declared Python dependencies.** Provision Python 3.11
   and 3.12 environments within the authorized workspace, install
   `.[test]`, and use `numpy==1.26.4` for Scientific authority. The current
   NumPy 2.4.2 results cannot substitute for the declared environment.
   Keep temporary files and build outputs inside this repository.
3. **Resolve qualification environment restrictions without weakening tests.**
   Run ASan+UBSan with leak detection in an environment where LeakSanitizer is
   supported. Run Boundary and airgap tests where creating an INET socket is
   permitted but external connectivity is isolated; the current host denies
   socket construction before the guard test can exercise its assertion.
4. **Supply the exact pinned donor checkout in the allowed workspace.**
   The repository pins donor commit
   `8eda162712ebe291f2b88b6906366075859bc485`. Run parity with
   `ELPIS_REQUIRE_DONOR=1` and an explicit permitted `ELPIS_DONOR_ROOT`.
   Do not invent a donor fixture or relax parity.
5. **Complete exact-head CI-equivalent qualification.** Use
   `.github/workflows/ci.yml` and `docs/CI_POLICY.md` as authority. Collect
   all twelve required jobs, plus the focused suites, for the final corrective
   head. No merge or fully-qualified claim is warranted from this handoff.

The external real-tokenizer test remains an existing asset-dependent skip.
If this asset is supplied, it must be explicit and within the permitted
workspace; do not pull unrelated model assets merely to change the skip count.

### Priority 1: review the safety boundaries before deployment

- Review the four short production changes first: record, store, gate and
  runtime evolution wiring. Confirm finalization compares the reserved
  assertion identity and that no path clears pending or retries execution.
- Review the persistence-version change deliberately. Existing format-v1
  stores need externally informed reconciliation; a generic automatic upgrade
  would not establish whether an unpublished side effect occurred.
- For operational use, specify who may act as the trusted reconciler and what
  external evidence establishes the result. This is an operational decision,
  not justification for adding receipts-as-history, queues, replay or a new
  runtime subsystem.
- Treat a failure in unchanged numerical tests separately. Preserve its output
  and classify it under CI policy before proposing any numerical change.
  No numerical defect or unrelated flaky test was established in this session.

### Scope for further development

There is no architectural redesign recommendation. Preserve the single live
ECS, fixed-size continuity register, runtime composition, HACF structural
memory, and inference communication boundary. The next useful DEV work is
qualification and review of this correction, not another implementation pass.

## Reproduction and handoff commands

Run from the repository root. Use a correctly provisioned interpreter and keep
all paths inside the authorized workspace.

```sh
git status --short
git log -3 --oneline
git diff 233cf82533e0ee1f0fe97c435eb768e2163ef6fa HEAD --stat

export TMPDIR="$PWD/build-corrective"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"
export ELPIS_REQUIRE_NATIVE=1
export ELPIS_NATIVE_BUILD="$PWD/build"

python -m pytest -rs \
  tests/continuity tests/evolution tests/integration/test_evolution.py \
  tests/integration/test_runtime_hot_path.py \
  tests/integration/test_codec_ecs_turn.py tests/boundary/test_one_ecs.py \
  --basetemp=build-corrective/pytest-review
```

Read `build-corrective/python311.log`, `airgap.log` and `scientific.log`
for final summaries before deciding whether any long-running check needs to
be started again. Native configure/build/test logs are named by compiler/build
mode in the same directory. The workflow remains the complete command
authority for full qualification.

## Closure status

Both corrective mechanisms are implemented and their permanent focused tests
pass. Full repository-equivalent qualification is **incomplete and not green**:
three suites were still running at handoff, and the documented dependency,
donor, socket-policy and LeakSanitizer environment blockers remain.
This review is a handoff, not permission to merge or a claim of completed
scientific qualification.
