# R2A — integration lifetime census (DEV mechanics only)

Authority: worktree `research/ecsc_hca_r0`, HEAD
`cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3`. Frozen canonical context SHA256:
`afa96f35c51e8130683eb5502e8e6478d7c113caf1f6116e4a703c6cd0aae2f0`.
This census precedes R2 implementation. No production integration or science.

The concrete call path is `context.successor(old_view, observations, limits,
through=..., dropped=...)` -> immutable, sorted, digest-sealed `ContextView` ->
`Baselines.build(view, same_authority, refreshed_checks, same_limits,
previous=old_graph)` -> `task.prepare(graph, task, meter)` -> B2 start/warm/observe.
`ContextView.verify` checks canonical membership, capacities and digest, not source
truth or the relation of a view to its alleged predecessor. `Baselines._build`
additionally checks predecessor digest, version + 1, monotone through/dropped,
exact retained BINDING grants, and equality of the entire Authority and Limits.
The allowlists include absent members: reintroducing an already granted member is
possible; adding a new grant during a successor is not. No grant migration exists.

`SourceCheck` observations can be missing, matching, or mismatching the pinned
synthetic source digest. Only these observations may refresh under fixed authority.
Policy includes source grants, schemas (including weights), scopes, kinds, epoch
and limits. Evidence includes policy, view digest and the ordered check tuple.
Source validity is VALID / STALE / UNKNOWN (or AUTHORITY_ONLY for no references).
It is separate from PROVISIONAL / BINDING and CANDIDATE / ADMITTED. Integer salience
does not promote any label and is never source verification or semantic truth.
Even an ADMITTED downstream factor can depend on CANDIDATE salience.

Task preparation selects scope and exact typed/status predicates, validates seeds
and refuses invalid seeded sources. Task seeds are public premises, never withheld
answer keys. Readout retains scoped bindings even when their sources are invalid;
it refuses insufficient binding capacity. Zero binding activation is not deletion.
An invalid intermediate must invalidate its descendants, not just its own readout.

Frozen B2 hashes the entire Task with exact evidence into `Activation.owner`.
Changing a seed, scope, budget, schedule, source checks or view coordinate changes
ownership. A different session refuses that activation with INVALID_STATE. This
is an established safety feature. R2 must never rewrite B2 owners or transplant
arbitrary activation bytes. A fresh B2 session is required at those boundaries;
same evidence and exactly identical Task support task-local warm recurrence.

| Event | Same research lineage? | Carryover requirement |
|---|---|---|
| Identical graph and Task | Yes | Only issued, bound state; repeated explicit WARM |
| Seed/predicate/budget/schedule change, same scope | Same evidence lineage | Frozen B2 fresh; R2 must invalidate changed dependencies |
| Same-policy verified immediate successor | Yes | Revalidate binding retention and compute old/new dependency damage |
| Checks refreshed on a verified successor | Yes | Invalidate changed cells plus all old/new downstream cells |
| Checks refreshed on identical view | Independent build supported by R1 | R2 candidate conservatively requires an explicit successor; no implicit carryover |
| Scope switch | Separate task context | B2S refuses; fresh experiment or explicit B0 recompute |
| Changed authority/epoch/weights/limits | No | Refuse migration, even if otherwise well formed |
| Unrelated/skipped/reversed predecessor | No | Refuse before state work |
| Removed optional axiom | Yes | Canonical retract also removes every incident relation |
| Removed relation only | No direct retract-relation observation | Can retract/reassert an optional member, leaving incident relations absent; no invented API |
| Removed/downgraded/renamed binding | No | Research BINDING_RETENTION rejects even a canonical required retract |
| Optional observation dropped | Yes, counters monotone | Missing observation is not evidence; compare actual published membership |

Canonical queue overflow can evict/drop optional observations and records cumulative
drops; required observations are refused when no capacity remains. R2 uses pure
successor calls, no lane worker. It can model the resulting dropped count, not
infer the content of a missing observation. Source manifest/version replacement,
new-source discovery, external source attestation, arbitrary relation retraction,
native n-ary admission, durable memory and cross-authority migration are unsupported.

Safety plan: compare selected old/new typed factors, member source states and seed
clamps. Mark changed cells and changed-factor outputs, then take directed reachability
through the union of OLD and NEW input-to-output dependencies. Reset every marked
cell to its new seed clamp. Cycles are traversed to closure, never stopped at a
single edge. No dependency can lead from an affected cell to a retained cell.
This permits unaffected components to retain activation, while a wholly affected
cycle may require total reset. Unaffected positive values are usable only when
their construction history is trusted. R2 will accept only its own immutable issued
snapshot, not a caller-supplied numeric payload. Explicit damage will carry a bounded
taint set, cleared through downstream invalidation before any subsequent observation.
Arbitrary external corruption with falsified taint is outside this in-process API;
there is no serialized-state import or new hashing registry.

B0I can use the same conservative dependency region, reset it, and recompute the
least integer fixed point within that region with unaffected boundary values held
fixed. Monotonicity, finite [0,1000] cells, nonexpansive weights and closure of the
region give equality to fresh B0, including unsupported cycles. B2S will use bounded
asynchronous local updates after invalidation; partial schedules may under-retrieve
and have no finite fairness guarantee. It must never describe reset as memory.
