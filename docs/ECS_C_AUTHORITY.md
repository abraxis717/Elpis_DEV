# ECS_C authority (R0)

Authority: `docs/ELPIS_MISSION.md` (Amendment 1), `docs/NATIVE_SPLIT_CENSUS.md` (sections 5a and 11, Stage C).
Contract: `native/ECS_C/authority/include/elpis/ecs_c_authority.h` (normative C ABI v1). Implementation:
`native/ECS_C/authority` (Rust crate `elpis_ecs_c_authority`; libraries `libelpis_ecs_c_authority.so` and
`libelpis_ecs_c_authority_testing.so`).

The ECS_C authority is the successor of standalone continuity. It holds the **complete materialized authority** of
ECS_C in one fixed-size record: the lineage, the entities and their lifecycles, sender attribution and logical
clocks, source versions, scopes and observation bounds, the cognition lineage (which retained-state identity is
authoritative, and of which format), the evolution authority, receipts and reconciliations. Every transition is a
typed event validated by one transition law before it is written; recovery replays the newest transition against
its predecessor. It keeps no log, no segments, no generations and no compaction archive.

**Status.** Built and mechanically tested as a library; its durability protocol through the native host is RUN PASS
under ECS_C Recovery Q0, for commit `117b41a` (RuntimeCore ABI v6) only, within that specification's scope. Composed
by RuntimeCore since ABI v6 (`elpis_runtime_create_ecs_c`; docs/RUNTIME_CORE.md, "The durable authority (ABI v6)"),
which publishes through it, admits typed evidence against its sources and records evidence summaries, and run by the
native host `elpis`, the production entry point (docs/NATIVE_HOST.md). Since ABI v7 it is RuntimeCore's only durable
authority, on the native host and the offline Python reference alike: standalone continuity retired into it
(docs/RUNTIME_CORE.md, "One durable authority (ABI v7)"; docs/ELPIS_MISSION.md, Amendment 2), and the crate carries
continuity's SHA-256, evolution law and digest and register reader as its own, with no dependency. The ABI v7
successor of Q0 is not run. Nothing in this document is a scientific claim.

## Ownership

ECS_C owns the authority; it performs no geometric learning and never touches ECS_G state: it names retained-state
identities (ELPISGK1 or ELPISEC1 digests), never the states. The crate depends on nothing but the Rust standard
library: SHA-256 (`sha256.rs`), the evolution authority and its digest (`register.rs`) and the legacy register reader
(`legacy.rs`) are its own, moved unchanged from standalone continuity when it retired into ECS_C (one implementation
each; until then the authority embedded the continuity crate for them). Its own C ABI is the crate's default `c-abi`
feature, so a crate that embeds it (RuntimeCore) does not re-export `elpis_ecs_c_*`; there is no build script. It
links no ECS_G, memory, RuntimeCore or codec code.

## Threat model

* **T0, accidents:** process death at any instruction, torn writes (any prefix of a record or a witness), a write
  that fails, a sync that fails with the bytes lost, a sync that fails although the bytes are durable.
* **T1, the state-directory adversary:** can read and write the state directory at will (forge or substitute
  records, copy back older genuine records, delete or truncate), but cannot read the deployment key and cannot
  write the authority directory.
* **Not defended:** a holder of the deployment key; code inside the process; an adversary who can write the
  authority directory; a device that loses acknowledged writes of *both* directories coherently. Sender attribution
  is semantic (the authorization matrix and per-sender sequences), not authentication of in-process callers.

The two protections are different in kind, and only their combination is claimed:

| Mechanism | What it is | What it defeats |
|---|---|---|
| checksum (SHA-256 under a domain, the last field, covering the MAC) | the record's integrity identity, recomputable by anyone | accidents: a write torn anywhere reads TORN |
| MAC (HMAC-SHA256 under a key derived from the 32-byte deployment key) | authentication | forgery and substitution: a checksummed record without a valid MAC reads UNAUTHENTICATED |
| rollback witness (authenticated, in the authority directory) | the newest acknowledged generation and digests | replay of older genuine records, a device that drops an acknowledged record write |

The witness is not a second writable digest beside the first: its independence is the access-control separation of
its directory, which every open verifies (owner = effective uid, mode 0700, no symlink, and neither directory inside
the other) together with the key file (mode exactly 0400). A deployment that places both directories under one
principal that T1 controls has integrity, not rollback protection, and is outside the claim.

## Files and bounds

| Directory | File | Size | Written |
|---|---|---|---|
| state | `authority.a`, `authority.b` | `R = 1344 + 160 × capacity` bytes each | the non-current slot, in place, per transition |
| authority | `deployment.key` (0400) | 32 bytes | once, at provisioning |
| authority | `witness.a`, `witness.b` | 240 bytes each | the non-current witness slot, in place, per transition |

Capacity is the number of entity rows (2 ≤ capacity ≤ 4096; the founding operator and runtime always occupy two).
Every bound of the persistence invariant is fixed at provisioning and tested over long sequences
(`long_sequences_keep_files_bytes_recovery_work_and_metadata_constant`, 100 000 iterations run locally):

| Bound | Value |
|---|---|
| files | exactly five; nothing is created, renamed or removed after provisioning |
| persistent bytes | `2R + 32 + 480` |
| per-transition writes | two in-place writes (`R + 240` bytes) and two `fdatasync`; no directory operation |
| recovery work | five reads (`2R + 512` bytes) and at most one replayed transition, whatever the history length |
| resident memory | the current record and one staging record: `O(R)` |
| retained metadata | fixed-size: counters and chain heads (receipts, evidence, reconciliations, reclaimed rows) |
| counters | refused `EXHAUSTED` before 2^63 − 1, never wrapped |
| entity rows | at most `capacity`; founding refuses `CAPACITY` before any write; an explicit operator ENTITY_RECLAIM of a TERMINATED row folds it into a chain head (identities are never reused: founding indices only grow) |

Capacity exhaustion never discards authoritative state; a larger capacity is a new, operator-provisioned deployment.

## Record (ELPECA01 v1) and witness (ELPECW01 v1)

Layouts are specified field by field in `native/ECS_C/authority/src/record.rs` and `src/model.rs` and summarized in
the header. A record carries the header (generation, predecessor generation, lineage, deployment id, predecessor
record digest and state root, state root, receipt), the materialized state (512 fixed bytes plus `capacity` entity
rows of 160 bytes), the one 512-byte event that produced it, then the MAC and the checksum. The checksum is the
record digest. The state root is SHA-256 over the lineage, generation, capacity and the state bytes. A witness pins
the generation, lineage, deployment id, record digest and state root of the newest acknowledged record.

Identity domains (all new): `elpis.ecs_c.authority.{record,record-mac,witness,witness-mac,state,receipt,
receipt-chain,evidence-chain,reconcile-chain,reclaim-chain,lineage,deployment}.v1`, `elpis.ecs_c.entity.v1`, and the
key labels `elpis.ecs_c.authority.key.{record,witness,deployment}.v1`. The evolution digest is continuity's
`elpis.continuity.evolution-authority.v2`, unchanged and tested equal.

## Entities, events and the transition law

Entities are OPERATOR, RUNTIME and SOURCE. Lifecycle: FOUNDED → ACTIVE ⇄ DORMANT, and any non-terminal state →
TERMINATED (final). The founding operator and runtime are founded ACTIVE at genesis and can never be terminated; the
authority always keeps an ACTIVE operator. An entity's identity is SHA-256 of the lineage, its founding index, kind
and label digest. A source carries a version (≥ 1), a scope digest, a per-turn observation bound (1..256) and the
32-bit memory `source_authority` its typed observations record (unique among the present source rows; freed only by
an explicit reclamation), so admission can map an observation to its source (`elpis_ecs_c_source_for_authority`).

Every event names a sender and a sender sequence that must equal the sender's watermark + 1 (`DUPLICATE` at or below
it, `SEQUENCE_GAP` above), and a receiver (the entity an entity-scoped event acts on; zero otherwise). Requests are
compare-and-publish: `STALE` unless the caller names the current generation and state root. The authorization
matrix:

| Event | Sender | Law |
|---|---|---|
| ENTITY_FOUND | operator | the receiver must be the identity the state founds next (`STALE` otherwise); `CAPACITY` when full; a source's `source_authority` held by another present source is `SCOPE_CONFLICT` |
| ENTITY_LIFECYCLE | operator | from must be current; the FSM above; founders never terminated; an ACTIVE operator remains |
| SOURCE_VERSION | operator | expected version must be current (`SOURCE_STALE`); versions only increase; new scope and bound |
| ANCHOR | runtime | once (`ALREADY_ANCHORED`); a K1 (ELPISGK1) or composite (ELPISEC1) identity |
| PUBLISH | runtime | from exactly the current identity and format (`STATE_MISMATCH`); folds an evidence summary (count, digest) into the evidence chain |
| IMPORT_COMPOSITE | runtime | the explicit move of a K1 lineage to a composite one |
| EVOLUTION_RESERVE / FINALIZE | runtime or operator | continuity's law: the exact expected authority, idle → pending → next idle; exhaustion refused at reservation |
| IMPORT_LEGACY | operator | a complete continuity v2/v3 record, verified by continuity's decoder, only into a fresh authority |
| RECONCILE_* , ENTITY_RECLAIM | operator | below |

Typed-evidence admission (`elpis_ecs_c_check_evidence`) is pure: the source must be an ACTIVE source at exactly
its current version and scope; the answer is its per-turn bound. Admission records nothing; a durable publication
records the evidence summary.

Each transition's receipt is SHA-256 over the lineage, the new and predecessor generations, the predecessor record
digest and state root, and the event bytes. The caller keeps receipts; ECS_C keeps only the chain head and count,
against which kept receipts can be verified (`elpis_ecs_c_receipt_chain`).

## Transition and recovery

A transition writes the complete next record into the slot that does not hold the current authority and syncs it
(the commit point), then writes the witness into the witness slot that does not hold the current pin and syncs it,
then acknowledges. A failed record write changes nothing (`PUBLICATION_REFUSED`, the handle stays usable). A failed
sync, or any failure after the record is durable, closes the handle (`PUBLICATION_UNCERTAIN`); reopening resolves to
exactly one authority.

Open reads the two record slots, the two witness slots and the key, classifies each slot EMPTY, TORN,
UNAUTHENTICATED, INVALID (authenticated but invalid) or VALID, takes the highest valid witness as the pin, and
decides as below.

**File sizes.**
- A slot or witness file shorter than its fixed size is a prefix and is classified TORN, like any other incomplete
  write. The transition that next rewrites it in place restores its size.
- A file longer than its fixed size is refused at open (`CORRUPT`): no in-place write produces one, and its length
  bounds nothing. So are a missing, extra, symlinked or swapped file.
- The capacity is the slots' common size. When a shortened slot leaves them none, it is the authenticated pin's.

| Situation | Disposition | Handle |
|---|---|---|
| the pinned record is present and its predecessor replays to it (or genesis recomputes) | CLEAN | normal |
| the pinned record is present, the other slot holds an incomplete write | TRAILING_INCOMPLETE | normal |
| a valid successor of the pin (the record synced, the witness not) | INTERRUPTED_COMPLETED: verified by replay, adopted, witness completed | normal |
| the pinned record is present, the other slot holds a checksummed record without a valid MAC | RESTRICTED_UNAUTHENTICATED | reads; only RECONCILE_UNAUTHENTICATED of exactly that slot and those bytes |
| the pinned record is present, its predecessor is not (zeroed, or a duplicate copy) | PREDECESSOR_UNAVAILABLE | normal (not replay-verified) |
| every authenticated record is older than the pin (copied-back files, a lying device) | ROLLBACK | candidate; only RECONCILE_ROLLBACK of exactly that pin |
| the pinned record is torn; an older authenticated one remains | CORRUPT_ACKNOWLEDGED | candidate; as above |
| the pinned record was replaced by an unauthenticated one | UNAUTHENTICATED_ACKNOWLEDGED | candidate; as above |
| no valid witness | WITNESS_MISSING | candidate; only RECONCILE_REPIN of exactly those witness classes and that record |
| authenticated records that are not one history (incl. another record at the pinned generation) | CONFLICT | refused |
| an authenticated record two or more generations ahead of the pin, or witnesses that disagree | WITNESS_INCONSISTENT | refused |
| an authenticated record that does not replay from its predecessor, or is invalid | REPLAY_MISMATCH | refused |
| no authenticated record | UNRECOVERABLE | refused |

The required recovery distinctions map as follows: complete valid → CLEAN; incomplete trailing → TRAILING_INCOMPLETE;
corrupt complete → CORRUPT_ACKNOWLEDGED; an interrupted transition → INTERRUPTED_COMPLETED (present once, never
duplicated; a retried request is `DUPLICATE`); lost sync → the older authority (CLEAN after `PUBLICATION_UNCERTAIN`),
or ROLLBACK when the device lost an acknowledged write; a partial cross-domain operation → `evolution_pending` in the
report (a reservation whose result must be established and finalized; never cleared automatically; the checkpoint /
authority pair of a durable publication is RuntimeCore's: the checkpoint slot first, then the record); conflicting candidate → CONFLICT; recovered older
than acknowledged → ROLLBACK; unauthorized higher-generation replacement → RESTRICTED_UNAUTHENTICATED or
UNAUTHENTICATED_ACKNOWLEDGED.

**Operator reconciliation** is explicit and exact: the reconciliation must name what recovery observed, and is itself
a transition (recorded in the reconciliation chain). RECONCILE_ROLLBACK adopts the newest authenticated older record
and continues at the abandoned generation + 1, so no generation is ever reused. RECONCILE_UNAUTHENTICATED overwrites
the substituted slot. RECONCILE_REPIN writes both witness slots for the newest authenticated record. Nothing is
reconciled automatically. In a reconcile-only or restricted mode, `elpis_ecs_c_read_reconciliation` reports the one
reconciliation the authority accepts: its event kind and every payload field it must name (the abandoned generation,
class and record digest of a rollback; the slot and observed bytes of an unauthenticated substitution; the witness
classes and current record of a repin). The operator still decides whether to send it, and who sends it.

## Provisioning and legacy import

Provisioning creates both directories owner-only (an existing authority directory must already be protected), draws
a fresh 32-byte deployment key and lineage nonce from the system's random source, and publishes the genesis record and
the first witness through temporary files; the existence of `witness.a` is the publication point. Before it,
provisioning may simply be repeated over its own debris; after it, `ALREADY_PROVISIONED`.

A standalone continuity register is imported explicitly: `elpis_ecs_c_legacy_record` reads its current record under
continuity's own lock (resolved exactly as continuity resolves it; the directory is never written), and an operator
IMPORT_LEGACY event carries the 176 bytes verbatim, so the import replays at recovery like any other transition and
records the legacy generation and record digest.

## Tests

* `ECS_C.cargo_test_authority` (Rust): HMAC RFC 4231 vectors; wire round trips and malformed events; deterministic
  genesis; the fixed file set and modes; provisioning interrupted at every step; protection and admission refusals;
  the lifecycle FSM, the authorization matrix, sequences, compare-and-publish, capacity and reclamation, source
  versions and scopes, the cognition lineage, the evolution law against continuity, exhaustion, rollback direction,
  legacy import (v2 and v3), receipts; every state invariant; the crash matrix (death at every step, record and
  witness writes torn at every boundary including inside the MAC and the checksum, failed and lying syncs, crashes
  while completing the witness at open, 300 random crash sequences); the T1 matrix (forged generation, forged
  evolution revision and head, a validly checksummed substitute, another deployment's record, a corrupt newest
  record, copied-back records, a lying device, swapped and duplicated slots, witness loss) and the key-holder cases
  that must refuse (conflicts, replay mismatches, invalid authenticated records, inconsistent witnesses); boundedness
  over long sequences. The suite was checked by 25 targeted mutations of the transition law, the record validation,
  recovery and admission; 24 fail a test, and the survivor is an equivalent mutant (a mixed-lineage pre-check that the
  pin and predecessor-binding checks enforce again).
* `ECS_C.test_ecs_c_authority_abi` and `_testing` (C): the ABI end to end through both libraries, with one crash
  through the C fault hooks.
* `tests/boundary/test_ecs_c_authority.py`: the header, the FFI and the ABI inventory declare one ABI; the
  persistence surface is the fixed file set; sizes agree. `tests/boundary/test_abi_inventory.py` checks the
  binaries' exports, SONAMEs and Elpis links.

## Not claimed

No qualification beyond ECS_C Recovery Q0's scope (commit `117b41a` only; `SIGKILL`, never power loss; the T1
adversary); no protection beyond the threat model; no scientific claim of any kind.
