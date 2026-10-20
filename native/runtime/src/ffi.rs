//! The stable C ABI (include/elpis/runtime.h).
//!
//! Every function returns 0 or a positive stable code: a durable-authority code (1..=17, 255; `CONTINUITY_*`, the names
//! ABI v1 gave them) or a RuntimeCore code (64..). A runtime handle serializes its calls internally (one lock per handle), except the durability lane's
//! wait and status reads, which take only the lane's lock: they never wait behind a HOT operation, and a HOT
//! operation never waits behind them.
//!
//! ABI v4 (docs/RUNTIME_CORE.md, "Live and durable authority"): a managed commit is volatile (no checkpoint, no
//! continuity publication); durable publication is an explicit request of one of two kinds, never confused:
//! `elpis_runtime_recoverable_request` (the complete state; needs an attached checkpoint store; ACKNOWLEDGED only
//! with a read-back verified slot) or `elpis_runtime_identity_request` (only the identity; no store attached;
//! IDENTITY_PUBLISHED, never ACKNOWLEDGED), served by the durability lane (`elpis_runtime_durable_wait`,
//! `elpis_runtime_durable_status`). QUERY reports the live
//! generation instead of a digest. The substrate tables carry the live-coordinate entries (ecsg_k1.h). A v3 caller
//! refuses to load the library (the version check), so no caller can mistake a volatile commit for a durable one.
//!
//! ABI v5 (docs/RUNTIME_CORE.md, "Observe-only turns"): `elpis_runtime_turn_begin_observe` opens a managed turn on a
//! composite substrate that runs no K1 schedule, so typed memory observations need no geometric learning; the fuel
//! budget gains `max_observations` (every managed turn's observation bound) and the counters `k1_observes`. Every
//! v4 entry point keeps its meaning; the envelope and continuity formats are unchanged. A v4 caller refuses to load
//! this library (the version check), so no caller reads the v5 budget or counters with the v4 layout.
//!
//! ABI v6 (docs/RUNTIME_CORE.md, "The durable authority"): a second durable-authority backend, the ECS_C authority
//! (`elpis_runtime_create_ecs_c`), with native typed-evidence admission, evidence summaries and operator events.
//!
//! ABI v7: the ECS_C authority is the only one. `elpis_runtime_create` (standalone continuity) is gone, and so is the
//! re-export of continuity's own C ABI (`elpis_continuity_*`); standalone continuity retired into ECS_C
//! (docs/NATIVE_SPLIT_CENSUS.md, section 9). The snapshot and evolution types and the durable-authority codes keep
//! their layouts, values and names (`elpis_continuity_snapshot`, `ELPIS_CONTINUITY_*`), now declared by runtime.h
//! itself. A v6 caller refuses to load this library (the version check), so none creates a continuity runtime.
//!
//! # Safety
//!
//! Every pointer argument is either NULL (refused with `RUNTIME_INVALID`, or the documented meaning) or valid
//! for its declared size: 32 bytes for a digest, `len` elements for a buffer, one struct otherwise. Runtime
//! handles come from `elpis_runtime_create_ecs_c` and are used until `elpis_runtime_destroy`. A substrate
//! descriptor's handle and function table must be live for the duration of the call; the table is copied.
//!
//! Managed ownership (ABI v3): RuntimeCore claims every K1 state it binds with its own lease (ecsg_k1.h "Managed
//! ownership"), so the bound state refuses every unmanaged mutation, and re-establishes the state's exact
//! retained-state identity inside every managed transaction begin. `elpis_runtime_release` gives the state back.
//!
//! Substrate lifetime (ABI v2): a successful `elpis_runtime_turn_begin` opens a native K1 transaction that
//! RuntimeCore ends with exactly one terminal native action, the commit or an abort, before it forgets the turn.
//! To end it on every path (commit, abort, a refused commit, close, reopen, destroy) it retains the descriptor's
//! handle, resident id and `txn_abort` entry from that begin until the turn ends. The native state must stay live,
//! and its library loaded, for that whole interval. The K1 FMS adapter enforces this itself (a resident state with
//! an open transaction cannot be closed, and its runtime cannot be destroyed while a state is registered); a
//! standalone K1 state must not be destroyed by its owner while a turn on it is open.
#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, OsStr};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::authority::{Cognition, Digest, EvolutionState, Snapshot, ZERO};
use crate::code::{Code, Error, Rt};
use crate::core::{Core, Counters, Recovery, Refusal, Stimulus, VolatileCommit};
use crate::durable::{Kind, Lane, Status, Ticket};
use crate::fuel::Budget;
use crate::substrate::{features, CSubstrate, Experience, Native, ScheduleOutcome, Transition};

pub const ABI_VERSION: u32 = 7;

/// `elpis_continuity_evolution` (runtime.h): the evolution authority (80 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CEvolution {
    pub revision: u64,
    pub head: Digest,
    pub pending: u8,
    pub reserved: [u8; 7],
    pub assertion: Digest,
}

/// `elpis_continuity_snapshot` (runtime.h): one verified durable authority (192 bytes). `k1_state_digest` keeps its
/// historical name; `state_format` says what it identifies. The two digests are outputs.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CSnapshot {
    pub generation: u64,
    pub anchored: u8,
    /// `StateFormat` of the anchored digest: 0 K1 (`ELPISGK1`), 1 composite (`ELPISEC1`).
    pub state_format: u8,
    pub reserved: [u8; 6],
    pub k1_state_digest: Digest,
    pub evolution: CEvolution,
    pub evolution_digest: Digest,
    pub record_digest: Digest,
}

const _: () = assert!(std::mem::size_of::<CEvolution>() == 80);

pub fn to_c_evolution(e: &EvolutionState) -> CEvolution {
    CEvolution {
        revision: e.revision(),
        head: e.head(),
        pending: e.assertion().is_some() as u8,
        reserved: [0; 7],
        assertion: e.assertion().unwrap_or(ZERO),
    }
}

/// The C form of a snapshot: its record digest is the continuity register record's (`register::Snapshot::digest`).
pub fn to_c(s: &Snapshot) -> CSnapshot {
    let (anchored, k1) = match s.cognition() {
        Cognition::Unanchored => (0, ZERO),
        Cognition::Anchored(d) => (1, d),
    };
    CSnapshot {
        generation: s.generation(),
        anchored,
        state_format: s.format() as u8,
        reserved: [0; 6],
        k1_state_digest: k1,
        evolution: to_c_evolution(&s.evolution()),
        evolution_digest: s.evolution().digest(),
        record_digest: s.digest(),
    }
}

/// A well-formed evolution authority, or `Corrupt` (malformed flag or reserved bytes, an idle authority carrying an
/// assertion, counter bounds or the genesis rule).
pub fn from_c_evolution(e: &CEvolution) -> Result<EvolutionState, Code> {
    if e.reserved != [0; 7] {
        return Err(Code::Corrupt);
    }
    let state = match e.pending {
        0 if e.assertion == ZERO => EvolutionState::idle(e.revision, e.head),
        1 => EvolutionState::pending(e.revision, e.head, e.assertion),
        _ => return Err(Code::Corrupt),
    };
    state.map_err(|_| Code::Corrupt)
}

/// Opaque runtime handle: the HOT state machine behind its lock, and the durability lane its wait and status
/// reads reach without that lock.
pub struct ElpisRuntime {
    core: Mutex<Core>,
    lane: Arc<Lane>,
}

/// `elpis_runtime_turn_begin_result`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CTurnBegin {
    pub schedule: ScheduleOutcome,
    /// The K1 status behind an `ECS_*` refusal (0 otherwise).
    pub k1_status: i32,
    pub reserved: u32,
}

/// `elpis_runtime_volatile_commit` (ABI v4): a managed commit is volatile.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CVolatileCommit {
    /// The native transition: epochs, generations (before/after), steps.
    pub transition: Transition,
    /// The live generation the commit established (= `transition.generation_after`; 0 when not committed).
    pub live_generation: u64,
    /// 1 when the native commit happened, else 0.
    pub committed: u32,
    /// The K1 status behind an `ECS_*` refusal (0 otherwise).
    pub k1_status: i32,
}

/// `elpis_runtime_query_result` (ABI v4).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CQuery {
    /// The live generation the answer was computed from (0 on refusal).
    pub live_generation: u64,
    /// The K1 status behind an `ECS_*` refusal (0 otherwise).
    pub k1_status: i32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<CQuery>() == 16);

/// `elpis_runtime_observe_begin_result` (ABI v5).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CObserveBegin {
    /// The live generation the observe-only turn was opened on (0 on refusal).
    pub live_generation: u64,
    /// The composite status behind an `ECS_*` refusal (0 otherwise).
    pub k1_status: i32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<CObserveBegin>() == 16);

/// `elpis_runtime_durability`: the durability lane as a status read sees it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CDurability {
    pub last_requested: Ticket,
    pub last_acknowledged: Ticket,
    /// The request the worker is running (zero sequence: none).
    pub running: Ticket,
    /// The live generation of the bound state (diagnostics mirror).
    pub live_generation: u64,
    /// 1 while a captured request waits for the worker.
    pub pending: u32,
    /// The lane's sticky fault (0: none): an uncertain publication or a simulated death, until reopen.
    pub fault: i32,
    /// Testing builds: the step the worker is parked at (0: none).
    pub parked: u32,
    /// 1 when `durable` holds the current durable authority (0 while closed or faulted).
    pub durable_known: u32,
    /// The attached checkpoint store's envelope capacity (0: none attached).
    pub checkpoint_capacity: u64,
    pub durable: CSnapshot,
}

const _: () = assert!(std::mem::size_of::<CSnapshot>() == 192);
const _: () = assert!(std::mem::size_of::<CDurability>() == 64 * 3 + 32 + 192);

/// `elpis_runtime_observe_result`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CObserve {
    /// The observation's relation key (zero on refusal).
    pub relation_key: Digest,
    /// The composite status behind an `ECS_*` refusal (0 otherwise).
    pub k1_status: i32,
    /// The memory's own status behind a composite REFUSED / UNQUALIFIED / INVALID / CAPACITY (0 otherwise).
    pub memory_status: i32,
}
const _: () = assert!(std::mem::size_of::<CObserve>() == 40);

/// `elpis_runtime_recovery`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CRecovery {
    /// 1 NOTHING_TO_RECOVER, 2 RESUMABLE, 3 CANDIDATE_UNRESOLVED, 4 CHECKPOINT_MISSING.
    pub disposition: u32,
    /// 1 when a verified slot holds the authorized identity.
    pub authorized_present: u32,
    pub authorized: Digest,
    pub candidate: Digest,
    /// The authorized envelope's size when resumable (copied when the caller's buffer holds it), else 0.
    pub envelope_bytes: u64,
}

const _: () = assert!(std::mem::size_of::<CRecovery>() == 80);
const _: () = assert!(std::mem::size_of::<CTurnBegin>() == 48);
const _: () = assert!(std::mem::size_of::<CVolatileCommit>() == 64);
const _: () = assert!(std::mem::size_of::<Counters>() == 136);
const _: () = assert!(std::mem::size_of::<Budget>() == 56);

fn rc(result: Result<(), Error>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

unsafe fn runtime<'a>(p: *mut ElpisRuntime) -> Result<MutexGuard<'a, Core>, Error> {
    let rt = p.as_ref().ok_or(Error::Runtime(Rt::Invalid))?;
    // A panic aborts the process (panic = "abort"), so the lock is never left poisoned by a live caller.
    Ok(rt.core.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The durability lane of a handle, without the handle's lock.
unsafe fn lane<'a>(p: *mut ElpisRuntime) -> Result<&'a Arc<Lane>, Error> {
    Ok(&p.as_ref().ok_or(Error::Runtime(Rt::Invalid))?.lane)
}

unsafe fn snapshot_out(out: *mut CSnapshot, s: &Snapshot) {
    if !out.is_null() {
        *out = to_c(s);
    }
}

unsafe fn digest<'a>(p: *const u8) -> Result<&'a Digest, Error> {
    (p as *const Digest).as_ref().ok_or(Error::Runtime(Rt::Invalid))
}

unsafe fn substrate(d: *const CSubstrate) -> Result<Native, Error> {
    let d = d.as_ref().ok_or(Error::Runtime(Rt::Invalid))?;
    Native::from_c(d).ok_or(Error::Runtime(Rt::Invalid))
}

unsafe fn slice<'a, T>(p: *const T, len: usize) -> Result<&'a [T], Error> {
    if p.is_null() {
        return Err(Rt::Invalid.into());
    }
    Ok(std::slice::from_raw_parts(p, len))
}

#[no_mangle]
pub extern "C" fn elpis_runtime_abi_version() -> u32 {
    ABI_VERSION
}

/// The S3 readout length for an input dimension (`elpis_ecsg_k1_features`), 0 outside 1..=64.
#[no_mangle]
pub extern "C" fn elpis_runtime_features(dim: usize) -> usize {
    features(dim)
}

/// The canonical cognitive fuel ceiling (`elpis_runtime_budget`), compiled in; every budget must narrow it.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_fuel_ceiling(out: *mut Budget) -> i32 {
    match out.as_mut() {
        Some(out) => {
            *out = crate::fuel::CEILING;
            0
        }
        None => Rt::Invalid as i32,
    }
}

/// ECS work units of an experience schedule on a `dim x width` state (docs/RUNTIME_CORE.md, fuel). Refused
/// (`RUNTIME_INVALID`) for an invalid shape or schedule, or a total beyond `u64`.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_work_units(
    dim: usize,
    width: usize,
    schedule: *const Experience,
    experiences: usize,
    out: *mut u64,
) -> i32 {
    let run = || -> Result<u64, Error> {
        let schedule = slice(schedule, experiences)?;
        crate::fuel::schedule_units(dim, width, schedule).ok_or(Error::Runtime(Rt::Invalid))
    };
    match (run(), out.as_mut()) {
        (Ok(units), Some(out)) => {
            *out = units;
            0
        }
        (Err(e), _) => e.code(),
        (_, None) => Rt::Invalid as i32,
    }
}

/// ECS work units of a QUERY of `rows` rows on a `dim x width` state.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_query_work_units(dim: usize, width: usize, rows: usize, out: *mut u64) -> i32 {
    match (crate::fuel::query_units(dim, width, rows), out.as_mut()) {
        (Some(units), Some(out)) => {
            *out = units;
            0
        }
        _ => Rt::Invalid as i32,
    }
}

/// The stable name of any code this library returns (`RUNTIME_OK` for 0), or NULL. Static storage.
#[no_mangle]
pub extern "C" fn elpis_runtime_code_name(code: i32) -> *const c_char {
    if code == 0 {
        return c"RUNTIME_OK".as_ptr();
    }
    match Rt::from_i32(code) {
        Some(r) => match r {
            Rt::Invalid => c"RUNTIME_INVALID",
            Rt::Closed => c"RUNTIME_CLOSED",
            Rt::SubstrateSwitch => c"COGNITION_SUBSTRATE_SWITCH",
            Rt::EcsState => c"ECS_STATE",
            Rt::EcsRefused => c"ECS_REFUSED",
            Rt::EcsStale => c"ECS_STALE",
            Rt::TurnOpen => c"RUNTIME_TURN_OPEN",
            Rt::TurnNotOpen => c"RUNTIME_TURN_NOT_OPEN",
            Rt::EvolutionInFlight => c"RUNTIME_EVOLUTION_IN_FLIGHT",
            Rt::EvolutionNotInFlight => c"RUNTIME_EVOLUTION_NOT_IN_FLIGHT",
            Rt::Fuel => c"COGNITION_FUEL_EXCEEDED",
            Rt::CheckpointInvalid => c"CHECKPOINT_INVALID",
            Rt::CheckpointIo => c"CHECKPOINT_IO",
            Rt::CheckpointShape => c"CHECKPOINT_SHAPE",
            Rt::CheckpointUnresolved => c"CHECKPOINT_UNRESOLVED",
            Rt::DurableBusy => c"RUNTIME_DURABLE_BUSY",
            Rt::DurablePending => c"RUNTIME_DURABLE_PENDING",
            Rt::RecoveryUnavailable => c"RUNTIME_RECOVERY_UNAVAILABLE",
            Rt::CheckpointAttached => c"RUNTIME_CHECKPOINT_ATTACHED",
            Rt::NotResident => c"RUNTIME_NOT_RESIDENT",
            Rt::Authority => c"RUNTIME_AUTHORITY",
            Rt::ObservationRefused => c"RUNTIME_OBSERVATION_REFUSED",
        }
        .as_ptr(),
        None => match Code::from_i32(code) {
            Some(c) => c.name().as_ptr(),
            None => std::ptr::null(),
        },
    }
}

// -- lifecycle ---------------------------------------------------------------------------------------------

/// Create a closed runtime over a provisioned ECS_C authority (`docs/ECS_C_AUTHORITY.md`; ABI v6): an absolute state
/// directory and its protected authority directory (`*_len` bytes each, no terminator needed). Provisioning and
/// operator reconciliation are the authority's own (`elpis/ecs_c_authority.h`), performed with no runtime open.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_create_ecs_c(
    state_dir: *const u8,
    state_len: usize,
    authority_dir: *const u8,
    authority_len: usize,
    out: *mut *mut ElpisRuntime,
) -> i32 {
    if state_dir.is_null() || authority_dir.is_null() || out.is_null() {
        return Rt::Invalid as i32;
    }
    *out = std::ptr::null_mut();
    let state = Path::new(OsStr::from_bytes(std::slice::from_raw_parts(state_dir, state_len)));
    let authority = Path::new(OsStr::from_bytes(std::slice::from_raw_parts(authority_dir, authority_len)));
    rc(Core::new_ecs_c(state, authority).map(|core| {
        let lane = core.lane();
        *out = Box::into_raw(Box::new(ElpisRuntime { core: Mutex::new(core), lane }))
    }))
}

/// `elpis_runtime_authority` (ABI v6).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CAuthority {
    /// `ELPIS_RUNTIME_BACKEND_ECS_C` (2), the only backend since ABI v7 (v6 also had 1, standalone continuity).
    pub backend: u32,
    /// The last ECS_C refusal (`ELPIS_ECS_C_*`; 0 none).
    pub ecs_c_code: i32,
    /// The ECS_C recovery disposition and mode of the last open.
    pub disposition: u32,
    pub mode: u32,
    pub replay_verified: u8,
    pub witness_completed: u8,
    pub reserved: [u8; 6],
    /// The authority's generation.
    pub generation: u64,
    pub receipt_count: u64,
    pub evidence_count: u64,
    pub entity_count: u32,
    pub capacity: u32,
    pub lineage: Digest,
    pub record_digest: Digest,
    pub state_root: Digest,
    pub receipt_head: Digest,
    pub evidence_head: Digest,
}

const _: () = assert!(std::mem::size_of::<CAuthority>() == 216);

/// `elpis_runtime_authority_entity` (ABI v6, `elpis_runtime_authority_entity_at`): one entity row of the held ECS_C authority, laid out as
/// `elpis_ecs_c_entity`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CAuthorityEntity {
    pub id: Digest,
    pub kind: u8,
    pub lifecycle: u8,
    pub reserved: [u8; 2],
    pub per_turn_limit: u32,
    pub founding_index: u64,
    pub founded_generation: u64,
    pub last_generation: u64,
    pub sender_watermark: u64,
    pub source_version: u64,
    pub label: Digest,
    pub scope: Digest,
    pub source_authority: u32,
    pub reserved2: u32,
}

const _: () = assert!(std::mem::size_of::<CAuthorityEntity>() == 152);

/// ABI v6: the durable authority's backend and, under ECS_C, its recovery report, current record and last refusal.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_authority_status(rt: *mut ElpisRuntime, out: *mut CAuthority) -> i32 {
    rc((|| {
        let s = runtime(rt)?.authority_status()?;
        let out = out.as_mut().ok_or(Error::Runtime(Rt::Invalid))?;
        *out = CAuthority {
            backend: s.backend,
            ecs_c_code: s.ecs_c_code,
            disposition: s.disposition,
            mode: s.mode,
            replay_verified: s.replay_verified as u8,
            witness_completed: s.witness_completed as u8,
            reserved: [0; 6],
            generation: s.generation,
            receipt_count: s.receipt_count,
            evidence_count: s.evidence_count,
            entity_count: s.entity_count,
            capacity: s.capacity,
            lineage: s.lineage,
            record_digest: s.record_digest,
            state_root: s.state_root,
            receipt_head: s.receipt_head,
            evidence_head: s.evidence_head,
        };
        Ok(())
    })())
}

/// ABI v6, ECS_C authority only: one operator event (512 bytes, `elpis/ecs_c_authority.h`: ENTITY_FOUND,
/// ENTITY_LIFECYCLE, SOURCE_VERSION, ENTITY_RECLAIM, IMPORT_LEGACY) while this runtime holds the authority; the
/// typed-evidence admission table follows it. `out_generation` (optional) receives the new authority generation.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_authority_transition(
    rt: *mut ElpisRuntime,
    event: *const u8,
    out_generation: *mut u64,
) -> i32 {
    rc((|| {
        if event.is_null() {
            return Err(Rt::Invalid.into());
        }
        let event = elpis_ecs_c_authority::Event::decode(std::slice::from_raw_parts(event, 512))
            .map_err(|_| Error::Runtime(Rt::Invalid))?;
        let receipt = runtime(rt)?.authority_transition(&event).map_err(|r| r.error)?;
        if let Some(out) = out_generation.as_mut() {
            *out = receipt.generation;
        }
        Ok(())
    })())
}

/// ABI v6: the held ECS_C authority's entity row at `index` (what an operator event is built from: identities,
/// lifecycles, sender watermarks). `RUNTIME_INVALID` under continuity and past the last row; refused closed.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_authority_entity_at(
    rt: *mut ElpisRuntime,
    index: u32,
    out: *mut CAuthorityEntity,
) -> i32 {
    rc((|| {
        let out = out.as_mut().ok_or(Error::Runtime(Rt::Invalid))?;
        let e = runtime(rt)?.authority_entity(index)?;
        *out = CAuthorityEntity {
            id: e.id,
            kind: e.kind as u8,
            lifecycle: e.lifecycle as u8,
            reserved: [0; 2],
            per_turn_limit: e.per_turn_limit,
            founding_index: e.founding_index,
            founded_generation: e.founded_generation,
            last_generation: e.last_generation,
            sender_watermark: e.sender_watermark,
            source_version: e.source_version,
            label: e.label,
            scope: e.scope,
            source_authority: e.source_authority,
            reserved2: 0,
        };
        Ok(())
    })())
}

/// ABI v6: the identity the next ENTITY_FOUND of `kind` and `label` founds on the held ECS_C authority (the
/// receiver that event must name). `RUNTIME_INVALID` under continuity and for an unknown kind; refused closed.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_authority_next_entity_id(
    rt: *mut ElpisRuntime,
    kind: u8,
    label: *const u8,
    out: *mut u8,
) -> i32 {
    rc((|| {
        let label = digest(label)?;
        let out = (out as *mut Digest).as_mut().ok_or(Error::Runtime(Rt::Invalid))?;
        *out = runtime(rt)?.authority_next_entity_id(kind, label)?;
        Ok(())
    })())
}

/// Close (if open) and free a runtime; sets `*rt` to NULL. An open managed turn is aborted natively first; a pending
/// durable request is abandoned and a running one completes before the worker stops.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_destroy(rt: *mut *mut ElpisRuntime) {
    if !rt.is_null() && !(*rt).is_null() {
        drop(Box::from_raw(*rt));
        *rt = std::ptr::null_mut();
    }
}

/// Open, or reopen after close (a fail-stop is cleared by close + open): resolves the current authority, binds
/// nothing. An already open runtime is refused (`CONTINUITY_OPEN`) and keeps its open turn.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_open(rt: *mut ElpisRuntime, out: *mut CSnapshot) -> i32 {
    rc(runtime(rt).and_then(|mut core| core.open()).map(|s| snapshot_out(out, &s)))
}

/// Close (a closed runtime is unchanged). An open managed turn is aborted natively first: nothing installed.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_close(rt: *mut ElpisRuntime) {
    if let Ok(mut core) = runtime(rt) {
        core.close();
    }
}

/// The fail-stop disposition (0: none). A closed or NULL runtime reports `RUNTIME_CLOSED`/`RUNTIME_INVALID`.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_fault(rt: *mut ElpisRuntime) -> i32 {
    match runtime(rt) {
        Err(e) => e.code(),
        Ok(core) if !core.is_open() => Rt::Closed as i32,
        Ok(core) => core.fault().map_or(0, |c| c as i32),
    }
}

/// The current durable authority: the last durably acknowledged record (readable while fail-stopped; refused with
/// the lane's fault while the durable outcome is unknown). The live state may be ahead of it.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_snapshot(rt: *mut ElpisRuntime, out: *mut CSnapshot) -> i32 {
    if out.is_null() {
        return Rt::Invalid as i32;
    }
    rc(runtime(rt).and_then(|core| core.snapshot()).map(|s| snapshot_out(out, &s)))
}

/// Copy RuntimeCore's crossing counters; reset them when `reset` is nonzero.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_read_counters(rt: *mut ElpisRuntime, out: *mut Counters, reset: i32) -> i32 {
    if out.is_null() {
        return Rt::Invalid as i32;
    }
    rc(runtime(rt).map(|mut core| *out = core.counters(reset != 0)))
}

// -- K1 lineage and the managed turn ------------------------------------------------------------------------

/// Explicitly anchor the first managed K1 lineage at the substrate's retained identity (reads K1 only).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_anchor(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    out: *mut CSnapshot,
) -> i32 {
    let run = || -> Result<Snapshot, Error> {
        let mut native = substrate(sub)?;
        let key = native.key();
        runtime(rt)?.anchor(&mut native, key).map_err(|r| r.error)
    };
    rc(run().map(|s| snapshot_out(out, &s)))
}

/// QUERY: answer `f_W(x)` for `rows` rows of the descriptor's dimension (`x_len = rows * dim`) from the lineage's
/// live K1 state, read-only: no transaction, no commit, no publication, no digest, no I/O. `out` receives `rows`
/// values; `result->live_generation` the live generation they were computed from (the call is refused and `out`
/// zeroed otherwise).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_query(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    x: *const f64,
    x_len: usize,
    out: *mut f64,
    rows: usize,
    budget: *const Budget,
    result: *mut CQuery,
) -> i32 {
    let run = || -> Result<u64, Refusal> {
        let mut native = substrate(sub)?;
        let key = native.key();
        let x = slice(x, x_len)?;
        let budget = budget.as_ref().ok_or(Refusal::from(Rt::Invalid))?;
        if out.is_null() {
            return Err(Rt::Invalid.into());
        }
        let out = std::slice::from_raw_parts_mut(out, rows);
        runtime(rt)?.query(&mut native, key, x, out, budget)
    };
    let (value, code) = match run() {
        Ok(live_generation) => (CQuery { live_generation, k1_status: 0, reserved: 0 }, 0),
        Err(r) => (CQuery { k1_status: r.k1_status, ..CQuery::default() }, r.error.code()),
    };
    if !result.is_null() {
        *result = value;
    }
    code
}

/// Begin an OBSERVE-ONLY managed turn on a composite substrate (ABI v5): the lineage is verified and bound as for
/// `elpis_runtime_turn_begin`, the composite transaction is opened at the live generation, and no K1 schedule runs.
/// Observations then enter through `elpis_runtime_turn_observe` (at most `budget->max_observations`); the turn ends
/// with `elpis_runtime_turn_commit` (volatile; refused `RUNTIME_INVALID`, the turn kept open, while nothing was
/// observed) or `elpis_runtime_turn_abort`. A K1 or K1 FMS substrate is refused (`RUNTIME_INVALID`).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_turn_begin_observe(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    budget: *const Budget,
    out: *mut CObserveBegin,
) -> i32 {
    let run = || -> Result<u64, Refusal> {
        let mut native = substrate(sub)?;
        let key = native.key();
        let budget = budget.as_ref().ok_or(Refusal::from(Rt::Invalid))?;
        runtime(rt)?.turn_begin_observe(&mut native, key, budget)
    };
    let (value, code) = match run() {
        Ok(live_generation) => (CObserveBegin { live_generation, k1_status: 0, reserved: 0 }, 0),
        Err(r) => (CObserveBegin { k1_status: r.k1_status, ..CObserveBegin::default() }, r.error.code()),
    };
    if !out.is_null() {
        *out = value;
    }
    code
}

unsafe fn refusal_begin(out: *mut CTurnBegin, r: &Refusal) -> i32 {
    if !out.is_null() {
        *out = CTurnBegin { schedule: r.schedule, k1_status: r.k1_status, reserved: 0 };
    }
    r.error.code()
}

/// Verify the K1 lineage, begin the native transaction and run the whole experience schedule on its candidate
/// (one native call). `x` holds `y_len` rows of the descriptor's dimension (`x_len = y_len * dim`); `s3_out`
/// receives the readout (`s3_len = elpis_runtime_features(dim)`). The turn stays open for commit or abort.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_turn_begin(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    x: *const f64,
    x_len: usize,
    y: *const f64,
    y_len: usize,
    schedule: *const Experience,
    experiences: usize,
    learning_rate: f64,
    budget: *const Budget,
    s3_out: *mut f64,
    s3_len: usize,
    out: *mut CTurnBegin,
) -> i32 {
    let run = || -> Result<ScheduleOutcome, Refusal> {
        let mut native = substrate(sub)?;
        let key = native.key();
        let budget = budget.as_ref().ok_or(Refusal::from(Rt::Invalid))?;
        let stimulus = Stimulus {
            x: slice(x, x_len)?,
            y: slice(y, y_len)?,
            schedule: slice(schedule, experiences)?,
            rate: learning_rate,
        };
        if s3_out.is_null() {
            return Err(Rt::Invalid.into());
        }
        let s3 = std::slice::from_raw_parts_mut(s3_out, s3_len);
        runtime(rt)?.turn_begin(&mut native, key, &stimulus, budget, s3)
    };
    match run() {
        Ok(schedule) => {
            if !out.is_null() {
                *out = CTurnBegin { schedule, k1_status: 0, reserved: 0 };
            }
            0
        }
        Err(r) => refusal_begin(out, &r),
    }
}

/// Commit the open turn natively: a VOLATILE commit (ABI v4). The candidate is installed and the live generation
/// advances by one; nothing is checkpointed, published or written. Durability is an explicit request
/// (`elpis_runtime_recoverable_request` or `elpis_runtime_identity_request`).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_turn_commit(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    out: *mut CVolatileCommit,
) -> i32 {
    let run = || -> Result<VolatileCommit, Refusal> {
        let mut native = substrate(sub)?;
        let key = native.key();
        runtime(rt)?.turn_commit(&mut native, key)
    };
    let (result, code) = match run() {
        Ok(c) => (
            CVolatileCommit {
                transition: c.transition,
                live_generation: c.live_generation,
                committed: 1,
                k1_status: 0,
            },
            0,
        ),
        Err(r) => (
            CVolatileCommit {
                transition: r.committed.unwrap_or_default(),
                live_generation: 0,
                committed: r.committed.is_some() as u32,
                k1_status: r.k1_status,
            },
            r.error.code(),
        ),
    };
    if !out.is_null() {
        *out = result;
    }
    code
}

// -- durable publication: the explicit WARM lane -----------------------------------------------------------------

/// Request RECOVERABLE durable publication of the live state of the described (bound, or bindable) substrate: the
/// complete envelope in the attached checkpoint store (written, synced, read back verified), then continuity. The
/// envelope is captured unsealed under the lease (the one synchronous step) and handed to the durability lane;
/// returns at once with `out` a PENDING ticket. ACKNOWLEDGED (always `recoverable` 1) means restart resumes exactly
/// this state. Refused with `RUNTIME_RECOVERY_UNAVAILABLE` (nothing bound or captured) unless a checkpoint store is
/// attached. A request that finds an earlier one still waiting supersedes it. Refused while a turn is open, on an
/// unanchored lineage, and with the lane's fault while it is faulted.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_recoverable_request(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    out: *mut Ticket,
) -> i32 {
    request(rt, sub, out, Kind::Recoverable)
}

/// Request IDENTITY-ONLY durable publication: continuity names the live state's identity; no store holds its bytes
/// (the caller keeps them). Resolves IDENTITY_PUBLISHED (always `recoverable` 0), never ACKNOWLEDGED. Refused with
/// `RUNTIME_CHECKPOINT_ATTACHED` (nothing bound or captured) while a checkpoint store is attached. Otherwise as
/// `elpis_runtime_recoverable_request`.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_identity_request(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    out: *mut Ticket,
) -> i32 {
    request(rt, sub, out, Kind::Identity)
}

unsafe fn request(rt: *mut ElpisRuntime, sub: *const CSubstrate, out: *mut Ticket, kind: Kind) -> i32 {
    let run = || -> Result<Ticket, Refusal> {
        let mut native = substrate(sub)?;
        let key = native.key();
        runtime(rt)?.durable_request(&mut native, key, kind)
    };
    match run() {
        Ok(t) => {
            if !out.is_null() {
                *out = t;
            }
            0
        }
        Err(r) => {
            if !out.is_null() {
                *out = Ticket::default();
            }
            r.error.code()
        }
    }
}

/// Wait (the caller's explicit choice) until request `sequence` resolves, at most `timeout_ms` milliseconds
/// (`UINT64_MAX`: without limit). Takes only the lane's lock: HOT operations proceed meanwhile. Returns 0 with the
/// resolved ticket, `RUNTIME_DURABLE_PENDING` with the still-pending ticket after a timeout, `RUNTIME_INVALID` for a
/// sequence never issued.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_durable_wait(
    rt: *mut ElpisRuntime,
    sequence: u64,
    timeout_ms: u64,
    out: *mut Ticket,
) -> i32 {
    let timeout = (timeout_ms != u64::MAX).then(|| Duration::from_millis(timeout_ms));
    match lane(rt).and_then(|l| l.wait(sequence, timeout)) {
        Ok(t) => {
            if !out.is_null() {
                *out = t;
            }
            if t.status() == Status::Pending {
                Rt::DurablePending as i32
            } else {
                0
            }
        }
        Err(e) => e.code(),
    }
}

/// The durability lane's state (lane lock only): the last requested and acknowledged tickets, the running one, the
/// live generation, the lane fault and the durable authority.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_durable_status(rt: *mut ElpisRuntime, out: *mut CDurability) -> i32 {
    if out.is_null() {
        return Rt::Invalid as i32;
    }
    match lane(rt) {
        Ok(l) => {
            let r = l.report();
            let durable = r.durable.unwrap_or(Snapshot::GENESIS);
            *out = CDurability {
                last_requested: r.last_requested,
                last_acknowledged: r.last_acknowledged,
                running: r.running,
                live_generation: r.live_generation,
                pending: r.pending,
                fault: r.fault,
                parked: r.parked,
                durable_known: r.durable.is_some() as u32,
                checkpoint_capacity: r.capacity,
                durable: to_c(&durable),
            };
            0
        }
        Err(e) => e.code(),
    }
}

/// The bound state's retained-state identity and live generation, read on request (cold path: one guarded native
/// call that hashes the state). Binds the state first if needed (verified against the durable identity).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_live_identity(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    digest: *mut u8,
    generation: *mut u64,
) -> i32 {
    if digest.is_null() || generation.is_null() {
        return Rt::Invalid as i32;
    }
    let run = || -> Result<(Digest, u64), Refusal> {
        let mut native = substrate(sub)?;
        let key = native.key();
        runtime(rt)?.live_identity(&mut native, key)
    };
    match run() {
        Ok((d, g)) => {
            std::ptr::copy_nonoverlapping(d.as_ptr(), digest, 32);
            *generation = g;
            0
        }
        Err(r) => r.error.code(),
    }
}

/// OBSERVE inside the open turn of a composite substrate (`ELPIS_RUNTIME_SUBSTRATE_COMPOSITE`): `observation` is the
/// caller's `elpis_ecs_mem_observation`, passed through unread. The observation becomes authoritative only with the
/// turn's commit; a refusal leaves the turn open and its candidate unchanged.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_turn_observe(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    observation: *const std::ffi::c_void,
    out: *mut CObserve,
) -> i32 {
    let (value, code) = match (|| -> Result<(Native, crate::substrate::Key), Error> {
        let native = substrate(sub)?;
        let key = native.key();
        Ok((native, key))
    })() {
        Err(e) => (CObserve::default(), e.code()),
        Ok((mut native, key)) => match runtime(rt) {
            Err(e) => (CObserve::default(), e.code()),
            Ok(mut core) => match core.turn_observe(&mut native, key, crate::substrate::Observation(observation)) {
                (Ok(relation_key), _) => (CObserve { relation_key, k1_status: 0, memory_status: 0 }, 0),
                (Err(r), memory_status) => {
                    (CObserve { relation_key: [0; 32], k1_status: r.k1_status, memory_status }, r.error.code())
                }
            },
        },
    };
    if !out.is_null() {
        *out = value;
    }
    code
}

/// LEGACY: move the open runtime's K1 lineage to the composite substrate made from exactly its authorized K1 state
/// with an empty memory (`elpis_ecs_composite_import_k1`). Requires an attached checkpoint store provisioned for
/// the composite envelope size; the envelope is durable before continuity publishes the import.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_import_composite(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    out: *mut CSnapshot,
) -> i32 {
    let run = || -> Result<Snapshot, Error> {
        let mut native = substrate(sub)?;
        let key = native.key();
        runtime(rt)?.import_composite(&mut native, key).map_err(|r| r.error)
    };
    rc(run().map(|s| snapshot_out(out, &s)))
}

/// Release managed ownership of the bound state (its K1 lease) and unbind it, so its owner may use it unmanaged.
/// Refused while a managed turn is open; a state that is not the bound one is refused; nothing bound is OK.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_release(rt: *mut ElpisRuntime, sub: *const CSubstrate) -> i32 {
    let run = || -> Result<(), Error> {
        let mut native = substrate(sub)?;
        let key = native.key();
        runtime(rt)?.release(&mut native, key).map_err(|r| r.error)
    };
    rc(run())
}

/// Abort the open turn of the described substrate (through the capability retained at begin): nothing is
/// installed. Another substrate is refused (`RUNTIME_TURN_NOT_OPEN`) and the turn stays open.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_turn_abort(rt: *mut ElpisRuntime, sub: *const CSubstrate) -> i32 {
    let run = || -> Result<(), Error> {
        let key = substrate(sub)?.key();
        runtime(rt)?.turn_abort(key).map_err(|r| r.error)
    };
    rc(run())
}

// -- K1 Recovery R0 ---------------------------------------------------------------------------------------------

unsafe fn path_arg<'a>(path: *const u8, len: usize) -> Result<&'a Path, Error> {
    if path.is_null() {
        return Err(Rt::Invalid.into());
    }
    Ok(Path::new(OsStr::from_bytes(std::slice::from_raw_parts(path, len))))
}

/// Operator provisioning of a K1 checkpoint store: create the absolute directory `path` (it must not exist) with
/// exactly two empty slot files sized for envelopes of `envelope_bytes` (the admitted K1 shape's), fully written
/// and synced. Never overwrites, resizes or adopts anything.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_checkpoint_provision(path: *const u8, len: usize, envelope_bytes: usize) -> i32 {
    rc(path_arg(path, len)
        .and_then(|dir| crate::checkpoint::Store::provision(dir, envelope_bytes).map_err(Error::Runtime)))
}

/// Attach a provisioned checkpoint store to an open runtime (no turn in flight).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_checkpoint_attach(rt: *mut ElpisRuntime, path: *const u8, len: usize) -> i32 {
    let run = || -> Result<(), Error> {
        let dir = path_arg(path, len)?;
        runtime(rt)?.checkpoint_attach(dir).map_err(|r| r.error)
    };
    rc(run())
}

/// What restart may do with the K1 lineage (read-only). When RESUMABLE and `envelope_len` holds it, the authorized
/// envelope is copied to `envelope` (restore it into a K1 state; its identity is `out->authorized`).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_checkpoint_recover(
    rt: *mut ElpisRuntime,
    out: *mut CRecovery,
    envelope: *mut u8,
    envelope_len: usize,
) -> i32 {
    if out.is_null() {
        return Rt::Invalid as i32;
    }
    let run = || -> Result<Recovery, Error> { runtime(rt)?.recover().map_err(|r| r.error) };
    match run() {
        Ok(r) => {
            let size = r.envelope.as_ref().map_or(0, Vec::len);
            if let Some(bytes) = &r.envelope {
                if !envelope.is_null() && envelope_len >= size {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), envelope, size);
                }
            }
            *out = CRecovery {
                disposition: r.disposition as u32,
                authorized_present: r.authorized_present as u32,
                authorized: r.authorized,
                candidate: r.candidate,
                envelope_bytes: size as u64,
            };
            0
        }
        Err(e) => e.code(),
    }
}

/// Operator reconciliation: discard the unresolved candidate (exactly `candidate`); the lineage stays at
/// continuity's authority.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_checkpoint_discard(rt: *mut ElpisRuntime, candidate: *const u8) -> i32 {
    let run = || -> Result<(), Error> {
        let candidate = digest(candidate)?;
        runtime(rt)?.checkpoint_discard(candidate).map_err(|r| r.error)
    };
    rc(run())
}

/// Operator reconciliation: adopt the unresolved candidate (exactly `candidate`) as the authority: continuity
/// publishes authorized -> candidate after its envelope verified. A publication failure fail-stops.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_checkpoint_adopt(
    rt: *mut ElpisRuntime,
    candidate: *const u8,
    out: *mut CSnapshot,
) -> i32 {
    let run = || -> Result<Snapshot, Error> {
        let candidate = digest(candidate)?;
        runtime(rt)?.checkpoint_adopt(candidate).map_err(|r| r.error)
    };
    rc(run().map(|s| snapshot_out(out, &s)))
}

// -- evolution ------------------------------------------------------------------------------------------------

/// The current idle evolution authority (`CONTINUITY_EVOLUTION_PENDING` while a reservation is pending).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_evolution_authority(rt: *mut ElpisRuntime, out: *mut CSnapshot) -> i32 {
    rc(runtime(rt).and_then(|core| core.evolution_authority()).map(|s| snapshot_out(out, &s)))
}

/// Durably reserve one assertion against the observed idle authority, before the caller executes it.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_evolution_reserve(
    rt: *mut ElpisRuntime,
    observed: *const CEvolution,
    assertion: *const u8,
    out: *mut CSnapshot,
) -> i32 {
    let run = || -> Result<Snapshot, Error> {
        let observed = observed.as_ref().ok_or(Error::Runtime(Rt::Invalid))?;
        let observed = from_c_evolution(observed).map_err(|_| Error::Authority(Code::AuthorityMismatch))?;
        let assertion = digest(assertion)?;
        runtime(rt)?.evolution_reserve(&observed, assertion)
    };
    rc(run().map(|s| snapshot_out(out, &s)))
}

/// Finalize the attempt in flight with its executed receipt digest.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_evolution_finalize(
    rt: *mut ElpisRuntime,
    receipt: *const u8,
    out: *mut CSnapshot,
) -> i32 {
    let run = || -> Result<Snapshot, Error> {
        let receipt = digest(receipt)?;
        runtime(rt)?.evolution_finalize(receipt)
    };
    rc(run().map(|s| snapshot_out(out, &s)))
}

/// The caller could not complete the attempt in flight: the reservation stays pending; the runtime fail-stops.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_evolution_abandon(rt: *mut ElpisRuntime) -> i32 {
    rc(runtime(rt).and_then(|mut core| core.evolution_abandon()))
}

/// Explicit reconciliation of a durable pending authority with an externally established receipt.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_evolution_reconcile(
    rt: *mut ElpisRuntime,
    expected: *const CEvolution,
    receipt: *const u8,
    out: *mut CSnapshot,
) -> i32 {
    let run = || -> Result<Snapshot, Error> {
        let expected = expected.as_ref().ok_or(Error::Runtime(Rt::Invalid))?;
        let expected = from_c_evolution(expected).map_err(|_| Error::Authority(Code::AuthorityMismatch))?;
        let receipt = digest(receipt)?;
        runtime(rt)?.evolution_reconcile(&expected, receipt)
    };
    rc(run().map(|s| snapshot_out(out, &s)))
}

// -- testing library only ---------------------------------------------------------------------------------------

#[cfg(feature = "testing")]
mod testing {
    use super::*;
    use crate::authority::IoCounters;

    /// Arm one fault of the authority's record (`authority::Store::testing_arm`: the actions ABI v6 shared with
    /// continuity's own testing hook).
    #[no_mangle]
    pub unsafe extern "C" fn elpis_runtime_testing_fault(
        rt: *mut ElpisRuntime,
        publication: u64,
        action: u32,
        arg: u64,
    ) -> i32 {
        rc(runtime(rt).and_then(|mut core| core.testing_arm(publication, action, arg)))
    }

    /// Hold (nonzero) or release (zero) the durability worker before it takes a new job.
    #[no_mangle]
    pub unsafe extern "C" fn elpis_runtime_testing_durable_hold(rt: *mut ElpisRuntime, hold: i32) -> i32 {
        rc(runtime(rt).map(|mut core| core.testing_durable_hold(hold != 0)))
    }

    /// Plan one fault of the next durable publication (docs/RUNTIME_CORE.md, "Durability crash matrix"):
    /// action 0 clear; 1 die (simulated process death) at `step`; 2 park at `step` (block until the process is
    /// killed or the runtime closes) with exactly that step's on-disk effect; 3 checkpoint write fails; 4 checkpoint
    /// write torn after `arg` bytes, then fails; 5 checkpoint sync fails; 6 continuity record write fails; 7 record
    /// torn after `arg` bytes, then fails; 8 record sync fails with the bytes lost; 9 record sync fails although
    /// durable; 10 checkpoint write and sync succeed but the bytes on disk differ (caught by the read-back). Steps: 1
    /// taken, 2 sealed, 3 slot torn (`arg` bytes), 4 slot written (unsynced), 5 slot synced, 6 record torn (`arg`
    /// bytes), 7 record written (unsynced), 8 record synced, 9 acknowledged. A death at steps 1..5 resolves the ticket
    /// ABANDONED (continuity's record never begun); at 6..8 UNCERTAIN (restart resolves it); at 9 the ticket keeps
    /// its acknowledgment.
    #[no_mangle]
    pub unsafe extern "C" fn elpis_runtime_testing_durable_fault(
        rt: *mut ElpisRuntime,
        action: u32,
        step: u32,
        arg: u64,
    ) -> i32 {
        use crate::durable::{Plan, Step};
        let plan = match action {
            0 => None,
            1 | 2 => match Step::from_u32(step) {
                Some(step) => Some(Plan::At { step, park: action == 2, arg }),
                None => return Rt::Invalid as i32,
            },
            3 => Some(Plan::SlotWriteFail),
            4 => Some(Plan::SlotTornFail(arg)),
            5 => Some(Plan::SlotSyncFail),
            6 => Some(Plan::RecordWriteFail),
            7 => Some(Plan::RecordTornFail(arg)),
            8 => Some(Plan::RecordSyncLost),
            9 => Some(Plan::RecordSyncDurable),
            10 => Some(Plan::SlotReadBackFail),
            _ => return Rt::Invalid as i32,
        };
        rc(runtime(rt).map(|mut core| core.testing_durable_plan(plan)))
    }

    /// Plan one fault of the next control operation of kind `op` (docs/RUNTIME_CORE.md, "Control-operation crash
    /// points"): 1 anchor, 2 legacy import, 3 operator discard, 4 operator adopt. These run outside the lane's step
    /// machine, on the caller's thread with the stores held. Action 0 clears; 1 dies (simulated) at `point`; 2 parks
    /// the caller at `point` with exactly that point's on-disk effect (it holds the handle: the wait ends only with
    /// the process, or after 300 s as a simulated death; the durability status reports `parked` = 100 * op + point).
    /// Points: 1 begun (checks passed, nothing written); 2 slot written, not synced (anchor, import: the envelope;
    /// discard: the zeroed header); 3 slot synced (anchor, import: continuity not begun; discard: not returned);
    /// 4 continuity record torn after `arg` bytes; 5 record written, not synced; 6 record synced, not returned
    /// (anchor, import, adopt). A discard has no record point and an adopt no slot point (`RUNTIME_INVALID`).
    #[no_mangle]
    pub unsafe extern "C" fn elpis_runtime_testing_control_fault(
        rt: *mut ElpisRuntime,
        op: u32,
        action: u32,
        point: u32,
        arg: u64,
    ) -> i32 {
        use crate::durable::control::Plan;
        let plan = match action {
            0 => None,
            1 | 2 => match Plan::new(op, point, action == 2, arg) {
                Some(plan) => Some(plan),
                None => return Rt::Invalid as i32,
            },
            _ => return Rt::Invalid as i32,
        };
        rc(runtime(rt).map(|mut core| core.testing_control_plan(plan)))
    }

    /// Copy the checkpoint store's I/O counters (zero when none is attached); reset them when `reset` is nonzero.
    #[no_mangle]
    pub unsafe extern "C" fn elpis_runtime_testing_checkpoint_io_counters(
        rt: *mut ElpisRuntime,
        out: *mut crate::checkpoint::probe::Counters,
        reset: i32,
    ) -> i32 {
        if out.is_null() {
            return Rt::Invalid as i32;
        }
        rc(runtime(rt).map(|mut core| *out = core.testing_checkpoint_counters(reset != 0)))
    }

    /// Copy the embedded store's I/O counters; reset them when `reset` is nonzero.
    #[no_mangle]
    pub unsafe extern "C" fn elpis_runtime_testing_io_counters(
        rt: *mut ElpisRuntime,
        out: *mut IoCounters,
        reset: i32,
    ) -> i32 {
        if out.is_null() {
            return Rt::Invalid as i32;
        }
        rc(runtime(rt).map(|mut core| *out = core.testing_authority_counters(reset != 0)))
    }
}
