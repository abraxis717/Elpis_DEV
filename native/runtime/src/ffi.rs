//! The stable C ABI (include/elpis/runtime.h), ABI v8.
//!
//! Every function returns 0 or a positive stable code: an authority code (2..=5, 9..=13, 15; `CONTINUITY_*`, the
//! names ABI v1 gave them) or a RuntimeCore code (64..). A runtime handle serializes its calls internally (one lock
//! per handle).
//!
//! ABI v8 (docs/RUNTIME_CORE.md; docs/ELPIS_MISSION.md, Amendment 3): the runtime is volatile only. A runtime is
//! created from a configuration (`elpis_runtime_create`), never from directories; each open starts a new volatile
//! ECS_C session (docs/ECS_C_SESSION.md) at a fresh genesis, and close ends it. Every v7 entry point of the
//! persistent runtime is gone, not stubbed: `elpis_runtime_create_ecs_c`, `elpis_runtime_import_composite`, the
//! durable requests (`recoverable_request`, `identity_request`, `durable_wait`, `durable_status`), the K1 checkpoint
//! store (`checkpoint_provision`, `_attach`, `_recover`, `_discard`, `_adopt`), `elpis_runtime_evolution_reconcile`
//! and the testing library's fault hooks; so are the substrate tables' `leased_capture` and `envelope_seal`. The
//! anchor, the evolution reservation and finalization and the operator's events are session transitions in memory.
//! A v7 caller refuses to load this library (the version check), so none can mistake it for a durable runtime.
//!
//! The history of v1..v7 (the live/durable split of v4, observe-only turns of v5, the ECS_C authority of v6 and v7)
//! is the historical specification (docs/history/RUNTIME_CORE_ABI_V7.md).
//!
//! # Safety
//!
//! Every pointer argument is either NULL (refused with `RUNTIME_INVALID`, or the documented meaning) or valid
//! for its declared size: 32 bytes for a digest, `len` elements for a buffer, one struct otherwise. Runtime
//! handles come from `elpis_runtime_create` and are used until `elpis_runtime_destroy`. A substrate descriptor's
//! handle and function table must be live for the duration of the call; the table is copied.
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

use std::ffi::c_char;
use std::sync::{Mutex, MutexGuard};

use crate::authority::{Cognition, Digest, EvolutionState, SessionTemplate, Snapshot, ZERO};
use crate::code::{Code, Error, Rt};
use crate::core::{Core, Counters, Refusal, Stimulus, VolatileCommit};
use crate::fuel::Budget;
use crate::substrate::{features, CSubstrate, Experience, Native, ScheduleOutcome, Transition};

pub const ABI_VERSION: u32 = 8;

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

/// `elpis_continuity_snapshot` (runtime.h): the session's authority view (192 bytes). `k1_state_digest` keeps its
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

/// Opaque runtime handle: the state machine behind its lock.
pub struct ElpisRuntime {
    core: Mutex<Core>,
}

/// `elpis_runtime_config` (ABI v8): what every session of a runtime is created with.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CConfig {
    /// `ELPIS_RUNTIME_ABI_V8`.
    pub abi_version: u32,
    /// The session's ECS_C entity capacity, 2..=4096.
    pub capacity: u32,
    /// Zero: every session draws a fresh nonce (getrandom(2)); otherwise each session's nonce is derived from it and
    /// the session's ordinal.
    pub nonce: Digest,
    pub operator_label: Digest,
    pub runtime_label: Digest,
}

const _: () = assert!(std::mem::size_of::<CConfig>() == 104);

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
const _: () = assert!(std::mem::size_of::<CSnapshot>() == 192);

const _: () = assert!(std::mem::size_of::<CTurnBegin>() == 48);
const _: () = assert!(std::mem::size_of::<CVolatileCommit>() == 64);
const _: () = assert!(std::mem::size_of::<Counters>() == 96);
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
            Rt::NotResident => c"RUNTIME_NOT_RESIDENT",
            Rt::Authority => c"RUNTIME_AUTHORITY",
            Rt::ObservationRefused => c"RUNTIME_OBSERVATION_REFUSED",
            Rt::NoMem => c"RUNTIME_NOMEM",
            Rt::Retired => c"RUNTIME_RETIRED",
        }
        .as_ptr(),
        None => match Code::from_i32(code) {
            Some(c) => c.name().as_ptr(),
            None => std::ptr::null(),
        },
    }
}

// -- lifecycle ---------------------------------------------------------------------------------------------

/// Create a closed runtime from `config` (ABI v8). No session exists until `elpis_runtime_open`; no path, file or
/// directory is involved. `RUNTIME_INVALID` for a NULL argument, another `abi_version` or a capacity outside
/// 2..=4096.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_create(config: *const CConfig, out: *mut *mut ElpisRuntime) -> i32 {
    if config.is_null() || out.is_null() {
        return Rt::Invalid as i32;
    }
    *out = std::ptr::null_mut();
    let c = &*config;
    if c.abi_version != ABI_VERSION {
        return Rt::Invalid as i32;
    }
    let template = SessionTemplate {
        capacity: c.capacity,
        nonce_seed: c.nonce,
        operator_label: c.operator_label,
        runtime_label: c.runtime_label,
    };
    rc(Core::new(template).map(|core| *out = Box::into_raw(Box::new(ElpisRuntime { core: Mutex::new(core) }))))
}

/// `elpis_runtime_authority` (ABI v8): the session's status.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CAuthority {
    /// `ELPIS_RUNTIME_BACKEND_ECS_C_SESSION`.
    pub backend: u32,
    /// The last session refusal (`ELPIS_ECS_C_*`; 0 none); while closed, the refusal of the last refused open.
    pub ecs_c_code: i32,
    pub entity_count: u32,
    pub capacity: u32,
    pub generation: u64,
    /// The session's process-unique instance (0: no session is open).
    pub instance: u64,
    pub receipt_count: u64,
    pub evidence_count: u64,
    /// The session's exact, constant memory footprint (0: no session is open).
    pub volatile_bytes: u64,
    /// Sessions this runtime has opened.
    pub sessions: u64,
    /// The session's next founding index and reclaimed-row count (what a relational view syncs from, with the rows).
    pub next_founding_index: u64,
    pub reclaimed_count: u64,
    pub session: Digest,
    pub commit: Digest,
    pub state_root: Digest,
    pub receipt_head: Digest,
    pub evidence_head: Digest,
}

const _: () = assert!(std::mem::size_of::<CAuthority>() == 240);

/// `elpis_runtime_authority_entity` (`elpis_runtime_authority_entity_at`): one entity row of the session, laid out as
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

/// The session's status (readable closed: the sessions opened and the last refused open's session code).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_authority_status(rt: *mut ElpisRuntime, out: *mut CAuthority) -> i32 {
    rc((|| {
        let s = runtime(rt)?.authority_status();
        let out = out.as_mut().ok_or(Error::Runtime(Rt::Invalid))?;
        *out = CAuthority {
            backend: s.backend,
            ecs_c_code: s.ecs_c_code,
            entity_count: s.entity_count,
            capacity: s.capacity,
            generation: s.generation,
            instance: s.instance,
            receipt_count: s.receipt_count,
            evidence_count: s.evidence_count,
            volatile_bytes: s.volatile_bytes,
            sessions: s.sessions,
            next_founding_index: s.next_founding_index,
            reclaimed_count: s.reclaimed_count,
            session: s.session,
            commit: s.commit,
            state_root: s.state_root,
            receipt_head: s.receipt_head,
            evidence_head: s.evidence_head,
        };
        Ok(())
    })())
}

/// One operator event (512 bytes, `elpis/ecs_c_types.h`: ENTITY_FOUND, ENTITY_LIFECYCLE, SOURCE_VERSION,
/// ENTITY_RECLAIM) on the open session, in memory; the typed-evidence admission table follows it. A legacy import or
/// a reconciliation is `RUNTIME_RETIRED`; RuntimeCore's own kinds are `RUNTIME_INVALID`. `out_generation`
/// (optional) receives the session's new generation.
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
        let event = elpis_ecs_c_session::model::Event::decode(std::slice::from_raw_parts(event, 512))
            .map_err(|_| Error::Runtime(Rt::Invalid))?;
        let receipt = runtime(rt)?.authority_transition(&event).map_err(|r| r.error)?;
        if let Some(out) = out_generation.as_mut() {
            *out = receipt.generation;
        }
        Ok(())
    })())
}

/// The session's entity row at `index` (what an operator event is built from: identities, lifecycles, sender
/// watermarks). `RUNTIME_INVALID` past the last row; refused closed.
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

/// The identity the next ENTITY_FOUND of `kind` and `label` founds in the session (the receiver that event must
/// name). `RUNTIME_INVALID` for an unknown kind; refused closed.
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

/// Close (if open) and free a runtime; sets `*rt` to NULL. An open managed turn is aborted natively first; the
/// session's memory is released.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_destroy(rt: *mut *mut ElpisRuntime) {
    if !rt.is_null() && !(*rt).is_null() {
        drop(Box::from_raw(*rt));
        *rt = std::ptr::null_mut();
    }
}

/// Open a closed runtime: a new session at a fresh genesis, nothing anchored or bound (a fail-stop is cleared by
/// close + open, with the session it belonged to). An already open runtime is refused (`CONTINUITY_OPEN`) and keeps
/// its open turn. `RUNTIME_NOMEM` when the session's memory cannot be reserved (the runtime stays closed).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_open(rt: *mut ElpisRuntime, out: *mut CSnapshot) -> i32 {
    rc(runtime(rt).and_then(|mut core| core.open()).map(|s| snapshot_out(out, &s)))
}

/// Close (a closed runtime is unchanged): the session ends. An open managed turn is aborted natively first: nothing
/// installed, nothing kept.
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

/// The session's authority view (readable while fail-stopped). The live state may be ahead of the anchored identity.
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

/// Explicitly anchor the session's lineage at the substrate's retained identity (reads K1 only; in memory).
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
/// live K1 state, read-only: no transaction, no commit, no digest, no I/O. `out` receives `rows`
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

/// Commit the open turn natively: a VOLATILE commit. The candidate is installed and the live generation advances by
/// one; nothing is written anywhere.
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

/// The bound state's retained-state identity and live generation, read on request (cold path: one guarded native
/// call that hashes the state). Binds the state first if needed (verified against the session's identity).
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

// -- evolution ------------------------------------------------------------------------------------------------

/// The current idle evolution authority (`CONTINUITY_EVOLUTION_PENDING` while a reservation is pending).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_evolution_authority(rt: *mut ElpisRuntime, out: *mut CSnapshot) -> i32 {
    rc(runtime(rt).and_then(|core| core.evolution_authority()).map(|s| snapshot_out(out, &s)))
}

/// Reserve one assertion in the session against the observed idle authority, before the caller executes it.
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

/// The caller could not complete the attempt in flight: the reservation stays pending; the runtime fail-stops for
/// the rest of the session.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_evolution_abandon(rt: *mut ElpisRuntime) -> i32 {
    rc(runtime(rt).and_then(|mut core| core.evolution_abandon()))
}
