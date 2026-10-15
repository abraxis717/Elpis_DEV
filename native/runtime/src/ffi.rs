//! The stable C ABI (include/elpis/runtime.h).
//!
//! Every function returns 0 or a positive stable code: a continuity code (1..=17, 255) or a RuntimeCore code
//! (64..). A runtime handle serializes its calls internally (one lock per handle), except the durability lane's
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
//! # Safety
//!
//! Every pointer argument is either NULL (refused with `RUNTIME_INVALID`, or the documented meaning) or valid
//! for its declared size: 32 bytes for a digest, `len` elements for a buffer, one struct otherwise. Runtime
//! handles come from `elpis_runtime_create` and are used until `elpis_runtime_destroy`. A substrate
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

use elpis_continuity::ffi::{from_c_evolution, to_c, CEvolution, CSnapshot};
use elpis_continuity::{Code, Digest, Snapshot};

use crate::code::{Error, Rt};
use crate::core::{Core, Counters, Recovery, Refusal, Stimulus, VolatileCommit};
use crate::durable::{Kind, Lane, Status, Ticket};
use crate::fuel::Budget;
use crate::substrate::{features, CSubstrate, Experience, Native, ScheduleOutcome, Transition};

pub const ABI_VERSION: u32 = 4;

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
const _: () = assert!(std::mem::size_of::<Counters>() == 128);
const _: () = assert!(std::mem::size_of::<Budget>() == 48);

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
        }
        .as_ptr(),
        None => match Code::from_i32(code) {
            Some(_) => elpis_continuity::ffi::elpis_continuity_code_name(code),
            None => std::ptr::null(),
        },
    }
}

// -- lifecycle ---------------------------------------------------------------------------------------------

/// Create a closed runtime over an absolute continuity directory (`len` bytes, no terminator needed).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_create(path: *const u8, len: usize, out: *mut *mut ElpisRuntime) -> i32 {
    if path.is_null() || out.is_null() {
        return Rt::Invalid as i32;
    }
    *out = std::ptr::null_mut();
    let dir = Path::new(OsStr::from_bytes(std::slice::from_raw_parts(path, len)));
    rc(Core::new(dir).map(|core| {
        let lane = core.lane();
        *out = Box::into_raw(Box::new(ElpisRuntime { core: Mutex::new(core), lane }))
    }))
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
        let observed = from_c_evolution(observed).map_err(|_| Error::Continuity(Code::AuthorityMismatch))?;
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
        let expected = from_c_evolution(expected).map_err(|_| Error::Continuity(Code::AuthorityMismatch))?;
        let receipt = digest(receipt)?;
        runtime(rt)?.evolution_reconcile(&expected, receipt)
    };
    rc(run().map(|s| snapshot_out(out, &s)))
}

// -- testing library only ---------------------------------------------------------------------------------------

#[cfg(feature = "testing")]
mod testing {
    use super::*;
    use elpis_continuity::store::probe::Counters as IoCounters;

    /// Arm one continuity fault (the actions of `elpis_continuity_testing_fault`).
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
        rc(runtime(rt).map(|mut core| *out = core.testing_continuity_counters(reset != 0)))
    }
}
