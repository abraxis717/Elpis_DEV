//! The stable C ABI (include/elpis/runtime.h).
//!
//! Every function returns 0 or a positive stable code: a continuity code (1..=17, 255) or a RuntimeCore code
//! (64..). A runtime handle serializes its calls internally (one lock per handle).
//!
//! # Safety
//!
//! Every pointer argument is either NULL (refused with `RUNTIME_INVALID`, or the documented meaning) or valid
//! for its declared size: 32 bytes for a digest, `len` elements for a buffer, one struct otherwise. Runtime
//! handles come from `elpis_runtime_create` and are used until `elpis_runtime_destroy`. A substrate
//! descriptor's handle and function table must be live for the duration of the call; the table is copied.
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
use std::sync::{Mutex, MutexGuard};

use elpis_continuity::ffi::{from_c_evolution, to_c, CEvolution, CSnapshot};
use elpis_continuity::{Code, Digest, Snapshot};

use crate::code::{Error, Rt};
use crate::core::{Core, Counters, Refusal, Stimulus};
use crate::substrate::{features, CSubstrate, CommitIdentity, Experience, Native, ScheduleOutcome};

pub const ABI_VERSION: u32 = 3;

/// Opaque runtime handle.
pub struct ElpisRuntime(Mutex<Core>);

/// `elpis_runtime_turn_begin_result`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CTurnBegin {
    pub schedule: ScheduleOutcome,
    /// The K1 status behind an `ECS_*` refusal (0 otherwise).
    pub k1_status: i32,
    pub reserved: u32,
}

/// `elpis_runtime_turn_commit_result`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CTurnCommit {
    pub identity: CommitIdentity,
    /// 1 when the native commit happened (also when its publication then failed), else 0.
    pub committed: u32,
    /// The K1 status behind an `ECS_*` refusal (0 otherwise).
    pub k1_status: i32,
}

/// `elpis_runtime_query_result`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CQuery {
    /// The retained-state identity the answer was computed from (zero on refusal).
    pub state_digest: Digest,
    /// The K1 status behind an `ECS_*` refusal (0 otherwise).
    pub k1_status: i32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<CQuery>() == 40);
const _: () = assert!(std::mem::size_of::<CTurnBegin>() == 48);
const _: () = assert!(std::mem::size_of::<CTurnCommit>() == 120);
const _: () = assert!(std::mem::size_of::<Counters>() == 64);

fn rc(result: Result<(), Error>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(e) => e.code(),
    }
}

unsafe fn runtime<'a>(p: *mut ElpisRuntime) -> Result<MutexGuard<'a, Core>, Error> {
    let rt = p.as_ref().ok_or(Error::Runtime(Rt::Invalid))?;
    // A panic aborts the process (panic = "abort"), so the lock is never left poisoned by a live caller.
    Ok(rt.0.lock().unwrap_or_else(|e| e.into_inner()))
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
    rc(Core::new(dir).map(|core| *out = Box::into_raw(Box::new(ElpisRuntime(Mutex::new(core))))))
}

/// Close (if open) and free a runtime; sets `*rt` to NULL. An open managed turn is aborted natively first.
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

/// The current durable authority (readable while fail-stopped).
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
/// authoritative K1 state, read-only: no transaction, no commit, no publication. `out` receives `rows` values;
/// `result->state_digest` the retained-state identity they were computed from (it equals the durable expected
/// identity, or the call is refused and `out` is zeroed).
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_query(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    x: *const f64,
    x_len: usize,
    out: *mut f64,
    rows: usize,
    result: *mut CQuery,
) -> i32 {
    let run = || -> Result<Digest, Refusal> {
        let mut native = substrate(sub)?;
        let key = native.key();
        let x = slice(x, x_len)?;
        if out.is_null() {
            return Err(Rt::Invalid.into());
        }
        let out = std::slice::from_raw_parts_mut(out, rows);
        runtime(rt)?.query(&mut native, key, x, out)
    };
    let (value, code) = match run() {
        Ok(state_digest) => (CQuery { state_digest, k1_status: 0, reserved: 0 }, 0),
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
    s3_out: *mut f64,
    s3_len: usize,
    out: *mut CTurnBegin,
) -> i32 {
    let run = || -> Result<ScheduleOutcome, Refusal> {
        let mut native = substrate(sub)?;
        let key = native.key();
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
        runtime(rt)?.turn_begin(&mut native, key, &stimulus, s3)
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

/// Commit the open turn natively and publish its retained-state identity to continuity. On a publication
/// failure the runtime fail-stops; `out->committed` is 1 and carries the committed identity.
#[no_mangle]
pub unsafe extern "C" fn elpis_runtime_turn_commit(
    rt: *mut ElpisRuntime,
    sub: *const CSubstrate,
    out: *mut CTurnCommit,
    snapshot: *mut CSnapshot,
) -> i32 {
    let run = || -> Result<(CommitIdentity, Snapshot), Refusal> {
        let mut native = substrate(sub)?;
        let key = native.key();
        runtime(rt)?.turn_commit(&mut native, key)
    };
    let (result, code) = match run() {
        Ok((identity, s)) => {
            snapshot_out(snapshot, &s);
            (CTurnCommit { identity, committed: 1, k1_status: 0 }, 0)
        }
        Err(r) => {
            let committed = r.committed.as_deref().copied().unwrap_or_default();
            (
                CTurnCommit { identity: committed, committed: r.committed.is_some() as u32, k1_status: r.k1_status },
                r.error.code(),
            )
        }
    };
    if !out.is_null() {
        *out = result;
    }
    code
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
