//! The native K1 retained state a runtime drives, reached through the K1 C ABI.
//!
//! RuntimeCore links no ECS code. A caller hands it a substrate descriptor: the native handle, the K1 function
//! table of its library (`ecsg_k1.h` for a standalone state, `ecsg_k1_fms.h` for an FMS-resident one), the
//! declared input dimension and the caller identity of the handle's owner. Every [`K1Ops`] call is exactly one
//! native K1 crossing; RuntimeCore computes no ECS mathematics and touches no ECS byte.
//!
//! From a successful managed-turn begin until that turn ends, RuntimeCore retains one capability over the state:
//! [`TxnAbort`], the native `txn_abort` of the turn's transaction (a copy of the descriptor's handle, resident id
//! and abort entry; never commit, schedule or any other operation). It is what ends the turn when the runtime is
//! closed, reopened or destroyed without an explicit commit or abort, so an open native transaction is never
//! forgotten. The C ABI (v2) makes the substrate's lifetime explicit: the state must stay live until the turn ends.

use std::ffi::{c_int, c_void};

pub type Digest = [u8; 32];

/// K1 status values (`elpis_ecsg_k1_status`); the FMS adapter returns them too, or `-100 + fms_status`.
pub const K1_OK: i32 = 0;
pub const K1_INVALID: i32 = -1;
pub const K1_NONFINITE: i32 = -2;
pub const K1_STALE: i32 = -3;
/// A concurrent overlapping call on the same state: refused before anything was done (transient by contract).
pub const K1_BUSY: i32 = -4;
pub const K1_CAPACITY: i32 = -5;
/// The state is managed under another lease (`ELPIS_ECSG_K1_LEASED`): refused before anything was touched.
pub const K1_LEASED: i32 = -8;
/// The FMS adapter's capacity refusal (`-100 + FMS_CAPACITY`).
pub const K1_FMS_CAPACITY: i32 = -107;

/// The highest K1 input dimension (`ELPIS_ECSG_K1_MAX_DIM`).
pub const MAX_DIM: usize = 64;
/// The most experiences one schedule holds (`ELPIS_ECSG_K1_MAX_EXPERIENCES`).
pub const MAX_EXPERIENCES: usize = 64;

/// `elpis_ecsg_k1_experience`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Experience {
    pub rows: u64,
    pub steps: u64,
}

/// `elpis_ecsg_k1_schedule_result` (`elpis_runtime_schedule_result`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScheduleOutcome {
    pub epoch_before: u64,
    pub epoch_after: u64,
    pub experiences_applied: u64,
    pub failed_experience: u64,
    pub failed_step: u64,
}

/// `elpis_ecsg_k1_transition`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Transition {
    pub epoch_before: u64,
    pub epoch_after: u64,
    pub generation_before: u64,
    pub generation_after: u64,
    pub steps: u64,
    pub failed_step: u64,
}

/// `elpis_ecsg_k1_commit_identity`: the committed transition and its exact retained-state identities.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommitIdentity {
    pub transition: Transition,
    pub state_before_digest: Digest,
    pub state_after_digest: Digest,
}

/// The S3 readout length of a K1 state of input dimension `dim` (`elpis_ecsg_k1_features`):
/// `d + d(d+1)/2 + d(d+1)(d+2)/6`, the number of degree-1..3 monomials. 0 outside 1..=64.
pub fn features(dim: usize) -> usize {
    if !(1..=MAX_DIM).contains(&dim) {
        return 0;
    }
    dim + dim * (dim + 1) / 2 + dim * (dim + 1) * (dim + 2) / 6
}

/// The native operations a runtime needs on one retained K1 state. Each returns the K1 status.
/// The native operations a runtime needs on one retained K1 state. Each returns the K1 status.
///
/// Every mutating operation is a *leased* entry point (ecsg_k1.h, "Managed ownership"): it presents RuntimeCore's
/// lease, and the state refuses it (`LEASED`) unless that lease holds the state. While RuntimeCore holds the lease
/// the state refuses every unmanaged mutating entry point, so the bound state cannot move outside the lineage.
pub trait K1Ops {
    fn state_digest(&mut self, out: &mut Digest) -> i32;
    /// The state's immutable shape `(dim, width)` (`shape`): read-only, no guard.
    fn shape(&mut self, dim: &mut usize, width: &mut usize) -> i32;
    /// QUERY bound to its identity (`query_identity`): `out[r] = f_W(x_r)` for `out.len()` rows of the declared
    /// dimension, and `digest` = the retained-state identity of the same authoritative state, in one native call.
    /// Read-only: no transaction, no commit, nothing written to the retained state.
    fn query_identity(&mut self, dim: usize, x: &[f64], out: &mut [f64], digest: &mut Digest) -> i32;
    /// Claim the state for `lease` (replacing any other lease; BUSY while a transaction is open).
    fn lease_claim(&mut self, lease: u64) -> i32;
    /// Release the state if `lease` holds it.
    fn lease_release(&mut self, lease: u64) -> i32;
    fn reserve(&mut self, lease: u64, max_rows: usize) -> i32;
    /// Begin a transaction and write the retained-state identity of exactly the source it copied (one call).
    fn txn_begin(&mut self, lease: u64, source: &mut Digest, token: &mut u64) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn txn_run_schedule(
        &mut self,
        lease: u64,
        token: u64,
        x: &[f64],
        y: &[f64],
        schedule: &[Experience],
        rate: f64,
        s3: &mut [f64],
        result: &mut ScheduleOutcome,
    ) -> i32;
    fn txn_commit_identity(&mut self, lease: u64, token: u64, out: &mut CommitIdentity) -> i32;
    fn txn_abort(&mut self, token: u64) -> i32;
    /// The portable envelope of the authoritative state (`snapshot_write`; read-only).
    fn snapshot_write(&mut self, out: &mut [u8]) -> i32;
    /// The portable envelope of the open transaction's candidate (`txn_snapshot_write`; read-only).
    fn txn_snapshot_write(&mut self, token: u64, out: &mut [u8]) -> i32;
    /// The owned abort capability over this same native state, retained by RuntimeCore from a successful turn
    /// begin until that turn ends. It may outlive the call that produced it, but never the open transaction it
    /// ends: the native state is live for exactly that interval (the substrate lifetime contract, runtime.h).
    fn retain_abort(&self) -> Box<dyn TxnAbort>;
}

/// The one operation RuntimeCore retains across calls: ending the open managed turn's native transaction.
pub trait TxnAbort: Send {
    /// `txn_abort` of the K1 ABI: OK (aborted, or no transaction open), INVALID (not this token's transaction),
    /// BUSY (a concurrent overlapping call; nothing done).
    fn txn_abort(&mut self, token: u64) -> i32;
}

/// What an open runtime binds its K1 lineage to: the native state and the caller's owner identity.
///
/// `owner` is the caller's identity for the object that owns the handle; it must stay unique while bound (the
/// Python adapter passes the identity of a wrapper it keeps alive), so a freed handle whose address is reused
/// by another state is not mistaken for the bound one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    pub kind: u32,
    pub handle: usize,
    pub id: u64,
    pub owner: u64,
    pub dim: usize,
}

pub const KIND_K1: u32 = 1;
pub const KIND_K1_FMS: u32 = 2;

type Status = c_int;
type State = *mut c_void;

/// `elpis_ecsg_k1_leased_txn_run_schedule`.
pub type K1RunSchedule = unsafe extern "C" fn(
    State,
    u64,
    u64,
    *const f64,
    *const f64,
    usize,
    *const Experience,
    usize,
    f64,
    *mut f64,
    usize,
    *mut ScheduleOutcome,
) -> Status;

/// `elpis_ecsg_k1_fms_leased_txn_run_schedule`.
pub type K1FmsRunSchedule = unsafe extern "C" fn(
    State,
    u64,
    u64,
    u64,
    *const f64,
    *const f64,
    usize,
    *const Experience,
    usize,
    f64,
    *mut f64,
    usize,
    *mut ScheduleOutcome,
) -> Status;

/// `elpis_ecsg_k1_query_identity`.
pub type K1QueryIdentity = unsafe extern "C" fn(State, usize, *const f64, usize, *mut f64, *mut u8) -> Status;

/// `elpis_ecsg_k1_fms_query_identity`.
pub type K1FmsQueryIdentity = unsafe extern "C" fn(State, u64, usize, *const f64, usize, *mut f64, *mut u8) -> Status;

/// The function table of a standalone K1 library (`ecsg_k1.h`). Every entry is required.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct K1Api {
    pub state_digest: Option<unsafe extern "C" fn(State, *mut u8) -> Status>,
    pub shape: Option<unsafe extern "C" fn(State, *mut usize, *mut usize) -> Status>,
    pub query_identity: Option<K1QueryIdentity>,
    pub lease_claim: Option<unsafe extern "C" fn(State, u64) -> Status>,
    pub lease_release: Option<unsafe extern "C" fn(State, u64) -> Status>,
    pub leased_reserve: Option<unsafe extern "C" fn(State, u64, usize) -> Status>,
    pub leased_txn_begin: Option<unsafe extern "C" fn(State, u64, *mut u8, *mut u64) -> Status>,
    pub leased_txn_run_schedule: Option<K1RunSchedule>,
    pub leased_txn_commit_identity: Option<unsafe extern "C" fn(State, u64, u64, *mut CommitIdentity) -> Status>,
    pub txn_abort: Option<unsafe extern "C" fn(State, u64) -> Status>,
    pub snapshot_write: Option<unsafe extern "C" fn(State, *mut u8, usize) -> Status>,
    pub txn_snapshot_write: Option<unsafe extern "C" fn(State, u64, *mut u8, usize) -> Status>,
}

/// The function table of the K1 FMS adapter (`ecsg_k1_fms.h`): the same operations on one resident state id.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct K1FmsApi {
    pub state_digest: Option<unsafe extern "C" fn(State, u64, *mut u8) -> Status>,
    pub shape: Option<unsafe extern "C" fn(State, u64, *mut usize, *mut usize) -> Status>,
    pub query_identity: Option<K1FmsQueryIdentity>,
    pub lease_claim: Option<unsafe extern "C" fn(State, u64, u64) -> Status>,
    pub lease_release: Option<unsafe extern "C" fn(State, u64, u64) -> Status>,
    pub leased_reserve: Option<unsafe extern "C" fn(State, u64, u64, usize) -> Status>,
    pub leased_txn_begin: Option<unsafe extern "C" fn(State, u64, u64, *mut u8, *mut u64) -> Status>,
    pub leased_txn_run_schedule: Option<K1FmsRunSchedule>,
    pub leased_txn_commit_identity: Option<unsafe extern "C" fn(State, u64, u64, u64, *mut CommitIdentity) -> Status>,
    pub txn_abort: Option<unsafe extern "C" fn(State, u64, u64) -> Status>,
    pub snapshot_write: Option<unsafe extern "C" fn(State, u64, *mut u8, usize) -> Status>,
    pub txn_snapshot_write: Option<unsafe extern "C" fn(State, u64, u64, *mut u8, usize) -> Status>,
}

/// `elpis_runtime_substrate`: one native K1 state as the caller describes it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CSubstrate {
    /// 1: standalone K1 (`elpis_ecsg_k1 *`); 2: FMS-resident K1 (`elpis_ecsg_k1_fms *` and `id`).
    pub kind: u32,
    pub reserved: u32,
    pub handle: *mut c_void,
    /// The resident state id (kind 2); 0 for a standalone state.
    pub id: u64,
    pub owner: u64,
    /// The state's input dimension, 1..=64, verified natively against the state's shape before any input is read.
    pub dim: u64,
    /// `const elpis_runtime_k1_api *` (kind 1) or `const elpis_runtime_k1_fms_api *` (kind 2).
    pub api: *const c_void,
}

#[derive(Clone, Copy)]
enum Table {
    K1(K1Api),
    Fms(K1FmsApi),
}

/// A validated descriptor: a [`K1Ops`] over the caller's native function table (copied by value).
#[derive(Clone, Copy)]
pub struct Native {
    key: Key,
    handle: *mut c_void,
    table: Table,
}

// SAFETY: a `Native` is a native state handle plus the K1 library's own entry points. The K1 and K1 FMS ABIs may
// be called from any thread: each state guards itself (an overlapping call is refused BUSY, never a data race).
// RuntimeCore moves a retained copy only together with the runtime handle that serializes its calls.
unsafe impl Send for Native {}

macro_rules! complete {
    ($t:expr) => {
        $t.state_digest.is_some()
            && $t.shape.is_some()
            && $t.query_identity.is_some()
            && $t.lease_claim.is_some()
            && $t.lease_release.is_some()
            && $t.leased_reserve.is_some()
            && $t.leased_txn_begin.is_some()
            && $t.leased_txn_run_schedule.is_some()
            && $t.leased_txn_commit_identity.is_some()
            && $t.txn_abort.is_some()
            && $t.snapshot_write.is_some()
            && $t.txn_snapshot_write.is_some()
    };
}

impl Native {
    /// Validate a descriptor (no native call). `None` for anything malformed.
    ///
    /// # Safety
    /// `d.api` must point to the table its `kind` names (it is copied), and `d.handle` must stay a live state of
    /// that library, with the library loaded, for as long as the returned value or a capability retained from it
    /// ([`K1Ops::retain_abort`]) is used.
    pub unsafe fn from_c(d: &CSubstrate) -> Option<Native> {
        let dim = usize::try_from(d.dim).ok()?;
        if d.reserved != 0 || d.handle.is_null() || d.api.is_null() || features(dim) == 0 {
            return None;
        }
        let table = match d.kind {
            KIND_K1 if d.id == 0 => {
                let t = *(d.api as *const K1Api);
                complete!(t).then_some(Table::K1(t))?
            }
            KIND_K1_FMS => {
                let t = *(d.api as *const K1FmsApi);
                complete!(t).then_some(Table::Fms(t))?
            }
            _ => return None,
        };
        let key = Key { kind: d.kind, handle: d.handle as usize, id: d.id, owner: d.owner, dim };
        Some(Native { key, handle: d.handle, table })
    }

    pub fn key(&self) -> Key {
        self.key
    }
}

// The entries were checked present by `from_c`; `expect` documents that invariant.
const TABLE: &str = "validated K1 function table";

/// One native call through the validated table: `$k1` for a standalone state, `$fms` (with the resident id) for
/// an FMS-resident one.
macro_rules! call {
    ($self:ident, $entry:ident, ($($arg:expr),*)) => {
        unsafe {
            match &$self.table {
                Table::K1(t) => t.$entry.expect(TABLE)($self.handle $(, $arg)*),
                Table::Fms(t) => t.$entry.expect(TABLE)($self.handle, $self.key.id $(, $arg)*),
            }
        }
    };
}

impl K1Ops for Native {
    fn state_digest(&mut self, out: &mut Digest) -> i32 {
        call!(self, state_digest, (out.as_mut_ptr()))
    }

    fn shape(&mut self, dim: &mut usize, width: &mut usize) -> i32 {
        call!(self, shape, (dim, width))
    }

    fn query_identity(&mut self, dim: usize, x: &[f64], out: &mut [f64], digest: &mut Digest) -> i32 {
        call!(self, query_identity, (dim, x.as_ptr(), out.len(), out.as_mut_ptr(), digest.as_mut_ptr()))
    }

    fn lease_claim(&mut self, lease: u64) -> i32 {
        call!(self, lease_claim, (lease))
    }

    fn lease_release(&mut self, lease: u64) -> i32 {
        call!(self, lease_release, (lease))
    }

    fn reserve(&mut self, lease: u64, max_rows: usize) -> i32 {
        call!(self, leased_reserve, (lease, max_rows))
    }

    fn txn_begin(&mut self, lease: u64, source: &mut Digest, token: &mut u64) -> i32 {
        call!(self, leased_txn_begin, (lease, source.as_mut_ptr(), token))
    }

    fn txn_run_schedule(
        &mut self,
        lease: u64,
        token: u64,
        x: &[f64],
        y: &[f64],
        schedule: &[Experience],
        rate: f64,
        s3: &mut [f64],
        result: &mut ScheduleOutcome,
    ) -> i32 {
        call!(
            self,
            leased_txn_run_schedule,
            (
                lease,
                token,
                x.as_ptr(),
                y.as_ptr(),
                y.len(),
                schedule.as_ptr(),
                schedule.len(),
                rate,
                s3.as_mut_ptr(),
                s3.len(),
                result
            )
        )
    }

    fn txn_commit_identity(&mut self, lease: u64, token: u64, out: &mut CommitIdentity) -> i32 {
        call!(self, leased_txn_commit_identity, (lease, token, out))
    }

    fn txn_abort(&mut self, token: u64) -> i32 {
        TxnAbort::txn_abort(self, token)
    }

    fn snapshot_write(&mut self, out: &mut [u8]) -> i32 {
        call!(self, snapshot_write, (out.as_mut_ptr(), out.len()))
    }

    fn txn_snapshot_write(&mut self, token: u64, out: &mut [u8]) -> i32 {
        call!(self, txn_snapshot_write, (token, out.as_mut_ptr(), out.len()))
    }

    fn retain_abort(&self) -> Box<dyn TxnAbort> {
        Box::new(*self)
    }
}

impl TxnAbort for Native {
    fn txn_abort(&mut self, token: u64) -> i32 {
        call!(self, txn_abort, (token))
    }
}
