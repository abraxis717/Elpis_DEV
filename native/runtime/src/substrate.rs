//! The native K1 retained state a runtime drives, reached through the K1 C ABI.
//!
//! RuntimeCore links no ECS code. A caller hands it a substrate descriptor: the native handle, the K1 function
//! table of its library (`ecsg_k1.h` for a standalone state, `ecsg_k1_fms.h` for an FMS-resident one), the
//! declared input dimension and the caller identity of the handle's owner. Every [`K1Ops`] call is exactly one
//! native K1 crossing; RuntimeCore computes no ECS mathematics and touches no ECS byte.

use std::ffi::{c_int, c_void};

pub type Digest = [u8; 32];

/// K1 status values (`elpis_ecsg_k1_status`); the FMS adapter returns them too, or `-100 + fms_status`.
pub const K1_OK: i32 = 0;
pub const K1_NONFINITE: i32 = -2;
pub const K1_STALE: i32 = -3;
pub const K1_CAPACITY: i32 = -5;
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
pub trait K1Ops {
    fn state_digest(&mut self, out: &mut Digest) -> i32;
    fn reserve(&mut self, max_rows: usize) -> i32;
    fn txn_begin(&mut self, token: &mut u64) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn txn_run_schedule(
        &mut self,
        token: u64,
        x: &[f64],
        y: &[f64],
        schedule: &[Experience],
        rate: f64,
        s3: &mut [f64],
        result: &mut ScheduleOutcome,
    ) -> i32;
    fn txn_commit_identity(&mut self, token: u64, out: &mut CommitIdentity) -> i32;
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

/// `elpis_ecsg_k1_txn_run_schedule`.
pub type K1RunSchedule = unsafe extern "C" fn(
    *mut c_void,
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

/// `elpis_ecsg_k1_fms_txn_run_schedule`.
pub type K1FmsRunSchedule = unsafe extern "C" fn(
    *mut c_void,
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

/// The function table of a standalone K1 library (`ecsg_k1.h`). Every entry is required.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct K1Api {
    pub state_digest: Option<unsafe extern "C" fn(*mut c_void, *mut u8) -> Status>,
    pub reserve: Option<unsafe extern "C" fn(*mut c_void, usize) -> Status>,
    pub txn_begin: Option<unsafe extern "C" fn(*mut c_void, *mut u64) -> Status>,
    pub txn_run_schedule: Option<K1RunSchedule>,
    pub txn_commit_identity: Option<unsafe extern "C" fn(*mut c_void, u64, *mut CommitIdentity) -> Status>,
    pub txn_abort: Option<unsafe extern "C" fn(*mut c_void, u64) -> Status>,
}

/// The function table of the K1 FMS adapter (`ecsg_k1_fms.h`): the same operations on one resident state id.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct K1FmsApi {
    pub state_digest: Option<unsafe extern "C" fn(*mut c_void, u64, *mut u8) -> Status>,
    pub reserve: Option<unsafe extern "C" fn(*mut c_void, u64, usize) -> Status>,
    pub txn_begin: Option<unsafe extern "C" fn(*mut c_void, u64, *mut u64) -> Status>,
    pub txn_run_schedule: Option<K1FmsRunSchedule>,
    pub txn_commit_identity: Option<unsafe extern "C" fn(*mut c_void, u64, u64, *mut CommitIdentity) -> Status>,
    pub txn_abort: Option<unsafe extern "C" fn(*mut c_void, u64, u64) -> Status>,
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
    /// The state's input dimension, 1..=64. K1 verifies it natively: the readout length it implies must equal
    /// the state's own before any input byte is read.
    pub dim: u64,
    /// `const elpis_runtime_k1_api *` (kind 1) or `const elpis_runtime_k1_fms_api *` (kind 2).
    pub api: *const c_void,
}

enum Table {
    K1(K1Api),
    Fms(K1FmsApi),
}

/// A validated descriptor: a [`K1Ops`] over the caller's native function table.
pub struct Native {
    key: Key,
    handle: *mut c_void,
    table: Table,
}

impl Native {
    /// Validate a descriptor (no native call). `None` for anything malformed.
    ///
    /// # Safety
    /// `d.api` must point to the table its `kind` names, and `d.handle` must stay a live state of that library
    /// for as long as the returned value is used.
    pub unsafe fn from_c(d: &CSubstrate) -> Option<Native> {
        let dim = usize::try_from(d.dim).ok()?;
        if d.reserved != 0 || d.handle.is_null() || d.api.is_null() || features(dim) == 0 {
            return None;
        }
        let table = match d.kind {
            KIND_K1 if d.id == 0 => {
                let t = *(d.api as *const K1Api);
                let complete = t.state_digest.is_some()
                    && t.reserve.is_some()
                    && t.txn_begin.is_some()
                    && t.txn_run_schedule.is_some()
                    && t.txn_commit_identity.is_some()
                    && t.txn_abort.is_some();
                complete.then_some(Table::K1(t))?
            }
            KIND_K1_FMS => {
                let t = *(d.api as *const K1FmsApi);
                let complete = t.state_digest.is_some()
                    && t.reserve.is_some()
                    && t.txn_begin.is_some()
                    && t.txn_run_schedule.is_some()
                    && t.txn_commit_identity.is_some()
                    && t.txn_abort.is_some();
                complete.then_some(Table::Fms(t))?
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

impl K1Ops for Native {
    fn state_digest(&mut self, out: &mut Digest) -> i32 {
        unsafe {
            match &self.table {
                Table::K1(t) => t.state_digest.expect(TABLE)(self.handle, out.as_mut_ptr()),
                Table::Fms(t) => t.state_digest.expect(TABLE)(self.handle, self.key.id, out.as_mut_ptr()),
            }
        }
    }

    fn reserve(&mut self, max_rows: usize) -> i32 {
        unsafe {
            match &self.table {
                Table::K1(t) => t.reserve.expect(TABLE)(self.handle, max_rows),
                Table::Fms(t) => t.reserve.expect(TABLE)(self.handle, self.key.id, max_rows),
            }
        }
    }

    fn txn_begin(&mut self, token: &mut u64) -> i32 {
        unsafe {
            match &self.table {
                Table::K1(t) => t.txn_begin.expect(TABLE)(self.handle, token),
                Table::Fms(t) => t.txn_begin.expect(TABLE)(self.handle, self.key.id, token),
            }
        }
    }

    fn txn_run_schedule(
        &mut self,
        token: u64,
        x: &[f64],
        y: &[f64],
        schedule: &[Experience],
        rate: f64,
        s3: &mut [f64],
        result: &mut ScheduleOutcome,
    ) -> i32 {
        unsafe {
            match &self.table {
                Table::K1(t) => t.txn_run_schedule.expect(TABLE)(
                    self.handle,
                    token,
                    x.as_ptr(),
                    y.as_ptr(),
                    y.len(),
                    schedule.as_ptr(),
                    schedule.len(),
                    rate,
                    s3.as_mut_ptr(),
                    s3.len(),
                    result,
                ),
                Table::Fms(t) => t.txn_run_schedule.expect(TABLE)(
                    self.handle,
                    self.key.id,
                    token,
                    x.as_ptr(),
                    y.as_ptr(),
                    y.len(),
                    schedule.as_ptr(),
                    schedule.len(),
                    rate,
                    s3.as_mut_ptr(),
                    s3.len(),
                    result,
                ),
            }
        }
    }

    fn txn_commit_identity(&mut self, token: u64, out: &mut CommitIdentity) -> i32 {
        unsafe {
            match &self.table {
                Table::K1(t) => t.txn_commit_identity.expect(TABLE)(self.handle, token, out),
                Table::Fms(t) => t.txn_commit_identity.expect(TABLE)(self.handle, self.key.id, token, out),
            }
        }
    }

    fn txn_abort(&mut self, token: u64) -> i32 {
        unsafe {
            match &self.table {
                Table::K1(t) => t.txn_abort.expect(TABLE)(self.handle, token),
                Table::Fms(t) => t.txn_abort.expect(TABLE)(self.handle, self.key.id, token),
            }
        }
    }
}
