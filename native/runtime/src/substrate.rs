//! The native K1 retained state a runtime drives, reached through the K1 C ABI.
//!
//! RuntimeCore links no ECS code. A caller hands it a substrate descriptor: the native handle, the K1 function
//! table of its library (`ecsg_k1.h` for a standalone state, `ecsg_k1_fms.h` for an FMS-resident one), the
//! declared input dimension and the caller identity of the handle's owner. Every [`K1Ops`] call is exactly one
//! native K1 crossing; RuntimeCore computes no ECS mathematics and touches no ECS byte.
//!
//! ABI v4 (docs/RUNTIME_CORE.md, "Live and durable authority"): the managed operations reach the state through its
//! live coordinate (ecsg_k1.h, "Live coordinate"): `leased_query`, `leased_txn_begin_at` and `leased_txn_commit`
//! bind every answer, begin and commit to the state's generation under RuntimeCore's lease and compute no digest;
//! `leased_identity` (the SHA-256 retained-state identity with the generation) is read once when a state is bound;
//! `leased_capture` and the pure `envelope_seal` produce the envelope that durable publication writes, the capture
//! on the caller's thread and the seal on the durability lane. The digest-bearing entries of v3 (`query_identity`,
//! `leased_txn_begin`, `leased_txn_commit_identity`, `snapshot_write`, `txn_snapshot_write`) are not in the v4
//! tables: RuntimeCore no longer calls them.
//!
//! From a successful managed-turn begin until that turn ends, RuntimeCore retains one capability over the state:
//! [`TxnAbort`], the native `txn_abort` of the turn's transaction (a copy of the descriptor's handle, resident id
//! and abort entry; never commit, schedule or any other operation). It is what ends the turn when the runtime is
//! closed, reopened or destroyed without an explicit commit or abort, so an open native transaction is never
//! forgotten. The C ABI (v2) makes the substrate's lifetime explicit: the state must stay live until the turn ends.
//!
//! An FMS-resident state is held resident for as long as it is bound (docs/RUNTIME_CORE.md, "HOT residency"):
//! binding takes the adapter's residency hold (`resident_hold`, only if the state is WARM now) before anything else
//! touches it, and from then on every operation on it pins resident-only, so no HOT operation ever promotes it,
//! reads the cold store or waits for a move. RuntimeCore retains a second capability over the state while bound:
//! [`ResidentRelease`], which drops that hold on release, close, reopen and destruction. A standalone K1 state and a
//! composite are RAM objects: their hold is a no-op.
//!
//! A third kind is the ECS composite retained state (`ecs_composite.h`, docs/ECS_COMPOSITE_STATE.md): K1 and the ECS
//! memory under one identity (`ELPISEC1`). Its table is the K1 table (same signatures and status numbers; the
//! digests are composite identities) plus one entry, `leased_txn_observe`, which hands an opaque observation record
//! to the composite's open transaction. RuntimeCore never reads that record: it passes the pointer through.

use std::ffi::{c_int, c_void};

pub use crate::authority::StateFormat;

/// A caller's observation record (`elpis_ecs_mem_observation *`), opaque to RuntimeCore: it is never read here,
/// only handed to the composite's `leased_txn_observe`. The C ABI caller guarantees it is valid for the call.
#[derive(Clone, Copy, Debug)]
pub struct Observation(pub *const c_void);

/// The memory `source_authority` a typed observation records (`elpis_ecs_mem_observation`: `abi_version` u32 at 0,
/// which must be `ELPIS_ECS_MEM_ABI_V1`; `source_authority` u32 at 24), or None for a NULL or foreign record. ABI v6
/// admission under the ECS_C authority reads it; the memory itself records it and never interprets it.
pub fn observation_source(observation: &Observation) -> Option<u32> {
    if observation.0.is_null() {
        return None;
    }
    let base = observation.0 as *const u8;
    // SAFETY: a non-NULL observation is a valid `elpis_ecs_mem_observation` (the C ABI contract of the native
    // observe it is passed to); both fields lie inside its fixed header.
    unsafe {
        if std::ptr::read_unaligned(base as *const u32) != 1 {
            return None;
        }
        Some(std::ptr::read_unaligned(base.add(24) as *const u32))
    }
}

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
/// The FMS-resident state is not WARM while residency is required (`ELPIS_ECSG_K1_FMS_NOT_RESIDENT`: demoted, or a
/// move in flight): refused without pinning, promoting, reading the cold store or waiting.
pub const K1_FMS_NOT_RESIDENT: i32 = -112;

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

/// The S3 readout length of a K1 state of input dimension `dim` (`elpis_ecsg_k1_features`):
/// `d + d(d+1)/2 + d(d+1)(d+2)/6`, the number of degree-1..3 monomials. 0 outside 1..=64.
pub fn features(dim: usize) -> usize {
    if !(1..=MAX_DIM).contains(&dim) {
        return 0;
    }
    dim + dim * (dim + 1) / 2 + dim * (dim + 1) * (dim + 2) / 6
}

/// The `ELPISGK1` envelope size of a `dim x width` K1 state (`elpis_ecsg_k1_envelope_bytes`): the 64-byte header,
/// W, H packed and a as binary64, and the 32-byte trailer. `None` on an invalid shape or an overflow. K1 verifies it
/// natively: a capture into a buffer of another size is refused (`leased_capture` INVALID, `envelope_seal`
/// INVALID), never written short.
pub fn k1_envelope_bytes(dim: usize, width: usize) -> Option<usize> {
    let f = features(dim);
    if f == 0 || width == 0 {
        return None;
    }
    let values = dim.checked_mul(width)?.checked_add(f.checked_mul(f + 1)? / 2)?.checked_add(f)?;
    values.checked_mul(8)?.checked_add(64 + 32)
}

/// The pure sealer of the substrate's envelope format (`envelope_seal`): writes the SHA-256 trailer(s) of an unsealed
/// envelope in place. No state; callable from any thread (the durability lane calls it).
pub type Sealer = unsafe extern "C" fn(*mut u8, usize) -> c_int;

/// The native operations a runtime needs on one retained K1 state. Each returns the K1 status.
///
/// Every mutating operation is a *leased* entry point (ecsg_k1.h, "Managed ownership"): it presents RuntimeCore's
/// lease, and the state refuses it (`LEASED`) unless that lease holds the state. While RuntimeCore holds the lease
/// the state refuses every unmanaged mutating entry point, so the bound state cannot move outside the lineage. The
/// live-coordinate entries (ecsg_k1.h, "Live coordinate") report the state's generation under the same guard as
/// their answer, begin, commit or capture, and refuse another lease before reading anything.
pub trait K1Ops {
    /// The retained-state identity of the authoritative state (unleased read; used to check a state before it is
    /// claimed, so a state that is not the lineage is never claimed).
    fn state_digest(&mut self, out: &mut Digest) -> i32;
    /// The state's immutable shape `(dim, width)` (`shape`): read-only, no guard.
    fn shape(&mut self, dim: &mut usize, width: &mut usize) -> i32;
    /// Claim the state for `lease` (replacing any other lease; BUSY while a transaction is open).
    fn lease_claim(&mut self, lease: u64) -> i32;
    /// Release the state if `lease` holds it.
    fn lease_release(&mut self, lease: u64) -> i32;
    fn reserve(&mut self, lease: u64, max_rows: usize) -> i32;
    /// The retained-state identity and the generation of the same state, for the lease holder (one guarded call).
    fn leased_identity(&mut self, lease: u64, digest: &mut Digest, generation: &mut u64) -> i32;
    /// QUERY for the lease holder (`leased_query`): `out[r] = f_W(x_r)` for `out.len()` rows of the declared
    /// dimension and the generation of the same authoritative state, in one native call. Read-only; no digest.
    fn leased_query(&mut self, lease: u64, dim: usize, x: &[f64], out: &mut [f64], generation: &mut u64) -> i32;
    /// Begin a transaction and report the generation of exactly the source it copied (no digest).
    fn txn_begin_at(&mut self, lease: u64, source_generation: &mut u64, token: &mut u64) -> i32;
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
    /// Commit the open transaction and report the transition (generations before and after; no digest).
    fn txn_commit(&mut self, lease: u64, token: u64, out: &mut Transition) -> i32;
    fn txn_abort(&mut self, token: u64) -> i32;
    /// The authoritative state's envelope UNSEALED (its trailer zero) and the generation of the same state, for the
    /// lease holder (one guarded call). `envelope_seal` completes it.
    fn leased_capture(&mut self, lease: u64, out: &mut [u8], generation: &mut u64) -> i32;
    /// The pure sealer of this substrate's envelope format.
    fn sealer(&self) -> Sealer;
    /// The owned abort capability over this same native state, retained by RuntimeCore from a successful turn
    /// begin until that turn ends. It may outlive the call that produced it, but never the open transaction it
    /// ends: the native state is live for exactly that interval (the substrate lifetime contract, runtime.h).
    fn retain_abort(&self) -> Box<dyn TxnAbort>;
    /// Hold the state resident while it is bound (FMS-resident states; a no-op for RAM states): only if it is WARM
    /// now, never promoting it; `K1_FMS_NOT_RESIDENT` otherwise. Every later operation then pins resident-only.
    fn resident_hold(&mut self) -> i32 {
        K1_OK
    }
    /// The owned capability that drops the hold [`K1Ops::resident_hold`] took (None for RAM states), retained by
    /// RuntimeCore for exactly as long as the state is bound.
    fn retain_resident(&self) -> Option<Box<dyn ResidentRelease>> {
        None
    }
    /// The retained-state format this substrate's identities name (`ELPISGK1` for K1 states).
    fn format(&self) -> StateFormat {
        StateFormat::K1
    }
    /// OBSERVE on the open transaction's memory candidate (composite substrates only; `observation` is the
    /// caller's opaque `elpis_ecs_mem_observation`, never read here). A K1 substrate has no memory: INVALID.
    fn txn_observe(
        &mut self,
        _lease: u64,
        _token: u64,
        _observation: Observation,
        _key: &mut Digest,
        _memory_status: &mut i32,
    ) -> i32 {
        K1_INVALID
    }
}

/// The one operation RuntimeCore retains across calls: ending the open managed turn's native transaction.
pub trait TxnAbort: Send {
    /// `txn_abort` of the K1 ABI: OK (aborted, or no transaction open), INVALID (not this token's transaction),
    /// BUSY (a concurrent overlapping call; nothing done).
    fn txn_abort(&mut self, token: u64) -> i32;
}

/// The other operation RuntimeCore retains across calls while a state is bound: dropping its residency hold.
pub trait ResidentRelease: Send {
    /// `resident_release` of the K1 FMS ABI: OK, INVALID (no hold), BUSY (a concurrent overlapping call), or an
    /// FMS refusal (the hold stands).
    fn resident_release(&mut self) -> i32;
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
/// An ECS composite retained state (`elpis_ecs_composite *`, docs/ECS_COMPOSITE_STATE.md).
pub const KIND_COMPOSITE: u32 = 3;

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

/// `elpis_ecsg_k1_leased_query`.
pub type K1LeasedQuery = unsafe extern "C" fn(State, u64, usize, *const f64, usize, *mut f64, *mut u64) -> Status;

/// `elpis_ecsg_k1_fms_leased_query`.
pub type K1FmsLeasedQuery =
    unsafe extern "C" fn(State, u64, u64, usize, *const f64, usize, *mut f64, *mut u64) -> Status;

/// The function table of a standalone K1 library (`ecsg_k1.h`), ABI v4. Every entry is required.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct K1Api {
    pub state_digest: Option<unsafe extern "C" fn(State, *mut u8) -> Status>,
    pub shape: Option<unsafe extern "C" fn(State, *mut usize, *mut usize) -> Status>,
    pub lease_claim: Option<unsafe extern "C" fn(State, u64) -> Status>,
    pub lease_release: Option<unsafe extern "C" fn(State, u64) -> Status>,
    pub leased_reserve: Option<unsafe extern "C" fn(State, u64, usize) -> Status>,
    pub leased_identity: Option<unsafe extern "C" fn(State, u64, *mut u8, *mut u64) -> Status>,
    pub leased_query: Option<K1LeasedQuery>,
    pub leased_txn_begin_at: Option<unsafe extern "C" fn(State, u64, *mut u64, *mut u64) -> Status>,
    pub leased_txn_run_schedule: Option<K1RunSchedule>,
    pub leased_txn_commit: Option<unsafe extern "C" fn(State, u64, u64, *mut Transition) -> Status>,
    pub txn_abort: Option<unsafe extern "C" fn(State, u64) -> Status>,
    pub leased_capture: Option<unsafe extern "C" fn(State, u64, *mut u8, usize, *mut u64) -> Status>,
    pub envelope_seal: Option<Sealer>,
}

/// `elpis_ecs_composite_leased_txn_observe` (the observation and key are opaque to RuntimeCore).
pub type CompositeObserve = unsafe extern "C" fn(State, u64, u64, *const c_void, *mut u8, *mut c_int) -> Status;

/// The function table of an ECS composite (`ecs_composite.h`): the K1 table over composite identities, then the
/// memory OBSERVE of the open transaction. Every entry is required.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CompositeApi {
    pub base: K1Api,
    pub leased_txn_observe: Option<CompositeObserve>,
}

/// The function table of the K1 FMS adapter (`ecsg_k1_fms.h`), ABI v4: the same operations on one resident state
/// id (the sealer is pure and takes no id).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct K1FmsApi {
    pub state_digest: Option<unsafe extern "C" fn(State, u64, *mut u8) -> Status>,
    pub shape: Option<unsafe extern "C" fn(State, u64, *mut usize, *mut usize) -> Status>,
    pub lease_claim: Option<unsafe extern "C" fn(State, u64, u64) -> Status>,
    pub lease_release: Option<unsafe extern "C" fn(State, u64, u64) -> Status>,
    pub leased_reserve: Option<unsafe extern "C" fn(State, u64, u64, usize) -> Status>,
    pub leased_identity: Option<unsafe extern "C" fn(State, u64, u64, *mut u8, *mut u64) -> Status>,
    pub leased_query: Option<K1FmsLeasedQuery>,
    pub leased_txn_begin_at: Option<unsafe extern "C" fn(State, u64, u64, *mut u64, *mut u64) -> Status>,
    pub leased_txn_run_schedule: Option<K1FmsRunSchedule>,
    pub leased_txn_commit: Option<unsafe extern "C" fn(State, u64, u64, u64, *mut Transition) -> Status>,
    pub txn_abort: Option<unsafe extern "C" fn(State, u64, u64) -> Status>,
    pub leased_capture: Option<unsafe extern "C" fn(State, u64, u64, *mut u8, usize, *mut u64) -> Status>,
    pub envelope_seal: Option<Sealer>,
    /// `elpis_ecsg_k1_fms_resident_hold` / `_resident_release` (docs/RUNTIME_CORE.md, "HOT residency").
    pub resident_hold: Option<unsafe extern "C" fn(State, u64) -> Status>,
    pub resident_release: Option<unsafe extern "C" fn(State, u64) -> Status>,
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
    /// `const elpis_runtime_k1_api *` (kind 1), `const elpis_runtime_k1_fms_api *` (kind 2) or
    /// `const elpis_runtime_composite_api *` (kind 3), ABI v4 layouts.
    pub api: *const c_void,
}

#[derive(Clone, Copy)]
enum Table {
    K1(K1Api),
    Fms(K1FmsApi),
    Composite(CompositeApi),
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
            && $t.lease_claim.is_some()
            && $t.lease_release.is_some()
            && $t.leased_reserve.is_some()
            && $t.leased_identity.is_some()
            && $t.leased_query.is_some()
            && $t.leased_txn_begin_at.is_some()
            && $t.leased_txn_run_schedule.is_some()
            && $t.leased_txn_commit.is_some()
            && $t.txn_abort.is_some()
            && $t.leased_capture.is_some()
            && $t.envelope_seal.is_some()
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
                (complete!(t) && t.resident_hold.is_some() && t.resident_release.is_some()).then_some(Table::Fms(t))?
            }
            KIND_COMPOSITE if d.id == 0 => {
                let t = *(d.api as *const CompositeApi);
                (complete!(t.base) && t.leased_txn_observe.is_some()).then_some(Table::Composite(t))?
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
                Table::Composite(t) => t.base.$entry.expect(TABLE)($self.handle $(, $arg)*),
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

    fn lease_claim(&mut self, lease: u64) -> i32 {
        call!(self, lease_claim, (lease))
    }

    fn lease_release(&mut self, lease: u64) -> i32 {
        call!(self, lease_release, (lease))
    }

    fn reserve(&mut self, lease: u64, max_rows: usize) -> i32 {
        call!(self, leased_reserve, (lease, max_rows))
    }

    fn leased_identity(&mut self, lease: u64, digest: &mut Digest, generation: &mut u64) -> i32 {
        call!(self, leased_identity, (lease, digest.as_mut_ptr(), generation))
    }

    fn leased_query(&mut self, lease: u64, dim: usize, x: &[f64], out: &mut [f64], generation: &mut u64) -> i32 {
        call!(self, leased_query, (lease, dim, x.as_ptr(), out.len(), out.as_mut_ptr(), generation))
    }

    fn txn_begin_at(&mut self, lease: u64, source_generation: &mut u64, token: &mut u64) -> i32 {
        call!(self, leased_txn_begin_at, (lease, source_generation, token))
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

    fn txn_commit(&mut self, lease: u64, token: u64, out: &mut Transition) -> i32 {
        call!(self, leased_txn_commit, (lease, token, out))
    }

    fn txn_abort(&mut self, token: u64) -> i32 {
        TxnAbort::txn_abort(self, token)
    }

    fn leased_capture(&mut self, lease: u64, out: &mut [u8], generation: &mut u64) -> i32 {
        call!(self, leased_capture, (lease, out.as_mut_ptr(), out.len(), generation))
    }

    fn sealer(&self) -> Sealer {
        match &self.table {
            Table::K1(t) => t.envelope_seal,
            Table::Fms(t) => t.envelope_seal,
            Table::Composite(t) => t.base.envelope_seal,
        }
        .expect(TABLE)
    }

    fn retain_abort(&self) -> Box<dyn TxnAbort> {
        Box::new(*self)
    }

    fn resident_hold(&mut self) -> i32 {
        match &self.table {
            // SAFETY: the validated FMS table's own entry over the descriptor's live handle and resident id.
            Table::Fms(t) => unsafe { t.resident_hold.expect(TABLE)(self.handle, self.key.id) },
            _ => K1_OK,
        }
    }

    fn retain_resident(&self) -> Option<Box<dyn ResidentRelease>> {
        matches!(self.table, Table::Fms(_)).then(|| Box::new(*self) as Box<dyn ResidentRelease>)
    }

    fn format(&self) -> StateFormat {
        match self.table {
            Table::Composite(_) => StateFormat::Composite,
            _ => StateFormat::K1,
        }
    }

    fn txn_observe(
        &mut self,
        lease: u64,
        token: u64,
        observation: Observation,
        key: &mut Digest,
        memory_status: &mut i32,
    ) -> i32 {
        match &self.table {
            Table::Composite(t) => unsafe {
                t.leased_txn_observe.expect(TABLE)(
                    self.handle,
                    lease,
                    token,
                    observation.0,
                    key.as_mut_ptr(),
                    memory_status,
                )
            },
            _ => K1_INVALID,
        }
    }
}

impl TxnAbort for Native {
    fn txn_abort(&mut self, token: u64) -> i32 {
        call!(self, txn_abort, (token))
    }
}

impl ResidentRelease for Native {
    fn resident_release(&mut self) -> i32 {
        match &self.table {
            // SAFETY: as `resident_hold`; the substrate lifetime contract keeps the state live while it is bound.
            Table::Fms(t) => unsafe { t.resident_release.expect(TABLE)(self.handle, self.key.id) },
            _ => K1_OK,
        }
    }
}
