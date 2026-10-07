//! The stable C ABI (include/elpis/continuity.h). No object framework, no events, no history.
//!
//! Every function returns `ELPIS_CONTINUITY_OK` (0) or a positive stable code. A store handle is owned by
//! one caller at a time; calls on one handle must not overlap.
//!
//! # Safety
//!
//! Every pointer argument is either NULL (refused with `CONTINUITY_INVALID`, or the documented meaning) or
//! valid for its declared size: 32 bytes for a digest, 176 for a record, `len` for a path or raw record,
//! one struct for a snapshot or evolution authority. Store handles come from `elpis_continuity_store_create`
//! and are used until `elpis_continuity_store_destroy`.
#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, OsStr};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use crate::error::Code;
use crate::record::{Cognition, Digest, EvolutionState, Snapshot, RECORD_SIZE, ZERO};
use crate::store::Store;

pub const ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CEvolution {
    pub revision: u64,
    pub head: Digest,
    pub pending: u8,
    pub reserved: [u8; 7],
    pub assertion: Digest,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CSnapshot {
    pub generation: u64,
    pub anchored: u8,
    pub reserved: [u8; 7],
    pub k1_state_digest: Digest,
    pub evolution: CEvolution,
    pub evolution_digest: Digest,
    pub record_digest: Digest,
}

const _: () = assert!(std::mem::size_of::<CEvolution>() == 80);
const _: () = assert!(std::mem::size_of::<CSnapshot>() == 192);

/// Opaque store handle.
pub struct ElpisContinuityStore(Store);

fn rc(result: Result<(), Code>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(code) => code as i32,
    }
}

fn to_c_evolution(e: &EvolutionState) -> CEvolution {
    CEvolution {
        revision: e.revision(),
        head: e.head(),
        pending: e.assertion().is_some() as u8,
        reserved: [0; 7],
        assertion: e.assertion().unwrap_or(ZERO),
    }
}

fn to_c(s: &Snapshot) -> CSnapshot {
    let (anchored, k1) = match s.cognition() {
        Cognition::Unanchored => (0, ZERO),
        Cognition::Anchored(d) => (1, d),
    };
    CSnapshot {
        generation: s.generation(),
        anchored,
        reserved: [0; 7],
        k1_state_digest: k1,
        evolution: to_c_evolution(&s.evolution()),
        evolution_digest: s.evolution().digest(),
        record_digest: s.digest(),
    }
}

/// A well-formed evolution authority, or `Corrupt` (malformed flag/reserved bytes, an idle authority
/// carrying an assertion, counter bounds or the genesis rule).
fn from_c_evolution(e: &CEvolution) -> Result<EvolutionState, Code> {
    if e.reserved != [0; 7] {
        return Err(Code::Corrupt);
    }
    match e.pending {
        0 if e.assertion == ZERO => EvolutionState::idle(e.revision, e.head),
        1 => EvolutionState::pending(e.revision, e.head, e.assertion),
        _ => Err(Code::Corrupt),
    }
}

fn from_c(s: &CSnapshot) -> Result<Snapshot, Code> {
    if s.reserved != [0; 7] {
        return Err(Code::Corrupt);
    }
    let cognition = match s.anchored {
        0 if s.k1_state_digest == ZERO => Cognition::Unanchored,
        1 => Cognition::Anchored(s.k1_state_digest),
        _ => return Err(Code::Corrupt),
    };
    Snapshot::new(s.generation, cognition, from_c_evolution(&s.evolution)?)
}

unsafe fn digest_arg<'a>(p: *const u8) -> Option<&'a Digest> {
    if p.is_null() {
        None
    } else {
        Some(&*(p as *const Digest))
    }
}

unsafe fn write_out(out: *mut CSnapshot, s: &Snapshot) {
    if !out.is_null() {
        *out = to_c(s);
    }
}

unsafe fn store<'a>(p: *mut ElpisContinuityStore) -> Result<&'a mut Store, Code> {
    p.as_mut().map(|s| &mut s.0).ok_or(Code::Invalid)
}

#[no_mangle]
pub extern "C" fn elpis_continuity_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn elpis_continuity_record_size() -> usize {
    RECORD_SIZE
}

/// The stable name of a code (`CONTINUITY_OK` for 0), or NULL for an unknown value. Static storage.
#[no_mangle]
pub extern "C" fn elpis_continuity_code_name(code: i32) -> *const c_char {
    if code == 0 {
        return c"CONTINUITY_OK".as_ptr();
    }
    match Code::from_i32(code) {
        Some(c) => match c {
            Code::Uninitialized => c"CONTINUITY_UNINITIALIZED",
            Code::Unanchored => c"CONTINUITY_UNANCHORED",
            Code::AlreadyAnchored => c"CONTINUITY_ALREADY_ANCHORED",
            Code::StateMismatch => c"CONTINUITY_STATE_MISMATCH",
            Code::Corrupt => c"CONTINUITY_CORRUPT",
            Code::PublicationRefused => c"CONTINUITY_PUBLICATION_REFUSED",
            Code::PublicationUncertain => c"CONTINUITY_PUBLICATION_UNCERTAIN",
            Code::Locked => c"CONTINUITY_LOCKED",
            Code::AuthorityMismatch => c"CONTINUITY_AUTHORITY_MISMATCH",
            Code::EvolutionPending => c"CONTINUITY_EVOLUTION_PENDING",
            Code::EvolutionNotPending => c"CONTINUITY_EVOLUTION_NOT_PENDING",
            Code::Exhausted => c"CONTINUITY_EXHAUSTED",
            Code::Invalid => c"CONTINUITY_INVALID",
            Code::Path => c"CONTINUITY_PATH",
            Code::Open => c"CONTINUITY_OPEN",
            Code::LegacyStorage => c"CONTINUITY_LEGACY_STORAGE",
            Code::Io => c"CONTINUITY_IO",
            Code::TestingProcessDeath => c"CONTINUITY_TESTING_PROCESS_DEATH",
        }
        .as_ptr(),
        None => std::ptr::null(),
    }
}

// -- pure record functions --------------------------------------------------------------------------

/// The evolution-authority v2 digest of a well-formed authority (`CONTINUITY_CORRUPT` otherwise).
#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_evolution_digest(e: *const CEvolution, out: *mut u8) -> i32 {
    if e.is_null() || out.is_null() {
        return Code::Invalid as i32;
    }
    rc(from_c_evolution(&*e).map(|e| *(out as *mut Digest) = e.digest()))
}

/// Encode a well-formed snapshot as one 176-byte record (`CONTINUITY_CORRUPT` for an invalid snapshot).
/// The digest fields of the input are ignored.
#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_record_encode(s: *const CSnapshot, out: *mut u8) -> i32 {
    if s.is_null() || out.is_null() {
        return Code::Invalid as i32;
    }
    rc(from_c(&*s).map(|s| *(out as *mut [u8; RECORD_SIZE]) = s.encode()))
}

/// Decode one record. `*empty` is 1 for an all-zero slot (then `*out` is untouched), else 0.
#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_record_decode(
    raw: *const u8,
    len: usize,
    out: *mut CSnapshot,
    empty: *mut i32,
) -> i32 {
    if raw.is_null() || out.is_null() || empty.is_null() {
        return Code::Invalid as i32;
    }
    rc(Snapshot::decode(std::slice::from_raw_parts(raw, len)).map(|decoded| match decoded {
        None => *empty = 1,
        Some(s) => {
            *empty = 0;
            *out = to_c(&s);
        }
    }))
}

// -- the store ---------------------------------------------------------------------------------------

/// Create a closed store for an absolute directory path (`len` bytes, no terminator needed).
#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_store_create(
    path: *const u8,
    len: usize,
    out: *mut *mut ElpisContinuityStore,
) -> i32 {
    if path.is_null() || out.is_null() {
        return Code::Invalid as i32;
    }
    *out = std::ptr::null_mut();
    let dir = Path::new(OsStr::from_bytes(std::slice::from_raw_parts(path, len)));
    rc(Store::new(dir).map(|s| *out = Box::into_raw(Box::new(ElpisContinuityStore(s)))))
}

/// Close (if open) and free a store; sets `*store` to NULL.
#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_store_destroy(store: *mut *mut ElpisContinuityStore) {
    if !store.is_null() && !(*store).is_null() {
        drop(Box::from_raw(*store));
        *store = std::ptr::null_mut();
    }
}

#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_store_open(s: *mut ElpisContinuityStore, out: *mut CSnapshot) -> i32 {
    rc(store(s).and_then(|s| s.open()).map(|snap| write_out(out, &snap)))
}

#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_store_close(s: *mut ElpisContinuityStore) {
    if let Ok(s) = store(s) {
        s.close();
    }
}

#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_store_snapshot(s: *mut ElpisContinuityStore, out: *mut CSnapshot) -> i32 {
    if out.is_null() {
        return Code::Invalid as i32;
    }
    rc(store(s).and_then(|s| s.snapshot()).map(|snap| write_out(out, &snap)))
}

/// Explicit anchor of the first K1 lineage. A NULL digest is `CONTINUITY_INVALID`.
#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_anchor_cognition(
    s: *mut ElpisContinuityStore,
    k1: *const u8,
    out: *mut CSnapshot,
) -> i32 {
    rc(store(s).and_then(|s| s.anchor_cognition(digest_arg(k1))).map(|snap| write_out(out, &snap)))
}

#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_commit_cognition(
    s: *mut ElpisContinuityStore,
    before: *const u8,
    after: *const u8,
    out: *mut CSnapshot,
) -> i32 {
    rc(store(s)
        .and_then(|s| s.commit_cognition(digest_arg(before), digest_arg(after)))
        .map(|snap| write_out(out, &snap)))
}

/// Durably reserve one assertion. A NULL or malformed `expected` cannot be the current authority
/// (`CONTINUITY_AUTHORITY_MISMATCH`); a NULL assertion is `CONTINUITY_INVALID`.
#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_reserve_evolution(
    s: *mut ElpisContinuityStore,
    expected: *const CEvolution,
    assertion: *const u8,
    out: *mut CSnapshot,
) -> i32 {
    let expected = expected.as_ref().and_then(|e| from_c_evolution(e).ok());
    rc(store(s)
        .and_then(|s| s.reserve_evolution(expected.as_ref(), digest_arg(assertion)))
        .map(|snap| write_out(out, &snap)))
}

/// Finalize the exact pending authority with the established receipt digest (also the explicit
/// reconciliation primitive after restart).
#[no_mangle]
pub unsafe extern "C" fn elpis_continuity_finalize_evolution(
    s: *mut ElpisContinuityStore,
    expected: *const CEvolution,
    receipt: *const u8,
    out: *mut CSnapshot,
) -> i32 {
    let expected = expected.as_ref().and_then(|e| from_c_evolution(e).ok());
    rc(store(s)
        .and_then(|s| s.finalize_evolution(expected.as_ref(), digest_arg(receipt)))
        .map(|snap| write_out(out, &snap)))
}

// -- testing library only -----------------------------------------------------------------------------

#[cfg(feature = "testing")]
mod testing {
    use super::*;
    use crate::store::probe::{Counters, Fault};
    use crate::store::Step;

    /// Arm one fault for the `publication`-th publication from now (0: the next open's initialization).
    /// Actions: 0 clear, 1 die at step `arg`, 2 write fails, 3 torn write of `arg` bytes then refusal,
    /// 4 torn write of `arg` bytes then death, 5 sync fails with the bytes lost, 6 sync fails but durable.
    #[no_mangle]
    pub unsafe extern "C" fn elpis_continuity_testing_fault(
        s: *mut ElpisContinuityStore,
        publication: u64,
        action: u32,
        arg: u64,
    ) -> i32 {
        let s = match store(s) {
            Ok(s) => s,
            Err(c) => return c as i32,
        };
        let fault = match action {
            0 => None,
            1 => match Step::from_u32(arg as u32) {
                Some(step) => Some(Fault::Die(step)),
                None => return Code::Invalid as i32,
            },
            2 => Some(Fault::WriteFail),
            3 => Some(Fault::TornFail(arg as usize)),
            4 => Some(Fault::TornDie(arg as usize)),
            5 => Some(Fault::SyncFailLost),
            6 => Some(Fault::SyncFailDurable),
            _ => return Code::Invalid as i32,
        };
        s.probe.publications = 0;
        s.probe.plan = fault.map(|f| (publication, f));
        0
    }

    /// Copy the store's I/O counters; reset them when `reset` is nonzero.
    #[no_mangle]
    pub unsafe extern "C" fn elpis_continuity_testing_counters(
        s: *mut ElpisContinuityStore,
        out: *mut Counters,
        reset: i32,
    ) -> i32 {
        let s = match store(s) {
            Ok(s) => s,
            Err(c) => return c as i32,
        };
        if out.is_null() {
            return Code::Invalid as i32;
        }
        *out = s.probe.counters;
        if reset != 0 {
            s.probe.counters = Counters::default();
        }
        0
    }
}
