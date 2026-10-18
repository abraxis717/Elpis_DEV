//! The stable C ABI (include/elpis/ecs_c_authority.h).
//!
//! Every function returns `ELPIS_ECS_C_OK` (0) or a positive stable code. A handle is owned by one caller at a
//! time; calls on one handle must not overlap.
//!
//! # Safety
//!
//! Every pointer argument is either NULL (refused with `ECS_C_INVALID`, unless documented as optional) or valid
//! for its declared size: 32 bytes for a digest, 512 for an event, 176 for a legacy record, `len` for a path, one
//! struct for an output. Handles come from `elpis_ecs_c_open` and are used until `elpis_ecs_c_close`.
#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, CString, OsStr};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::OnceLock;

use elpis_continuity::ffi::{from_c_evolution, to_c_evolution, CEvolution};
use elpis_continuity::RECORD_SIZE;

use crate::error::Code;
use crate::legacy::read_continuity;
use crate::model::{Cognition, Digest, Entity, EntityKind, Event, EVENT_BYTES, ZERO};
use crate::record::record_bytes;
use crate::store::{Authority, Receipt, Report};
use crate::{MAX_CAPACITY, MIN_CAPACITY};

pub const ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CProvisioned {
    pub capacity: u32,
    pub reserved: u32,
    pub record_bytes: u64,
    pub lineage: Digest,
    pub deployment: Digest,
    pub record_digest: Digest,
    pub state_root: Digest,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CReport {
    pub disposition: u32,
    pub mode: u32,
    pub slot_class: [u8; 2],
    pub witness_class: [u8; 2],
    pub replay_verified: u8,
    pub witness_completed: u8,
    pub evolution_pending: u8,
    pub reserved: u8,
    pub slot_generation: [u64; 2],
    pub witness_generation: u64,
    pub current_generation: u64,
    pub reads: u64,
    pub read_bytes: u64,
    pub replays: u64,
    pub observed: [Digest; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CStatus {
    pub generation: u64,
    pub record_bytes: u64,
    pub capacity: u32,
    pub entity_count: u32,
    pub mode: u32,
    pub anchored: u8,
    pub state_format: u8,
    pub legacy_imported: u8,
    pub legacy_format: u8,
    pub lineage: Digest,
    pub deployment: Digest,
    pub record_digest: Digest,
    pub state_root: Digest,
    pub identity: Digest,
    pub publications: u64,
    pub last_publication_generation: u64,
    pub evolution: CEvolution,
    pub evolution_digest: Digest,
    pub receipt_head: Digest,
    pub receipt_count: u64,
    pub evidence_head: Digest,
    pub evidence_count: u64,
    pub reconcile_head: Digest,
    pub reconcile_count: u64,
    pub reclaimed_head: Digest,
    pub reclaimed_count: u64,
    pub next_founding_index: u64,
    pub legacy_generation: u64,
    pub legacy_digest: Digest,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CEntity {
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CReceipt {
    pub generation: u64,
    pub predecessor_generation: u64,
    pub receipt_count: u64,
    pub founded_present: u8,
    pub reserved: [u8; 7],
    pub receipt: Digest,
    pub record_digest: Digest,
    pub state_root: Digest,
    pub receipt_head: Digest,
    pub founded: Digest,
}

const _: () = assert!(std::mem::size_of::<CProvisioned>() == 144);
const _: () = assert!(std::mem::size_of::<CReport>() == 136);
const _: () = assert!(std::mem::size_of::<CStatus>() == 528);
const _: () = assert!(std::mem::size_of::<CEntity>() == 152);
const _: () = assert!(std::mem::size_of::<CReceipt>() == 192);

/// Opaque authority handle.
pub struct ElpisEcsCAuthority(pub Authority);

fn rc(result: Result<(), Code>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(code) => code as i32,
    }
}

unsafe fn path<'a>(ptr: *const c_char, len: usize) -> Result<&'a Path, Code> {
    if ptr.is_null() || len == 0 {
        return Err(Code::Invalid);
    }
    let bytes = std::slice::from_raw_parts(ptr as *const u8, len);
    if bytes.contains(&0) {
        return Err(Code::Path);
    }
    Ok(Path::new(OsStr::from_bytes(bytes)))
}

unsafe fn digest(ptr: *const u8) -> Result<Digest, Code> {
    if ptr.is_null() {
        return Err(Code::Invalid);
    }
    Ok(*(ptr as *const Digest))
}

unsafe fn handle<'a>(h: *const ElpisEcsCAuthority) -> Result<&'a Authority, Code> {
    h.as_ref().map(|h| &h.0).ok_or(Code::Invalid)
}

unsafe fn handle_mut<'a>(h: *mut ElpisEcsCAuthority) -> Result<&'a mut Authority, Code> {
    h.as_mut().map(|h| &mut h.0).ok_or(Code::Invalid)
}

unsafe fn write<T>(out: *mut T, value: T) -> Result<(), Code> {
    if out.is_null() {
        return Err(Code::Invalid);
    }
    out.write(value);
    Ok(())
}

pub fn to_c_report(r: &Report) -> CReport {
    CReport {
        disposition: r.disposition,
        mode: r.mode,
        slot_class: r.slot_class,
        witness_class: r.witness_class,
        replay_verified: r.replay_verified as u8,
        witness_completed: r.witness_completed as u8,
        evolution_pending: r.evolution_pending as u8,
        reserved: 0,
        slot_generation: r.slot_generation,
        witness_generation: r.witness_generation,
        current_generation: r.current_generation,
        reads: r.reads,
        read_bytes: r.read_bytes,
        replays: r.replays,
        observed: r.observed,
    }
}

pub fn to_c_entity(e: &Entity) -> CEntity {
    CEntity {
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
    }
}

fn to_c_receipt(r: &Receipt) -> CReceipt {
    CReceipt {
        generation: r.generation,
        predecessor_generation: r.predecessor_generation,
        receipt_count: r.receipt_count,
        founded_present: r.founded.is_some() as u8,
        reserved: [0; 7],
        receipt: r.receipt,
        record_digest: r.record_digest,
        state_root: r.state_root,
        receipt_head: r.receipt_head,
        founded: r.founded.unwrap_or(ZERO),
    }
}

pub fn status(a: &Authority) -> Result<CStatus, Code> {
    let c = a.current()?;
    let s = &c.state;
    let (anchored, state_format, identity) = match s.cognition {
        Cognition::Unanchored => (0, 0, ZERO),
        Cognition::Anchored { format, identity } => (1, format as u8, identity),
    };
    let legacy = s.legacy;
    Ok(CStatus {
        generation: c.generation,
        record_bytes: record_bytes(c.capacity) as u64,
        capacity: c.capacity,
        entity_count: s.entities.len() as u32,
        mode: a.mode()?.code(),
        anchored,
        state_format,
        legacy_imported: legacy.is_some() as u8,
        legacy_format: legacy.map(|l| l.format as u8).unwrap_or(0),
        lineage: c.lineage,
        deployment: c.deployment,
        record_digest: c.digest,
        state_root: c.state_root,
        identity,
        publications: s.publications,
        last_publication_generation: s.last_publication_generation,
        evolution: to_c_evolution(&s.evolution),
        evolution_digest: s.evolution.digest(),
        receipt_head: s.receipt_head,
        receipt_count: s.receipt_count,
        evidence_head: s.evidence_head,
        evidence_count: s.evidence_count,
        reconcile_head: s.reconcile_head,
        reconcile_count: s.reconcile_count,
        reclaimed_head: s.reclaimed_head,
        reclaimed_count: s.reclaimed_count,
        next_founding_index: s.next_founding_index,
        legacy_generation: legacy.map(|l| l.generation).unwrap_or(0),
        legacy_digest: legacy.map(|l| l.record_digest).unwrap_or(ZERO),
    })
}

#[no_mangle]
pub extern "C" fn elpis_ecs_c_abi_version() -> u32 {
    ABI_VERSION
}

/// Bytes of one record slot for `capacity`, or 0 when the capacity is outside the provisionable range.
#[no_mangle]
pub extern "C" fn elpis_ecs_c_record_bytes(capacity: u32) -> u64 {
    if (MIN_CAPACITY..=MAX_CAPACITY).contains(&capacity) {
        record_bytes(capacity) as u64
    } else {
        0
    }
}

/// The stable name of a code (`ECS_C_OK` for 0), or NULL for an unknown value. Static storage.
#[no_mangle]
pub extern "C" fn elpis_ecs_c_code_name(code: i32) -> *const c_char {
    static NAMES: OnceLock<Vec<(i32, CString)>> = OnceLock::new();
    let names = NAMES.get_or_init(|| {
        let mut v = vec![(0, CString::new("ECS_C_OK").expect("name"))];
        v.extend(Code::ALL.iter().map(|c| (*c as i32, CString::new(c.name()).expect("name"))));
        v
    });
    names.iter().find(|(c, _)| *c == code).map(|(_, n)| n.as_ptr()).unwrap_or(std::ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_provision(
    state_dir: *const c_char,
    state_dir_len: usize,
    authority_dir: *const c_char,
    authority_dir_len: usize,
    capacity: u32,
    operator_label: *const u8,
    runtime_label: *const u8,
    out: *mut CProvisioned,
) -> i32 {
    rc((|| {
        let (sd, ad) = (path(state_dir, state_dir_len)?, path(authority_dir, authority_dir_len)?);
        let (op, rt) = (digest(operator_label)?, digest(runtime_label)?);
        let p = Authority::provision(sd, ad, capacity, &op, &rt)?;
        if !out.is_null() {
            out.write(CProvisioned {
                capacity: p.capacity,
                reserved: 0,
                record_bytes: p.record_bytes,
                lineage: p.lineage,
                deployment: p.deployment,
                record_digest: p.record_digest,
                state_root: p.state_root,
            });
        }
        Ok(())
    })())
}

/// Open and recover. On success `*out` is a handle (its mode may be restricted or reconcile-only: see the
/// report). On `ECS_C_RECOVERY_REFUSED` no handle is returned and the report names the disposition. `report` is
/// optional.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_open(
    state_dir: *const c_char,
    state_dir_len: usize,
    authority_dir: *const c_char,
    authority_dir_len: usize,
    out: *mut *mut ElpisEcsCAuthority,
    report: *mut CReport,
) -> i32 {
    if out.is_null() {
        return Code::Invalid as i32;
    }
    out.write(std::ptr::null_mut());
    let opened = (|| {
        let (sd, ad) = (path(state_dir, state_dir_len)?, path(authority_dir, authority_dir_len)?);
        Authority::new(sd, ad)
    })();
    let mut authority = match opened {
        Ok(a) => a,
        Err(code) => return code as i32,
    };
    match authority.open() {
        Ok(r) => {
            if !report.is_null() {
                report.write(to_c_report(&r));
            }
            out.write(Box::into_raw(Box::new(ElpisEcsCAuthority(authority))));
            0
        }
        Err(refusal) => {
            if let (false, Some(r)) = (report.is_null(), refusal.report.as_deref()) {
                report.write(to_c_report(r));
            }
            refusal.code as i32
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_close(h: *mut ElpisEcsCAuthority) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_read_status(h: *const ElpisEcsCAuthority, out: *mut CStatus) -> i32 {
    rc((|| write(out, status(handle(h)?)?))())
}

#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_read_report(h: *const ElpisEcsCAuthority, out: *mut CReport) -> i32 {
    rc((|| write(out, to_c_report(&handle(h)?.report()?)))())
}

#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_entity_at(h: *const ElpisEcsCAuthority, index: u32, out: *mut CEntity) -> i32 {
    rc((|| {
        let a = handle(h)?;
        let e = a.current()?.state.entities.get(index as usize).ok_or(Code::Invalid)?;
        write(out, to_c_entity(e))
    })())
}

#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_entity_find(
    h: *const ElpisEcsCAuthority,
    id: *const u8,
    out: *mut CEntity,
) -> i32 {
    rc((|| {
        let a = handle(h)?;
        let id = digest(id)?;
        let e = a.current()?.state.entity(&id).ok_or(Code::UnknownEntity)?;
        write(out, to_c_entity(e))
    })())
}

/// The present source row that owns the memory source-authority identifier `source_authority` (any lifecycle), or
/// `ECS_C_UNKNOWN_ENTITY`. Typed memory observations record this identifier; admission maps it to its source.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_source_for_authority(
    h: *const ElpisEcsCAuthority,
    source_authority: u32,
    out: *mut CEntity,
) -> i32 {
    rc((|| {
        let a = handle(h)?;
        let e = a.current()?.state.source_for_authority(source_authority).ok_or(Code::UnknownEntity)?;
        write(out, to_c_entity(e))
    })())
}

/// The identity the next ENTITY_FOUND of `kind` and `label` founds (the receiver such an event must name).
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_next_entity_id(
    h: *const ElpisEcsCAuthority,
    kind: u8,
    label: *const u8,
    out: *mut u8,
) -> i32 {
    rc((|| {
        let a = handle(h)?;
        let kind = EntityKind::from_u8(kind).ok_or(Code::Invalid)?;
        let id = a.next_entity_id(kind, &digest(label)?)?;
        write(out as *mut Digest, id)
    })())
}

#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_check_evidence(
    h: *const ElpisEcsCAuthority,
    source: *const u8,
    version: u64,
    scope: *const u8,
    out_per_turn_limit: *mut u32,
) -> i32 {
    rc((|| {
        let limit = handle(h)?.check_evidence(&digest(source)?, version, &digest(scope)?)?;
        if !out_per_turn_limit.is_null() {
            out_per_turn_limit.write(limit);
        }
        Ok(())
    })())
}

/// Compare-and-publish one 512-byte event. `out` is optional.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_transition(
    h: *mut ElpisEcsCAuthority,
    expected_generation: u64,
    expected_state_root: *const u8,
    event: *const u8,
    out: *mut CReceipt,
) -> i32 {
    rc((|| {
        let a = handle_mut(h)?;
        if event.is_null() {
            return Err(Code::Invalid);
        }
        let event = Event::decode(std::slice::from_raw_parts(event, EVENT_BYTES))?;
        let receipt = a.transition(expected_generation, &digest(expected_state_root)?, &event)?;
        if !out.is_null() {
            out.write(to_c_receipt(&receipt));
        }
        Ok(())
    })())
}

/// Read the current record of a standalone continuity directory (176 bytes) for an IMPORT_LEGACY event.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_legacy_record(directory: *const c_char, len: usize, out: *mut u8) -> i32 {
    rc((|| {
        let record = read_continuity(path(directory, len)?)?;
        write(out as *mut [u8; RECORD_SIZE], record)
    })())
}

/// The next receipt chain head (pure), for verifying kept receipts against `receipt_head`.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_receipt_chain(head: *const u8, receipt: *const u8, out: *mut u8) -> i32 {
    rc((|| write(out as *mut Digest, crate::model::receipt_chain(&digest(head)?, &digest(receipt)?)))())
}

/// `elpis.continuity.evolution-authority.v2` of an evolution authority (pure; byte-identical to continuity's).
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_evolution_digest(evolution: *const CEvolution, out: *mut u8) -> i32 {
    rc((|| {
        let e = from_c_evolution(evolution.as_ref().ok_or(Code::Invalid)?).map_err(|_| Code::Invalid)?;
        write(out as *mut Digest, e.digest())
    })())
}

#[cfg(feature = "testing")]
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_testing_arm(
    h: *mut ElpisEcsCAuthority,
    publication: u64,
    action: u32,
    arg: u64,
) -> i32 {
    rc((|| handle_mut(h)?.testing_arm(publication, action, arg))())
}

#[cfg(feature = "testing")]
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_testing_counters(
    h: *mut ElpisEcsCAuthority,
    reset: i32,
    out: *mut crate::store::Counters,
) -> i32 {
    rc((|| {
        let counters = handle_mut(h)?.testing_counters(reset != 0);
        write(out, counters)
    })())
}
