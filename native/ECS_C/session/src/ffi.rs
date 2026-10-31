//! The C ABI of the volatile session (include/elpis/ecs_c_session.h, ABI v1). Every function returns
//! `ELPIS_ECS_C_OK` (0) or a positive stable code, and writes its output only on success. A handle is owned by one
//! caller at a time; calls on one handle must not overlap. No function performs I/O.

use core::alloc::Layout;
use core::ffi::c_char;
use core::ptr;

use crate::error::Code;
use crate::model::{Cognition, Digest, Entity, EntityKind, Event, EVENT_BYTES};
use crate::register::EvolutionState;
use crate::session::{Coordinate, Session, SessionConfig, SessionReceipt};

pub const ABI_VERSION: u32 = 1;
const OK: i32 = 0;

fn code(c: Code) -> i32 {
    c as i32
}

#[repr(C)]
pub struct CConfig {
    pub abi_version: u32,
    pub capacity: u32,
    pub nonce: [u8; 32],
    pub operator_label: [u8; 32],
    pub runtime_label: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CCoordinate {
    pub session: [u8; 32],
    pub instance: u64,
    pub generation: u64,
    pub state_root: [u8; 32],
}

/// The entity row, laid out as `elpis_ecs_c_entity` (include/elpis/ecs_c_types.h).
#[repr(C)]
pub struct CEntity {
    pub id: [u8; 32],
    pub kind: u8,
    pub lifecycle: u8,
    pub reserved: [u8; 2],
    pub per_turn_limit: u32,
    pub founding_index: u64,
    pub founded_generation: u64,
    pub last_generation: u64,
    pub sender_watermark: u64,
    pub source_version: u64,
    pub label: [u8; 32],
    pub scope: [u8; 32],
    pub source_authority: u32,
    pub reserved2: u32,
}

/// The evolution authority, laid out as `elpis_ecs_c_evolution`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CEvolution {
    pub revision: u64,
    pub head: [u8; 32],
    pub pending: u8,
    pub reserved: [u8; 7],
    pub assertion: [u8; 32],
}

#[repr(C)]
pub struct CReceipt {
    pub generation: u64,
    pub predecessor_generation: u64,
    pub receipt_count: u64,
    pub founded_present: u8,
    pub reserved: [u8; 7],
    pub receipt: [u8; 32],
    pub commit: [u8; 32],
    pub state_root: [u8; 32],
    pub receipt_head: [u8; 32],
    pub founded: [u8; 32],
}

#[repr(C)]
pub struct CStatus {
    pub capacity: u32,
    pub entity_count: u32,
    pub instance: u64,
    pub generation: u64,
    pub session: [u8; 32],
    pub state_root: [u8; 32],
    pub commit: [u8; 32],
    pub anchored: u8,
    pub state_format: u8,
    pub reserved: [u8; 6],
    pub identity: [u8; 32],
    pub publications: u64,
    pub last_publication_generation: u64,
    pub evolution: CEvolution,
    pub evolution_digest: [u8; 32],
    pub receipt_head: [u8; 32],
    pub receipt_count: u64,
    pub evidence_head: [u8; 32],
    pub evidence_count: u64,
    pub reclaimed_head: [u8; 32],
    pub reclaimed_count: u64,
    pub next_founding_index: u64,
    pub volatile_bytes: u64,
    pub transitions: u64,
    pub refusals: u64,
}

const _: () = assert!(core::mem::size_of::<CConfig>() == 104);
const _: () = assert!(core::mem::size_of::<CCoordinate>() == 80);
const _: () = assert!(core::mem::size_of::<CEntity>() == 152);
const _: () = assert!(core::mem::size_of::<CEvolution>() == 80);
const _: () = assert!(core::mem::size_of::<CReceipt>() == 192);
const _: () = assert!(core::mem::size_of::<CStatus>() == 440);

fn c_entity(e: &Entity) -> CEntity {
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

fn c_evolution(e: &EvolutionState) -> CEvolution {
    let mut out =
        CEvolution { revision: e.revision(), head: e.head(), pending: 0, reserved: [0; 7], assertion: [0; 32] };
    if let Some(a) = e.assertion() {
        out.pending = 1;
        out.assertion = a;
    }
    out
}

fn evolution_from(c: &CEvolution) -> Result<EvolutionState, Code> {
    if c.reserved != [0; 7] {
        return Err(Code::Invalid);
    }
    match c.pending {
        0 if c.assertion == [0; 32] => EvolutionState::idle(c.revision, c.head),
        1 => EvolutionState::pending(c.revision, c.head, c.assertion),
        _ => Err(Code::Invalid),
    }
}

fn coordinate_from(c: &CCoordinate) -> Coordinate {
    Coordinate { session: c.session, instance: c.instance, generation: c.generation, state_root: c.state_root }
}

fn c_receipt(r: &SessionReceipt) -> CReceipt {
    CReceipt {
        generation: r.generation,
        predecessor_generation: r.predecessor_generation,
        receipt_count: r.receipt_count,
        founded_present: r.founded.is_some() as u8,
        reserved: [0; 7],
        receipt: r.receipt,
        commit: r.commit,
        state_root: r.state_root,
        receipt_head: r.receipt_head,
        founded: r.founded.unwrap_or([0; 32]),
    }
}

/// The stable name of a code ("ECS_C_OK" for 0), or NULL. Static storage.
fn name_of(value: i32) -> Option<&'static [u8]> {
    if value == OK {
        return Some(b"ECS_C_OK\0");
    }
    let c = Code::from_i32(value).or_else(|| Code::VOLATILE.iter().copied().find(|c| *c as i32 == value))?;
    Some(match c {
        Code::Invalid => b"ECS_C_INVALID\0",
        Code::Stale => b"ECS_C_STALE\0",
        Code::Unauthorized => b"ECS_C_UNAUTHORIZED\0",
        Code::UnknownEntity => b"ECS_C_UNKNOWN_ENTITY\0",
        Code::Lifecycle => b"ECS_C_LIFECYCLE\0",
        Code::Duplicate => b"ECS_C_DUPLICATE\0",
        Code::SequenceGap => b"ECS_C_SEQUENCE_GAP\0",
        Code::Capacity => b"ECS_C_CAPACITY\0",
        Code::Exhausted => b"ECS_C_EXHAUSTED\0",
        Code::SourceStale => b"ECS_C_SOURCE_STALE\0",
        Code::ScopeConflict => b"ECS_C_SCOPE_CONFLICT\0",
        Code::Unanchored => b"ECS_C_UNANCHORED\0",
        Code::AlreadyAnchored => b"ECS_C_ALREADY_ANCHORED\0",
        Code::StateMismatch => b"ECS_C_STATE_MISMATCH\0",
        Code::EvolutionMismatch => b"ECS_C_EVOLUTION_MISMATCH\0",
        Code::EvolutionPending => b"ECS_C_EVOLUTION_PENDING\0",
        Code::EvolutionNotPending => b"ECS_C_EVOLUTION_NOT_PENDING\0",
        Code::Corrupt => b"ECS_C_CORRUPT\0",
        Code::SessionMismatch => b"ECS_C_SESSION_MISMATCH\0",
        Code::PersistenceRetired => b"ECS_C_PERSISTENCE_RETIRED\0",
        Code::NoMem => b"ECS_C_NOMEM\0",
        Code::EntropyUnavailable => b"ECS_C_ENTROPY_UNAVAILABLE\0",
        // The durable authority's storage codes: a session never returns them.
        _ => return None,
    })
}

#[no_mangle]
pub extern "C" fn elpis_ecs_c_session_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn elpis_ecs_c_session_code_name(value: i32) -> *const c_char {
    name_of(value).map_or(ptr::null(), |s| s.as_ptr() as *const c_char)
}

/// Exact bytes a session of `capacity` reserves for its life, or 0 outside 2..=4096.
#[no_mangle]
pub extern "C" fn elpis_ecs_c_session_bytes(capacity: u32) -> u64 {
    Session::volatile_bytes_for(capacity) as u64
}

/// # Safety
/// `config` and `out` must be valid pointers (or null, refused `Invalid`).
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_create(config: *const CConfig, out: *mut *mut Session) -> i32 {
    if config.is_null() || out.is_null() {
        return code(Code::Invalid);
    }
    *out = ptr::null_mut();
    let c = &*config;
    if c.abi_version != ABI_VERSION {
        return code(Code::Invalid);
    }
    let session = match Session::create(&SessionConfig {
        capacity: c.capacity,
        nonce: c.nonce,
        operator_label: c.operator_label,
        runtime_label: c.runtime_label,
    }) {
        Ok(s) => s,
        Err(e) => return code(e),
    };
    // The handle is one fallible allocation: out of memory is a refusal, never an abort.
    let layout = Layout::new::<Session>();
    let p = alloc::alloc::alloc(layout) as *mut Session;
    if p.is_null() {
        drop(session);
        return code(Code::NoMem);
    }
    p.write(session);
    *out = p;
    OK
}

/// Release a session: its memory is freed (its state image cleared first). Null is ignored.
///
/// # Safety
/// `session` must be a handle from `elpis_ecs_c_session_create`, not yet destroyed.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_destroy(session: *mut Session) {
    if session.is_null() {
        return;
    }
    ptr::drop_in_place(session);
    alloc::alloc::dealloc(session as *mut u8, Layout::new::<Session>());
}

unsafe fn get<'a>(session: *const Session) -> Result<&'a Session, i32> {
    session.as_ref().ok_or(code(Code::Invalid))
}

/// # Safety
/// `session` must be a live handle (or null); `out` a valid pointer (or null).
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_read_coordinate(session: *const Session, out: *mut CCoordinate) -> i32 {
    let s = match get(session) {
        Ok(s) => s,
        Err(e) => return e,
    };
    if out.is_null() {
        return code(Code::Invalid);
    }
    let c = s.coordinate();
    out.write(CCoordinate {
        session: c.session,
        instance: c.instance,
        generation: c.generation,
        state_root: c.state_root,
    });
    OK
}

/// # Safety
/// As `elpis_ecs_c_session_read_coordinate`.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_read_status(session: *const Session, out: *mut CStatus) -> i32 {
    let s = match get(session) {
        Ok(s) => s,
        Err(e) => return e,
    };
    if out.is_null() {
        return code(Code::Invalid);
    }
    let st = s.state();
    let (anchored, state_format, identity) = match st.cognition {
        Cognition::Unanchored => (0u8, 0u8, [0u8; 32]),
        Cognition::Anchored { format, identity } => (1, format as u8, identity),
    };
    let (transitions, refusals) = s.counters();
    out.write(CStatus {
        capacity: s.capacity(),
        entity_count: st.entities.len() as u32,
        instance: s.instance(),
        generation: s.generation(),
        session: *s.lineage(),
        state_root: *s.state_root(),
        commit: *s.commit(),
        anchored,
        state_format,
        reserved: [0; 6],
        identity,
        publications: st.publications,
        last_publication_generation: st.last_publication_generation,
        evolution: c_evolution(&st.evolution),
        evolution_digest: st.evolution.digest(),
        receipt_head: st.receipt_head,
        receipt_count: st.receipt_count,
        evidence_head: st.evidence_head,
        evidence_count: st.evidence_count,
        reclaimed_head: st.reclaimed_head,
        reclaimed_count: st.reclaimed_count,
        next_founding_index: st.next_founding_index,
        volatile_bytes: s.volatile_bytes() as u64,
        transitions,
        refusals,
    });
    OK
}

/// # Safety
/// As `elpis_ecs_c_session_read_coordinate`.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_entity_at(session: *const Session, index: u32, out: *mut CEntity) -> i32 {
    let s = match get(session) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match (s.state().entities.get(index as usize), out.is_null()) {
        (Some(e), false) => {
            out.write(c_entity(e));
            OK
        }
        _ => code(Code::Invalid),
    }
}

/// # Safety
/// As `elpis_ecs_c_session_read_coordinate`; `id` a valid 32-byte buffer (or null).
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_entity_find(
    session: *const Session,
    id: *const [u8; 32],
    out: *mut CEntity,
) -> i32 {
    let s = match get(session) {
        Ok(s) => s,
        Err(e) => return e,
    };
    if id.is_null() || out.is_null() {
        return code(Code::Invalid);
    }
    match s.state().entity(&*id) {
        Some(e) => {
            out.write(c_entity(e));
            OK
        }
        None => code(Code::UnknownEntity),
    }
}

/// # Safety
/// As `elpis_ecs_c_session_read_coordinate`.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_source_for_authority(
    session: *const Session,
    source_authority: u32,
    out: *mut CEntity,
) -> i32 {
    let s = match get(session) {
        Ok(s) => s,
        Err(e) => return e,
    };
    if out.is_null() {
        return code(Code::Invalid);
    }
    match s.state().source_for_authority(source_authority) {
        Some(e) => {
            out.write(c_entity(e));
            OK
        }
        None => code(Code::UnknownEntity),
    }
}

/// # Safety
/// As `elpis_ecs_c_session_read_coordinate`; `label` and `out` valid 32-byte buffers (or null).
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_next_entity_id(
    session: *const Session,
    kind: u8,
    label: *const [u8; 32],
    out: *mut [u8; 32],
) -> i32 {
    let s = match get(session) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let kind = match EntityKind::from_u8(kind) {
        Some(k) => k,
        None => return code(Code::Invalid),
    };
    if label.is_null() || out.is_null() {
        return code(Code::Invalid);
    }
    out.write(s.next_entity_id(kind, &*label));
    OK
}

/// Typed-evidence admission (pure): OK with the per-turn bound, or why not.
///
/// # Safety
/// As `elpis_ecs_c_session_read_coordinate`; `source` and `scope` valid 32-byte buffers; `out_limit` valid or null.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_check_evidence(
    session: *const Session,
    source: *const [u8; 32],
    version: u64,
    scope: *const [u8; 32],
    out_limit: *mut u32,
) -> i32 {
    let s = match get(session) {
        Ok(s) => s,
        Err(e) => return e,
    };
    if source.is_null() || scope.is_null() {
        return code(Code::Invalid);
    }
    match s.check_evidence(&*source, version, &*scope) {
        Ok(limit) => {
            if !out_limit.is_null() {
                out_limit.write(limit);
            }
            OK
        }
        Err(e) => code(e),
    }
}

/// Compare-and-publish one 512-byte event, all or nothing. `out` (optional) receives the receipt.
///
/// # Safety
/// `session` a live handle; `expected` valid; `event` a valid 512-byte buffer; `out` valid or null.
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_transition(
    session: *mut Session,
    expected: *const CCoordinate,
    event: *const [u8; EVENT_BYTES],
    out: *mut CReceipt,
) -> i32 {
    let s = match session.as_mut() {
        Some(s) => s,
        None => return code(Code::Invalid),
    };
    if expected.is_null() || event.is_null() {
        return code(Code::Invalid);
    }
    let decoded = match Event::decode(&*event) {
        Ok(e) => e,
        Err(e) => return code(e),
    };
    match s.transition(&coordinate_from(&*expected), &decoded) {
        Ok(r) => {
            if !out.is_null() {
                out.write(c_receipt(&r));
            }
            OK
        }
        Err(e) => code(e),
    }
}

/// The next receipt chain head (pure): verify kept receipts against a session's receipt head.
///
/// # Safety
/// Valid 32-byte buffers (or null, refused).
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_receipt_chain(
    head: *const [u8; 32],
    receipt: *const [u8; 32],
    out: *mut [u8; 32],
) -> i32 {
    if head.is_null() || receipt.is_null() || out.is_null() {
        return code(Code::Invalid);
    }
    out.write(crate::model::receipt_chain(&*head, &*receipt));
    OK
}

/// `elpis.continuity.evolution-authority.v2` of an evolution authority (pure).
///
/// # Safety
/// Valid pointers (or null, refused).
#[no_mangle]
pub unsafe extern "C" fn elpis_ecs_c_session_evolution_digest(evolution: *const CEvolution, out: *mut Digest) -> i32 {
    if evolution.is_null() || out.is_null() {
        return code(Code::Invalid);
    }
    match evolution_from(&*evolution) {
        Ok(e) => {
            out.write(e.digest());
            OK
        }
        Err(e) => code(e),
    }
}
