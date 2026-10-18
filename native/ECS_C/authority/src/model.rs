//! The ECS_C authority model: the materialized state, the typed events, and the one transition law ([`apply`]).
//!
//! The same law validates a request before anything is written and replays the newest transition at recovery, so
//! a durable record is admitted only if its predecessor plus its event reproduce it byte for byte. Nothing here
//! performs I/O. Every structure is fixed-size for a provisioned capacity: the state holds at most `capacity`
//! entity rows, counters and chain heads, never a collection that grows with turns, events or time.
//!
//! Integers are little-endian. Every domain string ends in a NUL byte.

use elpis_continuity::sha256;
use elpis_continuity::{EvolutionState, Snapshot, StateFormat, MAX_COUNTER};

use crate::error::Code;

pub type Digest = [u8; 32];
pub const ZERO: Digest = [0; 32];

/// The founding operator and the founding runtime always occupy the first two rows.
pub const MIN_CAPACITY: u32 = 2;
/// The largest entity capacity a deployment may provision (a record of about 656 KiB).
pub const MAX_CAPACITY: u32 = 4096;
/// The largest per-turn observation bound a source may be granted (RuntimeCore's native ceiling).
pub const MAX_PER_TURN: u32 = 256;

pub const STATE_FIXED: usize = 512;
pub const ENTITY_BYTES: usize = 160;
pub const EVENT_BYTES: usize = 512;
pub const PAYLOAD_AT: usize = 80;
pub const PAYLOAD_BYTES: usize = EVENT_BYTES - PAYLOAD_AT;
/// The continuity register record an IMPORT_LEGACY event carries verbatim.
pub const LEGACY_RECORD_BYTES: usize = elpis_continuity::RECORD_SIZE;

pub const fn state_bytes(capacity: u32) -> usize {
    STATE_FIXED + capacity as usize * ENTITY_BYTES
}

pub mod domain {
    pub const ENTITY_ID: &[u8] = b"elpis.ecs_c.entity.v1\x00";
    pub const LINEAGE: &[u8] = b"elpis.ecs_c.authority.lineage.v1\x00";
    pub const STATE_ROOT: &[u8] = b"elpis.ecs_c.authority.state.v1\x00";
    pub const RECEIPT: &[u8] = b"elpis.ecs_c.authority.receipt.v1\x00";
    pub const RECEIPT_CHAIN: &[u8] = b"elpis.ecs_c.authority.receipt-chain.v1\x00";
    pub const EVIDENCE_CHAIN: &[u8] = b"elpis.ecs_c.authority.evidence-chain.v1\x00";
    pub const RECONCILE_CHAIN: &[u8] = b"elpis.ecs_c.authority.reconcile-chain.v1\x00";
    pub const RECLAIM_CHAIN: &[u8] = b"elpis.ecs_c.authority.reclaim-chain.v1\x00";
}

// -- byte helpers ------------------------------------------------------------------------------------------

pub(crate) fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

pub(crate) fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

pub(crate) fn digest_at(b: &[u8], at: usize) -> Digest {
    b[at..at + 32].try_into().expect("32 bytes")
}

pub(crate) fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

pub(crate) fn put_u64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

pub(crate) fn put_digest(b: &mut [u8], at: usize, d: &Digest) {
    b[at..at + 32].copy_from_slice(d);
}

pub(crate) fn zero(b: &[u8]) -> bool {
    b.iter().all(|&x| x == 0)
}

fn chain(domain: &[u8], head: &Digest, parts: &[&[u8]]) -> Digest {
    let mut all: Vec<&[u8]> = Vec::with_capacity(parts.len() + 2);
    all.push(domain);
    all.push(head);
    all.extend_from_slice(parts);
    sha256::digest(&all)
}

fn bump(v: u64, by: u64) -> Result<u64, Code> {
    v.checked_add(by).filter(|n| *n <= MAX_COUNTER).ok_or(Code::Exhausted)
}

fn format_from(v: u8) -> Option<StateFormat> {
    match v {
        0 => Some(StateFormat::K1),
        1 => Some(StateFormat::Composite),
        _ => None,
    }
}

// -- entities ----------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EntityKind {
    Operator = 1,
    Runtime = 2,
    Source = 3,
}

impl EntityKind {
    pub fn from_u8(v: u8) -> Option<EntityKind> {
        [EntityKind::Operator, EntityKind::Runtime, EntityKind::Source].into_iter().find(|k| *k as u8 == v)
    }
}

/// FOUNDED -> ACTIVE <-> DORMANT, and any non-terminal state -> TERMINATED (final).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Lifecycle {
    Founded = 1,
    Active = 2,
    Dormant = 3,
    Terminated = 4,
}

impl Lifecycle {
    pub fn from_u8(v: u8) -> Option<Lifecycle> {
        [Lifecycle::Founded, Lifecycle::Active, Lifecycle::Dormant, Lifecycle::Terminated]
            .into_iter()
            .find(|l| *l as u8 == v)
    }

    pub fn may_become(self, to: Lifecycle) -> bool {
        use Lifecycle::*;
        matches!(
            (self, to),
            (Founded, Active)
                | (Founded, Terminated)
                | (Active, Dormant)
                | (Active, Terminated)
                | (Dormant, Active)
                | (Dormant, Terminated)
        )
    }
}

/// The identity of the entity founded at `founding_index` of a lineage. Founding indices are never reused, so an
/// identity is never founded twice, including after its row is reclaimed.
pub fn entity_id(lineage: &Digest, founding_index: u64, kind: EntityKind, label: &Digest) -> Digest {
    sha256::digest(&[domain::ENTITY_ID, lineage, &founding_index.to_le_bytes(), &[kind as u8], label])
}

/// One entity row (160 bytes).
///
/// | offset | size | field |
/// |---|---|---|
/// | 0 | 32 | id |
/// | 32 | 1 | kind (1 operator, 2 runtime, 3 source) |
/// | 33 | 1 | lifecycle (1 founded, 2 active, 3 dormant, 4 terminated) |
/// | 34 | 2 | reserved, zero |
/// | 36 | 4 | per-turn observation bound (sources: 1..=256; others 0) |
/// | 40 | 8 | founding index |
/// | 48 | 8 | generation founded |
/// | 56 | 8 | generation of the last event that touched it |
/// | 64 | 8 | sender watermark (the last accepted sender sequence) |
/// | 72 | 8 | source version (sources: >= 1; others 0) |
/// | 80 | 32 | label digest (content-free) |
/// | 112 | 32 | scope digest (sources: nonzero; others zero) |
/// | 144 | 16 | reserved, zero |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entity {
    pub id: Digest,
    pub kind: EntityKind,
    pub lifecycle: Lifecycle,
    pub per_turn_limit: u32,
    pub founding_index: u64,
    pub founded_generation: u64,
    pub last_generation: u64,
    pub sender_watermark: u64,
    pub source_version: u64,
    pub label: Digest,
    pub scope: Digest,
}

impl Entity {
    pub fn encode(&self, out: &mut [u8]) {
        out[..ENTITY_BYTES].fill(0);
        put_digest(out, 0, &self.id);
        out[32] = self.kind as u8;
        out[33] = self.lifecycle as u8;
        put_u32(out, 36, self.per_turn_limit);
        put_u64(out, 40, self.founding_index);
        put_u64(out, 48, self.founded_generation);
        put_u64(out, 56, self.last_generation);
        put_u64(out, 64, self.sender_watermark);
        put_u64(out, 72, self.source_version);
        put_digest(out, 80, &self.label);
        put_digest(out, 112, &self.scope);
    }

    fn decode(b: &[u8]) -> Result<Entity, Code> {
        if !zero(&b[34..36]) || !zero(&b[144..160]) {
            return Err(Code::Corrupt);
        }
        Ok(Entity {
            id: digest_at(b, 0),
            kind: EntityKind::from_u8(b[32]).ok_or(Code::Corrupt)?,
            lifecycle: Lifecycle::from_u8(b[33]).ok_or(Code::Corrupt)?,
            per_turn_limit: u32_at(b, 36),
            founding_index: u64_at(b, 40),
            founded_generation: u64_at(b, 48),
            last_generation: u64_at(b, 56),
            sender_watermark: u64_at(b, 64),
            source_version: u64_at(b, 72),
            label: digest_at(b, 80),
            scope: digest_at(b, 112),
        })
    }

    fn check(&self, lineage: &Digest, generation: u64) -> Result<(), Code> {
        let source = self.kind == EntityKind::Source;
        let fields_ok = if source {
            (1..=MAX_PER_TURN).contains(&self.per_turn_limit)
                && (1..=MAX_COUNTER).contains(&self.source_version)
                && self.scope != ZERO
        } else {
            self.per_turn_limit == 0 && self.source_version == 0 && self.scope == ZERO
        };
        let ok = fields_ok
            && self.id == entity_id(lineage, self.founding_index, self.kind, &self.label)
            && self.founded_generation >= 1
            && self.founded_generation <= self.last_generation
            && self.last_generation <= generation
            && self.sender_watermark <= MAX_COUNTER;
        if ok {
            Ok(())
        } else {
            Err(Code::Corrupt)
        }
    }
}

// -- the materialized state ----------------------------------------------------------------------------------

/// The cognition lineage: no lineage yet, or the authoritative retained-state identity of one format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cognition {
    Unanchored,
    Anchored { format: StateFormat, identity: Digest },
}

/// Provenance of an explicit import from a standalone continuity register (v2 or v3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Legacy {
    pub generation: u64,
    pub format: StateFormat,
    pub anchored: bool,
    pub record_digest: Digest,
}

/// The complete materialized ECS_C authority. Every earlier transition is folded into it; only chain heads and
/// counts remember that they happened.
///
/// Fixed part (512 bytes), then `capacity` entity rows (unused rows zero):
///
/// | offset | size | field |
/// |---|---|---|
/// | 0 | 1 | cognition anchored (0/1) |
/// | 1 | 1 | state format (0 ELPISGK1, 1 ELPISEC1; 0 when unanchored) |
/// | 8 | 32 | authoritative retained-state identity (zero when unanchored) |
/// | 40 | 8 | publications (anchor, publish, import) |
/// | 48 | 8 | generation of the last publication (0 if none) |
/// | 56 | 8 | evolution revision |
/// | 64 | 32 | evolution head |
/// | 96 | 1 | evolution pending (0/1) |
/// | 104 | 32 | pending assertion (zero when idle) |
/// | 136 | 32 + 8 | receipt chain head, receipt count |
/// | 176 | 32 + 8 | evidence chain head, evidence count |
/// | 216 | 32 + 8 | reconciliation chain head, count |
/// | 256 | 32 + 8 | reclaimed-row chain head, count |
/// | 296 | 8 | next founding index |
/// | 304 | 4 | entity rows in use |
/// | 312 | 1 + 1 + 1 | legacy imported, legacy anchored, legacy format |
/// | 320 | 8 | legacy generation |
/// | 328 | 32 | legacy record digest |
///
/// Every other byte is reserved and zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub cognition: Cognition,
    pub publications: u64,
    pub last_publication_generation: u64,
    pub evolution: EvolutionState,
    pub receipt_head: Digest,
    pub receipt_count: u64,
    pub evidence_head: Digest,
    pub evidence_count: u64,
    pub reconcile_head: Digest,
    pub reconcile_count: u64,
    pub reclaimed_head: Digest,
    pub reclaimed_count: u64,
    pub next_founding_index: u64,
    pub legacy: Option<Legacy>,
    pub entities: Vec<Entity>,
}

impl State {
    pub fn entity(&self, id: &Digest) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == *id)
    }

    fn index_of(&self, id: &Digest) -> Option<usize> {
        self.entities.iter().position(|e| e.id == *id)
    }

    fn has_active_operator(&self) -> bool {
        self.entities.iter().any(|e| e.kind == EntityKind::Operator && e.lifecycle == Lifecycle::Active)
    }

    /// The per-turn bound a source grants, if typed evidence from `source` at `version` in `scope` is admissible
    /// now. Pure: nothing is recorded.
    pub fn check_evidence(&self, source: &Digest, version: u64, scope: &Digest) -> Result<u32, Code> {
        let e = self.entity(source).ok_or(Code::UnknownEntity)?;
        if e.kind != EntityKind::Source {
            return Err(Code::Unauthorized);
        }
        if e.lifecycle != Lifecycle::Active {
            return Err(Code::Lifecycle);
        }
        if version != e.source_version {
            return Err(Code::SourceStale);
        }
        if *scope != e.scope {
            return Err(Code::ScopeConflict);
        }
        Ok(e.per_turn_limit)
    }

    pub fn encode(&self, capacity: u32, out: &mut [u8]) {
        let out = &mut out[..state_bytes(capacity)];
        out.fill(0);
        if let Cognition::Anchored { format, identity } = self.cognition {
            out[0] = 1;
            out[1] = format as u8;
            put_digest(out, 8, &identity);
        }
        put_u64(out, 40, self.publications);
        put_u64(out, 48, self.last_publication_generation);
        put_u64(out, 56, self.evolution.revision());
        put_digest(out, 64, &self.evolution.head());
        if let Some(assertion) = self.evolution.assertion() {
            out[96] = 1;
            put_digest(out, 104, &assertion);
        }
        put_digest(out, 136, &self.receipt_head);
        put_u64(out, 168, self.receipt_count);
        put_digest(out, 176, &self.evidence_head);
        put_u64(out, 208, self.evidence_count);
        put_digest(out, 216, &self.reconcile_head);
        put_u64(out, 248, self.reconcile_count);
        put_digest(out, 256, &self.reclaimed_head);
        put_u64(out, 288, self.reclaimed_count);
        put_u64(out, 296, self.next_founding_index);
        put_u32(out, 304, self.entities.len() as u32);
        if let Some(legacy) = self.legacy {
            out[312] = 1;
            out[313] = legacy.anchored as u8;
            out[314] = legacy.format as u8;
            put_u64(out, 320, legacy.generation);
            put_digest(out, 328, &legacy.record_digest);
        }
        for (i, e) in self.entities.iter().enumerate() {
            let at = STATE_FIXED + i * ENTITY_BYTES;
            e.encode(&mut out[at..at + ENTITY_BYTES]);
        }
    }

    pub fn to_bytes(&self, capacity: u32) -> Vec<u8> {
        let mut out = vec![0u8; state_bytes(capacity)];
        self.encode(capacity, &mut out);
        out
    }

    /// A fully validated state, or `Corrupt`.
    pub fn decode(b: &[u8], capacity: u32, lineage: &Digest, generation: u64) -> Result<State, Code> {
        if b.len() != state_bytes(capacity) {
            return Err(Code::Corrupt);
        }
        let reserved = [2..8, 97..104, 308..312, 315..320, 360..STATE_FIXED];
        if reserved.iter().any(|r| !zero(&b[r.clone()])) {
            return Err(Code::Corrupt);
        }
        let cognition = match b[0] {
            0 if b[1] == 0 && digest_at(b, 8) == ZERO => Cognition::Unanchored,
            1 => Cognition::Anchored { format: format_from(b[1]).ok_or(Code::Corrupt)?, identity: digest_at(b, 8) },
            _ => return Err(Code::Corrupt),
        };
        let (revision, head, assertion) = (u64_at(b, 56), digest_at(b, 64), digest_at(b, 104));
        let evolution = match b[96] {
            0 if assertion == ZERO => EvolutionState::idle(revision, head),
            1 => EvolutionState::pending(revision, head, assertion),
            _ => return Err(Code::Corrupt),
        }
        .map_err(|_| Code::Corrupt)?;
        let legacy = match b[312] {
            0 if zero(&b[313..315]) && u64_at(b, 320) == 0 && digest_at(b, 328) == ZERO => None,
            1 if b[313] <= 1 => Some(Legacy {
                generation: u64_at(b, 320),
                format: format_from(b[314]).ok_or(Code::Corrupt)?,
                anchored: b[313] == 1,
                record_digest: digest_at(b, 328),
            }),
            _ => return Err(Code::Corrupt),
        };
        let count = u32_at(b, 304);
        if count > capacity {
            return Err(Code::Corrupt);
        }
        let mut entities = Vec::with_capacity(count as usize);
        for i in 0..capacity as usize {
            let row = &b[STATE_FIXED + i * ENTITY_BYTES..STATE_FIXED + (i + 1) * ENTITY_BYTES];
            if i < count as usize {
                entities.push(Entity::decode(row)?);
            } else if !zero(row) {
                return Err(Code::Corrupt);
            }
        }
        let state = State {
            cognition,
            publications: u64_at(b, 40),
            last_publication_generation: u64_at(b, 48),
            evolution,
            receipt_head: digest_at(b, 136),
            receipt_count: u64_at(b, 168),
            evidence_head: digest_at(b, 176),
            evidence_count: u64_at(b, 208),
            reconcile_head: digest_at(b, 216),
            reconcile_count: u64_at(b, 248),
            reclaimed_head: digest_at(b, 256),
            reclaimed_count: u64_at(b, 288),
            next_founding_index: u64_at(b, 296),
            legacy,
            entities,
        };
        state.check(capacity, lineage, generation)?;
        Ok(state)
    }

    /// The state invariants. A state that fails them is never written and never admitted.
    pub fn check(&self, capacity: u32, lineage: &Digest, generation: u64) -> Result<(), Code> {
        let corrupt = Err(Code::Corrupt);
        let n = self.entities.len();
        if n < 2 || n > capacity as usize {
            return corrupt;
        }
        let founders = [(0, EntityKind::Operator), (1, EntityKind::Runtime)];
        for (i, (index, kind)) in founders.into_iter().enumerate() {
            let e = &self.entities[i];
            if e.founding_index != index
                || e.kind != kind
                || e.founded_generation != 1
                || e.lifecycle == Lifecycle::Terminated
            {
                return corrupt;
            }
        }
        for (i, e) in self.entities.iter().enumerate() {
            e.check(lineage, generation)?;
            if i > 0 && e.founding_index <= self.entities[i - 1].founding_index {
                return corrupt;
            }
            if e.founding_index >= self.next_founding_index {
                return corrupt;
            }
        }
        if (n as u64).checked_add(self.reclaimed_count) != Some(self.next_founding_index) || !self.has_active_operator()
        {
            return corrupt;
        }
        let counters = [
            self.publications,
            self.last_publication_generation,
            self.receipt_count,
            self.evidence_count,
            self.reconcile_count,
            self.reclaimed_count,
            self.next_founding_index,
        ];
        if counters.iter().any(|c| *c > MAX_COUNTER) {
            return corrupt;
        }
        if let Cognition::Anchored { identity, .. } = self.cognition {
            if identity == ZERO || self.publications == 0 {
                return corrupt;
            }
        }
        if (self.publications == 0) != (self.last_publication_generation == 0)
            || self.last_publication_generation > generation
        {
            return corrupt;
        }
        if self.receipt_count == 0 || self.receipt_head == ZERO || self.receipt_count > generation {
            return corrupt;
        }
        let chains = [
            (self.evidence_count, self.evidence_head),
            (self.reconcile_count, self.reconcile_head),
            (self.reclaimed_count, self.reclaimed_head),
        ];
        if chains.iter().any(|(count, head)| (*count == 0) != (*head == ZERO)) {
            return corrupt;
        }
        if let Some(legacy) = self.legacy {
            if legacy.generation == 0 || legacy.generation > MAX_COUNTER || legacy.record_digest == ZERO {
                return corrupt;
            }
        }
        Ok(())
    }
}

// -- events --------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum EventKind {
    Genesis = 1,
    EntityFound = 2,
    EntityLifecycle = 3,
    SourceVersion = 4,
    Anchor = 5,
    Publish = 6,
    ImportComposite = 7,
    EvolutionReserve = 8,
    EvolutionFinalize = 9,
    ImportLegacy = 10,
    ReconcileRollback = 11,
    ReconcileUnauthenticated = 12,
    ReconcileRepin = 13,
    EntityReclaim = 14,
}

impl EventKind {
    pub const ALL: [EventKind; 14] = [
        EventKind::Genesis,
        EventKind::EntityFound,
        EventKind::EntityLifecycle,
        EventKind::SourceVersion,
        EventKind::Anchor,
        EventKind::Publish,
        EventKind::ImportComposite,
        EventKind::EvolutionReserve,
        EventKind::EvolutionFinalize,
        EventKind::ImportLegacy,
        EventKind::ReconcileRollback,
        EventKind::ReconcileUnauthenticated,
        EventKind::ReconcileRepin,
        EventKind::EntityReclaim,
    ];

    pub fn from_u16(v: u16) -> Option<EventKind> {
        EventKind::ALL.into_iter().find(|k| *k as u16 == v)
    }

    /// Bytes of the payload this kind uses; the rest of the payload is zero.
    pub fn payload_len(self) -> usize {
        match self {
            EventKind::Genesis => 96,
            EventKind::EntityFound => 80,
            EventKind::EntityLifecycle => 2,
            EventKind::SourceVersion => 56,
            EventKind::Anchor => 40,
            EventKind::Publish => 112,
            EventKind::ImportComposite => 64,
            EventKind::EvolutionReserve | EventKind::EvolutionFinalize => 112,
            EventKind::ImportLegacy => LEGACY_RECORD_BYTES,
            EventKind::ReconcileRollback => 48,
            EventKind::ReconcileUnauthenticated | EventKind::ReconcileRepin => 40,
            EventKind::EntityReclaim => 0,
        }
    }

    /// Events about one entity name it as the receiver; every other event is addressed to the authority itself
    /// (receiver zero).
    pub fn entity_scoped(self) -> bool {
        matches!(
            self,
            EventKind::EntityFound | EventKind::EntityLifecycle | EventKind::SourceVersion | EventKind::EntityReclaim
        )
    }

    /// The authorization matrix: which kind of entity may send this event.
    pub fn sender_may_be(self, kind: EntityKind) -> bool {
        use EventKind::*;
        match self {
            Genesis => false,
            EntityFound
            | EntityLifecycle
            | SourceVersion
            | ImportLegacy
            | ReconcileRollback
            | ReconcileUnauthenticated
            | ReconcileRepin
            | EntityReclaim => kind == EntityKind::Operator,
            Anchor | Publish | ImportComposite => kind == EntityKind::Runtime,
            EvolutionReserve | EvolutionFinalize => matches!(kind, EntityKind::Runtime | EntityKind::Operator),
        }
    }

    pub fn reconciliation(self) -> bool {
        matches!(self, EventKind::ReconcileRollback | EventKind::ReconcileUnauthenticated | EventKind::ReconcileRepin)
    }
}

/// One typed event (512 bytes on the wire):
///
/// | offset | size | field |
/// |---|---|---|
/// | 0 | 2 | kind |
/// | 2 | 6 | reserved, zero |
/// | 8 | 32 | sender entity id (zero only for GENESIS) |
/// | 40 | 8 | sender sequence (= the sender's watermark + 1; zero only for GENESIS) |
/// | 48 | 32 | receiver entity id (entity-scoped kinds), else zero |
/// | 80 | 432 | payload (kind-specific; unused bytes zero) |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub kind: EventKind,
    pub sender: Digest,
    pub seq: u64,
    pub receiver: Digest,
    pub payload: [u8; PAYLOAD_BYTES],
}

fn put_evolution(p: &mut [u8], at: usize, e: &EvolutionState) {
    put_u64(p, at, e.revision());
    put_digest(p, at + 8, &e.head());
    if let Some(assertion) = e.assertion() {
        p[at + 40] = 1;
        put_digest(p, at + 48, &assertion);
    }
}

fn evolution_from(p: &[u8], at: usize) -> Result<EvolutionState, Code> {
    if !zero(&p[at + 41..at + 48]) {
        return Err(Code::Invalid);
    }
    let (revision, head, assertion) = (u64_at(p, at), digest_at(p, at + 8), digest_at(p, at + 48));
    match p[at + 40] {
        0 if assertion == ZERO => EvolutionState::idle(revision, head),
        1 => EvolutionState::pending(revision, head, assertion),
        _ => return Err(Code::Invalid),
    }
    .map_err(|_| Code::Invalid)
}

impl Event {
    fn new(kind: EventKind, sender: &Digest, seq: u64, receiver: &Digest) -> Event {
        Event { kind, sender: *sender, seq, receiver: *receiver, payload: [0; PAYLOAD_BYTES] }
    }

    pub fn genesis(nonce: &Digest, operator_label: &Digest, runtime_label: &Digest) -> Event {
        let mut e = Event::new(EventKind::Genesis, &ZERO, 0, &ZERO);
        put_digest(&mut e.payload, 0, nonce);
        put_digest(&mut e.payload, 32, operator_label);
        put_digest(&mut e.payload, 64, runtime_label);
        e
    }

    /// Found an entity. `id` must be the identity the current state founds next ([`State::next_entity_id`]).
    #[allow(clippy::too_many_arguments)]
    pub fn entity_found(
        sender: &Digest,
        seq: u64,
        id: &Digest,
        kind: EntityKind,
        label: &Digest,
        scope: &Digest,
        source_version: u64,
        per_turn_limit: u32,
    ) -> Event {
        let mut e = Event::new(EventKind::EntityFound, sender, seq, id);
        e.payload[0] = kind as u8;
        put_u32(&mut e.payload, 4, per_turn_limit);
        put_u64(&mut e.payload, 8, source_version);
        put_digest(&mut e.payload, 16, label);
        put_digest(&mut e.payload, 48, scope);
        e
    }

    pub fn entity_lifecycle(sender: &Digest, seq: u64, target: &Digest, from: Lifecycle, to: Lifecycle) -> Event {
        let mut e = Event::new(EventKind::EntityLifecycle, sender, seq, target);
        e.payload[0] = from as u8;
        e.payload[1] = to as u8;
        e
    }

    #[allow(clippy::too_many_arguments)]
    pub fn source_version(
        sender: &Digest,
        seq: u64,
        source: &Digest,
        expected_version: u64,
        new_version: u64,
        per_turn_limit: u32,
        scope: &Digest,
    ) -> Event {
        let mut e = Event::new(EventKind::SourceVersion, sender, seq, source);
        put_u64(&mut e.payload, 0, expected_version);
        put_u64(&mut e.payload, 8, new_version);
        put_u32(&mut e.payload, 16, per_turn_limit);
        put_digest(&mut e.payload, 24, scope);
        e
    }

    pub fn anchor(sender: &Digest, seq: u64, format: StateFormat, identity: &Digest) -> Event {
        let mut e = Event::new(EventKind::Anchor, sender, seq, &ZERO);
        e.payload[0] = format as u8;
        put_digest(&mut e.payload, 8, identity);
        e
    }

    /// Publish `after` as the authoritative identity of a durable commit from `before`, with a summary of the
    /// typed evidence admitted since the previous publication (count and digest; both zero when none).
    #[allow(clippy::too_many_arguments)]
    pub fn publish(
        sender: &Digest,
        seq: u64,
        format: StateFormat,
        before: &Digest,
        after: &Digest,
        evidence_count: u64,
        evidence_digest: &Digest,
    ) -> Event {
        let mut e = Event::new(EventKind::Publish, sender, seq, &ZERO);
        e.payload[0] = format as u8;
        put_digest(&mut e.payload, 8, before);
        put_digest(&mut e.payload, 40, after);
        put_u64(&mut e.payload, 72, evidence_count);
        put_digest(&mut e.payload, 80, evidence_digest);
        e
    }

    pub fn import_composite(sender: &Digest, seq: u64, k1: &Digest, composite: &Digest) -> Event {
        let mut e = Event::new(EventKind::ImportComposite, sender, seq, &ZERO);
        put_digest(&mut e.payload, 0, k1);
        put_digest(&mut e.payload, 32, composite);
        e
    }

    pub fn evolution_reserve(sender: &Digest, seq: u64, expected: &EvolutionState, assertion: &Digest) -> Event {
        let mut e = Event::new(EventKind::EvolutionReserve, sender, seq, &ZERO);
        put_evolution(&mut e.payload, 0, expected);
        put_digest(&mut e.payload, 80, assertion);
        e
    }

    pub fn evolution_finalize(sender: &Digest, seq: u64, expected: &EvolutionState, result: &Digest) -> Event {
        let mut e = Event::new(EventKind::EvolutionFinalize, sender, seq, &ZERO);
        put_evolution(&mut e.payload, 0, expected);
        put_digest(&mut e.payload, 80, result);
        e
    }

    pub fn import_legacy(sender: &Digest, seq: u64, record: &[u8; LEGACY_RECORD_BYTES]) -> Event {
        let mut e = Event::new(EventKind::ImportLegacy, sender, seq, &ZERO);
        e.payload[..LEGACY_RECORD_BYTES].copy_from_slice(record);
        e
    }

    /// Operator acceptance of an authority older than the acknowledged one (`class`: 0 the acknowledged record is
    /// absent, 1 torn, 2 unauthenticated). The next generation is `abandoned_generation + 1`.
    pub fn reconcile_rollback(
        sender: &Digest,
        seq: u64,
        abandoned_generation: u64,
        abandoned_digest: &Digest,
        class: u8,
    ) -> Event {
        let mut e = Event::new(EventKind::ReconcileRollback, sender, seq, &ZERO);
        put_u64(&mut e.payload, 0, abandoned_generation);
        e.payload[8] = class;
        put_digest(&mut e.payload, 16, abandoned_digest);
        e
    }

    /// Operator acknowledgment of an unauthenticated record in the non-current slot (it is overwritten).
    pub fn reconcile_unauthenticated(sender: &Digest, seq: u64, slot: u8, class: u8, observed: &Digest) -> Event {
        let mut e = Event::new(EventKind::ReconcileUnauthenticated, sender, seq, &ZERO);
        e.payload[0] = slot;
        e.payload[1] = class;
        put_digest(&mut e.payload, 8, observed);
        e
    }

    /// Operator re-pin of the newest authenticated record after both witness slots were lost.
    pub fn reconcile_repin(sender: &Digest, seq: u64, witness_classes: [u8; 2], pinned: &Digest) -> Event {
        let mut e = Event::new(EventKind::ReconcileRepin, sender, seq, &ZERO);
        e.payload[0] = witness_classes[0];
        e.payload[1] = witness_classes[1];
        put_digest(&mut e.payload, 8, pinned);
        e
    }

    pub fn entity_reclaim(sender: &Digest, seq: u64, target: &Digest) -> Event {
        Event::new(EventKind::EntityReclaim, sender, seq, target)
    }

    pub fn encode(&self) -> [u8; EVENT_BYTES] {
        let mut out = [0u8; EVENT_BYTES];
        out[0..2].copy_from_slice(&(self.kind as u16).to_le_bytes());
        put_digest(&mut out, 8, &self.sender);
        put_u64(&mut out, 40, self.seq);
        put_digest(&mut out, 48, &self.receiver);
        out[PAYLOAD_AT..].copy_from_slice(&self.payload);
        out
    }

    /// A well-formed event, or `Invalid`.
    pub fn decode(b: &[u8]) -> Result<Event, Code> {
        if b.len() != EVENT_BYTES || !zero(&b[2..8]) {
            return Err(Code::Invalid);
        }
        let kind = EventKind::from_u16(u16::from_le_bytes([b[0], b[1]])).ok_or(Code::Invalid)?;
        let e = Event {
            kind,
            sender: digest_at(b, 8),
            seq: u64_at(b, 40),
            receiver: digest_at(b, 48),
            payload: b[PAYLOAD_AT..].try_into().expect("payload"),
        };
        e.check_shape()?;
        Ok(e)
    }

    fn check_shape(&self) -> Result<(), Code> {
        if !zero(&self.payload[self.kind.payload_len()..]) {
            return Err(Code::Invalid);
        }
        if !self.kind.entity_scoped() && self.receiver != ZERO {
            return Err(Code::Invalid);
        }
        Ok(())
    }
}

// -- the transition law ------------------------------------------------------------------------------------

/// The authority a transition starts from.
pub struct Prior<'a> {
    pub lineage: Digest,
    pub generation: u64,
    pub record_digest: Digest,
    pub state_root: Digest,
    pub state: &'a State,
}

/// The validated result of one transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Applied {
    pub generation: u64,
    pub state: State,
    pub receipt: Digest,
    /// The identity an ENTITY_FOUND founded.
    pub founded: Option<Digest>,
}

/// The receipt of the transition that produced `generation` from its predecessor by `event`.
pub fn receipt(
    lineage: &Digest,
    generation: u64,
    predecessor_generation: u64,
    predecessor_digest: &Digest,
    predecessor_state_root: &Digest,
    event: &[u8; EVENT_BYTES],
) -> Digest {
    sha256::digest(&[
        domain::RECEIPT,
        lineage,
        &generation.to_le_bytes(),
        &predecessor_generation.to_le_bytes(),
        predecessor_digest,
        predecessor_state_root,
        event,
    ])
}

/// The next receipt chain head: an operator who keeps receipts can recompute the head from them.
pub fn receipt_chain(head: &Digest, receipt: &Digest) -> Digest {
    chain(domain::RECEIPT_CHAIN, head, &[receipt])
}

impl State {
    /// The identity the next ENTITY_FOUND of `kind` and `label` founds.
    pub fn next_entity_id(&self, lineage: &Digest, kind: EntityKind, label: &Digest) -> Digest {
        entity_id(lineage, self.next_founding_index, kind, label)
    }
}

/// The genesis of a lineage: the founding operator and runtime (ACTIVE), no cognition, evolution at genesis.
pub fn genesis(deployment: &Digest, capacity: u32, event: &Event) -> Result<(Digest, State, Digest), Code> {
    if event.kind != EventKind::Genesis || event.sender != ZERO || event.seq != 0 || event.receiver != ZERO {
        return Err(Code::Invalid);
    }
    event.check_shape()?;
    if !(MIN_CAPACITY..=MAX_CAPACITY).contains(&capacity) {
        return Err(Code::Invalid);
    }
    let p = &event.payload;
    let (nonce, operator_label, runtime_label) = (digest_at(p, 0), digest_at(p, 32), digest_at(p, 64));
    if nonce == ZERO {
        return Err(Code::Invalid);
    }
    let lineage = sha256::digest(&[domain::LINEAGE, deployment, &nonce, &capacity.to_le_bytes()]);
    let founder = |index: u64, kind: EntityKind, label: Digest| Entity {
        id: entity_id(&lineage, index, kind, &label),
        kind,
        lifecycle: Lifecycle::Active,
        per_turn_limit: 0,
        founding_index: index,
        founded_generation: 1,
        last_generation: 1,
        sender_watermark: 0,
        source_version: 0,
        label,
        scope: ZERO,
    };
    let receipt = receipt(&lineage, 1, 0, &ZERO, &ZERO, &event.encode());
    let state = State {
        cognition: Cognition::Unanchored,
        publications: 0,
        last_publication_generation: 0,
        evolution: EvolutionState::GENESIS,
        receipt_head: receipt_chain(&ZERO, &receipt),
        receipt_count: 1,
        evidence_head: ZERO,
        evidence_count: 0,
        reconcile_head: ZERO,
        reconcile_count: 0,
        reclaimed_head: ZERO,
        reclaimed_count: 0,
        next_founding_index: 2,
        legacy: None,
        entities: vec![
            founder(0, EntityKind::Operator, operator_label),
            founder(1, EntityKind::Runtime, runtime_label),
        ],
    };
    state.check(capacity, &lineage, 1)?;
    Ok((lineage, state, receipt))
}

/// Apply one event to `prior`: the complete semantic validation of a transition, and its result. Pure.
pub fn apply(prior: &Prior, capacity: u32, event: &Event) -> Result<Applied, Code> {
    let kind = event.kind;
    if kind == EventKind::Genesis {
        return Err(Code::Invalid);
    }
    event.check_shape()?;
    let p = &event.payload;
    let generation = if kind == EventKind::ReconcileRollback {
        // An accepted rollback continues above the abandoned generation, so no generation is ever reused.
        let abandoned = u64_at(p, 0);
        if abandoned <= prior.generation || abandoned >= MAX_COUNTER {
            return Err(Code::Invalid);
        }
        abandoned + 1
    } else {
        bump(prior.generation, 1)?
    };
    let mut s = prior.state.clone();
    let sender = *s.entity(&event.sender).ok_or(Code::UnknownEntity)?;
    if !kind.sender_may_be(sender.kind) || sender.lifecycle != Lifecycle::Active {
        return Err(Code::Unauthorized);
    }
    if event.seq <= sender.sender_watermark {
        return Err(Code::Duplicate);
    }
    if event.seq != sender.sender_watermark + 1 {
        return Err(Code::SequenceGap);
    }
    if event.seq > MAX_COUNTER {
        return Err(Code::Exhausted);
    }
    let mut founded = None;
    let lineage = prior.lineage;
    let publish = |s: &mut State| -> Result<(), Code> {
        s.publications = bump(s.publications, 1)?;
        s.last_publication_generation = generation;
        Ok(())
    };
    match kind {
        EventKind::Genesis => unreachable!("refused above"),
        EventKind::EntityFound => {
            let ek = EntityKind::from_u8(p[0]).ok_or(Code::Invalid)?;
            if !zero(&p[1..4]) {
                return Err(Code::Invalid);
            }
            let (limit, version, label, scope) = (u32_at(p, 4), u64_at(p, 8), digest_at(p, 16), digest_at(p, 48));
            let shape_ok = if ek == EntityKind::Source {
                (1..=MAX_PER_TURN).contains(&limit) && (1..=MAX_COUNTER).contains(&version) && scope != ZERO
            } else {
                limit == 0 && version == 0 && scope == ZERO
            };
            if !shape_ok {
                return Err(Code::Invalid);
            }
            if s.entities.len() >= capacity as usize {
                return Err(Code::Capacity);
            }
            if s.next_founding_index >= MAX_COUNTER {
                return Err(Code::Exhausted);
            }
            let id = s.next_entity_id(&lineage, ek, &label);
            if event.receiver != id {
                return Err(Code::Stale);
            }
            s.entities.push(Entity {
                id,
                kind: ek,
                lifecycle: Lifecycle::Founded,
                per_turn_limit: limit,
                founding_index: s.next_founding_index,
                founded_generation: generation,
                last_generation: generation,
                sender_watermark: 0,
                source_version: version,
                label,
                scope,
            });
            s.next_founding_index += 1;
            founded = Some(id);
        }
        EventKind::EntityLifecycle => {
            let ti = s.index_of(&event.receiver).ok_or(Code::UnknownEntity)?;
            let from = Lifecycle::from_u8(p[0]).ok_or(Code::Invalid)?;
            let to = Lifecycle::from_u8(p[1]).ok_or(Code::Invalid)?;
            let target = &mut s.entities[ti];
            if target.lifecycle != from || !from.may_become(to) {
                return Err(Code::Lifecycle);
            }
            if to == Lifecycle::Terminated && target.founding_index < 2 {
                return Err(Code::Lifecycle);
            }
            target.lifecycle = to;
            target.last_generation = generation;
            if !s.has_active_operator() {
                return Err(Code::Lifecycle);
            }
        }
        EventKind::SourceVersion => {
            let ti = s.index_of(&event.receiver).ok_or(Code::UnknownEntity)?;
            let (expected, version, limit, scope) = (u64_at(p, 0), u64_at(p, 8), u32_at(p, 16), digest_at(p, 24));
            let target = &mut s.entities[ti];
            if target.kind != EntityKind::Source || !zero(&p[20..24]) {
                return Err(Code::Invalid);
            }
            if target.lifecycle == Lifecycle::Terminated {
                return Err(Code::Lifecycle);
            }
            if expected != target.source_version {
                return Err(Code::SourceStale);
            }
            if version <= target.source_version
                || version > MAX_COUNTER
                || !(1..=MAX_PER_TURN).contains(&limit)
                || scope == ZERO
            {
                return Err(Code::Invalid);
            }
            target.source_version = version;
            target.per_turn_limit = limit;
            target.scope = scope;
            target.last_generation = generation;
        }
        EventKind::Anchor => {
            let format = format_from(p[0]).ok_or(Code::Invalid)?;
            let identity = digest_at(p, 8);
            if !zero(&p[1..8]) || identity == ZERO {
                return Err(Code::Invalid);
            }
            if s.cognition != Cognition::Unanchored {
                return Err(Code::AlreadyAnchored);
            }
            s.cognition = Cognition::Anchored { format, identity };
            publish(&mut s)?;
        }
        EventKind::Publish => {
            let format = format_from(p[0]).ok_or(Code::Invalid)?;
            let (before, after, count, digest) = (digest_at(p, 8), digest_at(p, 40), u64_at(p, 72), digest_at(p, 80));
            if !zero(&p[1..8]) || after == ZERO || (count == 0) != (digest == ZERO) {
                return Err(Code::Invalid);
            }
            match s.cognition {
                Cognition::Unanchored => return Err(Code::Unanchored),
                Cognition::Anchored { format: f, identity } if f != format || identity != before => {
                    return Err(Code::StateMismatch)
                }
                Cognition::Anchored { .. } => {}
            }
            s.cognition = Cognition::Anchored { format, identity: after };
            publish(&mut s)?;
            if count > 0 {
                s.evidence_count = bump(s.evidence_count, count)?;
                s.evidence_head = chain(
                    domain::EVIDENCE_CHAIN,
                    &s.evidence_head,
                    &[&generation.to_le_bytes(), &count.to_le_bytes(), &digest],
                );
            }
        }
        EventKind::ImportComposite => {
            let (k1, composite) = (digest_at(p, 0), digest_at(p, 32));
            if k1 == ZERO || composite == ZERO {
                return Err(Code::Invalid);
            }
            match s.cognition {
                Cognition::Unanchored => return Err(Code::Unanchored),
                Cognition::Anchored { format: StateFormat::K1, identity } if identity == k1 => {}
                Cognition::Anchored { .. } => return Err(Code::StateMismatch),
            }
            s.cognition = Cognition::Anchored { format: StateFormat::Composite, identity: composite };
            publish(&mut s)?;
        }
        EventKind::EvolutionReserve => {
            let expected = evolution_from(p, 0)?;
            let assertion = digest_at(p, 80);
            if expected != s.evolution {
                return Err(Code::EvolutionMismatch);
            }
            let (revision, head) = match expected {
                EvolutionState::Pending { .. } => return Err(Code::EvolutionPending),
                EvolutionState::Idle { revision, head } => (revision, head),
            };
            // The reservation and its finalization each need one generation, the finalization one revision:
            // refuse exhaustion before an irreversible attempt, not at finalization.
            if generation >= MAX_COUNTER || revision >= MAX_COUNTER {
                return Err(Code::Exhausted);
            }
            s.evolution = EvolutionState::pending(revision, head, assertion).map_err(|_| Code::Invalid)?;
        }
        EventKind::EvolutionFinalize => {
            let expected = evolution_from(p, 0)?;
            let result = digest_at(p, 80);
            if expected != s.evolution {
                return Err(Code::EvolutionMismatch);
            }
            let revision = match expected {
                EvolutionState::Idle { .. } => return Err(Code::EvolutionNotPending),
                EvolutionState::Pending { revision, .. } => revision,
            };
            if result == ZERO {
                return Err(Code::Invalid);
            }
            s.evolution = EvolutionState::idle(revision + 1, result).map_err(|_| Code::Invalid)?;
        }
        EventKind::ImportLegacy => {
            let snapshot = match Snapshot::decode(&p[..LEGACY_RECORD_BYTES]) {
                Ok(Some(snapshot)) => snapshot,
                _ => return Err(Code::LegacyRefused),
            };
            if s.legacy.is_some()
                || s.cognition != Cognition::Unanchored
                || s.evolution != EvolutionState::GENESIS
                || s.publications != 0
            {
                return Err(Code::LegacyRefused);
            }
            let anchored = match snapshot.cognition() {
                elpis_continuity::Cognition::Anchored(identity) => {
                    s.cognition = Cognition::Anchored { format: snapshot.format(), identity };
                    publish(&mut s)?;
                    true
                }
                elpis_continuity::Cognition::Unanchored => false,
            };
            s.evolution = snapshot.evolution();
            s.legacy = Some(Legacy {
                generation: snapshot.generation(),
                format: snapshot.format(),
                anchored,
                record_digest: snapshot.digest(),
            });
        }
        EventKind::ReconcileRollback => {
            if p[8] > 2 || !zero(&p[9..16]) || digest_at(p, 16) == ZERO {
                return Err(Code::Invalid);
            }
        }
        EventKind::ReconcileUnauthenticated => {
            if p[0] > 1 || p[1] != crate::record::SlotClass::Unauthenticated as u8 || !zero(&p[2..8]) {
                return Err(Code::Invalid);
            }
            if digest_at(p, 8) == ZERO {
                return Err(Code::Invalid);
            }
        }
        EventKind::ReconcileRepin => {
            let valid = crate::record::SlotClass::Valid as u8;
            if p[0] > valid || p[1] > valid || !zero(&p[2..8]) {
                return Err(Code::Invalid);
            }
            if digest_at(p, 8) != prior.record_digest {
                return Err(Code::Stale);
            }
        }
        EventKind::EntityReclaim => {
            let ti = s.index_of(&event.receiver).ok_or(Code::UnknownEntity)?;
            if s.entities[ti].lifecycle != Lifecycle::Terminated {
                return Err(Code::Lifecycle);
            }
            let mut row = [0u8; ENTITY_BYTES];
            s.entities[ti].encode(&mut row);
            s.reclaimed_count = bump(s.reclaimed_count, 1)?;
            s.reclaimed_head = chain(domain::RECLAIM_CHAIN, &s.reclaimed_head, &[&row]);
            s.entities.remove(ti);
        }
    }
    let encoded = event.encode();
    if kind.reconciliation() {
        s.reconcile_count = bump(s.reconcile_count, 1)?;
        s.reconcile_head = chain(domain::RECONCILE_CHAIN, &s.reconcile_head, &[&generation.to_le_bytes(), &encoded]);
    }
    let si = s.index_of(&event.sender).expect("the sender is ACTIVE, so it was not reclaimed");
    s.entities[si].sender_watermark = event.seq;
    s.entities[si].last_generation = generation;
    let receipt = receipt(&lineage, generation, prior.generation, &prior.record_digest, &prior.state_root, &encoded);
    s.receipt_count = bump(s.receipt_count, 1)?;
    s.receipt_head = receipt_chain(&s.receipt_head, &receipt);
    // The law never produces a state its own validator refuses; if it did, nothing is written.
    s.check(capacity, &lineage, generation)?;
    Ok(Applied { generation, state: s, receipt, founded })
}
