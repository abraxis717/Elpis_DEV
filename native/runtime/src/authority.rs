//! The durable authority RuntimeCore publishes to (docs/RUNTIME_CORE.md, "The durable authority (ABI v7)"): the
//! ECS_C authority (`elpis_runtime_create_ecs_c`; docs/ECS_C_AUTHORITY.md), embedded without its C ABI.
//!
//! RuntimeCore ABI v6 had a second backend, standalone continuity; ABI v7 retired it (docs/NATIVE_SPLIT_CENSUS.md,
//! section 9). The authority changes where the lineage and the evolution authority live, never what is published
//! when. Every publication is one typed event sent by the authority's founding RUNTIME entity at its next sender
//! sequence, compare-and-publish against the current generation and state root; RuntimeCore is the authority's only
//! writer while it holds it open.
//!
//! Error domain. The call sites speak the durable-authority code space ([`Code`]: continuity's codes, kept with their
//! values and names). An ECS_C refusal with an equivalent maps to it. At open, any other is `RUNTIME_AUTHORITY`; in a
//! transition the one realistic other (the founding runtime is no longer ACTIVE) is `CONTINUITY_AUTHORITY_MISMATCH`.
//! The exact ECS_C code is kept ([`Status::ecs_c_code`]).
//!
//! The view. RuntimeCore reports its durable authority as a [`Snapshot`]: the generation, the cognition lineage and
//! its format, and the evolution authority, in the shape of the continuity register record ECS_C owns
//! (`elpis_ecs_c_authority::register`), so every digest it reports is unchanged.

use std::path::Path;

use elpis_ecs_c_authority as ecs_c;

pub use ecs_c::register::{Cognition, Digest, EvolutionState, Snapshot, StateFormat, ZERO};

use crate::code::{Code, Error, Rt};

/// The durable authority of one runtime handle.
pub struct Store {
    authority: ecs_c::Authority,
    /// The last ECS_C refusal (RuntimeCore reports a mapped code; this is the exact one).
    last: Option<ecs_c::Code>,
    /// The recovery report of the last open, kept when the open was refused (the operator's reason to reconcile).
    report: Option<ecs_c::Report>,
    /// The evidence summary the next PUBLISH carries, with the committed-observation total it accounts up to (set
    /// by the durability lane for that publication only).
    evidence: Option<(u64, u64, Digest)>,
    /// The committed-observation total the last successful publication accounted for (since this open).
    published_observed: u64,
}

/// The typed-evidence admission table RuntimeCore applies on the HOT lane: the ECS_C authority's ACTIVE
/// sources, by the memory source-authority identifier their observations record, with their per-turn bounds. A copy
/// of the authority's state (at most its entity capacity), refreshed whenever the authority changes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Admission {
    sources: Vec<(u32, u32)>,
}

impl Admission {
    /// The per-turn bound of the ACTIVE source that owns `source_authority`, if any.
    pub fn bound(&self, source_authority: u32) -> Option<u32> {
        self.sources.iter().find(|(a, _)| *a == source_authority).map(|(_, limit)| *limit)
    }
}

/// What `elpis_runtime_authority_status` reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Status {
    /// Always [`BACKEND_ECS_C`] (ABI v6 also had 1, standalone continuity).
    pub backend: u32,
    /// The last ECS_C refusal (0: none).
    pub ecs_c_code: i32,
    /// The ECS_C recovery report of the last open.
    pub disposition: u32,
    pub mode: u32,
    pub replay_verified: bool,
    pub witness_completed: bool,
    pub generation: u64,
    pub lineage: Digest,
    pub record_digest: Digest,
    pub state_root: Digest,
    pub receipt_head: Digest,
    pub receipt_count: u64,
    pub evidence_count: u64,
    /// The chain head of the evidence summaries publications recorded (docs/ECS_C_AUTHORITY.md, "Entities, events
    /// and the transition law").
    pub evidence_head: Digest,
    pub entity_count: u32,
    pub capacity: u32,
}

/// `elpis_runtime_authority.backend`: the ECS_C authority.
pub const BACKEND_ECS_C: u32 = 2;

/// The memory component's identity recorded in an ECS composite envelope (`ELPISEC1`, offset 80;
/// docs/ECS_COMPOSITE_STATE.md); `None` for any other envelope.
pub fn composite_memory_identity(envelope: &[u8]) -> Option<Digest> {
    if envelope.len() < 128 || &envelope[..8] != b"ELPISEC1" {
        return None;
    }
    envelope[80..112].try_into().ok()
}

/// An ECS_C refusal in RuntimeCore's error domain.
pub fn map(code: ecs_c::Code) -> Error {
    use ecs_c::Code as E;
    let c = match code {
        E::Invalid => Code::Invalid,
        E::Unanchored => Code::Unanchored,
        E::AlreadyAnchored => Code::AlreadyAnchored,
        E::StateMismatch => Code::StateMismatch,
        E::EvolutionMismatch => Code::AuthorityMismatch,
        E::EvolutionPending => Code::EvolutionPending,
        E::EvolutionNotPending => Code::EvolutionNotPending,
        E::Exhausted => Code::Exhausted,
        E::PublicationRefused => Code::PublicationRefused,
        E::PublicationUncertain => Code::PublicationUncertain,
        E::Locked => Code::Locked,
        E::Io => Code::Io,
        E::Path => Code::Path,
        E::Open => Code::Open,
        E::Closed => Code::Uninitialized,
        E::Corrupt => Code::Corrupt,
        E::TestingProcessDeath => Code::TestingProcessDeath,
        _ => return Error::Runtime(Rt::Authority),
    };
    Error::Authority(c)
}

/// An ECS_C transition refusal in the durable-authority code space (see the module header).
fn transition_code(code: ecs_c::Code) -> Code {
    match map(code) {
        Error::Authority(c) => c,
        Error::Runtime(_) => Code::AuthorityMismatch,
    }
}

/// The view of an ECS_C authority record: its generation, cognition lineage, format and evolution.
pub fn snapshot_of(record: &ecs_c::Record) -> Result<Snapshot, Code> {
    let (cognition, format) = match record.state.cognition {
        ecs_c::Cognition::Unanchored => (Cognition::Unanchored, StateFormat::K1),
        ecs_c::Cognition::Anchored { format, identity } => (Cognition::Anchored(identity), format),
    };
    Snapshot::with_format(record.generation, cognition, format, record.state.evolution).map_err(|_| Code::Corrupt)
}

impl Store {
    fn current(&self) -> Result<&ecs_c::Record, Code> {
        self.authority.current().map_err(transition_code)
    }

    /// The founding runtime and its next sender sequence.
    fn runtime(&self) -> Result<(Digest, u64), Code> {
        let rt = &self.current()?.state.entities[1];
        Ok((rt.id, rt.sender_watermark + 1))
    }

    fn send(&mut self, event: ecs_c::Event) -> Result<(Snapshot, ecs_c::Receipt), Code> {
        let (generation, root) = {
            let c = self.current()?;
            (c.generation, c.state_root)
        };
        match self.authority.transition(generation, &root, &event) {
            Ok(receipt) => Ok((snapshot_of(self.current()?)?, receipt)),
            Err(code) => {
                self.last = Some(code);
                Err(transition_code(code))
            }
        }
    }

    fn publish(&mut self, event: ecs_c::Event) -> Result<Snapshot, Code> {
        self.send(event).map(|(snapshot, _)| snapshot)
    }

    pub fn ecs_c(state_dir: &Path, authority_dir: &Path) -> Result<Store, Error> {
        let authority = ecs_c::Authority::new(state_dir, authority_dir).map_err(map)?;
        Ok(Store { authority, last: None, report: None, evidence: None, published_observed: 0 })
    }

    /// Open: the authority resolves its current state. It opens only in its normal mode: a restricted or
    /// reconcile-only authority, or one recovery refused, is `RUNTIME_AUTHORITY` until the operator reconciles it
    /// with the runtime closed (docs/ECS_C_AUTHORITY.md, "Transition and recovery").
    pub fn open(&mut self) -> Result<Snapshot, Error> {
        self.last = None;
        self.evidence = None;
        self.published_observed = 0; // RuntimeCore's committed-observation total restarts with the open
        match self.authority.open() {
            Ok(report) => self.report = Some(report),
            Err(refusal) => {
                self.last = Some(refusal.code);
                self.report = refusal.report.map(|r| *r);
                return Err(map(refusal.code));
            }
        }
        if self.authority.mode().map_err(map)? != ecs_c::Mode::Normal {
            self.last = Some(ecs_c::Code::ReconcileRequired);
            self.authority.close();
            return Err(Error::Runtime(Rt::Authority));
        }
        Ok(snapshot_of(self.current()?)?)
    }

    pub fn close(&mut self) {
        self.authority.close()
    }

    pub fn snapshot(&self) -> Result<Snapshot, Code> {
        snapshot_of(self.current()?)
    }

    pub fn anchor_cognition_as(&mut self, format: StateFormat, digest: Option<&Digest>) -> Result<Snapshot, Code> {
        let digest = *digest.ok_or(Code::Invalid)?;
        let (rt, seq) = self.runtime()?;
        self.publish(ecs_c::Event::anchor(&rt, seq, format, &digest))
    }

    pub fn commit_cognition(&mut self, before: Option<&Digest>, after: Option<&Digest>) -> Result<Snapshot, Code> {
        let format = self.snapshot()?.format();
        self.commit_cognition_as(format, before, after)
    }

    /// Publish a committed identity. The PUBLISH carries the evidence summary the lane set for this publication
    /// ([`Store::set_evidence`]), consumed whatever the outcome.
    pub fn commit_cognition_as(
        &mut self,
        format: StateFormat,
        before: Option<&Digest>,
        after: Option<&Digest>,
    ) -> Result<Snapshot, Code> {
        let (total, count, summary) = self.evidence.take().unwrap_or((self.published_observed, 0, ZERO));
        let (before, after) = (*before.ok_or(Code::Invalid)?, *after.ok_or(Code::Invalid)?);
        let (rt, seq) = self.runtime()?;
        let published = self.publish(ecs_c::Event::publish(&rt, seq, format, &before, &after, count, &summary))?;
        self.published_observed = total;
        Ok(published)
    }

    pub fn import_cognition(&mut self, k1: Option<&Digest>, composite: Option<&Digest>) -> Result<Snapshot, Code> {
        let (k1, composite) = (*k1.ok_or(Code::Invalid)?, *composite.ok_or(Code::Invalid)?);
        let (rt, seq) = self.runtime()?;
        self.publish(ecs_c::Event::import_composite(&rt, seq, &k1, &composite))
    }

    pub fn reserve_evolution(
        &mut self,
        expected: Option<&EvolutionState>,
        assertion: Option<&Digest>,
    ) -> Result<Snapshot, Code> {
        let expected = *expected.ok_or(Code::AuthorityMismatch)?;
        let assertion = *assertion.ok_or(Code::Invalid)?;
        let (rt, seq) = self.runtime()?;
        self.publish(ecs_c::Event::evolution_reserve(&rt, seq, &expected, &assertion))
    }

    pub fn finalize_evolution(
        &mut self,
        expected: Option<&EvolutionState>,
        receipt: Option<&Digest>,
    ) -> Result<Snapshot, Code> {
        let expected = *expected.ok_or(Code::AuthorityMismatch)?;
        let receipt = *receipt.ok_or(Code::Invalid)?;
        let (rt, seq) = self.runtime()?;
        self.publish(ecs_c::Event::evolution_finalize(&rt, seq, &expected, &receipt))
    }

    /// The evidence summary the next publication of `envelope` carries: the typed observations committed since the
    /// last publication (`observed_total` is RuntimeCore's committed-observation total at the capture) and the
    /// identity of the memory that holds them, the composite envelope's memory component (zero with a zero count).
    /// Computed on the durability lane, never on the HOT lane.
    pub fn set_evidence(&mut self, observed_total: u64, envelope: &[u8]) {
        let count = observed_total.saturating_sub(self.published_observed);
        self.evidence = Some(match (count, composite_memory_identity(envelope)) {
            (0, _) | (_, None) => (observed_total, 0, ZERO),
            (count, Some(memory)) => (observed_total, count, memory),
        });
    }

    /// The HOT lane's admission table (`None` only when no current record can be read).
    pub fn admission(&self) -> Option<Admission> {
        let state = &self.authority.current().ok()?.state;
        let sources = state
            .entities
            .iter()
            .filter(|x| x.kind == ecs_c::EntityKind::Source && x.lifecycle == ecs_c::Lifecycle::Active)
            .map(|x| (x.source_authority, x.per_turn_limit))
            .collect();
        Some(Admission { sources })
    }

    /// An operator event while RuntimeCore holds the authority open: founding, lifecycle, source versions,
    /// reclamation and the legacy import. RuntimeCore's own events (anchor, publish, import, evolution) and the
    /// reconciliations (performed with the runtime closed) are refused (`RUNTIME_INVALID`).
    pub fn operator_transition(&mut self, event: &ecs_c::Event) -> Result<ecs_c::Receipt, Error> {
        use ecs_c::EventKind as K;
        if !matches!(
            event.kind,
            K::EntityFound | K::EntityLifecycle | K::SourceVersion | K::EntityReclaim | K::ImportLegacy
        ) {
            return Err(Rt::Invalid.into());
        }
        let (generation, root) = {
            let c = self.current()?;
            (c.generation, c.state_root)
        };
        self.authority.transition(generation, &root, event).map_err(|code| {
            self.last = Some(code);
            map(code)
        })
    }

    /// While held: the entity row at `index` of the current record (the operator's view for building an event:
    /// identities, lifecycles, sender watermarks). `RUNTIME_INVALID` past the last row.
    pub fn entity_at(&self, index: u32) -> Result<ecs_c::Entity, Error> {
        let entity = self.current()?.state.entities.get(index as usize).copied();
        entity.ok_or_else(|| Rt::Invalid.into())
    }

    /// While held: the identity the next ENTITY_FOUND of `kind` and `label` founds (the receiver that event must
    /// name). `RUNTIME_INVALID` for an unknown kind.
    pub fn next_entity_id(&self, kind: u8, label: &Digest) -> Result<Digest, Error> {
        let kind = ecs_c::EntityKind::from_u8(kind).ok_or(Rt::Invalid)?;
        self.authority.next_entity_id(kind, label).map_err(|code| transition_code(code).into())
    }

    pub fn status(&self) -> Status {
        let mut status = Status {
            backend: BACKEND_ECS_C,
            ecs_c_code: self.last.map(|c| c as i32).unwrap_or(0),
            ..Status::default()
        };
        if let Some(r) = self.report {
            status.disposition = r.disposition;
            status.mode = r.mode;
            status.replay_verified = r.replay_verified;
            status.witness_completed = r.witness_completed;
            status.generation = r.current_generation;
        }
        if let Ok(c) = self.authority.current() {
            status.generation = c.generation;
            status.lineage = c.lineage;
            status.record_digest = c.digest;
            status.state_root = c.state_root;
            status.receipt_head = c.state.receipt_head;
            status.receipt_count = c.state.receipt_count;
            status.evidence_count = c.state.evidence_count;
            status.evidence_head = c.state.evidence_head;
            status.entity_count = c.state.entities.len() as u32;
            status.capacity = c.capacity;
        }
        status
    }
}

/// The authority's I/O counters as `elpis_runtime_testing_io_counters` reports them: the layout ABI v6's continuity
/// counters had (the testing library only).
#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct IoCounters {
    pub opens: u64,
    pub preads: u64,
    pub pread_bytes: u64,
    pub pwrites: u64,
    pub pwrite_bytes: u64,
    pub data_syncs: u64,
    pub dir_syncs: u64,
    pub renames: u64,
    pub unlinks: u64,
}

#[cfg(feature = "testing")]
impl Store {
    /// Arm the authority's record fault (0 clear; 1 die at a step: 0 begun, 1 written, 2 synced; 2..=6 write fails,
    /// torn then refused, torn then death, sync fails with the bytes lost, sync fails although durable).
    pub fn testing_arm(&mut self, publication: u64, action: u32, arg: u64) -> Result<(), Code> {
        if action > 6 || (action == 1 && arg > 2) {
            return Err(Code::Invalid);
        }
        self.authority.testing_arm(publication, action, arg).map_err(transition_code)
    }

    /// The authority's I/O counters.
    pub fn testing_counters(&mut self, reset: bool) -> IoCounters {
        let c = self.authority.testing_counters(reset);
        IoCounters {
            opens: c.opens,
            preads: c.preads,
            pread_bytes: c.pread_bytes,
            pwrites: c.pwrites,
            pwrite_bytes: c.pwrite_bytes,
            data_syncs: c.data_syncs,
            dir_syncs: c.dir_syncs,
            renames: c.renames,
            unlinks: c.unlinks,
        }
    }
}
