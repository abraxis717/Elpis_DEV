//! The durable authority RuntimeCore publishes to (docs/RUNTIME_CORE.md, "The durable authority (ABI v6)").
//!
//! Two backends behind one interface, the continuity store's: standalone continuity (`elpis_runtime_create`, the
//! ABI v5 path, unchanged) and the ECS_C authority (`elpis_runtime_create_ecs_c`, ABI v6; docs/ECS_C_AUTHORITY.md).
//! RuntimeCore's laws are the same over either: the backend changes where the lineage and the evolution authority
//! live, never what is published when. Under ECS_C every publication is one typed event sent by the authority's
//! founding RUNTIME entity at its next sender sequence, compare-and-publish against the current generation and state
//! root; RuntimeCore is the authority's only writer while it holds it open.
//!
//! Error domain. The call sites speak continuity's codes. An ECS_C refusal with a continuity equivalent maps to it.
//! At open, any other is `RUNTIME_AUTHORITY`; in a transition the one realistic other (the founding runtime is no
//! longer ACTIVE) is `CONTINUITY_AUTHORITY_MISMATCH`. The exact ECS_C code is kept ([`Status::ecs_c_code`]).

use std::path::Path;

use elpis_continuity::{Code, Cognition, Digest, EvolutionState, Snapshot, StateFormat};
use elpis_ecs_c_authority as ecs_c;

use crate::code::{Error, Rt};

/// The durable authority of one runtime handle.
pub enum Store {
    Continuity(Box<elpis_continuity::Store>),
    EcsC(Box<EcsC>),
}

/// The ECS_C backend.
pub struct EcsC {
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

/// The typed-evidence admission table RuntimeCore applies on the HOT lane under the ECS_C authority: the ACTIVE
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
    /// 1 continuity, 2 ECS_C.
    pub backend: u32,
    /// The last ECS_C refusal (0: none, or continuity).
    pub ecs_c_code: i32,
    /// The ECS_C recovery report of the last open (zeros for continuity).
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
    Error::Continuity(c)
}

/// An ECS_C transition refusal in continuity's code space (see the module header).
fn transition_code(code: ecs_c::Code) -> Code {
    match map(code) {
        Error::Continuity(c) => c,
        Error::Runtime(_) => Code::AuthorityMismatch,
    }
}

/// The continuity-form view of an ECS_C authority: its generation, cognition lineage, format and evolution.
pub fn snapshot_of(record: &ecs_c::Record) -> Result<Snapshot, Code> {
    let (cognition, format) = match record.state.cognition {
        ecs_c::Cognition::Unanchored => (Cognition::Unanchored, StateFormat::K1),
        ecs_c::Cognition::Anchored { format, identity } => (Cognition::Anchored(identity), format),
    };
    Snapshot::with_format(record.generation, cognition, format, record.state.evolution)
}

impl EcsC {
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
}

impl Store {
    pub fn continuity(directory: &Path) -> Result<Store, Code> {
        Ok(Store::Continuity(Box::new(elpis_continuity::Store::new(directory)?)))
    }

    pub fn ecs_c(state_dir: &Path, authority_dir: &Path) -> Result<Store, Error> {
        let authority = ecs_c::Authority::new(state_dir, authority_dir).map_err(map)?;
        Ok(Store::EcsC(Box::new(EcsC { authority, last: None, report: None, evidence: None, published_observed: 0 })))
    }

    /// Open: the authority resolves its current state. ECS_C opens only in its normal mode: a restricted or
    /// reconcile-only authority, or one recovery refused, is `RUNTIME_AUTHORITY` until the operator reconciles it
    /// with the runtime closed (docs/ECS_C_AUTHORITY.md, "Transition and recovery").
    pub fn open(&mut self) -> Result<Snapshot, Error> {
        match self {
            Store::Continuity(s) => Ok(s.open()?),
            Store::EcsC(e) => {
                e.last = None;
                e.evidence = None;
                e.published_observed = 0; // RuntimeCore's committed-observation total restarts with the open
                match e.authority.open() {
                    Ok(report) => e.report = Some(report),
                    Err(refusal) => {
                        e.last = Some(refusal.code);
                        e.report = refusal.report.map(|r| *r);
                        return Err(map(refusal.code));
                    }
                }
                if e.authority.mode().map_err(map)? != ecs_c::Mode::Normal {
                    e.last = Some(ecs_c::Code::ReconcileRequired);
                    e.authority.close();
                    return Err(Error::Runtime(Rt::Authority));
                }
                Ok(snapshot_of(e.current()?)?)
            }
        }
    }

    pub fn close(&mut self) {
        match self {
            Store::Continuity(s) => s.close(),
            Store::EcsC(e) => e.authority.close(),
        }
    }

    pub fn snapshot(&self) -> Result<Snapshot, Code> {
        match self {
            Store::Continuity(s) => s.snapshot(),
            Store::EcsC(e) => snapshot_of(e.current()?),
        }
    }

    pub fn anchor_cognition_as(&mut self, format: StateFormat, digest: Option<&Digest>) -> Result<Snapshot, Code> {
        match self {
            Store::Continuity(s) => s.anchor_cognition_as(format, digest),
            Store::EcsC(e) => {
                let digest = *digest.ok_or(Code::Invalid)?;
                let (rt, seq) = e.runtime()?;
                e.publish(ecs_c::Event::anchor(&rt, seq, format, &digest))
            }
        }
    }

    pub fn commit_cognition(&mut self, before: Option<&Digest>, after: Option<&Digest>) -> Result<Snapshot, Code> {
        let format = self.snapshot()?.format();
        self.commit_cognition_as(format, before, after)
    }

    /// Publish a committed identity. Under ECS_C the PUBLISH carries the evidence summary the lane set for this
    /// publication ([`Store::set_evidence`]), consumed whatever the outcome.
    pub fn commit_cognition_as(
        &mut self,
        format: StateFormat,
        before: Option<&Digest>,
        after: Option<&Digest>,
    ) -> Result<Snapshot, Code> {
        match self {
            Store::Continuity(s) => s.commit_cognition_as(format, before, after),
            Store::EcsC(e) => {
                let (total, count, summary) = e.evidence.take().unwrap_or((e.published_observed, 0, [0; 32]));
                let (before, after) = (*before.ok_or(Code::Invalid)?, *after.ok_or(Code::Invalid)?);
                let (rt, seq) = e.runtime()?;
                let published = e.publish(ecs_c::Event::publish(&rt, seq, format, &before, &after, count, &summary))?;
                e.published_observed = total;
                Ok(published)
            }
        }
    }

    pub fn import_cognition(&mut self, k1: Option<&Digest>, composite: Option<&Digest>) -> Result<Snapshot, Code> {
        match self {
            Store::Continuity(s) => s.import_cognition(k1, composite),
            Store::EcsC(e) => {
                let (k1, composite) = (*k1.ok_or(Code::Invalid)?, *composite.ok_or(Code::Invalid)?);
                let (rt, seq) = e.runtime()?;
                e.publish(ecs_c::Event::import_composite(&rt, seq, &k1, &composite))
            }
        }
    }

    pub fn reserve_evolution(
        &mut self,
        expected: Option<&EvolutionState>,
        assertion: Option<&Digest>,
    ) -> Result<Snapshot, Code> {
        match self {
            Store::Continuity(s) => s.reserve_evolution(expected, assertion),
            Store::EcsC(e) => {
                let expected = *expected.ok_or(Code::AuthorityMismatch)?;
                let assertion = *assertion.ok_or(Code::Invalid)?;
                let (rt, seq) = e.runtime()?;
                e.publish(ecs_c::Event::evolution_reserve(&rt, seq, &expected, &assertion))
            }
        }
    }

    pub fn finalize_evolution(
        &mut self,
        expected: Option<&EvolutionState>,
        receipt: Option<&Digest>,
    ) -> Result<Snapshot, Code> {
        match self {
            Store::Continuity(s) => s.finalize_evolution(expected, receipt),
            Store::EcsC(e) => {
                let expected = *expected.ok_or(Code::AuthorityMismatch)?;
                let receipt = *receipt.ok_or(Code::Invalid)?;
                let (rt, seq) = e.runtime()?;
                e.publish(ecs_c::Event::evolution_finalize(&rt, seq, &expected, &receipt))
            }
        }
    }

    // -- ECS_C only ------------------------------------------------------------------------------------------

    /// The evidence summary the next publication of `envelope` carries: the typed observations committed since the
    /// last publication (`observed_total` is RuntimeCore's committed-observation total at the capture) and the
    /// identity of the memory that holds them, the composite envelope's memory component (zero with a zero count).
    /// Computed on the durability lane, never on the HOT lane. Ignored by continuity, which keeps no evidence.
    pub fn set_evidence(&mut self, observed_total: u64, envelope: &[u8]) {
        if let Store::EcsC(e) = self {
            let count = observed_total.saturating_sub(e.published_observed);
            e.evidence = Some(match (count, composite_memory_identity(envelope)) {
                (0, _) | (_, None) => (observed_total, 0, [0; 32]),
                (count, Some(memory)) => (observed_total, count, memory),
            });
        }
    }

    /// The HOT lane's admission table: `None` under continuity (admission stays the boundary's).
    pub fn admission(&self) -> Option<Admission> {
        match self {
            Store::Continuity(_) => None,
            Store::EcsC(e) => {
                let state = &e.authority.current().ok()?.state;
                let sources = state
                    .entities
                    .iter()
                    .filter(|x| x.kind == ecs_c::EntityKind::Source && x.lifecycle == ecs_c::Lifecycle::Active)
                    .map(|x| (x.source_authority, x.per_turn_limit))
                    .collect();
                Some(Admission { sources })
            }
        }
    }

    /// An operator event while RuntimeCore holds the authority open: founding, lifecycle, source versions,
    /// reclamation and the legacy import. RuntimeCore's own events (anchor, publish, import, evolution) and the
    /// reconciliations (performed with the runtime closed) are refused (`RUNTIME_INVALID`).
    pub fn operator_transition(&mut self, event: &ecs_c::Event) -> Result<ecs_c::Receipt, Error> {
        use ecs_c::EventKind as K;
        let Store::EcsC(e) = self else {
            return Err(Rt::Invalid.into());
        };
        if !matches!(
            event.kind,
            K::EntityFound | K::EntityLifecycle | K::SourceVersion | K::EntityReclaim | K::ImportLegacy
        ) {
            return Err(Rt::Invalid.into());
        }
        let (generation, root) = {
            let c = e.current()?;
            (c.generation, c.state_root)
        };
        e.authority.transition(generation, &root, event).map_err(|code| {
            e.last = Some(code);
            map(code)
        })
    }

    pub fn status(&self) -> Status {
        match self {
            Store::Continuity(s) => {
                let generation = s.snapshot().map(|x| x.generation()).unwrap_or(0);
                Status { backend: 1, generation, ..Status::default() }
            }
            Store::EcsC(e) => {
                let mut status =
                    Status { backend: 2, ecs_c_code: e.last.map(|c| c as i32).unwrap_or(0), ..Status::default() };
                if let Some(r) = e.report {
                    status.disposition = r.disposition;
                    status.mode = r.mode;
                    status.replay_verified = r.replay_verified;
                    status.witness_completed = r.witness_completed;
                    status.generation = r.current_generation;
                }
                if let Ok(c) = e.authority.current() {
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
    }
}

#[cfg(feature = "testing")]
impl Store {
    /// Arm the authority's record fault. Continuity's actions and publication steps map one to one onto the ECS_C
    /// record slot's (0 clear; 1 die at a step: 0 begun, 1 written, 2 synced; 2..=6 write fails, torn then refused,
    /// torn then death, sync fails with the bytes lost, sync fails although durable). Continuity's initialization
    /// steps have no ECS_C counterpart: an ECS_C authority is provisioned, never initialized at open.
    pub fn testing_arm(&mut self, publication: u64, action: u32, arg: u64) -> Result<(), Code> {
        match self {
            Store::Continuity(s) => s.testing_arm(publication, action, arg),
            Store::EcsC(e) => {
                if action > 6 || (action == 1 && arg > 2) {
                    return Err(Code::Invalid);
                }
                e.authority.testing_arm(publication, action, arg).map_err(transition_code)
            }
        }
    }

    /// The authority's I/O counters, in continuity's layout.
    pub fn testing_counters(&mut self, reset: bool) -> elpis_continuity::store::probe::Counters {
        match self {
            Store::Continuity(s) => s.testing_counters(reset),
            Store::EcsC(e) => {
                let c = e.authority.testing_counters(reset);
                elpis_continuity::store::probe::Counters {
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
    }
}
