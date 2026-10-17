//! RuntimeCore: the one state machine behind an open Elpis runtime.
//!
//! ```text
//!   closed --open--> open --close--> closed
//!                     |
//!                     +-- fail-stop (a continuity code): every lineage- or authority-dependent operation is
//!                         refused with that code until close + open (restart reconciles against continuity)
//! ```
//!
//! Two lanes (docs/RUNTIME_CORE.md, "Live and durable authority"):
//!
//! * HOT (this module, the caller's thread, under the runtime handle's lock): binding, QUERY, the managed turn,
//!   OBSERVE, release. It performs no file I/O and never waits for the durability lane. Its authority is the
//!   **live coordinate** of the bound state: the native generation, under this runtime's managed lease, as last
//!   established by RuntimeCore itself (verified against the durable identity once, when the state is bound).
//! * WARM (`durable`, one worker thread): durable publication of a captured live state (checkpoint slot, then the
//!   continuity record), explicitly requested; and the stores every control operation uses.
//!
//! Laws (each is a test in `tests.rs`):
//! * K1 lineage: the first managed lineage is anchored explicitly, durably, at the state's retained identity; an
//!   open runtime binds its lineage to one native state, verified once against the durable expected identity (a
//!   mismatch fail-stops; a mismatched state is never claimed); another state is refused
//!   (`COGNITION_SUBSTRATE_SWITCH`) without a native call.
//! * Live and durable authority are distinct coordinates. The live state may advance beyond the last durable
//!   acknowledgment; every HOT operation is checked against the live coordinate, never against continuity. A
//!   crash loses at most the volatile work since the last acknowledgment; restart binds only a state whose identity
//!   is the durable one (an unacknowledged live state is a mismatch, never adopted).
//! * QUERY: read-only. One native call answers `f_W(x)` and reports the generation of the state it answered from,
//!   under the lease; it must be the live generation (otherwise the lineage is lost: fail-stop, answer withheld).
//! * Managed turn (LEARN): begin (the source generation must be the live one) -> one native schedule on the
//!   candidate -> (the boundary decodes) -> native commit, which advances the live generation by exactly one. A
//!   VOLATILE commit: no checkpoint, no continuity publication, no I/O. Every refusal before the commit leaves the
//!   complete retained state byte-for-byte unchanged (the native transaction law).
//! * Durable publication: `durable_request` captures the live state (unsealed, under the lease, at the live
//!   generation) and hands it to the durability lane; it returns at once. A RECOVERABLE request (checkpoint store
//!   required) is ACKNOWLEDGED only after the slot is read back verified and continuity names it: acknowledgment
//!   means recoverable (docs/K1_RECOVERY_R0.md). An IDENTITY request (no store attached) resolves
//!   IDENTITY_PUBLISHED, never ACKNOWLEDGED. Never implied by a turn.
//! * Turn lifecycle: once a native transaction is open, exactly one terminal native action ends it before
//!   RuntimeCore forgets the turn: the commit, or an abort. There is no third disposition. An explicit
//!   `turn_abort`, a refused commit, `close`, a reopen and destruction (`Drop`) all abort through the capability
//!   retained at begin ([`crate::substrate::TxnAbort`]); an abort refused BUSY (a concurrent overlapping call) is
//!   retried, never dropped. An implicit abort installs nothing and publishes nothing.
//! * Evolution: one attempt at a time. Its assertion is durably reserved (synchronously, on the caller's thread)
//!   before the boundary may execute it and finalized with the executed receipt; a failed reservation or
//!   finalization fail-stops, and an attempt the boundary could not complete leaves the durable reservation pending
//!   (fail-stop `CONTINUITY_EVOLUTION_PENDING`); execution is never retried or inferred. Volatile cognition never
//!   relaxes this.
//! * Composite lineages (docs/ECS_COMPOSITE_STATE.md): a substrate of the ECS composite kind is bound, queried,
//!   learned, durably published and recovered exactly as a K1 state; continuity records the lineage's format, and a
//!   substrate of another format never stands for it (`CONTINUITY_STATE_MISMATCH`). The memory changes only through
//!   `turn_observe` inside the managed turn. RuntimeCore exposes no memory LEARN or CONSOLIDATE (unqualified).
//! * Nothing else is durable or retained: no history, receipts, events or trajectory.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use elpis_continuity::{Code, Cognition, Digest, EvolutionState, Snapshot, StateFormat, Store};

use crate::checkpoint::{self, Disposition};
use crate::code::{Error, Rt};
use crate::durable::{Kind, Lane, Ticket};
use crate::fuel::{self, Budget};
use crate::substrate::{
    features, k1_envelope_bytes, Experience, K1Ops, Key, Observation, ResidentRelease, ScheduleOutcome, Transition,
    TxnAbort, K1_BUSY, K1_CAPACITY, K1_FMS_CAPACITY, K1_FMS_NOT_RESIDENT, K1_INVALID, K1_LEASED, K1_NONFINITE,
    K1_STALE, KIND_COMPOSITE, MAX_EXPERIENCES,
};

/// The next managed lease: unique and nonzero per RuntimeCore in this process (an aliasing defence between
/// cooperating callers, not a secret; the K1 header's "Managed ownership").
fn next_lease() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    (u64::from(std::process::id()) << 40) ^ n | 1 << 63
}

/// Native crossings and publications RuntimeCore performed (diagnostics; never authority). ABI v4 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub k1_state_digests: u64,
    pub k1_reserves: u64,
    pub k1_txn_begins: u64,
    pub k1_run_schedules: u64,
    pub k1_commits: u64,
    pub k1_aborts: u64,
    /// Continuity publications (control operations and the durability lane).
    pub publications: u64,
    pub k1_queries: u64,
    pub k1_shapes: u64,
    pub k1_lease_claims: u64,
    /// `leased_identity` crossings (binding, `live_identity`).
    pub k1_identities: u64,
    /// `leased_capture` crossings (durable requests and control-plane durability).
    pub k1_captures: u64,
    pub durable_requests: u64,
    pub durable_superseded: u64,
    pub durable_acknowledged: u64,
    pub durable_failed: u64,
    /// `leased_txn_observe` crossings (ABI v5): OBSERVE inside a managed turn, LEARN or observe-only.
    pub k1_observes: u64,
}

/// An admitted, native-ready stimulus: `x` holds `y.len()` rows of the state's dimension, row-major.
pub struct Stimulus<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub schedule: &'a [Experience],
    pub rate: f64,
}

/// A refusal and the native evidence behind it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub error: Error,
    /// The K1 status that caused it (0: no native refusal was involved).
    pub k1_status: i32,
    pub schedule: ScheduleOutcome,
    /// The committed native transition when the refusal came after the native commit (never in v4 except a
    /// generation that does not continue the live coordinate, which fail-stops).
    pub committed: Option<Transition>,
}

impl From<Error> for Refusal {
    fn from(error: Error) -> Self {
        Refusal { error, k1_status: 0, schedule: ScheduleOutcome::default(), committed: None }
    }
}

impl From<Code> for Refusal {
    fn from(c: Code) -> Self {
        Error::from(c).into()
    }
}

impl From<Rt> for Refusal {
    fn from(r: Rt) -> Self {
        Error::from(r).into()
    }
}

fn ecs(error: Rt, k1_status: i32, schedule: ScheduleOutcome) -> Refusal {
    Refusal { error: error.into(), k1_status, schedule, committed: None }
}

/// The open managed turn: the bound substrate, its native transaction token and the retained capability that
/// ends that transaction (abort only). It exists exactly while the native transaction is open. An observe-only
/// turn (ABI v5) ran no K1 schedule; every turn counts the observations it recorded against the bound its budget
/// admitted at begin.
struct OpenTurn {
    key: Key,
    token: u64,
    abort: Box<dyn TxnAbort>,
    observe_only: bool,
    observations: u64,
    max_observations: u64,
}

/// The bound state and its live coordinate: the native generation RuntimeCore last established for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bound {
    key: Key,
    generation: u64,
}

/// A managed commit: volatile. The native transition (epochs, generations) and the live generation it established.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VolatileCommit {
    pub transition: Transition,
    pub live_generation: u64,
}

/// The one terminal abort of an open native transaction (one crossing per attempt, counted). BUSY is a concurrent
/// overlapping call on the same state that did nothing: the transaction is still open, so the abort is retried
/// until the state admits it. Any other status leaves no transaction of this token open (aborted now, or
/// already discarded natively); the authoritative state is unchanged whatever it returns.
fn terminate(counters: &mut Counters, mut abort: impl FnMut() -> i32) {
    loop {
        counters.k1_aborts += 1;
        if abort() != K1_BUSY {
            return;
        }
        std::thread::yield_now();
    }
}

pub struct Core {
    lane: Arc<Lane>,
    /// The live generation's diagnostics mirror (the durability status reads it without the handle's lock).
    live_mirror: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
    /// This runtime's managed lease: claimed on every state it binds, presented on every K1 mutation.
    lease: u64,
    open: bool,
    fault: Option<Code>,
    bound: Option<Bound>,
    /// The residency hold of the bound FMS-resident state (docs/RUNTIME_CORE.md, "HOT residency"): taken before a
    /// state is bound, dropped whenever it is unbound (release, close, reopen, destruction). None for RAM states.
    resident: Option<Box<dyn ResidentRelease>>,
    /// The live identity an explicit release gave back (format and retained-state identity, read under the lease
    /// just before releasing it). Rebinding accepts it besides the durable identity, so volatile work survives an
    /// explicit release without being made durable; restart never sees it (it is never persisted).
    released: Option<(StateFormat, Digest)>,
    /// The width read natively for one state (immutable for that state's lifetime); fuel admission needs it.
    shape: Option<(Key, usize)>,
    turn: Option<OpenTurn>,
    evolution: Option<EvolutionState>,
    counters: Counters,
}

/// What [`Core::recover`] found: the disposition, the identities it is about and, when resumable, the envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recovery {
    pub disposition: Disposition,
    /// Continuity's identity (zero when unanchored).
    pub authorized: Digest,
    /// Whether a verified slot holds the authorized identity.
    pub authorized_present: bool,
    /// The newer unauthorized candidate's identity (zero when none).
    pub candidate: Digest,
    /// The authorized envelope, when the disposition is [`Disposition::Resumable`].
    pub envelope: Option<Vec<u8>>,
}

impl Core {
    /// A closed runtime over one continuity directory (an absolute path).
    pub fn new(continuity_dir: &Path) -> Result<Core, Error> {
        let live_mirror = Arc::new(AtomicU64::new(0));
        Ok(Core {
            lane: Arc::new(Lane::new(Store::new(continuity_dir)?, Arc::clone(&live_mirror))),
            live_mirror,
            worker: None,
            lease: next_lease(),
            open: false,
            fault: None,
            bound: None,
            resident: None,
            released: None,
            shape: None,
            turn: None,
            evolution: None,
            counters: Counters::default(),
        })
    }

    /// The durability lane (the C ABI's wait and status read it without the handle's lock).
    pub fn lane(&self) -> Arc<Lane> {
        Arc::clone(&self.lane)
    }

    // -- lifecycle ----------------------------------------------------------------------------------------

    /// Open (or reopen after close): the store resolves the current authority; nothing is bound; the durability
    /// worker starts. An open runtime is refused (`CONTINUITY_OPEN`) and keeps everything, its open turn included.
    pub fn open(&mut self) -> Result<Snapshot, Error> {
        if self.open {
            return Err(Code::Open.into());
        }
        let snapshot = self.lane.open()?;
        self.reset();
        self.open = true;
        let lane = Arc::clone(&self.lane);
        self.worker = Some(
            std::thread::Builder::new()
                .name("elpis-durable".into())
                .spawn(move || lane.work())
                .map_err(|_| Error::Runtime(Rt::Invalid))?,
        );
        Ok(snapshot)
    }

    /// Close. An open managed turn is aborted natively first (nothing installed, nothing published). A pending
    /// durable request is abandoned; a running one completes first (its I/O cannot be interrupted).
    pub fn close(&mut self) {
        self.reset();
        self.lane.shutdown();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.lane.close();
        self.open = false;
    }

    /// Forget the open runtime's bindings. Never forgets an open native transaction: it ends it first.
    fn reset(&mut self) {
        self.end_turn();
        self.fault = None;
        self.bound = None;
        self.unhold();
        self.released = None;
        self.shape = None;
        self.evolution = None;
    }

    /// Abort the open managed turn, if any, through the capability retained at its begin.
    fn end_turn(&mut self) {
        if let Some(turn) = self.turn.take() {
            self.terminate(turn);
        }
    }

    fn terminate(&mut self, turn: OpenTurn) {
        let OpenTurn { token, mut abort, .. } = turn;
        terminate(&mut self.counters, || abort.txn_abort(token));
    }

    /// Whether a managed turn (an open native transaction) is held.
    pub fn turn_is_open(&self) -> bool {
        self.turn.is_some()
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// The fail-stop disposition, if any (a simulated process death on the durability lane is one).
    pub fn fault(&self) -> Option<Code> {
        if self.lane.is_dead() {
            return Some(Code::TestingProcessDeath);
        }
        self.fault
    }

    fn live(&self) -> Result<(), Error> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        match self.fault() {
            Some(code) => Err(code.into()),
            None => Ok(()),
        }
    }

    /// The current durable authority (readable while fail-stopped: it is what restart will see; refused while the
    /// durability lane does not know it: an uncertain publication).
    pub fn snapshot(&self) -> Result<Snapshot, Error> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        self.lane.durable()
    }

    pub fn counters(&mut self, reset: bool) -> Counters {
        let warm = self.lane.counters(reset);
        let mut c = self.counters;
        c.publications += warm.publications;
        c.durable_requests = warm.requests;
        c.durable_superseded = warm.superseded;
        c.durable_acknowledged = warm.acknowledged;
        c.durable_failed = warm.failed + warm.uncertain;
        if reset {
            self.counters = Counters::default();
        }
        c
    }

    /// The live generation of the bound state (None when nothing is bound).
    pub fn live_generation(&self) -> Option<u64> {
        self.bound.map(|b| b.generation)
    }

    fn set_live(&mut self, key: Key, generation: u64) {
        self.bound = Some(Bound { key, generation });
        self.live_mirror.store(generation, Ordering::Relaxed);
    }

    // -- K1 lineage -----------------------------------------------------------------------------------------

    fn digest(&mut self, sub: &mut dyn K1Ops) -> Result<Digest, Refusal> {
        let mut out = [0u8; 32];
        self.counters.k1_state_digests += 1;
        match sub.state_digest(&mut out) {
            0 => Ok(out),
            rc => Err(ecs(Rt::EcsState, rc, ScheduleOutcome::default())),
        }
    }

    /// The retained-state identity and generation of the claimed state, in one guarded call under the lease.
    fn leased_identity(&mut self, sub: &mut dyn K1Ops) -> Result<(Digest, u64), Refusal> {
        let (mut digest, mut generation) = ([0u8; 32], 0u64);
        self.counters.k1_identities += 1;
        match sub.leased_identity(self.lease, &mut digest, &mut generation) {
            0 => Ok((digest, generation)),
            rc => Err(ecs(Rt::EcsState, rc, ScheduleOutcome::default())),
        }
    }

    /// The durable expected identity of the anchored lineage.
    fn expected(&self) -> Result<Digest, Refusal> {
        match self.lane.durable()?.cognition() {
            Cognition::Unanchored => Err(Code::Unanchored.into()),
            Cognition::Anchored(d) => Ok(d),
        }
    }

    /// Explicitly and durably anchor the first managed lineage at the state's retained identity (reads the state
    /// only). The state is claimed, its identity and generation read in one guarded call, its complete envelope made
    /// durable in the checkpoint store when one is attached, and only then continuity anchors it. This is a control
    /// operation: it waits for its own I/O (anchoring is the durable baseline every restart verifies against).
    pub fn anchor(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<Snapshot, Refusal> {
        self.live()?;
        if self.lane.durable()?.cognition() != Cognition::Unanchored {
            return Err(Code::AlreadyAnchored.into());
        }
        self.hold(sub)?;
        if let Err(r) = self.claim(sub) {
            self.unhold();
            return Err(r);
        }
        let anchored = self.anchor_claimed(sub, key);
        if anchored.is_err() {
            let _ = sub.lease_release(self.lease); // nothing was bound
            self.unhold();
        }
        anchored
    }

    fn anchor_claimed(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<Snapshot, Refusal> {
        let (identity, generation) = self.leased_identity(sub)?;
        let envelope = match self.lane.capacity() {
            Some(capacity) => Some(self.capture(sub, capacity, generation, true)?),
            None => None,
        };
        let format = sub.format();
        let published = self.lane.with_stores(|st| -> Result<Snapshot, Refusal> {
            if let (Some(store), Some(mut envelope)) = (st.checkpoint.as_mut(), envelope) {
                // SAFETY: the substrate library's pure sealer over a buffer this call owns.
                if unsafe { (sub.sealer())(envelope.as_mut_ptr(), envelope.len()) } != 0
                    || checkpoint::envelope_identity(&envelope) != Some(identity)
                {
                    return Err(Rt::CheckpointShape.into());
                }
                if store.holding(&identity).is_none() {
                    store.write_envelope(&envelope, None)?;
                }
            }
            Ok(st.continuity.anchor_cognition_as(format, Some(&identity))?)
        })?;
        match published {
            Ok(s) => {
                self.lane.count_publication();
                self.set_live(key, generation);
                Ok(s)
            }
            Err(r) => {
                if matches!(r.error, Error::Continuity(Code::PublicationUncertain | Code::TestingProcessDeath)) {
                    self.fault = Some(match r.error {
                        Error::Continuity(c) => c,
                        _ => unreachable!(),
                    });
                }
                Err(r)
            }
        }
    }

    /// Capture the bound state's envelope (unsealed) at the live generation `generation` into a fresh lane buffer.
    /// `provisioned`: `size` is the checkpoint store's capacity, so a capture the state refuses as malformed
    /// (`INVALID`: another envelope size) is `CHECKPOINT_SHAPE` (never resized for).
    fn capture(
        &mut self,
        sub: &mut dyn K1Ops,
        size: usize,
        generation: u64,
        provisioned: bool,
    ) -> Result<Vec<u8>, Refusal> {
        let mut buffer = self.lane.buffer(size);
        let mut captured = 0u64;
        self.counters.k1_captures += 1;
        match sub.leased_capture(self.lease, &mut buffer, &mut captured) {
            0 if captured == generation => Ok(buffer),
            0 => {
                self.lane.give_back(buffer);
                Err(self.lineage_lost(0))
            }
            K1_LEASED => {
                self.lane.give_back(buffer);
                Err(self.lineage_lost(K1_LEASED))
            }
            K1_INVALID if provisioned => {
                self.lane.give_back(buffer);
                Err(ecs(Rt::CheckpointShape, K1_INVALID, ScheduleOutcome::default()))
            }
            rc => {
                self.lane.give_back(buffer);
                Err(ecs(Rt::EcsRefused, rc, ScheduleOutcome::default()))
            }
        }
    }

    /// Claim managed ownership of the state (its K1 lease). Refused, without a fail-stop, while an unmanaged
    /// transaction is open on it (BUSY): the state is not adopted mid-transaction.
    fn claim(&mut self, sub: &mut dyn K1Ops) -> Result<(), Refusal> {
        self.counters.k1_lease_claims += 1;
        match sub.lease_claim(self.lease) {
            0 => Ok(()),
            rc => Err(ecs(Rt::EcsRefused, rc, ScheduleOutcome::default())),
        }
    }

    /// The bound state is not provably the lineage (its generation or identity is not the live one, or its managed
    /// lease was taken): fail-stop. Nothing outside the lineage is ever built upon or published.
    fn lineage_lost(&mut self, k1_status: i32) -> Refusal {
        self.fault = Some(Code::StateMismatch);
        Refusal { k1_status, ..Refusal::from(Code::StateMismatch) }
    }

    /// Bind the lineage to this state, or verify it is the bound one. Binding checks the state's identity before
    /// claiming it (a mismatched state is never claimed), claims it, then reads identity and generation together
    /// under the lease: the identity must still be the one checked, and the generation becomes the live coordinate.
    /// The state must be the durable identity (what a fresh open or restart resumes: only acknowledged state) or the
    /// live identity an explicit release of this open runtime gave back (its volatile work, never persisted).
    fn reconcile(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<(), Refusal> {
        if let Some(bound) = self.bound {
            return if bound.key == key { Ok(()) } else { Err(Rt::SubstrateSwitch.into()) };
        }
        // Residency first: an FMS-resident state that is not WARM now is refused before anything touches it (the
        // identity check below would otherwise materialize it synchronously on this, the HOT, path).
        self.hold(sub)?;
        let bound = self.bind(sub, key);
        if bound.is_err() {
            self.unhold();
        }
        bound
    }

    /// Take the residency hold of a state about to be bound (a no-op for RAM states). Refused, without a fail-stop
    /// and without touching the state further, with `RUNTIME_NOT_RESIDENT` when it is not WARM now: the caller
    /// materializes it explicitly (a WARM operation) and retries.
    fn hold(&mut self, sub: &mut dyn K1Ops) -> Result<(), Refusal> {
        self.unhold(); // at most one hold, and only for the state being bound
        if self.resident.is_some() {
            // An earlier hold could not be dropped yet (a concurrent overlapping call): never stack a second one.
            return Err(ecs(Rt::EcsRefused, K1_BUSY, ScheduleOutcome::default()));
        }
        match sub.resident_hold() {
            0 => {
                self.resident = sub.retain_resident();
                Ok(())
            }
            K1_FMS_NOT_RESIDENT => Err(Refusal { k1_status: K1_FMS_NOT_RESIDENT, ..Refusal::from(Rt::NotResident) }),
            rc => Err(ecs(Rt::EcsRefused, rc, ScheduleOutcome::default())),
        }
    }

    /// Drop the residency hold, if any. A refused release (BUSY: a concurrent overlapping call on the state) keeps
    /// the capability, so the hold is released by the next unbind or bind attempt, or at destruction; meanwhile the
    /// state merely stays resident.
    fn unhold(&mut self) {
        if let Some(mut resident) = self.resident.take() {
            if resident.resident_release() != 0 {
                self.resident = Some(resident);
            }
        }
    }

    fn bind(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<(), Refusal> {
        let durable = self.lane.durable()?;
        let anchored = match durable.cognition() {
            Cognition::Unanchored => return Err(Code::Unanchored.into()),
            Cognition::Anchored(d) => d,
        };
        // An identity of one format never stands for a state of another (a K1 state for a composite lineage).
        let (format, expected) = (sub.format(), self.digest(sub)?);
        if (format, expected) != (durable.format(), anchored) && Some((format, expected)) != self.released {
            self.fault = Some(Code::StateMismatch);
            return Err(Code::StateMismatch.into());
        }
        self.claim(sub)?;
        let (identity, generation) = match self.leased_identity(sub) {
            Ok(v) => v,
            Err(r) if r.k1_status == K1_LEASED => return Err(self.lineage_lost(K1_LEASED)),
            Err(r) => {
                let _ = sub.lease_release(self.lease);
                return Err(r);
            }
        };
        if identity != expected {
            let _ = sub.lease_release(self.lease); // it moved between the check and the claim
            self.fault = Some(Code::StateMismatch);
            return Err(Code::StateMismatch.into());
        }
        self.released = None;
        self.set_live(key, generation);
        Ok(())
    }

    /// Release managed ownership of the bound state (its lease) and unbind it, so its owner may use it unmanaged
    /// again. Refused while a managed turn is open; another state is refused (`COGNITION_SUBSTRATE_SWITCH`). A lease
    /// already taken by another owner is reported, and the state unbound. The live identity is read under the lease
    /// first (one hash: a control operation, never on the HOT lane) and kept, so rebinding the exact bytes it gave
    /// back resumes the live lineage: volatile work since the last durable acknowledgment is neither published nor
    /// lost by a release. Rebinding the durable state instead abandons that work, as a restart would.
    pub fn release(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<(), Refusal> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        let live = match self.bound {
            Some(bound) if bound.key == key => bound.generation,
            Some(_) => return Err(Rt::SubstrateSwitch.into()),
            None => return Ok(()),
        };
        self.bound = None;
        self.shape = None;
        // A lease already taken (or a generation that moved under it) gives back no live identity: the state is not
        // provably the lineage, and rebinding falls back to the durable identity.
        self.released = match self.leased_identity(sub) {
            Ok((identity, generation)) if generation == live => Some((sub.format(), identity)),
            _ => None,
        };
        let released = sub.lease_release(self.lease);
        self.unhold(); // the state is unbound: it may be demoted again
        match released {
            0 => Ok(()),
            rc => Err(ecs(Rt::EcsRefused, rc, ScheduleOutcome::default())),
        }
    }

    /// The state's width, read natively once per state (cold path; a pure getter) and verified against the declared
    /// dimension: a state of another dimension is refused (`ECS_REFUSED`, K1 `INVALID`) before any input is read.
    fn width(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<usize, Refusal> {
        if let Some((cached, width)) = self.shape {
            if cached == key {
                return Ok(width);
            }
        }
        let (mut dim, mut width) = (0usize, 0usize);
        self.counters.k1_shapes += 1;
        match sub.shape(&mut dim, &mut width) {
            0 if dim == key.dim && width > 0 => {
                self.shape = Some((key, width));
                Ok(width)
            }
            0 => Err(ecs(Rt::EcsRefused, K1_INVALID, ScheduleOutcome::default())),
            rc => Err(ecs(Rt::EcsState, rc, ScheduleOutcome::default())),
        }
    }

    /// The bound state's identity, read on request (cold path: one `leased_identity` crossing, which hashes the
    /// state): the retained-state identity and the generation together, verified against the live coordinate.
    pub fn live_identity(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<(Digest, u64), Refusal> {
        self.live()?;
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        self.reconcile(sub, key)?;
        let live = self.bound.expect("bound").generation;
        match self.leased_identity(sub) {
            Ok((d, g)) if g == live => Ok((d, g)),
            Ok(_) => Err(self.lineage_lost(0)),
            Err(r) if r.k1_status == K1_LEASED => Err(self.lineage_lost(K1_LEASED)),
            Err(r) => Err(r),
        }
    }

    // -- the canonical read-only QUERY ---------------------------------------------------------------------------

    /// QUERY: `f_W(x)` from the lineage's live K1 state, read-only (docs/COGNITION_R0.md).
    ///
    /// One native crossing answers and reports the generation of the same state under this runtime's lease
    /// (`leased_query`): it must be the live generation. A state whose lease was taken, or whose generation moved
    /// outside RuntimeCore, fail-stops (`CONTINUITY_STATE_MISMATCH`) and its answer is never returned. No
    /// transaction is opened, nothing is committed, published or written, no digest is computed, and the durable
    /// authority is not consulted (except, once, to bind). `out` receives the answer (one value per row; zeroed on
    /// any refusal). Returns the live generation the answer was computed from.
    pub fn query(
        &mut self,
        sub: &mut dyn K1Ops,
        key: Key,
        x: &[f64],
        out: &mut [f64],
        budget: &Budget,
    ) -> Result<u64, Refusal> {
        let answered = self.query_inner(sub, key, x, out, budget);
        if answered.is_err() {
            out.fill(0.0); // no answer leaves a refused query, whatever refused it
        }
        answered
    }

    fn query_inner(
        &mut self,
        sub: &mut dyn K1Ops,
        key: Key,
        x: &[f64],
        out: &mut [f64],
        budget: &Budget,
    ) -> Result<u64, Refusal> {
        self.live()?;
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        let rows = out.len();
        if rows == 0 || rows.checked_mul(key.dim) != Some(x.len()) {
            return Err(Rt::Invalid.into());
        }
        match self.bound {
            Some(bound) if bound.key != key => return Err(Rt::SubstrateSwitch.into()),
            Some(_) => {}
            None => {
                self.expected()?; // unanchored: refused before any native call
            }
        }
        let width = self.width(sub, key)?;
        fuel::admit_query(budget, key.dim, width, rows).map_err(|_| Refusal::from(Rt::Fuel))?;
        self.reconcile(sub, key)?;
        let live = self.bound.expect("bound").generation;
        let mut generation = 0u64;
        self.counters.k1_queries += 1;
        match sub.leased_query(self.lease, key.dim, x, out, &mut generation) {
            0 if generation == live => Ok(generation),
            0 => Err(self.lineage_lost(0)),
            K1_LEASED => Err(self.lineage_lost(K1_LEASED)),
            rc => Err(ecs(Rt::EcsRefused, rc, ScheduleOutcome::default())),
        }
    }

    // -- the managed canonical turn -----------------------------------------------------------------------------

    /// Verify the lineage, begin the native transaction and run the whole schedule on its candidate (one native
    /// call; capacity beyond the state's reservation grows it on the cold path first). `s3` receives the readout.
    /// The transaction stays open for [`Core::turn_commit`] or [`Core::turn_abort`].
    pub fn turn_begin(
        &mut self,
        sub: &mut dyn K1Ops,
        key: Key,
        stimulus: &Stimulus,
        budget: &Budget,
        s3: &mut [f64],
    ) -> Result<ScheduleOutcome, Refusal> {
        self.live()?;
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        let rows = stimulus.y.len();
        let schedule = stimulus.schedule;
        let scheduled = schedule.iter().try_fold(0u64, |n, e| n.checked_add(e.rows));
        if schedule.is_empty()
            || schedule.len() > MAX_EXPERIENCES
            || rows == 0
            || scheduled != Some(rows as u64)
            || rows.checked_mul(key.dim) != Some(stimulus.x.len())
            || s3.len() != features(key.dim)
        {
            return Err(Rt::Invalid.into());
        }
        match self.bound {
            Some(bound) if bound.key != key => return Err(Rt::SubstrateSwitch.into()),
            Some(_) => {}
            None => {
                self.expected()?; // unanchored: refused before any native call
            }
        }
        // Total fuel, before the identity check, any reserve, the transaction or a candidate mutation.
        let width = self.width(sub, key)?;
        fuel::admit_schedule(budget, key.dim, width, schedule).map_err(|_| Refusal::from(Rt::Fuel))?;
        self.reconcile(sub, key)?;
        let abort = sub.retain_abort();
        let (token, result) = self.run_schedule(sub, stimulus, s3)?;
        let max_observations = budget.max_observations;
        self.turn = Some(OpenTurn { key, token, abort, observe_only: false, observations: 0, max_observations });
        Ok(result)
    }

    /// Begin an OBSERVE-ONLY managed turn on a composite substrate (ABI v5): the lineage is verified and bound exactly
    /// as for [`Core::turn_begin`], and the native composite transaction is opened at the live generation, but no K1
    /// experience schedule runs: the K1 component's candidate is the unchanged copy of the live K1 state. The turn
    /// then admits typed observations ([`Core::turn_observe`], at most the budget's `max_observations`) and ends by
    /// [`Core::turn_commit`] (volatile; the live generation advances by one, as for every committed transaction, and
    /// the K1 retained state stays byte-identical) or [`Core::turn_abort`]. A K1 or K1 FMS substrate has no memory
    /// (`RUNTIME_INVALID`, before any native call). Returns the live generation the turn was opened on.
    pub fn turn_begin_observe(&mut self, sub: &mut dyn K1Ops, key: Key, budget: &Budget) -> Result<u64, Refusal> {
        self.live()?;
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        if key.kind != KIND_COMPOSITE || sub.format() != StateFormat::Composite {
            return Err(Rt::Invalid.into());
        }
        match self.bound {
            Some(bound) if bound.key != key => return Err(Rt::SubstrateSwitch.into()),
            Some(_) => {}
            None => {
                self.expected()?; // unanchored: refused before any native call
            }
        }
        // The shape is verified natively, then the budget, before the identity check or the transaction.
        self.width(sub, key)?;
        fuel::admit_observe_turn(budget).map_err(|_| Refusal::from(Rt::Fuel))?;
        self.reconcile(sub, key)?;
        let abort = sub.retain_abort();
        let token = self.begin(sub)?;
        let max_observations = budget.max_observations;
        self.turn = Some(OpenTurn { key, token, abort, observe_only: true, observations: 0, max_observations });
        Ok(self.bound.expect("bound").generation)
    }

    /// Begin the managed transaction under this runtime's lease; the same guarded K1 call reports the generation of
    /// exactly the source the candidate was copied from: it must be the live generation. A lost lease (another owner
    /// claimed the state) or another generation fail-stops before the candidate is touched; the transaction just
    /// opened is aborted.
    fn begin(&mut self, sub: &mut dyn K1Ops) -> Result<u64, Refusal> {
        let live = self.bound.expect("bound").generation;
        let (mut token, mut source) = (0u64, 0u64);
        self.counters.k1_txn_begins += 1;
        match sub.txn_begin_at(self.lease, &mut source, &mut token) {
            0 if source == live => Ok(token),
            0 => {
                self.abort(sub, token);
                Err(self.lineage_lost(0))
            }
            K1_LEASED => Err(self.lineage_lost(K1_LEASED)),
            rc => Err(ecs(Rt::EcsRefused, rc, ScheduleOutcome::default())),
        }
    }

    /// Abort a transaction this call opened (before it became the turn): the same terminal law.
    fn abort(&mut self, sub: &mut dyn K1Ops, token: u64) {
        terminate(&mut self.counters, || sub.txn_abort(token));
    }

    fn schedule_once(
        &mut self,
        sub: &mut dyn K1Ops,
        token: u64,
        stimulus: &Stimulus,
        s3: &mut [f64],
    ) -> (i32, ScheduleOutcome) {
        let mut result = ScheduleOutcome::default();
        self.counters.k1_run_schedules += 1;
        let rc = sub.txn_run_schedule(
            self.lease,
            token,
            stimulus.x,
            stimulus.y,
            stimulus.schedule,
            stimulus.rate,
            s3,
            &mut result,
        );
        (rc, result)
    }

    fn run_schedule(
        &mut self,
        sub: &mut dyn K1Ops,
        stimulus: &Stimulus,
        s3: &mut [f64],
    ) -> Result<(u64, ScheduleOutcome), Refusal> {
        let mut token = self.begin(sub)?;
        let (mut rc, mut result) = self.schedule_once(sub, token, stimulus, s3);
        if rc == K1_CAPACITY || rc == K1_FMS_CAPACITY {
            // Recoverable and checked before the candidate is touched: the transaction is open and unchanged.
            // Grow the reservation outside any transaction (cold path), then run the schedule once more.
            self.abort(sub, token);
            let rows = stimulus.schedule.iter().map(|e| e.rows).max().unwrap_or(0);
            let rows = usize::try_from(rows).map_err(|_| Refusal::from(Rt::Invalid))?;
            self.counters.k1_reserves += 1;
            match sub.reserve(self.lease, rows) {
                0 => {}
                K1_LEASED => return Err(self.lineage_lost(K1_LEASED)),
                rc => return Err(ecs(Rt::EcsRefused, rc, ScheduleOutcome::default())),
            }
            token = self.begin(sub)?;
            (rc, result) = self.schedule_once(sub, token, stimulus, s3);
        }
        if rc != 0 {
            // STALE, or NONFINITE from the candidate-mutating schedule, discards the transaction natively;
            // every other refusal leaves it open, and the turn aborts it.
            if rc != K1_STALE && rc != K1_NONFINITE {
                self.abort(sub, token);
            }
            return Err(ecs(Rt::EcsRefused, rc, result));
        }
        Ok((token, result))
    }

    fn take_turn(&mut self, key: Key) -> Result<OpenTurn, Refusal> {
        match &self.turn {
            Some(turn) if turn.key == key => Ok(self.turn.take().expect("open turn")),
            _ => Err(Rt::TurnNotOpen.into()),
        }
    }

    /// Commit the open turn natively: a VOLATILE commit. The candidate is installed and the live generation
    /// advances by exactly one; nothing is checkpointed, published or written, and the durability lane is not
    /// touched. A fail-stopped runtime, or a refused commit, aborts the turn instead: either way the turn has ended
    /// when this returns, unless it was refused before the turn was taken (`RUNTIME_CLOSED`,
    /// `RUNTIME_TURN_NOT_OPEN`). Durability is a separate, explicit request ([`Core::durable_request`]).
    pub fn turn_commit(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<VolatileCommit, Refusal> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        // An observe-only turn that recorded nothing has nothing to commit: refused before the turn is taken, so it
        // stays open for an observation or an abort.
        if matches!(&self.turn, Some(t) if t.key == key && t.observe_only && t.observations == 0) {
            return Err(Rt::Invalid.into());
        }
        let turn = self.take_turn(key)?;
        if let Some(code) = self.fault() {
            self.terminate(turn);
            return Err(code.into());
        }
        let observe_only = turn.observe_only;
        let live = self.bound.expect("a turn is begun on the bound state").generation;
        let mut transition = Transition::default();
        self.counters.k1_commits += 1;
        let rc = sub.txn_commit(self.lease, turn.token, &mut transition);
        if rc != 0 {
            // STALE discards the transaction natively; any other refusal leaves it open.
            if rc != K1_STALE {
                self.terminate(turn);
            }
            if rc == K1_LEASED {
                return Err(self.lineage_lost(K1_LEASED));
            }
            let error = if rc == K1_STALE { Rt::EcsStale } else { Rt::EcsRefused };
            return Err(ecs(error, rc, ScheduleOutcome::default()));
        }
        if transition.generation_before != live
            || transition.generation_after != live.wrapping_add(1)
            || (observe_only && (transition.steps != 0 || transition.epoch_after != transition.epoch_before))
        {
            // The state committed is not the continuation of the live coordinate (or an observe-only commit reports
            // K1 learning it never ran): never built upon.
            self.fault = Some(Code::StateMismatch);
            return Err(Refusal { committed: Some(transition), ..Refusal::from(Code::StateMismatch) });
        }
        self.set_live(key, transition.generation_after);
        Ok(VolatileCommit { transition, live_generation: transition.generation_after })
    }

    /// OBSERVE inside the open managed turn of a composite substrate: the observation (opaque here) is recorded on
    /// the turn's memory candidate, which becomes live only with the turn's commit (and durable only by an explicit
    /// durable publication). A refusal is recoverable: the turn stays open and its candidate unchanged. A K1
    /// substrate has no memory (`RUNTIME_INVALID`). Returns the relation key; the second value is the memory's own
    /// status behind a composite refusal (0 otherwise).
    pub fn turn_observe(
        &mut self,
        sub: &mut dyn K1Ops,
        key: Key,
        observation: Observation,
    ) -> (Result<Digest, Refusal>, i32) {
        if !self.open {
            return (Err(Rt::Closed.into()), 0);
        }
        let (token, recorded, max) = match &self.turn {
            Some(turn) if turn.key == key => (turn.token, turn.observations, turn.max_observations),
            _ => return (Err(Rt::TurnNotOpen.into()), 0),
        };
        if let Some(code) = self.fault() {
            return (Err(code.into()), 0);
        }
        if sub.format() != StateFormat::Composite || observation.0.is_null() {
            return (Err(Rt::Invalid.into()), 0);
        }
        // The turn's observation bound (ABI v5), before the native call: recoverable, the turn stays open.
        if fuel::admit_observation(recorded, max).is_err() {
            return (Err(Rt::Fuel.into()), 0);
        }
        let (mut relation, mut memory_status) = ([0u8; 32], 0i32);
        self.counters.k1_observes += 1;
        match sub.txn_observe(self.lease, token, observation, &mut relation, &mut memory_status) {
            0 => {
                if let Some(turn) = self.turn.as_mut() {
                    turn.observations += 1;
                }
                (Ok(relation), 0)
            }
            K1_LEASED => (Err(self.lineage_lost(K1_LEASED)), 0),
            rc => (Err(ecs(Rt::EcsRefused, rc, ScheduleOutcome::default())), memory_status),
        }
    }

    /// Abort the open turn of this substrate: nothing is installed. Another substrate's key is refused
    /// (`RUNTIME_TURN_NOT_OPEN`) and leaves the turn open.
    pub fn turn_abort(&mut self, key: Key) -> Result<(), Refusal> {
        let turn = self.take_turn(key)?;
        self.terminate(turn);
        Ok(())
    }

    // -- durable publication: an explicit request, served by the durability lane -----------------------------

    /// The capture size of the bound substrate's envelope: the attached checkpoint store's capacity, else the K1
    /// envelope size of its shape (verified natively by the capture and the seal). A composite without a checkpoint
    /// store is refused (`CHECKPOINT_INVALID`): its memory capacity is not known here.
    fn capture_size(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<usize, Refusal> {
        if let Some(capacity) = self.lane.capacity() {
            return Ok(capacity);
        }
        if key.kind == KIND_COMPOSITE {
            return Err(Rt::CheckpointInvalid.into());
        }
        let width = self.width(sub, key)?;
        k1_envelope_bytes(key.dim, width).ok_or_else(|| Rt::Invalid.into())
    }

    /// Request durable publication of the live state (binding it first if needed). The state's envelope is captured
    /// UNSEALED under this runtime's lease at the live generation (`leased_capture`: the one synchronous step) and
    /// handed to the durability lane, which seals, checkpoints and publishes it. Returns at once with a PENDING
    /// ticket; it never waits for the lane. `kind` is what the caller asks to make durable: the complete state
    /// ([`Kind::Recoverable`], refused with `RUNTIME_RECOVERY_UNAVAILABLE` unless a checkpoint store is attached) or
    /// only its identity ([`Kind::Identity`], refused with `RUNTIME_CHECKPOINT_ATTACHED` while one is attached).
    /// Refused, before anything is bound or captured, while a managed turn is open, while the lane is faulted (its
    /// fault code) and on an unanchored lineage (anchoring is explicit). A request that finds an earlier one still
    /// waiting supersedes it (bounded: one waiting job).
    pub fn durable_request(&mut self, sub: &mut dyn K1Ops, key: Key, kind: Kind) -> Result<Ticket, Refusal> {
        self.live()?;
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        if matches!(self.bound, Some(bound) if bound.key != key) {
            return Err(Rt::SubstrateSwitch.into());
        }
        if let Some(code) = self.lane.fault() {
            return Err(code.into());
        }
        match (kind, self.lane.capacity().is_some()) {
            (Kind::Recoverable, false) => return Err(Rt::RecoveryUnavailable.into()),
            (Kind::Identity, true) => return Err(Rt::CheckpointAttached.into()),
            _ => {}
        }
        self.reconcile(sub, key)?;
        let size = self.capture_size(sub, key)?;
        let live = self.bound.expect("bound").generation;
        let provisioned = self.lane.capacity().is_some();
        let envelope = self.capture(sub, size, live, provisioned)?;
        Ok(self.lane.enqueue(kind, live, sub.format(), envelope, sub.sealer())?)
    }

    // -- K1 Recovery R0: the bounded checkpoint store -------------------------------------------------------------

    /// Attach the provisioned checkpoint store (an open runtime with no turn in flight and no durable publication
    /// pending or running: an identity-only request admitted without a store never runs with one). From then on the
    /// anchor and every durable publication persist the complete envelope before continuity names it.
    pub fn checkpoint_attach(&mut self, directory: &Path) -> Result<(), Refusal> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        if self.lane.busy() {
            return Err(Rt::DurableBusy.into());
        }
        let store = checkpoint::Store::open(directory)?;
        self.lane.with_stores(|st| st.checkpoint = Some(store))?;
        Ok(())
    }

    pub fn checkpoint_attached(&self) -> bool {
        self.lane.capacity().is_some()
    }

    /// What restart may do with the lineage. Continuity decides; the slots are evidence. Read-only.
    pub fn recover(&mut self) -> Result<Recovery, Refusal> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        self.lane.with_stores(|st| -> Result<Recovery, Refusal> {
            let authorized = match st.continuity.snapshot()?.cognition() {
                Cognition::Unanchored => None,
                Cognition::Anchored(d) => Some(d),
            };
            let store = st.checkpoint.as_mut().ok_or(Rt::CheckpointInvalid)?;
            let candidate = authorized.and_then(|d| store.unresolved(&d));
            let mut out = Recovery {
                disposition: Disposition::NothingToRecover,
                authorized: authorized.unwrap_or([0u8; 32]),
                authorized_present: false,
                candidate: candidate.map_or([0u8; 32], |(_, s)| s.identity),
                envelope: None,
            };
            let authorized = match authorized {
                Some(d) => d,
                None => return Ok(out),
            };
            let held = store.holding(&authorized);
            out.authorized_present = held.is_some();
            out.disposition = if candidate.is_some() {
                Disposition::CandidateUnresolved
            } else if let Some(i) = held {
                match store.envelope(i)? {
                    Some((slot, envelope)) if slot.identity == authorized => {
                        out.envelope = Some(envelope.to_vec());
                        Disposition::Resumable
                    }
                    _ => Disposition::CheckpointMissing, // it no longer verifies
                }
            } else {
                Disposition::CheckpointMissing
            };
            Ok(out)
        })?
    }

    /// Operator reconciliation: discard the unresolved candidate `candidate` (withdraw its slot). The lineage stays
    /// at continuity's authority. Refused unless `candidate` is exactly the unresolved candidate, and while a durable
    /// publication is pending or running (`RUNTIME_DURABLE_BUSY`).
    pub fn checkpoint_discard(&mut self, candidate: &Digest) -> Result<(), Refusal> {
        self.live()?;
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        if self.lane.busy() {
            return Err(Rt::DurableBusy.into());
        }
        self.lane.with_stores(|st| -> Result<(), Refusal> {
            let authorized = match st.continuity.snapshot()?.cognition() {
                Cognition::Unanchored => return Err(Rt::CheckpointInvalid.into()),
                Cognition::Anchored(d) => d,
            };
            let store = st.checkpoint.as_mut().ok_or(Rt::CheckpointInvalid)?;
            match store.unresolved(&authorized) {
                Some((slot, s)) if s.identity == *candidate => Ok(store.retract(slot)?),
                _ => Err(Rt::CheckpointInvalid.into()),
            }
        })?
    }

    /// Operator reconciliation: adopt the unresolved candidate `candidate` as the lineage's authority (continuity
    /// publishes authorized -> candidate). Its complete envelope is verified first. A publication failure fail-stops.
    /// Refused while a state is bound or a durable publication is pending or running.
    pub fn checkpoint_adopt(&mut self, candidate: &Digest) -> Result<Snapshot, Refusal> {
        self.live()?;
        if self.turn.is_some() || self.bound.is_some() {
            return Err(Rt::TurnOpen.into()); // adopting under a bound state would contradict its identity
        }
        if self.lane.busy() {
            return Err(Rt::DurableBusy.into());
        }
        let adopted = self.lane.with_stores(|st| -> Result<Snapshot, Refusal> {
            let authorized = match st.continuity.snapshot()?.cognition() {
                Cognition::Unanchored => return Err(Code::Unanchored.into()),
                Cognition::Anchored(d) => d,
            };
            let store = st.checkpoint.as_mut().ok_or(Rt::CheckpointInvalid)?;
            let slot = match store.unresolved(&authorized) {
                Some((slot, s)) if s.identity == *candidate => slot,
                _ => return Err(Rt::CheckpointInvalid.into()),
            };
            match store.envelope(slot)? {
                Some((s, _)) if s.identity == *candidate => {}
                _ => return Err(Rt::CheckpointInvalid.into()),
            }
            Ok(st.continuity.commit_cognition(Some(&authorized), Some(candidate))?)
        })?;
        match adopted {
            Ok(snapshot) => {
                self.lane.count_publication();
                self.released = None; // the durable authority moved: a released live identity no longer applies
                Ok(snapshot)
            }
            Err(r) => {
                if let Error::Continuity(
                    code @ (Code::PublicationRefused | Code::PublicationUncertain | Code::TestingProcessDeath),
                ) = r.error
                {
                    self.fault = Some(code);
                }
                Err(r)
            }
        }
    }

    /// LEGACY import (docs/ECS_COMPOSITE_STATE.md): move this runtime's K1 lineage to the composite substrate
    /// `sub`, made by `elpis_ecs_composite_import_k1` from exactly the authorized K1 state with an empty memory.
    /// Explicit and authority-checked: the lineage must be K1-format and anchored, no turn open and nothing bound,
    /// and a checkpoint store provisioned for the composite envelope size must be attached. The composite is claimed,
    /// its envelope captured and sealed; it must name the authorized K1 identity and hold an empty memory (otherwise
    /// the claim is released, `CONTINUITY_STATE_MISMATCH`); it is made durable in the checkpoint store before
    /// continuity publishes the import. Then the composite is bound under this runtime's lease.
    pub fn import_composite(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<Snapshot, Refusal> {
        self.live()?;
        if self.turn.is_some() || self.bound.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        if self.lane.busy() {
            return Err(Rt::DurableBusy.into());
        }
        let durable = self.lane.durable()?;
        let k1 = match durable.cognition() {
            Cognition::Unanchored => return Err(Code::Unanchored.into()),
            Cognition::Anchored(d) => d,
        };
        if durable.format() != StateFormat::K1 || sub.format() != StateFormat::Composite {
            return Err(Code::StateMismatch.into());
        }
        let capacity = self.lane.capacity().ok_or(Rt::CheckpointInvalid)?;
        self.claim(sub)?;
        let imported = self.import_claimed(sub, key, &k1, capacity);
        if imported.is_err() {
            let _ = sub.lease_release(self.lease);
        }
        imported
    }

    fn import_claimed(
        &mut self,
        sub: &mut dyn K1Ops,
        key: Key,
        k1: &Digest,
        capacity: usize,
    ) -> Result<Snapshot, Refusal> {
        let (identity, generation) = self.leased_identity(sub)?;
        let mut envelope = match self.capture(sub, capacity, generation, true) {
            Ok(e) => e,
            Err(r) if r.error == Error::Continuity(Code::StateMismatch) => {
                self.fault = None; // a capture refusal of a state never bound is not a lineage loss
                return Err(Rt::CheckpointShape.into());
            }
            Err(r) => return Err(r),
        };
        // SAFETY: the substrate library's pure sealer over a buffer this call owns.
        if unsafe { (sub.sealer())(envelope.as_mut_ptr(), envelope.len()) } != 0
            || checkpoint::envelope_identity(&envelope) != Some(identity)
        {
            return Err(Rt::CheckpointShape.into());
        }
        if checkpoint::composite_import_identity(&envelope, k1) != Some(identity) {
            return Err(Code::StateMismatch.into()); // not the authorized K1 state with an empty memory
        }
        let published = self.lane.with_stores(|st| -> Result<Snapshot, Refusal> {
            let store = st.checkpoint.as_mut().ok_or(Rt::CheckpointInvalid)?;
            if store.unresolved(k1).is_some() {
                return Err(Rt::CheckpointUnresolved.into());
            }
            if store.holding(&identity).is_none() {
                store.write_envelope(&envelope, Some(k1))?;
            }
            Ok(st.continuity.import_cognition(Some(k1), Some(&identity))?)
        })?;
        match published {
            Ok(s) => {
                self.lane.count_publication();
                self.released = None;
                self.set_live(key, generation);
                self.shape = None;
                Ok(s)
            }
            Err(r) => {
                if let Error::Continuity(code @ (Code::PublicationUncertain | Code::TestingProcessDeath)) = r.error {
                    self.fault = Some(code);
                }
                Err(r)
            }
        }
    }

    // -- evolution: the durable reservation is synchronous and precedes execution --------------------------------

    /// The current idle evolution authority an assertion must be built against.
    pub fn evolution_authority(&self) -> Result<Snapshot, Error> {
        self.live()?;
        let snapshot = self.lane.durable()?;
        if snapshot.evolution().assertion().is_some() {
            return Err(Code::EvolutionPending.into());
        }
        Ok(snapshot)
    }

    /// Durably reserve the exact assertion against the authority the caller validated it with. Only after
    /// this returns may the boundary execute the attempt, once. Synchronous: the reservation is durable (written
    /// and synced) when it returns; volatile cognition never stands in for it.
    pub fn evolution_reserve(&mut self, observed: &EvolutionState, assertion: &Digest) -> Result<Snapshot, Error> {
        self.live()?;
        if self.evolution.is_some() {
            return Err(Rt::EvolutionInFlight.into());
        }
        if self.lane.durable()?.evolution().assertion().is_some() {
            return Err(Code::EvolutionPending.into());
        }
        match self.lane.with_stores(|st| st.continuity.reserve_evolution(Some(observed), Some(assertion)))? {
            Ok(snapshot) => {
                self.lane.count_publication();
                self.evolution = Some(snapshot.evolution());
                Ok(snapshot)
            }
            Err(code) => {
                // Nothing executed; whether the reservation became durable is for restart to resolve.
                self.fault = Some(code);
                Err(code.into())
            }
        }
    }

    /// Finalize the attempt in flight with the receipt its execution produced.
    pub fn evolution_finalize(&mut self, receipt: &Digest) -> Result<Snapshot, Error> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        let pending = self.evolution.take().ok_or(Rt::EvolutionNotInFlight)?;
        if let Some(code) = self.fault() {
            return Err(code.into());
        }
        match self.lane.with_stores(|st| st.continuity.finalize_evolution(Some(&pending), Some(receipt)))? {
            Ok(snapshot) => {
                self.lane.count_publication();
                Ok(snapshot)
            }
            Err(code) => {
                // The attempt executed; its finalization is not certain. Never re-executed.
                self.fault = Some(code);
                Err(code.into())
            }
        }
    }

    /// The boundary could not complete the attempt in flight (an exception or an invalid result may follow an
    /// external effect): the durable reservation stays pending and the runtime fail-stops.
    pub fn evolution_abandon(&mut self) -> Result<(), Error> {
        if self.evolution.take().is_none() {
            return Err(Rt::EvolutionNotInFlight.into());
        }
        self.fault.get_or_insert(Code::EvolutionPending);
        Ok(())
    }

    /// Explicit reconciliation: finalize a durable pending authority with an externally established receipt.
    pub fn evolution_reconcile(&mut self, expected: &EvolutionState, receipt: &Digest) -> Result<Snapshot, Error> {
        self.live()?;
        if self.evolution.is_some() {
            return Err(Rt::EvolutionInFlight.into());
        }
        match self.lane.with_stores(|st| st.continuity.finalize_evolution(Some(expected), Some(receipt)))? {
            Ok(snapshot) => {
                self.lane.count_publication();
                Ok(snapshot)
            }
            Err(code) => {
                if matches!(code, Code::PublicationRefused | Code::PublicationUncertain | Code::TestingProcessDeath) {
                    self.fault = Some(code);
                }
                Err(code.into())
            }
        }
    }

    // -- testing builds -------------------------------------------------------------------------------------------

    #[cfg(feature = "testing")]
    pub fn testing_arm(&mut self, publication: u64, action: u32, arg: u64) -> Result<(), Error> {
        self.lane.with_stores(|st| st.continuity.testing_arm(publication, action, arg))??;
        Ok(())
    }

    #[cfg(feature = "testing")]
    pub fn testing_continuity_counters(&mut self, reset: bool) -> elpis_continuity::store::probe::Counters {
        self.lane.with_stores(|st| st.continuity.testing_counters(reset)).unwrap_or_default()
    }

    #[cfg(feature = "testing")]
    pub fn testing_checkpoint_counters(&mut self, reset: bool) -> checkpoint::probe::Counters {
        self.lane
            .with_stores(|st| match st.checkpoint.as_mut() {
                Some(c) => {
                    let counters = c.probe.counters;
                    if reset {
                        c.probe.counters = checkpoint::probe::Counters::default();
                    }
                    counters
                }
                None => checkpoint::probe::Counters::default(),
            })
            .unwrap_or_default()
    }

    /// Hold (or release) the durability worker before it takes a new job (testing builds: the HOT lane must not
    /// depend on it).
    #[cfg(any(test, feature = "testing"))]
    pub fn testing_durable_hold(&mut self, hold: bool) {
        self.lane.hold(hold);
    }

    #[cfg(feature = "testing")]
    pub fn testing_durable_plan(&mut self, plan: Option<crate::durable::Plan>) {
        self.lane.plan(plan);
    }
}

impl Drop for Core {
    /// Destruction ends an open managed turn exactly as `close` does: one native abort through the retained
    /// capability, never a forgotten transaction; then the durability worker stops (a pending request is
    /// abandoned, a running one completes) and the stores release themselves.
    fn drop(&mut self) {
        self.end_turn();
        self.unhold();
        self.lane.shutdown();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.lane.close();
    }
}
