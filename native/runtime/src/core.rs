//! RuntimeCore: the one state machine behind an open Elpis runtime.
//!
//! ```text
//!   closed --open (a new session)--> open --close (the session ends)--> closed
//!                                     |
//!                                     +-- fail-stop: every lineage- or authority-dependent operation is refused
//!                                         with that code until close + open, which starts a new session
//! ```
//!
//! ABI v8 (docs/RUNTIME_CORE.md, docs/ELPIS_MISSION.md, Amendment 3): one lane, volatile only. Every operation runs
//! on the caller's thread under the runtime handle's lock and performs no file I/O. The authority is a volatile ECS_C
//! session ([`crate::authority`]): opened at a fresh genesis, it holds the lineage's cognition identity, the
//! evolution authority and the sources typed evidence is admitted against, in memory, until the runtime closes.
//! There is no durability lane, no checkpoint, no publication, no recovery and no resumption: a closed runtime's
//! session is gone, and the next open starts another.
//!
//! Laws (each is a test in `tests.rs`):
//! * K1 lineage: the session's lineage is anchored explicitly at the state's retained identity (in memory); the open
//!   runtime binds its lineage to one native state, verified once against the session's identity (a mismatch
//!   fail-stops; a mismatched state is never claimed); another state is refused (`COGNITION_SUBSTRATE_SWITCH`)
//!   without a native call.
//! * The live coordinate: the bound state's native generation, under this runtime's managed lease, as last
//!   established by RuntimeCore itself. Every operation is checked against it. The session's identity is the
//!   anchored one; the live state advances beyond it by volatile commits, which the session never needs to name.
//! * QUERY: read-only. One native call answers `f_W(x)` and reports the generation of the state it answered from,
//!   under the lease; it must be the live generation (otherwise the lineage is lost: fail-stop, answer withheld).
//! * Managed turn (LEARN): begin (the source generation must be the live one) -> one native schedule on the
//!   candidate -> (the boundary decodes) -> native commit, which advances the live generation by exactly one. No
//!   I/O. Every refusal before the commit leaves the complete retained state byte-for-byte unchanged (the native
//!   transaction law).
//! * Turn lifecycle: once a native transaction is open, exactly one terminal native action ends it before
//!   RuntimeCore forgets the turn: the commit, or an abort. There is no third disposition. An explicit
//!   `turn_abort`, a refused commit, `close`, a reopen and destruction (`Drop`) all abort through the capability
//!   retained at begin ([`crate::substrate::TxnAbort`]); an abort refused BUSY (a concurrent overlapping call) is
//!   retried, never dropped. An implicit abort installs nothing.
//! * Evolution: one attempt at a time. Its assertion is reserved in the session (synchronously) before the boundary
//!   may execute it and finalized with the executed receipt; an attempt the boundary could not complete leaves the
//!   reservation pending and fail-stops (`CONTINUITY_EVOLUTION_PENDING`) for the rest of the session; execution is
//!   never retried or inferred.
//! * Typed evidence: an observation is admitted only from an ACTIVE source of the session within its per-turn bound,
//!   checked before the native call.
//! * Composite lineages (docs/ECS_COMPOSITE_STATE.md): a substrate of the ECS composite kind is bound, queried and
//!   learned exactly as a K1 state; the session records the lineage's format, and a substrate of another format never
//!   stands for it (`CONTINUITY_STATE_MISMATCH`). The memory changes only through `turn_observe` inside the managed
//!   turn. RuntimeCore exposes no memory LEARN or CONSOLIDATE (unqualified).
//! * Nothing is persistent and nothing is retained beyond the session: no history, receipts, events or trajectory.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::authority::{
    Admission, Authority, Cognition, Digest, EvolutionState, SessionTemplate, Snapshot, StateFormat, Status,
};
use crate::code::{Code, Error, Rt};
use crate::fuel::{self, Budget};
use crate::substrate::{
    features, observation_source, Experience, K1Ops, Key, Observation, ResidentRelease, ScheduleOutcome, Transition,
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

/// Native crossings and session transitions RuntimeCore performed (diagnostics; never authority). ABI v8 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub k1_state_digests: u64,
    pub k1_reserves: u64,
    pub k1_txn_begins: u64,
    pub k1_run_schedules: u64,
    pub k1_commits: u64,
    pub k1_aborts: u64,
    /// Accepted session transitions RuntimeCore sent (the anchor, evolution reservations and finalizations, and the
    /// operator's events).
    pub session_transitions: u64,
    pub k1_queries: u64,
    pub k1_shapes: u64,
    pub k1_lease_claims: u64,
    /// `leased_identity` crossings (anchor, binding, release, `live_identity`).
    pub k1_identities: u64,
    /// `leased_txn_observe` crossings: OBSERVE inside a managed turn, LEARN or observe-only.
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
    /// The committed native transition when the refusal came after the native commit (only a generation that does
    /// not continue the live coordinate, which fail-stops).
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
    /// The observations this turn recorded per source (source authority, count); bounded by `max_observations`.
    per_source: Vec<(u32, u32)>,
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
    /// What every session of this runtime is created with.
    template: SessionTemplate,
    /// Sessions opened so far (the ordinal of the open one, while open).
    sessions: u64,
    /// The open session (None while closed).
    authority: Option<Authority>,
    /// The session refusal of the last refused open (reported by `authority_status` while closed).
    refused: Option<i32>,
    /// This runtime's managed lease: claimed on every state it binds, presented on every K1 mutation.
    lease: u64,
    fault: Option<Code>,
    bound: Option<Bound>,
    /// The residency hold of the bound FMS-resident state (docs/RUNTIME_CORE.md, "HOT residency"): taken before a
    /// state is bound, dropped whenever it is unbound (release, close, reopen, destruction). None for RAM states.
    resident: Option<Box<dyn ResidentRelease>>,
    /// The live identity an explicit release gave back (format and retained-state identity, read under the lease
    /// just before releasing it). Rebinding accepts it besides the session's anchored identity, so volatile work
    /// survives an explicit release; it lives only as long as the session.
    released: Option<(StateFormat, Digest)>,
    /// The width read natively for one state (immutable for that state's lifetime); fuel admission needs it.
    shape: Option<(Key, usize)>,
    turn: Option<OpenTurn>,
    evolution: Option<EvolutionState>,
    counters: Counters,
    /// The typed-evidence admission table (the session's ACTIVE sources and their per-turn bounds), a copy refreshed
    /// at open and after every operator transition.
    admission: Admission,
}

impl Core {
    /// A closed runtime whose sessions are created from `template`. `RUNTIME_INVALID` for a capacity outside the
    /// session's bounds. No session exists until `open`.
    pub fn new(template: SessionTemplate) -> Result<Core, Error> {
        template.validate()?;
        Ok(Core {
            template,
            sessions: 0,
            authority: None,
            refused: None,
            lease: next_lease(),
            fault: None,
            bound: None,
            resident: None,
            released: None,
            shape: None,
            turn: None,
            evolution: None,
            counters: Counters::default(),
            admission: Admission::default(),
        })
    }

    // -- lifecycle ----------------------------------------------------------------------------------------

    /// Open: a new session at a fresh genesis; nothing anchored, nothing bound. An open runtime is refused
    /// (`CONTINUITY_OPEN`) and keeps everything, its open turn included. `RUNTIME_NOMEM` when the session's memory
    /// cannot be reserved (`RUNTIME_AUTHORITY` when no nonce can be drawn): the runtime stays closed.
    pub fn open(&mut self) -> Result<Snapshot, Error> {
        if self.authority.is_some() {
            return Err(Code::Open.into());
        }
        let authority = match Authority::create(&self.template, self.sessions + 1) {
            Ok(a) => a,
            Err((error, code)) => {
                self.refused = Some(code as i32);
                return Err(error);
            }
        };
        self.reset();
        self.sessions += 1;
        self.refused = None;
        self.admission = authority.admission();
        let snapshot = authority.snapshot();
        self.authority = Some(authority);
        Ok(snapshot)
    }

    /// Close: the session ends. An open managed turn is aborted natively first (nothing installed); the bound
    /// state's lease and residency hold are given back by its owner's release or destruction, never written anywhere.
    pub fn close(&mut self) {
        self.reset();
        self.authority = None;
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
        self.admission = Admission::default();
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
        self.authority.is_some()
    }

    /// The fail-stop disposition, if any.
    pub fn fault(&self) -> Option<Code> {
        self.fault
    }

    fn live(&self) -> Result<(), Error> {
        if self.authority.is_none() {
            return Err(Rt::Closed.into());
        }
        match self.fault {
            Some(code) => Err(code.into()),
            None => Ok(()),
        }
    }

    fn authority(&self) -> Result<&Authority, Error> {
        self.authority.as_ref().ok_or(Error::Runtime(Rt::Closed))
    }

    fn authority_mut(&mut self) -> Result<&mut Authority, Error> {
        self.authority.as_mut().ok_or(Error::Runtime(Rt::Closed))
    }

    /// The session's view (readable while fail-stopped).
    pub fn snapshot(&self) -> Result<Snapshot, Error> {
        Ok(self.authority()?.snapshot())
    }

    pub fn counters(&mut self, reset: bool) -> Counters {
        let c = self.counters;
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

    /// The session's anchored identity of the lineage.
    fn expected(&self) -> Result<Digest, Refusal> {
        match self.authority()?.snapshot().cognition() {
            Cognition::Unanchored => Err(Code::Unanchored.into()),
            Cognition::Anchored(d) => Ok(d),
        }
    }

    /// Explicitly anchor the session's lineage at the state's retained identity (reads the state only). The state is
    /// claimed, its identity and generation read in one guarded call, and the session records the anchor (in
    /// memory); the state is bound at that generation.
    pub fn anchor(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<Snapshot, Refusal> {
        self.live()?;
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        if self.authority()?.snapshot().cognition() != Cognition::Unanchored {
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
        let format = sub.format();
        let snapshot = self.authority_mut()?.anchor_cognition_as(format, &identity)?;
        self.counters.session_transitions += 1;
        self.released = None;
        self.set_live(key, generation);
        Ok(snapshot)
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
    /// The state must be the session's anchored identity or the live identity an explicit release of this session
    /// gave back (its volatile work).
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
        let session = self.authority()?.snapshot();
        let anchored = match session.cognition() {
            Cognition::Unanchored => return Err(Code::Unanchored.into()),
            Cognition::Anchored(d) => d,
        };
        // An identity of one format never stands for a state of another (a K1 state for a composite lineage).
        let (format, expected) = (sub.format(), self.digest(sub)?);
        if (format, expected) != (session.format(), anchored) && Some((format, expected)) != self.released {
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
    /// back resumes the live lineage: volatile work is not lost by a release. Rebinding the anchored state instead
    /// abandons that work.
    pub fn release(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<(), Refusal> {
        if self.authority.is_none() {
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
        // provably the lineage, and rebinding falls back to the anchored identity.
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
    /// transaction is opened, nothing is committed or written, no digest is computed, and the session is not
    /// consulted (except, once, to bind). `out` receives the answer (one value per row; zeroed on
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
        self.turn = Some(OpenTurn {
            key,
            token,
            abort,
            observe_only: false,
            observations: 0,
            max_observations,
            per_source: Vec::new(),
        });
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
        self.turn = Some(OpenTurn {
            key,
            token,
            abort,
            observe_only: true,
            observations: 0,
            max_observations,
            per_source: Vec::new(),
        });
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
    /// advances by exactly one; nothing is written anywhere. A fail-stopped runtime, or a refused commit, aborts the
    /// turn instead: either way the turn has ended when this returns, unless it was refused before the turn was taken
    /// (`RUNTIME_CLOSED`, `RUNTIME_TURN_NOT_OPEN`).
    pub fn turn_commit(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<VolatileCommit, Refusal> {
        if self.authority.is_none() {
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
    /// the turn's memory candidate, which becomes live only with the turn's commit (in memory). A refusal is recoverable: the turn stays open and its candidate unchanged. A K1
    /// substrate has no memory (`RUNTIME_INVALID`). Returns the relation key; the second value is the memory's own
    /// status behind a composite refusal (0 otherwise).
    pub fn turn_observe(
        &mut self,
        sub: &mut dyn K1Ops,
        key: Key,
        observation: Observation,
    ) -> (Result<Digest, Refusal>, i32) {
        if self.authority.is_none() {
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
        // The observation's source must be a present ACTIVE source of the session within its per-turn bound, before
        // the native call (recoverable; the candidate is unchanged).
        let Some(source) = observation_source(&observation) else {
            return (Err(Rt::Invalid.into()), 0);
        };
        let Some(bound) = self.admission.bound(source) else {
            return (Err(Rt::ObservationRefused.into()), 0);
        };
        {
            let turn = self.turn.as_ref().expect("checked above");
            let used = turn.per_source.iter().find(|(s, _)| *s == source).map(|(_, n)| *n).unwrap_or(0);
            if used >= bound {
                return (Err(Rt::ObservationRefused.into()), 0);
            }
        }
        let (mut relation, mut memory_status) = ([0u8; 32], 0i32);
        self.counters.k1_observes += 1;
        match sub.txn_observe(self.lease, token, observation, &mut relation, &mut memory_status) {
            0 => {
                if let Some(turn) = self.turn.as_mut() {
                    turn.observations += 1;
                    match turn.per_source.iter_mut().find(|(s, _)| *s == source) {
                        Some((_, n)) => *n += 1,
                        None => turn.per_source.push((source, 1)),
                    }
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

    // -- the session authority (ABI v8) ------------------------------------------------------------------------

    /// The session's status; while closed, the number of sessions opened and the session refusal of the last refused
    /// open, if any.
    pub fn authority_status(&self) -> Status {
        match &self.authority {
            Some(a) => a.status(self.sessions),
            None => Status {
                backend: crate::authority::BACKEND_ECS_C_SESSION,
                ecs_c_code: self.refused.unwrap_or(0),
                sessions: self.sessions,
                ..Status::default()
            },
        }
    }

    /// An operator event on the open session (founding, lifecycle, source versions, reclamation;
    /// docs/ECS_C_SESSION.md). Refused while fail-stopped, while a managed turn is open (its admission must not
    /// change under it), for RuntimeCore's own kinds (`RUNTIME_INVALID`) and, as retired, for the persistence-only
    /// kinds (`RUNTIME_RETIRED`). The admission table is refreshed from the session afterwards.
    pub fn authority_transition(
        &mut self,
        event: &elpis_ecs_c_session::model::Event,
    ) -> Result<elpis_ecs_c_session::SessionReceipt, Refusal> {
        self.live()?;
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        let authority = self.authority_mut()?;
        let result = authority.operator_transition(event);
        let admission = authority.admission();
        self.admission = admission;
        if result.is_ok() {
            self.counters.session_transitions += 1;
        }
        Ok(result?)
    }

    /// The session's entity row at `index` (read-only; what an operator event is built from). Refused closed and
    /// fail-stopped.
    pub fn authority_entity(&self, index: u32) -> Result<elpis_ecs_c_session::model::Entity, Error> {
        self.live()?;
        self.authority()?.entity_at(index)
    }

    pub fn authority_next_entity_id(&self, kind: u8, label: &Digest) -> Result<Digest, Error> {
        self.live()?;
        self.authority()?.next_entity_id(kind, label)
    }

    // -- evolution: the reservation precedes execution ------------------------------------------------------------

    /// The current idle evolution authority an assertion must be built against.
    pub fn evolution_authority(&self) -> Result<Snapshot, Error> {
        self.live()?;
        let snapshot = self.authority()?.snapshot();
        if snapshot.evolution().assertion().is_some() {
            return Err(Code::EvolutionPending.into());
        }
        Ok(snapshot)
    }

    /// Reserve the exact assertion in the session against the authority the caller validated it with. Only after
    /// this returns may the boundary execute the attempt, once. A refused reservation changes nothing and executed
    /// nothing (no fail-stop): the caller may read the authority again.
    pub fn evolution_reserve(&mut self, observed: &EvolutionState, assertion: &Digest) -> Result<Snapshot, Error> {
        self.live()?;
        if self.evolution.is_some() {
            return Err(Rt::EvolutionInFlight.into());
        }
        if self.authority()?.snapshot().evolution().assertion().is_some() {
            return Err(Code::EvolutionPending.into());
        }
        let snapshot = self.authority_mut()?.reserve_evolution(observed, assertion)?;
        self.counters.session_transitions += 1;
        self.evolution = Some(snapshot.evolution());
        Ok(snapshot)
    }

    /// Finalize the attempt in flight with the receipt its execution produced. A refused finalization fail-stops for
    /// the rest of the session (the attempt executed; it is never re-executed).
    pub fn evolution_finalize(&mut self, receipt: &Digest) -> Result<Snapshot, Error> {
        if self.authority.is_none() {
            return Err(Rt::Closed.into());
        }
        let pending = self.evolution.take().ok_or(Rt::EvolutionNotInFlight)?;
        if let Some(code) = self.fault {
            return Err(code.into());
        }
        match self.authority_mut()?.finalize_evolution(&pending, receipt) {
            Ok(snapshot) => {
                self.counters.session_transitions += 1;
                Ok(snapshot)
            }
            Err(code) => {
                self.fault = Some(code);
                Err(code.into())
            }
        }
    }

    /// The boundary could not complete the attempt in flight (an exception or an invalid result may follow an
    /// external effect): the reservation stays pending and the runtime fail-stops for the rest of the session.
    pub fn evolution_abandon(&mut self) -> Result<(), Error> {
        if self.evolution.take().is_none() {
            return Err(Rt::EvolutionNotInFlight.into());
        }
        self.fault.get_or_insert(Code::EvolutionPending);
        Ok(())
    }
}

impl Drop for Core {
    /// Destruction ends an open managed turn exactly as `close` does (one native abort through the retained
    /// capability, never a forgotten transaction), drops the residency hold, and the session's memory is released.
    fn drop(&mut self) {
        self.end_turn();
        self.unhold();
    }
}
