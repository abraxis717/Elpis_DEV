//! RuntimeCore: the one state machine behind an open Elpis runtime.
//!
//! ```text
//!   closed --open--> open --close--> closed
//!                     |
//!                     +-- fail-stop (a continuity code): every lineage- or authority-dependent operation is
//!                         refused with that code until close + open (restart reconciles against continuity)
//! ```
//!
//! Laws (each is a test in `tests.rs`):
//! * K1 lineage: the first managed lineage is anchored explicitly at the state's retained identity; an open
//!   runtime binds its lineage to one native state, verified once against the durable expected identity
//!   (a mismatch fail-stops); another state is refused (`COGNITION_SUBSTRATE_SWITCH`) without a native call.
//! * QUERY: read-only. One native call answers `f_W(x)` and identifies the state it answered from; the identity
//!   must be the durable expected one (a mismatch fail-stops and the answer is withheld). No transaction, no
//!   commit, no publication: the retained state and continuity are unchanged.
//! * Managed turn (LEARN): begin -> one native schedule on the candidate -> (the boundary decodes) -> native commit ->
//!   one continuity publication of the committed identity. A K1 commit is final and never rolled back: a
//!   publication that is refused or uncertain fail-stops the runtime; restart sees either the old authority
//!   (the moved state is then a mismatch) or the new one. Every refusal before the commit leaves the complete
//!   retained state byte-for-byte unchanged (the native transaction law).
//! * Turn lifecycle: once a native transaction is open, exactly one terminal native action ends it before
//!   RuntimeCore forgets the turn: the commit, or an abort. There is no third disposition. An explicit
//!   `turn_abort`, a refused commit, `close`, a reopen and destruction (`Drop`) all abort through the capability
//!   retained at begin ([`crate::substrate::TxnAbort`]); an abort refused BUSY (a concurrent overlapping call) is
//!   retried, never dropped. An implicit abort installs nothing and publishes nothing: the authoritative K1 state
//!   (W, epoch, generation, H, a), its retained-state digest and continuity are unchanged.
//! * Evolution: one attempt at a time. Its assertion is durably reserved before the boundary may execute it
//!   and finalized with the executed receipt; a failed reservation or finalization fail-stops, and an attempt
//!   the boundary could not complete leaves the durable reservation pending (fail-stop
//!   `CONTINUITY_EVOLUTION_PENDING`); execution is never retried or inferred.
//! * Nothing else is durable or retained: no history, receipts, events or trajectory.

use std::path::Path;

use elpis_continuity::{Code, Cognition, Digest, EvolutionState, Snapshot, Store};

use crate::code::{Error, Rt};
use crate::fuel::{self, Budget};
use crate::substrate::{
    features, CommitIdentity, Experience, K1Ops, Key, ScheduleOutcome, TxnAbort, K1_BUSY, K1_CAPACITY, K1_FMS_CAPACITY,
    K1_INVALID, K1_NONFINITE, K1_STALE, MAX_EXPERIENCES,
};

/// Native crossings and publications RuntimeCore performed (diagnostics; never authority).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub k1_state_digests: u64,
    pub k1_reserves: u64,
    pub k1_txn_begins: u64,
    pub k1_run_schedules: u64,
    pub k1_commits: u64,
    pub k1_aborts: u64,
    pub publications: u64,
    pub k1_queries: u64,
    pub k1_shapes: u64,
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
    /// The committed K1 transition when the refusal came after the native commit (publication failure).
    pub committed: Option<Box<CommitIdentity>>,
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
/// ends that transaction (abort only). It exists exactly while the native transaction is open.
struct OpenTurn {
    key: Key,
    token: u64,
    abort: Box<dyn TxnAbort>,
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
    store: Store,
    open: bool,
    fault: Option<Code>,
    bound: Option<Key>,
    /// The width read natively for one state (immutable for that state's lifetime); fuel admission needs it.
    shape: Option<(Key, usize)>,
    turn: Option<OpenTurn>,
    evolution: Option<EvolutionState>,
    counters: Counters,
}

impl Core {
    /// A closed runtime over one continuity directory (an absolute path).
    pub fn new(continuity_dir: &Path) -> Result<Core, Error> {
        Ok(Core {
            store: Store::new(continuity_dir)?,
            open: false,
            fault: None,
            bound: None,
            shape: None,
            turn: None,
            evolution: None,
            counters: Counters::default(),
        })
    }

    // -- lifecycle ----------------------------------------------------------------------------------------

    /// Open (or reopen after close): the store resolves the current authority; nothing is bound. An open runtime
    /// is refused (`CONTINUITY_OPEN`) and keeps everything, its open turn included.
    pub fn open(&mut self) -> Result<Snapshot, Error> {
        let snapshot = self.store.open()?;
        self.reset();
        self.open = true;
        Ok(snapshot)
    }

    /// Close. An open managed turn is aborted natively first (nothing installed, nothing published).
    pub fn close(&mut self) {
        self.reset();
        self.store.close();
        self.open = false;
    }

    /// Forget the open runtime's bindings. Never forgets an open native transaction: it ends it first.
    fn reset(&mut self) {
        self.end_turn();
        self.fault = None;
        self.bound = None;
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

    /// The fail-stop disposition, if any.
    pub fn fault(&self) -> Option<Code> {
        self.fault
    }

    fn live(&self) -> Result<(), Error> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        match self.fault {
            Some(code) => Err(code.into()),
            None => Ok(()),
        }
    }

    /// The current durable authority (readable while fail-stopped: it is what restart will see).
    pub fn snapshot(&self) -> Result<Snapshot, Error> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        Ok(self.store.snapshot()?)
    }

    pub fn counters(&mut self, reset: bool) -> Counters {
        let c = self.counters;
        if reset {
            self.counters = Counters::default();
        }
        c
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

    /// Explicitly anchor the first managed K1 lineage at the state's retained identity (reads K1 only).
    pub fn anchor(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<Snapshot, Refusal> {
        self.live()?;
        if self.store.snapshot()?.cognition() != Cognition::Unanchored {
            return Err(Code::AlreadyAnchored.into());
        }
        let identity = self.digest(sub)?;
        match self.store.anchor_cognition(Some(&identity)) {
            Ok(s) => {
                self.counters.publications += 1;
                self.bound = Some(key);
                Ok(s)
            }
            Err(code) => {
                if matches!(code, Code::PublicationUncertain | Code::TestingProcessDeath) {
                    self.fault = Some(code);
                }
                Err(code.into())
            }
        }
    }

    /// Bind the lineage to this state, or verify it is the bound one.
    fn reconcile(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<(), Refusal> {
        if let Some(bound) = self.bound {
            return if bound == key { Ok(()) } else { Err(Rt::SubstrateSwitch.into()) };
        }
        let expected = match self.store.snapshot()?.cognition() {
            Cognition::Unanchored => return Err(Code::Unanchored.into()),
            Cognition::Anchored(d) => d,
        };
        if self.digest(sub)? != expected {
            self.fault = Some(Code::StateMismatch);
            return Err(Code::StateMismatch.into());
        }
        self.bound = Some(key);
        Ok(())
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

    // -- the canonical read-only QUERY ---------------------------------------------------------------------------

    /// QUERY: `f_W(x)` from the lineage's authoritative K1 state, read-only (docs/COGNITION_R0.md).
    ///
    /// One native crossing answers and identifies in one guarded K1 call (`query_identity`), so the answer and the
    /// identity it is checked against come from the same bytes. The identity must equal the durable expected
    /// identity: an unanchored lineage is refused (`CONTINUITY_UNANCHORED`); a state this lineage never committed
    /// (another state, or the bound one moved out of band) fail-stops (`CONTINUITY_STATE_MISMATCH`) and its answer
    /// is never returned. No transaction is opened, nothing is committed or published, and nothing durable or
    /// retained changes: W, epoch, H, a, continuity. `out` receives the answer (one value per row; zeroed on any
    /// refusal). Returns the retained-state identity the answer was computed from.
    pub fn query(
        &mut self,
        sub: &mut dyn K1Ops,
        key: Key,
        x: &[f64],
        out: &mut [f64],
        budget: &Budget,
    ) -> Result<Digest, Refusal> {
        self.live()?;
        if self.turn.is_some() {
            return Err(Rt::TurnOpen.into());
        }
        let rows = out.len();
        if rows == 0 || rows.checked_mul(key.dim) != Some(x.len()) {
            return Err(Rt::Invalid.into());
        }
        if matches!(self.bound, Some(bound) if bound != key) {
            return Err(Rt::SubstrateSwitch.into());
        }
        let expected = match self.store.snapshot()?.cognition() {
            Cognition::Unanchored => return Err(Code::Unanchored.into()),
            Cognition::Anchored(d) => d,
        };
        let width = self.width(sub, key)?;
        fuel::admit_query(budget, key.dim, width, rows).map_err(|_| Refusal::from(Rt::Fuel))?;
        let mut identity = [0u8; 32];
        self.counters.k1_queries += 1;
        let rc = sub.query_identity(key.dim, x, out, &mut identity);
        if rc != 0 {
            out.fill(0.0);
            return Err(ecs(Rt::EcsRefused, rc, ScheduleOutcome::default()));
        }
        if identity != expected {
            out.fill(0.0);
            self.fault = Some(Code::StateMismatch);
            return Err(Code::StateMismatch.into());
        }
        self.bound = Some(key);
        Ok(identity)
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
        if matches!(self.bound, Some(bound) if bound != key) {
            return Err(Rt::SubstrateSwitch.into());
        }
        if self.store.snapshot()?.cognition() == Cognition::Unanchored {
            return Err(Code::Unanchored.into());
        }
        // Total fuel, before the identity check, any reserve, the transaction or a candidate mutation.
        let width = self.width(sub, key)?;
        fuel::admit_schedule(budget, key.dim, width, schedule).map_err(|_| Refusal::from(Rt::Fuel))?;
        self.reconcile(sub, key)?;
        let abort = sub.retain_abort();
        let (token, result) = self.run_schedule(sub, stimulus, s3)?;
        self.turn = Some(OpenTurn { key, token, abort });
        Ok(result)
    }

    fn begin(&mut self, sub: &mut dyn K1Ops) -> Result<u64, Refusal> {
        let mut token = 0u64;
        self.counters.k1_txn_begins += 1;
        match sub.txn_begin(&mut token) {
            0 => Ok(token),
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
        let rc = sub.txn_run_schedule(token, stimulus.x, stimulus.y, stimulus.schedule, stimulus.rate, s3, &mut result);
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
            let reserved = sub.reserve(rows);
            if reserved != 0 {
                return Err(ecs(Rt::EcsRefused, reserved, ScheduleOutcome::default()));
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

    /// Commit the open turn natively, then publish the committed identity to continuity. A publication that
    /// fails fail-stops the runtime; the K1 commit stands (the refusal carries it). A fail-stopped runtime, or a
    /// refused commit, aborts the turn instead: either way the turn has ended when this returns, unless it was
    /// refused before the turn was taken (`RUNTIME_CLOSED`, `RUNTIME_TURN_NOT_OPEN`).
    pub fn turn_commit(&mut self, sub: &mut dyn K1Ops, key: Key) -> Result<(CommitIdentity, Snapshot), Refusal> {
        if !self.open {
            return Err(Rt::Closed.into());
        }
        let turn = self.take_turn(key)?;
        if let Some(code) = self.fault {
            self.terminate(turn);
            return Err(code.into());
        }
        let mut identity = CommitIdentity::default();
        self.counters.k1_commits += 1;
        let rc = sub.txn_commit_identity(turn.token, &mut identity);
        if rc != 0 {
            // STALE discards the transaction natively; any other refusal leaves it open.
            if rc != K1_STALE {
                self.terminate(turn);
            }
            let error = if rc == K1_STALE { Rt::EcsStale } else { Rt::EcsRefused };
            return Err(ecs(error, rc, ScheduleOutcome::default()));
        }
        match self.store.commit_cognition(Some(&identity.state_before_digest), Some(&identity.state_after_digest)) {
            Ok(snapshot) => {
                self.counters.publications += 1;
                Ok((identity, snapshot))
            }
            Err(code) => {
                self.fault = Some(code);
                Err(Refusal { committed: Some(Box::new(identity)), ..Refusal::from(code) })
            }
        }
    }

    /// Abort the open turn of this substrate: nothing is installed. Another substrate's key is refused
    /// (`RUNTIME_TURN_NOT_OPEN`) and leaves the turn open.
    pub fn turn_abort(&mut self, key: Key) -> Result<(), Refusal> {
        let turn = self.take_turn(key)?;
        self.terminate(turn);
        Ok(())
    }

    // -- evolution ------------------------------------------------------------------------------------------------

    /// The current idle evolution authority an assertion must be built against.
    pub fn evolution_authority(&self) -> Result<Snapshot, Error> {
        self.live()?;
        let snapshot = self.store.snapshot()?;
        if snapshot.evolution().assertion().is_some() {
            return Err(Code::EvolutionPending.into());
        }
        Ok(snapshot)
    }

    /// Durably reserve the exact assertion against the authority the caller validated it with. Only after
    /// this returns may the boundary execute the attempt, once.
    pub fn evolution_reserve(&mut self, observed: &EvolutionState, assertion: &Digest) -> Result<Snapshot, Error> {
        self.live()?;
        if self.evolution.is_some() {
            return Err(Rt::EvolutionInFlight.into());
        }
        if self.store.snapshot()?.evolution().assertion().is_some() {
            return Err(Code::EvolutionPending.into());
        }
        match self.store.reserve_evolution(Some(observed), Some(assertion)) {
            Ok(snapshot) => {
                self.counters.publications += 1;
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
        if let Some(code) = self.fault {
            return Err(code.into());
        }
        match self.store.finalize_evolution(Some(&pending), Some(receipt)) {
            Ok(snapshot) => {
                self.counters.publications += 1;
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
        match self.store.finalize_evolution(Some(expected), Some(receipt)) {
            Ok(snapshot) => {
                self.counters.publications += 1;
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
        Ok(self.store.testing_arm(publication, action, arg)?)
    }

    #[cfg(feature = "testing")]
    pub fn testing_continuity_counters(&mut self, reset: bool) -> elpis_continuity::store::probe::Counters {
        self.store.testing_counters(reset)
    }
}

impl Drop for Core {
    /// Destruction ends an open managed turn exactly as `close` does: one native abort through the retained
    /// capability, never a forgotten transaction. The embedded store releases itself.
    fn drop(&mut self) {
        self.end_turn();
    }
}
