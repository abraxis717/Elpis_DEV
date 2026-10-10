//! RuntimeCore laws against a deterministic in-memory stand-in for one native K1 state.
//!
//! The stand-in follows the refusal contract of `ecsg_k1.h` (CAPACITY/INVALID keep the transaction open,
//! NONFINITE from the schedule and STALE discard it, a commit installs the complete candidate or nothing) but
//! computes nothing of K1's mathematics; the real native state is exercised by the Python integration lane
//! (tests/integration). Fault injection needs the `testing` feature (`cargo test --features testing`).

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use elpis_continuity::{Code, Cognition, EvolutionState};

use crate::code::{Error, Rt};
use crate::core::{Core, Counters, Stimulus};
use crate::fuel::{self, Budget, CEILING};
use crate::substrate::{
    features, CommitIdentity, Digest, Experience, K1Ops, Key, ScheduleOutcome, TxnAbort, K1_BUSY, K1_CAPACITY,
    K1_LEASED, K1_NONFINITE, K1_STALE,
};

const DIM: usize = 2;
/// The stand-in's declared width (fuel admission reads it natively once per state).
const WIDTH: usize = 3;

struct Txn {
    token: u64,
    lease: u64,
    generation: u64,
    w: Vec<f64>,
    epoch: u64,
}

/// One retained "state": W, epoch and a generation that every commit advances.
struct State {
    w: Vec<f64>,
    epoch: u64,
    generation: u64,
    max_rows: usize,
    txn: Option<Txn>,
    /// The managed lease (0: unmanaged), as ecsg_k1.h: while set, only that lease's calls mutate.
    lease: u64,
    next_token: u64,
    digest_fails: bool,
    /// The next aborts refused BUSY (a concurrent overlapping call that did nothing).
    busy_aborts: u32,
    /// A recoverable commit refusal that leaves the transaction open (0: none).
    refuse_commit: i32,
    calls: Vec<&'static str>,
}

impl State {
    fn identity(&self) -> Digest {
        // FNV-1a over the retained values, widened to 32 bytes: an identity, not a cryptographic digest.
        let mut out = [0u8; 32];
        let mut h: u64 = 0xcbf29ce484222325;
        for v in self.w.iter().map(|v| v.to_bits()).chain([self.epoch]) {
            for b in v.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        for (i, chunk) in out.chunks_mut(8).enumerate() {
            chunk.copy_from_slice(&(h.rotate_left(i as u32 * 7) ^ i as u64).to_le_bytes());
        }
        out
    }

    fn abort(&mut self, token: u64) -> i32 {
        self.calls.push("txn_abort");
        if self.busy_aborts > 0 {
            self.busy_aborts -= 1;
            return K1_BUSY;
        }
        match &self.txn {
            Some(t) if t.token == token => {
                self.txn = None;
                0
            }
            Some(_) => -1,
            None => 0,
        }
    }
}

/// The caller's handle on one stand-in state; RuntimeCore's retained abort capability shares the same state, as
/// a retained native handle names the same native state.
struct Fake(Arc<Mutex<State>>);

impl Fake {
    fn new(seed: f64) -> Fake {
        Fake(Arc::new(Mutex::new(State {
            w: vec![seed, -seed, 0.5],
            epoch: 0,
            generation: 0,
            max_rows: 8,
            txn: None,
            lease: 0,
            next_token: 1,
            digest_fails: false,
            busy_aborts: 0,
            refuse_commit: 0,
            calls: Vec::new(),
        })))
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.0.lock().unwrap()
    }

    fn identity(&self) -> Digest {
        self.state().identity()
    }

    fn calls(&self) -> Vec<&'static str> {
        self.state().calls.clone()
    }

    fn clear_calls(&self) {
        self.state().calls.clear();
    }

    fn txn_open(&self) -> bool {
        self.state().txn.is_some()
    }

    fn epoch(&self) -> u64 {
        self.state().epoch
    }

    fn generation(&self) -> u64 {
        self.state().generation
    }

    fn max_rows(&self) -> usize {
        self.state().max_rows
    }

    /// Another owner claims the state (its lease is replaced).
    fn steal(&self) {
        self.state().lease = 0xdead;
    }

    fn lease(&self) -> u64 {
        self.state().lease
    }

    /// An unmanaged caller attempts a direct transition: refused while the state is leased (ecsg_k1.h).
    fn unmanaged_learn(&self) -> i32 {
        let mut s = self.state();
        if s.lease != 0 {
            return K1_LEASED;
        }
        s.w[0] += 1.0;
        s.epoch += 1;
        s.generation += 1;
        0
    }

    /// Someone commits a transition that bypasses every guard (a state moved while nobody could see it, e.g.
    /// before the claim): the identity check must catch it.
    fn interlope(&self) {
        let mut s = self.state();
        s.w[0] += 1.0;
        s.epoch += 1;
        s.generation += 1;
    }

    fn retained(&self) -> (Vec<u64>, u64) {
        let s = self.state();
        (s.w.iter().map(|v| v.to_bits()).collect(), s.epoch)
    }
}

struct FakeAbort(Arc<Mutex<State>>);

impl TxnAbort for FakeAbort {
    fn txn_abort(&mut self, token: u64) -> i32 {
        self.0.lock().unwrap().abort(token)
    }
}

impl K1Ops for Fake {
    fn state_digest(&mut self, out: &mut Digest) -> i32 {
        let mut s = self.state();
        s.calls.push("state_digest");
        if s.digest_fails {
            return -1;
        }
        *out = s.identity();
        0
    }

    fn reserve(&mut self, lease: u64, max_rows: usize) -> i32 {
        let mut s = self.state();
        s.calls.push("reserve");
        if s.lease != lease {
            return K1_LEASED;
        }
        if s.txn.is_some() {
            return K1_BUSY;
        }
        s.max_rows = s.max_rows.max(max_rows);
        0
    }

    fn txn_begin(&mut self, lease: u64, source: &mut Digest, token: &mut u64) -> i32 {
        let mut s = self.state();
        s.calls.push("txn_begin");
        if s.lease != lease {
            return K1_LEASED;
        }
        if s.txn.is_some() {
            return K1_BUSY;
        }
        *token = s.next_token;
        *source = s.identity();
        s.next_token += 1;
        let txn = Txn { token: *token, lease, generation: s.generation, w: s.w.clone(), epoch: s.epoch };
        s.txn = Some(txn);
        0
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
        let mut s = self.state();
        s.calls.push("txn_run_schedule");
        let max_rows = s.max_rows;
        let txn = match s.txn.as_mut() {
            Some(t) if t.lease != lease => return K1_LEASED,
            Some(t) if t.token == token => t,
            _ => return -1,
        };
        if s3.len() != features(DIM) || schedule.iter().map(|e| e.rows).sum::<u64>() != y.len() as u64 {
            return -1;
        }
        if schedule.iter().any(|e| e.rows as usize > max_rows) {
            return K1_CAPACITY;
        }
        if x.iter().chain(y).any(|v| !v.is_finite()) {
            s.txn = None;
            return K1_NONFINITE;
        }
        result.epoch_before = txn.epoch;
        for e in schedule {
            txn.epoch += e.steps;
        }
        txn.w[0] += rate * x.iter().sum::<f64>();
        txn.w[2] += rate * y.iter().sum::<f64>();
        result.epoch_after = txn.epoch;
        result.experiences_applied = schedule.len() as u64;
        s3.fill(txn.w[0]);
        0
    }

    fn txn_commit_identity(&mut self, lease: u64, token: u64, out: &mut CommitIdentity) -> i32 {
        let mut s = self.state();
        s.calls.push("txn_commit_identity");
        if matches!(&s.txn, Some(t) if t.lease != lease) {
            return K1_LEASED;
        }
        if s.refuse_commit != 0 {
            return std::mem::take(&mut s.refuse_commit);
        }
        let txn = match s.txn.take() {
            Some(t) if t.token == token => t,
            other => {
                s.txn = other;
                return -1;
            }
        };
        if txn.generation != s.generation {
            return K1_STALE;
        }
        out.state_before_digest = s.identity();
        out.transition.epoch_before = s.epoch;
        out.transition.generation_before = s.generation;
        s.w = txn.w;
        s.epoch = txn.epoch;
        s.generation += 1;
        out.transition.epoch_after = s.epoch;
        out.transition.generation_after = s.generation;
        out.state_after_digest = s.identity();
        0
    }

    fn txn_abort(&mut self, token: u64) -> i32 {
        self.state().abort(token)
    }

    fn query_identity(&mut self, dim: usize, x: &[f64], out: &mut [f64], digest: &mut Digest) -> i32 {
        let mut s = self.state();
        s.calls.push("query_identity");
        if dim != DIM || x.len() != out.len() * DIM {
            return -1;
        }
        if x.iter().any(|v| !v.is_finite()) {
            return K1_NONFINITE;
        }
        // Reads the retained values only (a stand-in answer, not K1's forward map).
        for (r, o) in out.iter_mut().enumerate() {
            *o = s.w[0] * x[r * DIM] + s.w[1] * x[r * DIM + 1] + s.w[2];
        }
        *digest = s.identity();
        0
    }

    fn lease_claim(&mut self, lease: u64) -> i32 {
        let mut s = self.state();
        s.calls.push("lease_claim");
        if s.txn.is_some() {
            return K1_BUSY;
        }
        s.lease = lease;
        0
    }

    fn lease_release(&mut self, lease: u64) -> i32 {
        let mut s = self.state();
        s.calls.push("lease_release");
        if s.lease != lease {
            return K1_LEASED;
        }
        if s.txn.is_some() {
            return K1_BUSY;
        }
        s.lease = 0;
        0
    }

    fn shape(&mut self, dim: &mut usize, width: &mut usize) -> i32 {
        let mut s = self.state();
        s.calls.push("shape");
        *dim = DIM;
        *width = WIDTH;
        0
    }

    fn retain_abort(&self) -> Box<dyn TxnAbort> {
        Box::new(FakeAbort(Arc::clone(&self.0)))
    }
}

fn query(core: &mut Core, sub: &mut Fake, k: Key, rows: usize) -> Result<(Vec<f64>, Digest), Error> {
    let x: Vec<f64> = (0..rows * DIM).map(|i| 0.25 * (i as f64 + 1.0)).collect();
    let mut out = vec![f64::NAN; rows];
    let digest = core.query(sub, k, &x, &mut out, &CEILING).map_err(|r| r.error)?;
    Ok((out, digest))
}

fn key(owner: u64) -> Key {
    Key { kind: 1, handle: 0x1000 + owner as usize, id: 0, owner, dim: DIM }
}

struct Input {
    x: Vec<f64>,
    y: Vec<f64>,
    schedule: Vec<Experience>,
}

fn input(rows: u64, steps: u64) -> Input {
    let n = rows as usize;
    Input {
        x: (0..n * DIM).map(|i| 0.01 * (i as f64 + 1.0)).collect(),
        y: (0..n).map(|i| 0.1 * i as f64).collect(),
        schedule: vec![Experience { rows, steps }; 1],
    }
}

fn stimulus(i: &Input) -> Stimulus<'_> {
    Stimulus { x: &i.x, y: &i.y, schedule: &i.schedule, rate: 0.01 }
}

fn turn(core: &mut Core, sub: &mut Fake, k: Key) -> Result<CommitIdentity, Error> {
    let i = input(4, 3);
    let mut s3 = vec![0.0; features(DIM)];
    core.turn_begin(sub, k, &stimulus(&i), &CEILING, &mut s3).map_err(|r| r.error)?;
    core.turn_commit(sub, k).map(|(id, _)| id).map_err(|r| r.error)
}

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("elpis-runtime-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&p);
        Dir(p)
    }

    fn core(&self) -> Core {
        let mut c = Core::new(&self.0).expect("absolute");
        c.open().expect("open");
        c
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn anchored(core: &Core) -> Option<Digest> {
    match core.snapshot().unwrap().cognition() {
        Cognition::Unanchored => None,
        Cognition::Anchored(d) => Some(d),
    }
}

#[test]
fn features_is_the_k1_readout_length() {
    assert_eq!(
        (features(0), features(1), features(2), features(6), features(64), features(65)),
        (0, 3, 9, 83, 47904, 0)
    );
}

#[test]
fn a_closed_runtime_refuses_everything() {
    let dir = Dir::new("closed");
    let mut core = Core::new(&dir.0).unwrap();
    let mut sub = Fake::new(0.1);
    assert_eq!(core.anchor(&mut sub, key(1)).unwrap_err().error, Rt::Closed.into());
    assert_eq!(core.evolution_authority().unwrap_err(), Rt::Closed.into());
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Rt::Closed.into());
    assert!(sub.calls().is_empty());
    assert!(Core::new(std::path::Path::new("relative")).is_err());
}

#[test]
fn anchor_is_explicit_once_reads_k1_only_and_binds() {
    let dir = Dir::new("anchor");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let before = sub.retained();
    let initial = core.snapshot().unwrap().generation();
    let s = core.anchor(&mut sub, key(1)).unwrap();
    assert_eq!(s.generation(), initial + 1);
    assert_eq!(anchored(&core), Some(sub.identity()));
    // Reads K1 only; then claims managed ownership (the lease) of the state it binds.
    assert_eq!((sub.retained(), sub.calls()), (before, vec!["state_digest", "lease_claim"]));
    assert_ne!(sub.lease(), 0);
    assert_eq!(core.anchor(&mut sub, key(1)).unwrap_err().error, Code::AlreadyAnchored.into());
    assert_eq!(core.fault(), None);
}

#[test]
fn an_unanchored_turn_is_refused_before_any_k1_call() {
    let dir = Dir::new("unanchored");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::Unanchored.into());
    assert!(sub.calls().is_empty() && sub.epoch() == 0);
    assert_eq!(anchored(&core), None);
}

#[test]
fn a_turn_publishes_exactly_the_committed_identity_with_three_k1_crossings() {
    let dir = Dir::new("turn");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap(); // warm
    sub.clear_calls();
    core.counters(true);
    let generation = core.snapshot().unwrap().generation();
    let id = turn(&mut core, &mut sub, key(1)).unwrap();
    assert_eq!(id.state_after_digest, sub.identity());
    assert_eq!(anchored(&core), Some(sub.identity()));
    assert_eq!(core.snapshot().unwrap().generation(), generation + 1);
    // The warm managed turn: begin, one schedule, commit; no digest (the binding holds), one publication.
    assert_eq!(sub.calls(), ["txn_begin", "txn_run_schedule", "txn_commit_identity"]);
    let expected =
        Counters { k1_txn_begins: 1, k1_run_schedules: 1, k1_commits: 1, publications: 1, ..Counters::default() };
    assert_eq!(core.counters(false), expected);
}

#[test]
fn restart_resumes_on_the_matching_state_with_one_identity_check() {
    let dir = Dir::new("restart");
    let mut sub = Fake::new(0.1);
    {
        let mut core = dir.core();
        core.anchor(&mut sub, key(1)).unwrap();
        turn(&mut core, &mut sub, key(1)).unwrap();
    }
    let mut core = dir.core();
    let generation = core.snapshot().unwrap().generation();
    sub.clear_calls();
    turn(&mut core, &mut sub, key(7)).unwrap(); // a fresh runtime binds whichever matching wrapper it is given
    assert_eq!(sub.calls()[..2], ["shape", "state_digest"]); // read-only: fuel admission, then identity
    assert_eq!(core.snapshot().unwrap().generation(), generation + 1);
}

#[test]
fn a_mismatched_state_fail_stops_before_any_mutation() {
    let dir = Dir::new("mismatch");
    let mut sub = Fake::new(0.1);
    dir.core().anchor(&mut sub, key(1)).unwrap();
    let mut other = Fake::new(0.2);
    let mut core = dir.core();
    let durable = core.snapshot().unwrap();
    let before = other.retained();
    assert_eq!(turn(&mut core, &mut other, key(2)).unwrap_err(), Code::StateMismatch.into());
    // Read-only crossings only: the shape (fuel admission) and the identity.
    assert_eq!((other.retained(), other.calls()), (before, vec!["shape", "state_digest"]));
    assert_eq!(core.fault(), Some(Code::StateMismatch));
    // Fail-stopped for every lineage-dependent operation, the matching state included; nothing synthesized.
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
    assert_eq!(core.evolution_authority().unwrap_err(), Code::StateMismatch.into());
    assert_eq!(core.snapshot().unwrap(), durable);
    // Reopen reconciles: the matching state resumes.
    core.close();
    core.open().unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
}

#[test]
fn an_open_runtime_refuses_another_state_without_a_native_call() {
    let dir = Dir::new("switch");
    let mut core = dir.core();
    let (mut first, mut second) = (Fake::new(0.1), Fake::new(0.1)); // identical bytes, different owner
    core.anchor(&mut first, key(1)).unwrap();
    assert_eq!(turn(&mut core, &mut second, key(2)).unwrap_err(), Rt::SubstrateSwitch.into());
    assert!(second.calls().is_empty());
    assert_eq!(core.fault(), None); // a refusal, not a fail-stop
                                    // A reused handle address under another owner is another state.
    let reused = Key { owner: 9, ..key(1) };
    assert_eq!(turn(&mut core, &mut second, reused).unwrap_err(), Rt::SubstrateSwitch.into());
    turn(&mut core, &mut first, key(1)).unwrap();
}

#[test]
fn refusals_before_the_commit_leave_the_retained_state_unchanged() {
    let dir = Dir::new("refusals");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    let durable = core.snapshot().unwrap();
    let before = sub.retained();
    let mut s3 = vec![0.0; features(DIM)];

    // NONFINITE input: discarded natively, no abort crossing.
    let mut poison = input(2, 1);
    poison.x[1] = f64::NAN;
    sub.clear_calls();
    let r = core.turn_begin(&mut sub, key(1), &stimulus(&poison), &CEILING, &mut s3).unwrap_err();
    assert_eq!((r.error, r.k1_status), (Rt::EcsRefused.into(), K1_NONFINITE));
    assert_eq!(sub.calls(), ["txn_begin", "txn_run_schedule"]);

    // Malformed buffers are refused before any native call.
    sub.clear_calls();
    let bad = Input { x: vec![0.0; 3], ..input(2, 1) };
    assert_eq!(
        core.turn_begin(&mut sub, key(1), &stimulus(&bad), &CEILING, &mut s3).unwrap_err().error,
        Rt::Invalid.into()
    );
    let mut short = vec![0.0; 2];
    assert_eq!(
        core.turn_begin(&mut sub, key(1), &stimulus(&input(2, 1)), &CEILING, &mut short).unwrap_err().error,
        Rt::Invalid.into()
    );
    assert!(sub.calls().is_empty());

    // The boundary aborts (a decode refusal): nothing installed.
    core.turn_begin(&mut sub, key(1), &stimulus(&input(2, 1)), &CEILING, &mut s3).unwrap();
    assert_eq!(
        core.turn_begin(&mut sub, key(1), &stimulus(&input(2, 1)), &CEILING, &mut s3).unwrap_err().error,
        Rt::TurnOpen.into()
    );
    assert_eq!(core.turn_commit(&mut sub, key(2)).unwrap_err().error, Rt::TurnNotOpen.into());
    core.turn_abort(key(1)).unwrap();
    assert_eq!(core.turn_abort(key(1)).unwrap_err().error, Rt::TurnNotOpen.into());

    // The source state moved during the turn: STALE, the candidate is discarded natively.
    core.turn_begin(&mut sub, key(1), &stimulus(&input(2, 1)), &CEILING, &mut s3).unwrap();
    sub.interlope();
    let moved = sub.retained();
    let r = core.turn_commit(&mut sub, key(1)).unwrap_err();
    assert_eq!((r.error, r.k1_status, r.committed), (Rt::EcsStale.into(), K1_STALE, None));
    assert!(!sub.txn_open() && sub.retained() == moved && moved != before);
    assert_eq!(core.snapshot().unwrap(), durable); // nothing published
    assert_eq!(core.fault(), None);
}

#[test]
fn rows_beyond_the_reservation_grow_it_on_the_cold_path() {
    let dir = Dir::new("capacity");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    sub.clear_calls();
    let big = input(12, 2);
    let mut s3 = vec![0.0; features(DIM)];
    core.turn_begin(&mut sub, key(1), &stimulus(&big), &CEILING, &mut s3).unwrap();
    core.turn_commit(&mut sub, key(1)).unwrap();
    assert_eq!(
        sub.calls(),
        [
            "shape",
            "txn_begin",
            "txn_run_schedule",
            "txn_abort",
            "reserve",
            "txn_begin",
            "txn_run_schedule",
            "txn_commit_identity"
        ]
    );
    assert_eq!(sub.max_rows(), 12);
}

#[test]
fn evolution_reserves_executes_once_and_finalizes() {
    let dir = Dir::new("evolution");
    let mut core = dir.core();
    let authority = core.evolution_authority().unwrap().evolution();
    let assertion = [7u8; 32];
    let pending = core.evolution_reserve(&authority, &assertion).unwrap().evolution();
    assert_eq!(pending.assertion(), Some(assertion));
    assert_eq!(core.evolution_reserve(&authority, &assertion).unwrap_err(), Rt::EvolutionInFlight.into());
    let done = core.evolution_finalize(&[9u8; 32]).unwrap().evolution();
    assert_eq!((done.revision(), done.head(), done.assertion()), (1, [9u8; 32], None));
    assert_eq!(core.evolution_finalize(&[9u8; 32]).unwrap_err(), Rt::EvolutionNotInFlight.into());
    // A stale observation cannot reserve; it fail-stops (nothing executed).
    assert_eq!(core.evolution_reserve(&authority, &assertion).unwrap_err(), Code::AuthorityMismatch.into());
    assert_eq!(core.fault(), Some(Code::AuthorityMismatch));
}

#[test]
fn an_abandoned_attempt_stays_pending_and_is_reconciled_explicitly_once() {
    let dir = Dir::new("abandon");
    let assertion = [3u8; 32];
    {
        let mut core = dir.core();
        let authority = core.evolution_authority().unwrap().evolution();
        core.evolution_reserve(&authority, &assertion).unwrap();
        core.evolution_abandon().unwrap();
        assert_eq!(core.fault(), Some(Code::EvolutionPending));
        assert_eq!(core.evolution_authority().unwrap_err(), Code::EvolutionPending.into());
    }
    let mut core = dir.core();
    let pending = core.snapshot().unwrap().evolution();
    assert_eq!(pending.assertion(), Some(assertion));
    assert_eq!(core.evolution_authority().unwrap_err(), Code::EvolutionPending.into());
    assert_eq!(core.evolution_reserve(&pending, &assertion).unwrap_err(), Code::EvolutionPending.into());
    assert_eq!(core.fault(), None);
    let done = core.evolution_reconcile(&pending, &[5u8; 32]).unwrap().evolution();
    assert_eq!(done, EvolutionState::idle(1, [5u8; 32]).unwrap());
    assert_eq!(core.evolution_reconcile(&pending, &[5u8; 32]).unwrap_err(), Code::AuthorityMismatch.into());
    assert_eq!(core.fault(), None); // an operator refusal, not a fail-stop
    assert_eq!(core.evolution_authority().unwrap().evolution(), done);
}

// -- turn lifecycle: an open native transaction is never forgotten ------------------------------------------------

/// What an implicit abort must leave untouched: the retained K1 state, its generation and digest, and continuity.
struct Witness {
    retained: (Vec<u64>, u64),
    generation: u64,
    identity: Digest,
    durable: elpis_continuity::Snapshot,
}

fn witness(core: &Core, sub: &Fake) -> Witness {
    Witness {
        retained: sub.retained(),
        generation: sub.generation(),
        identity: sub.identity(),
        durable: core.snapshot().unwrap(),
    }
}

fn unchanged(w: &Witness, dir: &Dir, sub: &Fake) {
    assert_eq!((sub.retained(), sub.generation(), sub.identity()), (w.retained.clone(), w.generation, w.identity));
    // Continuity as restart will see it: the committed identity of the last committed turn, nothing newer.
    let mut fresh = Core::new(&dir.0).unwrap();
    assert_eq!(fresh.open().unwrap(), w.durable);
}

/// An anchored runtime with one committed turn and a second turn begun (schedule ran on the candidate).
fn begun(dir: &Dir) -> (Core, Fake, Witness) {
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    let w = witness(&core, &sub);
    let i = input(4, 3);
    let mut s3 = vec![0.0; features(DIM)];
    core.turn_begin(&mut sub, key(1), &stimulus(&i), &CEILING, &mut s3).unwrap();
    assert!(core.turn_is_open() && sub.txn_open());
    core.counters(true);
    sub.clear_calls();
    (core, sub, w)
}

fn aborts(core: &mut Core) -> u64 {
    core.counters(false).k1_aborts
}

#[test]
fn an_explicit_abort_ends_the_turn_once_and_installs_nothing() {
    let dir = Dir::new("life-abort");
    let (mut core, mut sub, w) = begun(&dir);
    core.turn_abort(key(1)).unwrap();
    assert_eq!((sub.calls(), aborts(&mut core)), (vec!["txn_abort"], 1));
    assert!(!core.turn_is_open() && !sub.txn_open());
    core.close();
    assert_eq!(sub.calls(), ["txn_abort"]); // close finds nothing left to end
    unchanged(&w, &dir, &sub);
    core.open().unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
}

#[test]
fn a_commit_ends_the_turn_without_an_abort() {
    let dir = Dir::new("life-commit");
    let (mut core, mut sub, w) = begun(&dir);
    let (id, snapshot) = core.turn_commit(&mut sub, key(1)).unwrap();
    assert_eq!((sub.calls(), aborts(&mut core)), (vec!["txn_commit_identity"], 0));
    assert!(!core.turn_is_open() && !sub.txn_open());
    assert_eq!((id.state_before_digest, id.state_after_digest), (w.identity, sub.identity()));
    assert_eq!(sub.generation(), w.generation + 1);
    assert_eq!(snapshot.cognition(), Cognition::Anchored(sub.identity()));
    drop(core);
    assert_eq!(sub.calls(), ["txn_commit_identity"]);
}

#[test]
fn close_aborts_an_open_turn_natively_and_changes_no_authority() {
    let dir = Dir::new("life-close");
    let (mut core, mut sub, w) = begun(&dir);
    core.close();
    assert_eq!((sub.calls(), aborts(&mut core)), (vec!["txn_abort"], 1));
    assert!(!core.turn_is_open() && !sub.txn_open() && !core.is_open());
    unchanged(&w, &dir, &sub);
    // The same live state takes the next turn: a transaction can begin again.
    core.open().unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    assert_eq!(sub.generation(), w.generation + 1);
}

#[test]
fn destruction_aborts_an_open_turn_natively_and_changes_no_authority() {
    let dir = Dir::new("life-drop");
    let (core, mut sub, w) = begun(&dir);
    drop(core);
    assert_eq!(sub.calls(), ["txn_abort"]);
    assert!(!sub.txn_open());
    unchanged(&w, &dir, &sub);
    let mut core = dir.core();
    turn(&mut core, &mut sub, key(1)).unwrap();
}

#[test]
fn reopening_an_open_runtime_is_refused_and_keeps_the_turn() {
    let dir = Dir::new("life-reopen");
    let (mut core, mut sub, w) = begun(&dir);
    assert_eq!(core.open().unwrap_err(), Code::Open.into());
    assert!(sub.calls().is_empty() && core.turn_is_open() && sub.txn_open());
    // Not forgotten: the turn is still the caller's to finish.
    core.turn_commit(&mut sub, key(1)).unwrap();
    assert_eq!((aborts(&mut core), sub.generation()), (0, w.generation + 1));
}

#[test]
fn close_then_open_after_a_fail_stop_aborts_the_open_turn() {
    let dir = Dir::new("life-failstop");
    let (mut core, mut sub, w) = begun(&dir);
    // An evolution attempt completes; a reservation against the authority it replaced fail-stops, turn open.
    let stale = core.evolution_authority().unwrap().evolution();
    core.evolution_reserve(&stale, &[1u8; 32]).unwrap();
    core.evolution_finalize(&[2u8; 32]).unwrap();
    let durable = core.snapshot().unwrap();
    assert_eq!(core.evolution_reserve(&stale, &[3u8; 32]).unwrap_err(), Code::AuthorityMismatch.into());
    assert_eq!(core.fault(), Some(Code::AuthorityMismatch));
    assert!(core.turn_is_open() && sub.txn_open());
    core.close();
    assert_eq!((sub.calls(), aborts(&mut core)), (vec!["txn_abort"], 1));
    assert!(!sub.txn_open());
    let w = Witness { durable, ..w };
    unchanged(&w, &dir, &sub);
    core.open().unwrap();
    assert_eq!(core.fault(), None);
    turn(&mut core, &mut sub, key(1)).unwrap();
}

#[test]
fn a_fail_stopped_commit_aborts_the_turn() {
    let dir = Dir::new("life-failstop-commit");
    let (mut core, mut sub, w) = begun(&dir);
    let authority = core.evolution_authority().unwrap().evolution();
    let current = core.evolution_reserve(&authority, &[1u8; 32]).unwrap();
    core.evolution_abandon().unwrap(); // fail-stop CONTINUITY_EVOLUTION_PENDING, the turn still open
    let r = core.turn_commit(&mut sub, key(1)).unwrap_err();
    assert_eq!((r.error, r.committed), (Code::EvolutionPending.into(), None));
    assert_eq!((sub.calls(), aborts(&mut core)), (vec!["txn_abort"], 1));
    assert!(!core.turn_is_open() && !sub.txn_open());
    let w = Witness { durable: current, ..w };
    core.close();
    unchanged(&w, &dir, &sub);
}

#[test]
fn a_recoverable_commit_refusal_aborts_the_open_transaction() {
    let dir = Dir::new("life-commit-refused");
    let (mut core, mut sub, w) = begun(&dir);
    sub.state().refuse_commit = K1_BUSY; // the transaction stays open natively
    let r = core.turn_commit(&mut sub, key(1)).unwrap_err();
    assert_eq!((r.error, r.k1_status, r.committed), (Rt::EcsRefused.into(), K1_BUSY, None));
    assert_eq!(sub.calls(), ["txn_commit_identity", "txn_abort"]);
    assert!(!core.turn_is_open() && !sub.txn_open());
    core.close();
    unchanged(&w, &dir, &sub);
}

#[test]
fn an_abort_refused_busy_is_retried_until_it_ends_the_transaction() {
    for path in ["abort", "close", "drop"] {
        let dir = Dir::new(&format!("life-busy-{path}"));
        let (mut core, sub, w) = begun(&dir);
        sub.state().busy_aborts = 2;
        match path {
            "abort" => {
                core.turn_abort(key(1)).unwrap();
                core.close();
            }
            "close" => core.close(),
            _ => drop(core),
        }
        assert_eq!(sub.calls(), ["txn_abort"; 3]);
        assert!(!sub.txn_open());
        unchanged(&w, &dir, &sub);
    }
}

#[test]
fn another_substrates_abort_is_refused_and_the_turn_stays_open_until_close() {
    let dir = Dir::new("life-wrong-key");
    let (mut core, sub, w) = begun(&dir);
    assert_eq!(core.turn_abort(key(2)).unwrap_err().error, Rt::TurnNotOpen.into());
    assert!(sub.calls().is_empty() && core.turn_is_open() && sub.txn_open());
    core.close();
    assert_eq!(sub.calls(), ["txn_abort"]);
    unchanged(&w, &dir, &sub);
}

#[test]
fn lifecycle_without_a_turn_makes_no_native_call() {
    let dir = Dir::new("life-idle");
    let mut sub = Fake::new(0.1);
    let mut core = Core::new(&dir.0).unwrap();
    core.close(); // close of a never-opened runtime
    core.open().unwrap();
    core.anchor(&mut sub, key(1)).unwrap();
    sub.clear_calls();
    core.close();
    core.close(); // double close
    core.open().unwrap(); // open -> close -> open
    core.close();
    core.open().unwrap();
    drop(core); // destroy with no turn
    assert!(sub.calls().is_empty());
    assert_eq!(Core::new(&dir.0).unwrap().counters(false), Counters::default());
    // Double close after a turn: one abort, then nothing.
    let dir = Dir::new("life-idle-turn");
    let (mut core, sub, _) = begun(&dir);
    core.close();
    core.close();
    drop(core);
    assert_eq!(sub.calls(), ["txn_abort"]);
}

#[test]
fn a_cold_path_abort_refused_busy_is_retried_before_the_reservation_grows() {
    let dir = Dir::new("life-capacity-busy");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    sub.state().busy_aborts = 1;
    sub.clear_calls();
    let big = input(12, 2);
    let mut s3 = vec![0.0; features(DIM)];
    core.turn_begin(&mut sub, key(1), &stimulus(&big), &CEILING, &mut s3).unwrap();
    assert_eq!(
        sub.calls(),
        [
            "shape",
            "txn_begin",
            "txn_run_schedule",
            "txn_abort",
            "txn_abort",
            "reserve",
            "txn_begin",
            "txn_run_schedule"
        ]
    );
    core.turn_abort(key(1)).unwrap();
    assert!(!sub.txn_open());
}

#[cfg(feature = "testing")]
mod faults {
    use super::*;

    const DIE: u32 = 1;
    const WRITE_FAIL: u32 = 2;
    const SYNC_FAIL_LOST: u32 = 5;
    const SYNC_FAIL_DURABLE: u32 = 6;

    #[test]
    fn a_failed_publication_after_the_k1_commit_keeps_k1_and_fail_stops() {
        for (action, expected, durable_new) in [
            (WRITE_FAIL, Code::PublicationRefused, false),
            (SYNC_FAIL_LOST, Code::PublicationUncertain, false),
            (SYNC_FAIL_DURABLE, Code::PublicationUncertain, true),
        ] {
            let dir = Dir::new(&format!("publication-{action}"));
            let mut sub = Fake::new(0.1);
            let anchor = {
                let mut core = dir.core();
                core.anchor(&mut sub, key(1)).unwrap();
                let anchor = anchored(&core).unwrap();
                core.testing_arm(1, action, 0).unwrap();
                let i = input(4, 3);
                let mut s3 = vec![0.0; features(DIM)];
                core.turn_begin(&mut sub, key(1), &stimulus(&i), &CEILING, &mut s3).unwrap();
                let r = core.turn_commit(&mut sub, key(1)).unwrap_err();
                assert_eq!(r.error, expected.into());
                // The K1 commit stands and is reported.
                assert_eq!(r.committed.unwrap().state_after_digest, sub.identity());
                assert_ne!(sub.identity(), anchor);
                let committed = sub.retained();
                sub.clear_calls();
                assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), expected.into());
                assert!(sub.calls().is_empty() && sub.retained() == committed);
                anchor
            };
            let mut core = dir.core();
            if durable_new {
                assert_eq!(anchored(&core), Some(sub.identity()));
                turn(&mut core, &mut sub, key(1)).unwrap();
            } else {
                assert_eq!(anchored(&core), Some(anchor));
                assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
            }
        }
    }

    #[test]
    fn evolution_publication_failures_fail_stop_and_never_reexecute() {
        for phase in [1u64, 2] {
            let dir = Dir::new(&format!("evolution-fault-{phase}"));
            let mut core = dir.core();
            let authority = core.evolution_authority().unwrap().evolution();
            core.testing_arm(phase, WRITE_FAIL, 0).unwrap();
            let reserved = core.evolution_reserve(&authority, &[1u8; 32]);
            if phase == 1 {
                assert_eq!(reserved.unwrap_err(), Code::PublicationRefused.into());
                assert_eq!(core.evolution_finalize(&[2u8; 32]).unwrap_err(), Rt::EvolutionNotInFlight.into());
            } else {
                reserved.unwrap();
                assert_eq!(core.evolution_finalize(&[2u8; 32]).unwrap_err(), Code::PublicationRefused.into());
            }
            assert_eq!(core.fault(), Some(Code::PublicationRefused));
            assert_eq!(core.evolution_authority().unwrap_err(), Code::PublicationRefused.into());
            drop(core);
            let core = dir.core();
            let e = core.snapshot().unwrap().evolution();
            assert_eq!(e.assertion().is_some(), phase == 2); // a refused finalization leaves the reservation
        }
    }

    #[test]
    fn process_death_during_publication_fail_stops() {
        let dir = Dir::new("death");
        let mut core = dir.core();
        let authority = core.evolution_authority().unwrap().evolution();
        core.testing_arm(1, DIE, 1).unwrap(); // after the record is written, before it is synced
        assert_eq!(core.evolution_reserve(&authority, &[1u8; 32]).unwrap_err(), Code::TestingProcessDeath.into());
        assert_eq!(core.fault(), Some(Code::TestingProcessDeath));
    }

    #[test]
    fn a_warm_managed_turn_writes_one_record_and_syncs_once() {
        let dir = Dir::new("io");
        let mut core = dir.core();
        let mut sub = Fake::new(0.1);
        core.anchor(&mut sub, key(1)).unwrap();
        turn(&mut core, &mut sub, key(1)).unwrap();
        core.testing_continuity_counters(true);
        turn(&mut core, &mut sub, key(1)).unwrap();
        let io = core.testing_continuity_counters(false);
        assert_eq!((io.opens, io.preads, io.pwrites, io.pwrite_bytes, io.data_syncs), (0, 0, 1, 176, 1));
        assert_eq!((io.dir_syncs, io.renames, io.unlinks), (0, 0, 0));
    }
}

// -- QUERY: the canonical read-only operation ---------------------------------------------------------------------

#[test]
fn a_query_answers_from_the_anchored_state_and_changes_nothing() {
    let dir = Dir::new("query");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    let (retained, generation) = (sub.retained(), sub.generation());
    let durable = core.snapshot().unwrap();
    sub.clear_calls();
    core.counters(true);
    let (answer, identity) = query(&mut core, &mut sub, key(1), 3).unwrap();
    assert_eq!(identity, sub.identity());
    assert!(answer.iter().all(|v| v.is_finite()));
    // Cold: the state's shape is read once (fuel admission); then one native crossing answers and identifies.
    assert_eq!(sub.calls(), ["shape", "query_identity"]);
    assert_eq!(core.counters(false), Counters { k1_queries: 1, k1_shapes: 1, ..Counters::default() });
    // Warm: one native crossing; no digest-then-forward window, no transaction, no commit, no publication.
    sub.clear_calls();
    core.counters(true);
    let (warm, _) = query(&mut core, &mut sub, key(1), 3).unwrap();
    assert_eq!((sub.calls(), warm), (vec!["query_identity"], answer.clone()));
    assert_eq!(core.counters(false), Counters { k1_queries: 1, ..Counters::default() });
    assert_eq!((sub.retained(), sub.generation(), sub.txn_open()), (retained, generation, false));
    assert_eq!(core.snapshot().unwrap(), durable);
    // Deterministic and repeatable: the same state answers the same way.
    assert_eq!(query(&mut core, &mut sub, key(1), 3).unwrap(), (answer, identity));
    assert_eq!(core.snapshot().unwrap(), durable);
    assert_eq!(core.fault(), None);
}

#[test]
fn a_query_never_publishes_and_never_anchors() {
    let dir = Dir::new("query-unanchored");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let durable = core.snapshot().unwrap();
    assert_eq!(query(&mut core, &mut sub, key(1), 2).unwrap_err(), Code::Unanchored.into());
    assert!(sub.calls().is_empty());
    assert_eq!(core.snapshot().unwrap(), durable);
    assert_eq!(anchored(&core), None);
    assert_eq!(core.fault(), None);
}

#[test]
fn a_query_of_a_state_the_lineage_never_committed_fail_stops_and_withholds_the_answer() {
    let dir = Dir::new("query-mismatch");
    let mut sub = Fake::new(0.1);
    dir.core().anchor(&mut sub, key(1)).unwrap();
    let mut core = dir.core();
    let durable = core.snapshot().unwrap();
    let mut other = Fake::new(0.2);
    let x = vec![0.5; 2 * DIM];
    let mut out = vec![7.0; 2];
    let refused = core.query(&mut other, key(2), &x, &mut out, &CEILING).unwrap_err();
    assert_eq!(refused.error, Code::StateMismatch.into());
    assert_eq!(out, [0.0, 0.0]);
    assert_eq!(core.fault(), Some(Code::StateMismatch));
    assert_eq!(core.snapshot().unwrap(), durable);
    // Fail-stopped: the matching state is refused too until reopen.
    assert_eq!(query(&mut core, &mut sub, key(1), 1).unwrap_err(), Code::StateMismatch.into());
}

#[test]
fn a_query_of_a_bound_state_moved_out_of_band_fail_stops() {
    let dir = Dir::new("query-moved");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    query(&mut core, &mut sub, key(1), 1).unwrap();
    sub.interlope();
    let durable = core.snapshot().unwrap();
    assert_eq!(query(&mut core, &mut sub, key(1), 1).unwrap_err(), Code::StateMismatch.into());
    assert_eq!(core.fault(), Some(Code::StateMismatch));
    assert_eq!(core.snapshot().unwrap(), durable);
}

#[test]
fn a_refused_query_changes_nothing_and_does_not_fail_stop() {
    let dir = Dir::new("query-refused");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    let (retained, durable) = (sub.retained(), core.snapshot().unwrap());
    let mut out = vec![1.0; 1];
    let refused = core.query(&mut sub, key(1), &[f64::NAN, 0.0], &mut out, &CEILING).unwrap_err();
    assert_eq!((refused.error, refused.k1_status), (Rt::EcsRefused.into(), K1_NONFINITE));
    assert_eq!(out, [0.0]);
    // Malformed shapes are refused before any native call.
    sub.clear_calls();
    assert_eq!(core.query(&mut sub, key(1), &[0.0; 3], &mut out, &CEILING).unwrap_err().error, Rt::Invalid.into());
    assert_eq!(core.query(&mut sub, key(1), &[], &mut [], &CEILING).unwrap_err().error, Rt::Invalid.into());
    assert!(sub.calls().is_empty());
    assert_eq!((sub.retained(), core.snapshot().unwrap(), core.fault()), (retained, durable, None));
}

#[test]
fn a_query_is_refused_while_a_managed_turn_is_open_and_another_state_is_a_switch() {
    let dir = Dir::new("query-turn");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    let i = input(2, 1);
    let mut s3 = vec![0.0; features(DIM)];
    core.turn_begin(&mut sub, key(1), &stimulus(&i), &CEILING, &mut s3).unwrap();
    assert_eq!(query(&mut core, &mut sub, key(1), 1).unwrap_err(), Rt::TurnOpen.into());
    core.turn_abort(key(1)).unwrap();
    let mut other = Fake::new(0.1);
    other.clear_calls();
    assert_eq!(query(&mut core, &mut other, key(2), 1).unwrap_err(), Rt::SubstrateSwitch.into());
    assert!(other.calls().is_empty());
}

// -- total cognitive fuel ------------------------------------------------------------------------------------------

/// The work-unit formula, written out independently of `fuel.rs` (a second derivation the tests compare against).
fn units(dim: u128, width: u128, schedule: &[(u128, u128)]) -> u128 {
    let f = dim + dim * (dim + 1) / 2 + dim * (dim + 1) * (dim + 2) / 6;
    schedule
        .iter()
        .map(|&(r, k)| k * (2 * r * dim * width + 2 * dim * width * f + f * f) + (r * f * f + dim * width * f))
        .sum()
}

#[test]
fn work_units_are_the_documented_integer_formula() {
    let s = |v: &[(u64, u64)]| v.iter().map(|&(rows, steps)| Experience { rows, steps }).collect::<Vec<_>>();
    for (dim, width, sched) in
        [(6usize, 36usize, vec![(4u64, 3u64), (4, 3)]), (2, 3, vec![(1, 1)]), (6, 36, vec![(256, 1 << 20)])]
    {
        let want =
            units(dim as u128, width as u128, &sched.iter().map(|&(r, k)| (r as u128, k as u128)).collect::<Vec<_>>());
        assert_eq!(fuel::schedule_units(dim, width, &s(&sched)).map(u128::from), Some(want));
    }
    // The fixture turn of the integration tests (2 experiences x 4 rows x 3 steps on 6 x 36).
    assert_eq!(fuel::schedule_units(6, 36, &s(&[(4, 3), (4, 3)])), Some(357_806));
    assert_eq!(fuel::query_units(6, 36, 3), Some(648));
    // Totals beyond u64 are refused, never wrapped.
    assert_eq!(fuel::schedule_units(64, usize::MAX / 2, &s(&[(1, 1)])), None);
    assert_eq!(fuel::schedule_units(6, 36, &s(&[(u64::MAX, 1)])), None);
    assert_eq!(fuel::schedule_units(6, 36, &s(&[(1, u64::MAX)])), None);
    assert_eq!(fuel::query_units(6, usize::MAX, 2), None);
    assert_eq!(fuel::schedule_units(0, 36, &s(&[(1, 1)])), None);
    assert_eq!(fuel::schedule_units(6, 36, &[]), None);
}

#[test]
fn budgets_bind_totals_at_their_exact_integer_boundary() {
    let sched = [Experience { rows: 4, steps: 3 }, Experience { rows: 4, steps: 3 }];
    let need = fuel::schedule_units(DIM, WIDTH, &sched).unwrap();
    let at = Budget { max_work_units: need, ..CEILING };
    assert_eq!(fuel::admit_schedule(&at, DIM, WIDTH, &sched), Ok(need));
    let under = Budget { max_work_units: need - 1, ..CEILING };
    assert_eq!(fuel::admit_schedule(&under, DIM, WIDTH, &sched), Err(fuel::Over::WorkUnits));
    for (budget, over) in [
        (Budget { max_steps: 5, ..CEILING }, fuel::Over::Steps),
        (Budget { max_rows: 7, ..CEILING }, fuel::Over::Rows),
        (Budget { max_experience_rows: 3, ..CEILING }, fuel::Over::ExperienceRows),
        (Budget { max_experiences: 1, ..CEILING }, fuel::Over::Experiences),
        (Budget { max_work_units: CEILING.max_work_units + 1, ..CEILING }, fuel::Over::Budget),
        (Budget { max_steps: 0, ..CEILING }, fuel::Over::Budget),
    ] {
        assert_eq!(fuel::admit_schedule(&budget, DIM, WIDTH, &sched), Err(over));
    }
    assert_eq!(fuel::admit_schedule(&Budget { max_steps: 6, max_rows: 8, ..CEILING }, DIM, WIDTH, &sched), Ok(need));
    // The per-field bounds alone admit 64 x 2^20 steps; the totals do not.
    let huge = vec![Experience { rows: 256, steps: 1 << 20 }; 64];
    assert!(fuel::admit_schedule(&CEILING, 6, 36, &huge).is_err());
    assert_eq!(fuel::admit_query(&Budget { max_query_rows: 2, ..CEILING }, DIM, WIDTH, 3), Err(fuel::Over::QueryRows));
    assert_eq!(
        fuel::admit_query(&Budget { max_query_rows: 3, ..CEILING }, DIM, WIDTH, 3),
        Ok(3 * (DIM * WIDTH) as u64)
    );
}

#[test]
fn excessive_work_is_refused_before_reserve_begin_schedule_commit_and_publication() {
    let dir = Dir::new("fuel");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    let durable = core.snapshot().unwrap();
    let retained = sub.retained();
    let big = input(12, 2); // beyond the stand-in's 8 reserved rows: admitted, it would reserve
    let need = fuel::schedule_units(DIM, WIDTH, big_schedule(&big)).unwrap();
    for budget in [
        Budget { max_work_units: need - 1, ..CEILING },
        Budget { max_steps: 1, ..CEILING },
        Budget { max_experience_rows: 11, ..CEILING },
        Budget { max_query_rows: CEILING.max_query_rows + 1, ..CEILING },
    ] {
        sub.clear_calls();
        core.counters(true);
        let mut s3 = vec![0.0; features(DIM)];
        let refused = core.turn_begin(&mut sub, key(1), &stimulus(&big), &budget, &mut s3).unwrap_err();
        assert_eq!(refused.error, Rt::Fuel.into());
        // No reserve, no transaction, no schedule, no commit, no publication; at most the read-only shape.
        assert!(sub.calls().iter().all(|c| *c == "shape"), "{:?}", sub.calls());
        let c = core.counters(false);
        assert_eq!((c.k1_reserves, c.k1_txn_begins, c.k1_run_schedules, c.k1_commits, c.publications), (0, 0, 0, 0, 0));
        assert_eq!((sub.max_rows(), sub.retained(), sub.txn_open()), (8, retained.clone(), false));
        assert_eq!((core.snapshot().unwrap(), core.fault(), core.turn_is_open()), (durable, None, false));
    }
    // A query beyond its rows is refused before the native query.
    sub.clear_calls();
    let x = vec![0.5; 3 * DIM];
    let mut out = vec![0.0; 3];
    let refused = core.query(&mut sub, key(1), &x, &mut out, &Budget { max_query_rows: 2, ..CEILING }).unwrap_err();
    assert_eq!(refused.error, Rt::Fuel.into());
    assert!(!sub.calls().contains(&"query_identity"));
    // Within its budget the same schedule runs.
    let mut s3 = vec![0.0; features(DIM)];
    core.turn_begin(&mut sub, key(1), &stimulus(&big), &Budget { max_work_units: need, ..CEILING }, &mut s3).unwrap();
    core.turn_commit(&mut sub, key(1)).unwrap();
}

fn big_schedule(i: &Input) -> &[Experience] {
    &i.schedule
}

// -- managed ownership: the bound state's K1 lease ------------------------------------------------------------------

#[test]
fn a_bound_state_refuses_unmanaged_mutation_and_the_lineage_continues() {
    let dir = Dir::new("lease-refuse");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    let retained = sub.retained();
    assert_eq!(sub.unmanaged_learn(), K1_LEASED);
    assert_eq!(sub.retained(), retained);
    turn(&mut core, &mut sub, key(1)).unwrap();
    assert_eq!((anchored(&core), core.fault()), (Some(sub.identity()), None));
}

#[test]
fn a_lost_lease_fail_stops_before_any_k1_mutation_or_publication() {
    let dir = Dir::new("lease-lost");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    sub.steal();
    let (retained, durable) = (sub.retained(), core.snapshot().unwrap());
    sub.clear_calls();
    core.counters(true);
    let i = input(4, 3);
    let mut s3 = vec![0.0; features(DIM)];
    let refused = core.turn_begin(&mut sub, key(1), &stimulus(&i), &CEILING, &mut s3).unwrap_err();
    assert_eq!((refused.error, refused.k1_status), (Code::StateMismatch.into(), K1_LEASED));
    assert_eq!(sub.calls(), ["txn_begin"]); // refused natively: nothing begun, nothing scheduled
    assert_eq!((sub.retained(), core.snapshot().unwrap()), (retained, durable));
    assert_eq!((core.fault(), core.counters(false).publications), (Some(Code::StateMismatch), 0));
}

#[test]
fn a_bound_state_that_moved_is_caught_inside_the_transaction_begin() {
    // A move no guard could refuse (it bypasses the lease): the identity re-established by the begin catches it,
    // before the schedule touches the candidate. Nothing is built on it and nothing is published.
    let dir = Dir::new("lease-moved");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    sub.interlope();
    let (retained, durable) = (sub.retained(), core.snapshot().unwrap());
    sub.clear_calls();
    let i = input(4, 3);
    let mut s3 = vec![0.0; features(DIM)];
    let refused = core.turn_begin(&mut sub, key(1), &stimulus(&i), &CEILING, &mut s3).unwrap_err();
    assert_eq!(refused.error, Code::StateMismatch.into());
    assert_eq!(sub.calls(), ["txn_begin", "txn_abort"]);
    assert!(!sub.txn_open());
    assert_eq!(
        (sub.retained(), core.snapshot().unwrap(), core.fault()),
        (retained, durable, Some(Code::StateMismatch))
    );
}

#[test]
fn release_gives_the_state_back_and_rebinding_verifies_it_again() {
    let dir = Dir::new("lease-release");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    let mut other = Fake::new(0.3);
    assert_eq!(core.release(&mut other, key(2)).unwrap_err().error, Rt::SubstrateSwitch.into());
    core.release(&mut sub, key(1)).unwrap();
    assert_eq!(sub.lease(), 0);
    assert_eq!(sub.unmanaged_learn(), 0); // unmanaged again: it moves outside the lineage
    sub.clear_calls();
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
    assert_eq!(sub.calls(), ["shape", "state_digest"]); // verified again, refused before any claim or mutation
    assert_eq!(sub.lease(), 0); // a mismatched state is never claimed
}

#[test]
fn a_state_with_an_open_unmanaged_transaction_is_not_adopted() {
    let dir = Dir::new("lease-busy");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let mut token = 0u64;
    let mut source = [0u8; 32];
    assert_eq!(sub.txn_begin(0, &mut source, &mut token), 0); // an unmanaged transaction, left open
    let durable = core.snapshot().unwrap();
    let refused = core.anchor(&mut sub, key(1)).unwrap_err();
    assert_eq!((refused.error, refused.k1_status), (Rt::EcsRefused.into(), K1_BUSY));
    assert_eq!((core.snapshot().unwrap(), sub.lease(), core.fault()), (durable, 0, None));
}

#[test]
fn a_second_runtime_binding_the_same_state_makes_the_first_fail_stop() {
    let (a, b) = (Dir::new("lease-two-a"), Dir::new("lease-two-b"));
    let (mut first, mut second) = (a.core(), b.core());
    let mut sub = Fake::new(0.1);
    first.anchor(&mut sub, key(1)).unwrap();
    turn(&mut first, &mut sub, key(1)).unwrap();
    second.anchor(&mut sub, key(1)).unwrap(); // proves its own lineage, then takes the lease
    let retained = sub.retained();
    assert_eq!(turn(&mut first, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
    assert_eq!((sub.retained(), first.fault()), (retained, Some(Code::StateMismatch)));
    turn(&mut second, &mut sub, key(1)).unwrap();
}
