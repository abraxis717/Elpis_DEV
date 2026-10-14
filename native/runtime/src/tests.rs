//! RuntimeCore laws against a deterministic in-memory stand-in for one native K1 state.
//!
//! The stand-in follows the refusal contract of `ecsg_k1.h` (CAPACITY/INVALID keep the transaction open,
//! NONFINITE from the schedule and STALE discard it, a commit installs the complete candidate or nothing) and its
//! live coordinate (a generation advanced by exactly one per committed transition, reported under the lease) but
//! computes nothing of K1's mathematics; the real native state is exercised by the C ABI tests and the Python
//! integration lane. Fault injection needs the `testing` feature (`cargo test --features testing`).

use std::ffi::c_int;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use elpis_continuity::{Code, Cognition, EvolutionState};

use crate::code::{Error, Rt};
use crate::core::{Core, Counters, Stimulus, VolatileCommit};
use crate::durable::{Status, Ticket};
use crate::fuel::{self, Budget, CEILING};
use crate::substrate::{
    features, k1_envelope_bytes, Digest, Experience, K1Ops, Key, ScheduleOutcome, Sealer, Transition, TxnAbort,
    K1_BUSY, K1_CAPACITY, K1_INVALID, K1_LEASED, K1_NONFINITE, K1_STALE,
};

const DIM: usize = 2;
/// The stand-in's declared width (fuel admission reads it natively once per state).
const WIDTH: usize = 3;
/// The stand-in's envelope: the K1 envelope size of its shape (as K1's: the magic, the retained values, zero
/// padding, a SHA-256 trailer over everything before it).
const ENVELOPE: usize = 576;

struct Txn {
    token: u64,
    lease: u64,
    generation: u64,
    w: Vec<f64>,
    epoch: u64,
}

/// One retained "state": W, epoch and a generation that every committed transition advances by one.
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

fn envelope_of(w: &[f64], epoch: u64) -> Vec<u8> {
    let mut e = unsealed(w, epoch);
    seal_in_place(&mut e);
    e
}

fn unsealed(w: &[f64], epoch: u64) -> Vec<u8> {
    let mut e = vec![0u8; ENVELOPE];
    e[..8].copy_from_slice(b"ELPISGK1");
    for (i, v) in w.iter().enumerate() {
        e[16 + 8 * i..24 + 8 * i].copy_from_slice(&v.to_bits().to_le_bytes());
    }
    e[40..48].copy_from_slice(&epoch.to_le_bytes());
    e
}

fn seal_in_place(e: &mut [u8]) {
    let n = e.len();
    let trailer = elpis_continuity::sha256::digest(&[&e[..n - 32]]);
    e[n - 32..].copy_from_slice(&trailer);
}

/// The stand-in's `envelope_seal`: pure; refuses another size or magic (as K1's validates the header and size).
unsafe extern "C" fn fake_seal(p: *mut u8, n: usize) -> c_int {
    if p.is_null() || n != ENVELOPE {
        return K1_INVALID;
    }
    let e = std::slice::from_raw_parts_mut(p, n);
    if &e[..8] != b"ELPISGK1" {
        return K1_INVALID;
    }
    seal_in_place(e);
    0
}

impl State {
    /// The retained-state identity: the envelope's SHA-256 trailer (K1's law).
    fn identity(&self) -> Digest {
        let e = envelope_of(&self.w, self.epoch);
        let mut out = [0u8; 32];
        out.copy_from_slice(&e[ENVELOPE - 32..]);
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

    fn envelope(&self) -> Vec<u8> {
        let s = self.state();
        envelope_of(&s.w, s.epoch)
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

    /// Someone commits a transition that bypasses every guard (a state moved while nobody could see it): the live
    /// coordinate must catch it.
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

    /// The stand-in answer `f_W(x)` of its current W (what any query of the live state must return).
    fn answer(&self, x: &[f64]) -> Vec<f64> {
        let s = self.state();
        (0..x.len() / DIM).map(|r| s.w[0] * x[r * DIM] + s.w[1] * x[r * DIM + 1] + s.w[2]).collect()
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

    fn shape(&mut self, dim: &mut usize, width: &mut usize) -> i32 {
        let mut s = self.state();
        s.calls.push("shape");
        *dim = DIM;
        *width = WIDTH;
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

    fn leased_identity(&mut self, lease: u64, digest: &mut Digest, generation: &mut u64) -> i32 {
        let mut s = self.state();
        s.calls.push("leased_identity");
        if s.lease != lease {
            return K1_LEASED;
        }
        if s.digest_fails {
            return -1;
        }
        *digest = s.identity();
        *generation = s.generation;
        0
    }

    fn leased_query(&mut self, lease: u64, dim: usize, x: &[f64], out: &mut [f64], generation: &mut u64) -> i32 {
        let mut s = self.state();
        s.calls.push("leased_query");
        if dim != DIM || x.len() != out.len() * DIM {
            return -1;
        }
        if s.lease != lease {
            return K1_LEASED;
        }
        if x.iter().any(|v| !v.is_finite()) {
            return K1_NONFINITE;
        }
        // Reads the retained values only (a stand-in answer, not K1's forward map).
        for (r, o) in out.iter_mut().enumerate() {
            *o = s.w[0] * x[r * DIM] + s.w[1] * x[r * DIM + 1] + s.w[2];
        }
        *generation = s.generation;
        0
    }

    fn txn_begin_at(&mut self, lease: u64, source_generation: &mut u64, token: &mut u64) -> i32 {
        let mut s = self.state();
        s.calls.push("txn_begin_at");
        if s.lease != lease {
            return K1_LEASED;
        }
        if s.txn.is_some() {
            return K1_BUSY;
        }
        *token = s.next_token;
        *source_generation = s.generation;
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

    fn txn_commit(&mut self, lease: u64, token: u64, out: &mut Transition) -> i32 {
        let mut s = self.state();
        s.calls.push("txn_commit");
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
        out.epoch_before = s.epoch;
        out.generation_before = s.generation;
        s.w = txn.w;
        s.epoch = txn.epoch;
        s.generation += 1;
        out.epoch_after = s.epoch;
        out.generation_after = s.generation;
        out.steps = out.epoch_after - out.epoch_before;
        0
    }

    fn txn_abort(&mut self, token: u64) -> i32 {
        self.state().abort(token)
    }

    fn leased_capture(&mut self, lease: u64, out: &mut [u8], generation: &mut u64) -> i32 {
        let mut s = self.state();
        s.calls.push("leased_capture");
        if out.len() < ENVELOPE {
            return K1_INVALID;
        }
        if s.lease != lease {
            return K1_LEASED;
        }
        out[..ENVELOPE].copy_from_slice(&unsealed(&s.w, s.epoch));
        *generation = s.generation;
        0
    }

    fn sealer(&self) -> Sealer {
        fake_seal
    }

    fn retain_abort(&self) -> Box<dyn TxnAbort> {
        Box::new(FakeAbort(Arc::clone(&self.0)))
    }
}

fn query(core: &mut Core, sub: &mut Fake, k: Key, rows: usize) -> Result<(Vec<f64>, u64), Error> {
    let x = query_rows(rows);
    let mut out = vec![f64::NAN; rows];
    let generation = core.query(sub, k, &x, &mut out, &CEILING).map_err(|r| r.error)?;
    Ok((out, generation))
}

fn query_rows(rows: usize) -> Vec<f64> {
    (0..rows * DIM).map(|i| 0.25 * (i as f64 + 1.0)).collect()
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

fn turn(core: &mut Core, sub: &mut Fake, k: Key) -> Result<VolatileCommit, Error> {
    let i = input(4, 3);
    let mut s3 = vec![0.0; features(DIM)];
    core.turn_begin(sub, k, &stimulus(&i), &CEILING, &mut s3).map_err(|r| r.error)?;
    core.turn_commit(sub, k).map_err(|r| r.error)
}

/// Request durable publication of the live state and wait (the caller's explicit choice) for its resolution.
fn publish(core: &mut Core, sub: &mut Fake, k: Key) -> Result<Ticket, Error> {
    let t = core.durable_request(sub, k).map_err(|r| r.error)?;
    assert_eq!(t.status(), Status::Pending);
    core.lane().wait(t.sequence, Some(Duration::from_secs(30)))
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

/// The durable authority as a restart would read it from disk (the two slot files, read directly: the runtime under
/// test may hold the directory's lock).
fn on_disk(dir: &Dir) -> elpis_continuity::Snapshot {
    let read = |n: &str| -> [u8; 176] { std::fs::read(dir.0.join(n)).unwrap().try_into().unwrap() };
    elpis_continuity::store::resolve(&[read("continuity.a"), read("continuity.b")]).unwrap().0
}

#[test]
fn features_and_envelope_bytes_are_k1s() {
    assert_eq!(
        (features(0), features(1), features(2), features(6), features(64), features(65)),
        (0, 3, 9, 83, 47904, 0)
    );
    // elpis_ecsg_k1_envelope_bytes: 64 + 8 * (d*w + F(F+1)/2 + F) + 32.
    assert_eq!(k1_envelope_bytes(6, 1024), Some(77_800));
    assert_eq!(k1_envelope_bytes(6, 36), Some(64 + 8 * (216 + 3486 + 83) + 32));
    assert_eq!(k1_envelope_bytes(DIM, WIDTH), Some(ENVELOPE));
    assert_eq!((k1_envelope_bytes(0, 3), k1_envelope_bytes(2, 0), k1_envelope_bytes(65, 1)), (None, None, None));
}

#[test]
fn a_closed_runtime_refuses_everything() {
    let dir = Dir::new("closed");
    let mut core = Core::new(&dir.0).unwrap();
    let mut sub = Fake::new(0.1);
    assert_eq!(core.anchor(&mut sub, key(1)).unwrap_err().error, Rt::Closed.into());
    assert_eq!(core.evolution_authority().unwrap_err(), Rt::Closed.into());
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Rt::Closed.into());
    assert_eq!(core.durable_request(&mut sub, key(1)).unwrap_err().error, Rt::Closed.into());
    assert_eq!(core.snapshot().unwrap_err(), Rt::Closed.into());
    assert!(sub.calls().is_empty());
    assert!(Core::new(std::path::Path::new("relative")).is_err());
}

#[test]
fn anchor_is_explicit_once_durable_reads_the_state_only_and_binds() {
    let dir = Dir::new("anchor");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let before = sub.retained();
    let initial = core.snapshot().unwrap().generation();
    let s = core.anchor(&mut sub, key(1)).unwrap();
    assert_eq!(s.generation(), initial + 1);
    assert_eq!(anchored(&core), Some(sub.identity()));
    assert_eq!(on_disk(&dir), s); // durable when it returns: the anchor is the lineage's durable baseline
                                  // Claims managed ownership (the lease), then reads identity and generation together under it; mutates nothing.
    assert_eq!((sub.retained(), sub.calls()), (before, vec!["lease_claim", "leased_identity"]));
    assert_ne!(sub.lease(), 0);
    assert_eq!(core.live_generation(), Some(sub.generation()));
    assert_eq!(core.anchor(&mut sub, key(1)).unwrap_err().error, Code::AlreadyAnchored.into());
    assert_eq!(core.fault(), None);
}

#[test]
fn an_unanchored_turn_is_refused_before_any_k1_call() {
    let dir = Dir::new("unanchored");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::Unanchored.into());
    assert_eq!(core.durable_request(&mut sub, key(1)).unwrap_err().error, Code::Unanchored.into());
    assert!(sub.calls().is_empty() && sub.epoch() == 0);
    assert_eq!(anchored(&core), None);
}

#[test]
fn a_managed_turn_is_volatile_three_k1_crossings_no_digest_no_publication() {
    let dir = Dir::new("turn");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap(); // warm
    sub.clear_calls();
    core.counters(true);
    let generation = sub.generation();
    let c = turn(&mut core, &mut sub, key(1)).unwrap();
    // The live coordinate advanced by exactly one; the commit reports it.
    assert_eq!((c.transition.generation_before, c.transition.generation_after), (generation, generation + 1));
    assert_eq!((c.live_generation, core.live_generation()), (sub.generation(), Some(sub.generation())));
    // The durable authority did not move: no checkpoint, no publication, nothing on disk.
    assert_eq!(core.snapshot().unwrap(), anchor);
    assert_eq!(on_disk(&dir), anchor);
    assert_ne!(sub.identity(), anchored(&core).unwrap());
    // The warm managed turn: begin, one schedule, commit; no digest anywhere.
    assert_eq!(sub.calls(), ["txn_begin_at", "txn_run_schedule", "txn_commit"]);
    let expected = Counters { k1_txn_begins: 1, k1_run_schedules: 1, k1_commits: 1, ..Counters::default() };
    assert_eq!(core.counters(false), expected);
}

#[test]
fn queries_between_volatile_transitions_answer_from_the_live_state() {
    let dir = Dir::new("live-query");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    for n in 0..20 {
        turn(&mut core, &mut sub, key(1)).unwrap();
        sub.clear_calls();
        let (answer, generation) = query(&mut core, &mut sub, key(1), 3).unwrap();
        // The legitimate current live state, not the last durable one: exactly the live W's answer.
        assert_eq!(answer, sub.answer(&query_rows(3)), "turn {n}");
        assert_eq!(generation, sub.generation());
        assert_eq!(sub.calls(), ["leased_query"]); // one crossing; no digest; no lane
    }
    assert_eq!((core.snapshot().unwrap(), core.fault()), (anchor, None));
}

#[test]
fn restart_without_a_durable_publication_never_adopts_the_volatile_state() {
    let dir = Dir::new("restart-volatile");
    let mut sub = Fake::new(0.1);
    let anchor = {
        let mut core = dir.core();
        let a = core.anchor(&mut sub, key(1)).unwrap();
        turn(&mut core, &mut sub, key(1)).unwrap();
        turn(&mut core, &mut sub, key(1)).unwrap();
        a
    }; // the process ends with volatile work only
    let mut core = dir.core();
    assert_eq!(core.snapshot().unwrap(), anchor); // restart sees the last durable authority
    sub.clear_calls();
    let lease = sub.lease();
    assert_eq!(turn(&mut core, &mut sub, key(7)).unwrap_err(), Code::StateMismatch.into());
    // The advanced state is checked before any claim and never adopted: nothing claimed, nothing mutated.
    assert_eq!(sub.calls(), ["shape", "state_digest"]);
    assert_eq!((sub.lease(), core.fault()), (lease, Some(Code::StateMismatch)));
}

#[test]
fn restart_after_a_durable_publication_resumes_with_one_identity_check() {
    let dir = Dir::new("restart");
    let mut sub = Fake::new(0.1);
    {
        let mut core = dir.core();
        core.anchor(&mut sub, key(1)).unwrap();
        turn(&mut core, &mut sub, key(1)).unwrap();
        let t = publish(&mut core, &mut sub, key(1)).unwrap();
        assert_eq!((t.status(), t.identity, t.recoverable), (Status::Acknowledged, sub.identity(), 0));
    }
    let mut core = dir.core();
    let durable = core.snapshot().unwrap();
    assert_eq!(durable.cognition(), Cognition::Anchored(sub.identity()));
    sub.clear_calls();
    turn(&mut core, &mut sub, key(7)).unwrap(); // a fresh runtime binds whichever matching wrapper it is given
                                                // Read-only checks first (fuel shape, identity), then the claim and the identity + generation under the lease.
    assert_eq!(sub.calls()[..4], ["shape", "state_digest", "lease_claim", "leased_identity"]);
    assert_eq!(core.snapshot().unwrap(), durable); // the turn itself is volatile
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
    // Read-only crossings only: the shape (fuel admission) and the identity; never claimed.
    assert_eq!((other.retained(), other.calls(), other.lease()), (before, vec!["shape", "state_digest"], 0));
    assert_eq!(core.fault(), Some(Code::StateMismatch));
    // Fail-stopped for every lineage-dependent operation, the matching state included; nothing synthesized.
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
    assert_eq!(core.evolution_authority().unwrap_err(), Code::StateMismatch.into());
    assert_eq!(core.durable_request(&mut sub, key(1)).unwrap_err().error, Code::StateMismatch.into());
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
    assert_eq!(core.durable_request(&mut second, key(2)).unwrap_err().error, Rt::SubstrateSwitch.into());
    assert!(second.calls().is_empty());
    assert_eq!(core.fault(), None); // a refusal, not a fail-stop
    let reused = Key { owner: 9, ..key(1) }; // a reused handle address under another owner is another state
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
    assert_eq!(sub.calls(), ["txn_begin_at", "txn_run_schedule"]);
    assert_eq!(sub.retained(), before);

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
    assert_eq!(core.durable_request(&mut sub, key(1)).unwrap_err().error, Rt::TurnOpen.into());
    core.turn_abort(key(1)).unwrap();
    assert_eq!(core.turn_abort(key(1)).unwrap_err().error, Rt::TurnNotOpen.into());
    assert_eq!(sub.retained(), before);

    // The source state moved during the turn: STALE, the candidate is discarded natively.
    core.turn_begin(&mut sub, key(1), &stimulus(&input(2, 1)), &CEILING, &mut s3).unwrap();
    sub.interlope();
    let moved = sub.retained();
    let r = core.turn_commit(&mut sub, key(1)).unwrap_err();
    assert_eq!((r.error, r.k1_status, r.committed), (Rt::EcsStale.into(), K1_STALE, None));
    assert!(!sub.txn_open() && sub.retained() == moved && moved != before);
    assert_eq!(core.snapshot().unwrap(), durable); // nothing published
    assert_eq!(core.fault(), None);
    // The moved state is no longer the live coordinate: the next begin catches it and fail-stops.
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
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
            "txn_begin_at",
            "txn_run_schedule",
            "txn_abort",
            "reserve",
            "txn_begin_at",
            "txn_run_schedule",
            "txn_commit"
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
    // The reservation is durable when reserve returns (before the boundary may execute).
    assert_eq!(on_disk(&dir).evolution().assertion(), Some(assertion));
    assert_eq!(core.evolution_reserve(&authority, &assertion).unwrap_err(), Rt::EvolutionInFlight.into());
    let done = core.evolution_finalize(&[9u8; 32]).unwrap().evolution();
    assert_eq!((done.revision(), done.head(), done.assertion()), (1, [9u8; 32], None));
    assert_eq!(on_disk(&dir).evolution(), done);
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

#[test]
fn volatile_cognition_never_makes_a_pending_evolution_attempt_retryable() {
    // An attempt reserved, then abandoned (it may have had an external effect). Neither volatile turns nor a durable
    // publication of cognition clears, finalizes or retries it: they carry the pending authority unchanged.
    let dir = Dir::new("evolution-volatile");
    let mut sub = Fake::new(0.1);
    let assertion = [4u8; 32];
    {
        let mut core = dir.core();
        core.anchor(&mut sub, key(1)).unwrap();
        let authority = core.evolution_authority().unwrap().evolution();
        core.evolution_reserve(&authority, &assertion).unwrap();
        turn(&mut core, &mut sub, key(1)).unwrap(); // cognition stays HOT while the attempt is in flight
        core.evolution_abandon().unwrap();
        assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::EvolutionPending.into());
    }
    let mut core = dir.core();
    let pending = core.snapshot().unwrap().evolution();
    assert_eq!(pending.assertion(), Some(assertion));
    assert_eq!(core.evolution_reserve(&pending, &[8u8; 32]).unwrap_err(), Code::EvolutionPending.into());
    // A state matching the durable lineage (the anchored bytes) thinks and is durably published meanwhile.
    let mut fresh = Fake::new(0.1);
    turn(&mut core, &mut fresh, key(2)).unwrap();
    let t = publish(&mut core, &mut fresh, key(2)).unwrap();
    assert_eq!(t.status(), Status::Acknowledged);
    assert_eq!(on_disk(&dir).evolution(), pending); // the reservation is still pending, untouched
    assert_eq!(core.evolution_authority().unwrap_err(), Code::EvolutionPending.into());
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
    // Continuity as restart will see it: unchanged by the turn.
    assert_eq!(on_disk(dir), w.durable);
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
    // The state moved (volatile) past the anchor: a new open must not adopt it.
    core.open().unwrap();
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
}

#[test]
fn a_commit_ends_the_turn_without_an_abort() {
    let dir = Dir::new("life-commit");
    let (mut core, mut sub, w) = begun(&dir);
    let c = core.turn_commit(&mut sub, key(1)).unwrap();
    assert_eq!((sub.calls(), aborts(&mut core)), (vec!["txn_commit"], 0));
    assert!(!core.turn_is_open() && !sub.txn_open());
    assert_eq!((c.transition.generation_before, c.transition.generation_after), (w.generation, w.generation + 1));
    assert_eq!(sub.generation(), w.generation + 1);
    assert_eq!(core.snapshot().unwrap(), w.durable); // volatile
    drop(core);
    assert_eq!(sub.calls(), ["txn_commit"]);
}

#[test]
fn close_aborts_an_open_turn_natively_and_changes_no_authority() {
    let dir = Dir::new("life-close");
    let (mut core, sub, w) = begun(&dir);
    core.close();
    assert_eq!((sub.calls(), aborts(&mut core)), (vec!["txn_abort"], 1));
    assert!(!core.turn_is_open() && !sub.txn_open() && !core.is_open());
    unchanged(&w, &dir, &sub);
}

#[test]
fn destruction_aborts_an_open_turn_natively_and_changes_no_authority() {
    let dir = Dir::new("life-drop");
    let (core, sub, w) = begun(&dir);
    drop(core);
    assert_eq!(sub.calls(), ["txn_abort"]);
    assert!(!sub.txn_open());
    unchanged(&w, &dir, &sub);
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
    let (mut core, sub, w) = begun(&dir);
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
    assert_eq!(sub.calls(), ["txn_commit", "txn_abort"]);
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
            "txn_begin_at",
            "txn_run_schedule",
            "txn_abort",
            "txn_abort",
            "reserve",
            "txn_begin_at",
            "txn_run_schedule"
        ]
    );
    core.turn_abort(key(1)).unwrap();
    assert!(!sub.txn_open());
}

// -- QUERY: the canonical read-only operation ---------------------------------------------------------------------

#[test]
fn a_query_answers_from_the_live_state_and_changes_nothing() {
    let dir = Dir::new("query");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    let (retained, generation) = (sub.retained(), sub.generation());
    let durable = core.snapshot().unwrap();
    sub.clear_calls();
    core.counters(true);
    let (answer, g) = query(&mut core, &mut sub, key(1), 3).unwrap();
    assert_eq!((g, answer.clone()), (generation, sub.answer(&query_rows(3))));
    // Cold: the state's shape is read once (fuel admission); then one native crossing answers.
    assert_eq!(sub.calls(), ["shape", "leased_query"]);
    assert_eq!(core.counters(false), Counters { k1_queries: 1, k1_shapes: 1, ..Counters::default() });
    // Warm: one native crossing; no digest, no transaction, no commit, no publication.
    sub.clear_calls();
    core.counters(true);
    let (warm, _) = query(&mut core, &mut sub, key(1), 3).unwrap();
    assert_eq!((sub.calls(), warm), (vec!["leased_query"], answer.clone()));
    assert_eq!(core.counters(false), Counters { k1_queries: 1, ..Counters::default() });
    assert_eq!((sub.retained(), sub.generation(), sub.txn_open()), (retained, generation, false));
    assert_eq!(core.snapshot().unwrap(), durable);
    // Deterministic and repeatable: the same state answers the same way.
    assert_eq!(query(&mut core, &mut sub, key(1), 3).unwrap(), (answer, generation));
    assert_eq!((core.snapshot().unwrap(), core.fault()), (durable, None));
}

#[test]
fn a_query_never_publishes_and_never_anchors() {
    let dir = Dir::new("query-unanchored");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let durable = core.snapshot().unwrap();
    assert_eq!(query(&mut core, &mut sub, key(1), 2).unwrap_err(), Code::Unanchored.into());
    assert!(sub.calls().is_empty());
    assert_eq!((core.snapshot().unwrap(), anchored(&core), core.fault()), (durable, None, None));
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
    assert_eq!((core.fault(), core.snapshot().unwrap()), (Some(Code::StateMismatch), durable));
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
    sub.interlope(); // its generation moved outside RuntimeCore
    let durable = core.snapshot().unwrap();
    let x = query_rows(1);
    let mut out = vec![3.0; 1];
    assert_eq!(core.query(&mut sub, key(1), &x, &mut out, &CEILING).unwrap_err().error, Code::StateMismatch.into());
    assert_eq!(out, [0.0]); // the answer of a state outside the lineage is withheld
    assert_eq!((core.fault(), core.snapshot().unwrap()), (Some(Code::StateMismatch), durable));
}

#[test]
fn a_query_of_a_state_whose_lease_was_taken_fail_stops() {
    let dir = Dir::new("query-stolen");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    sub.steal();
    let refused = core.query(&mut sub, key(1), &query_rows(1), &mut [0.0], &CEILING).unwrap_err();
    assert_eq!((refused.error, refused.k1_status), (Code::StateMismatch.into(), K1_LEASED));
    assert_eq!(core.fault(), Some(Code::StateMismatch));
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
fn excessive_work_is_refused_before_reserve_begin_schedule_and_commit() {
    let dir = Dir::new("fuel");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    let durable = core.snapshot().unwrap();
    let retained = sub.retained();
    let big = input(12, 2); // beyond the stand-in's 8 reserved rows: admitted, it would reserve
    let need = fuel::schedule_units(DIM, WIDTH, &big.schedule).unwrap();
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
        // No reserve, no transaction, no schedule, no commit; at most the read-only shape.
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
    assert!(!sub.calls().contains(&"leased_query"));
    // Within its budget the same schedule runs.
    let mut s3 = vec![0.0; features(DIM)];
    core.turn_begin(&mut sub, key(1), &stimulus(&big), &Budget { max_work_units: need, ..CEILING }, &mut s3).unwrap();
    core.turn_commit(&mut sub, key(1)).unwrap();
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
    let (identity, generation) = core.live_identity(&mut sub, key(1)).unwrap();
    assert_eq!((identity, generation, core.fault()), (sub.identity(), sub.generation(), None));
}

#[test]
fn a_lost_lease_fail_stops_before_any_k1_mutation() {
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
    assert_eq!(sub.calls(), ["txn_begin_at"]); // refused natively: nothing begun, nothing scheduled
    assert_eq!((sub.retained(), core.snapshot().unwrap()), (retained, durable));
    assert_eq!((core.fault(), core.counters(false).publications), (Some(Code::StateMismatch), 0));
    assert_eq!(core.durable_request(&mut sub, key(1)).unwrap_err().error, Code::StateMismatch.into());
}

#[test]
fn a_bound_state_that_moved_is_caught_inside_the_transaction_begin() {
    // A move no guard could refuse (it bypasses the lease): the generation re-established by the begin catches it,
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
    assert_eq!(sub.calls(), ["txn_begin_at", "txn_abort"]);
    assert!(!sub.txn_open());
    assert_eq!(
        (sub.retained(), core.snapshot().unwrap(), core.fault()),
        (retained, durable, Some(Code::StateMismatch))
    );
}

#[test]
fn a_moved_state_is_never_captured_for_durable_publication() {
    let dir = Dir::new("lease-moved-durable");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    sub.interlope();
    assert_eq!(core.durable_request(&mut sub, key(1)).unwrap_err().error, Code::StateMismatch.into());
    assert_eq!((core.fault(), on_disk(&dir)), (Some(Code::StateMismatch), anchor));
    assert_eq!(core.lane().report().last_requested, Ticket::default()); // nothing reached the lane
}

#[test]
fn release_gives_the_state_back_and_rebinding_verifies_it_again() {
    let dir = Dir::new("lease-release");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    publish(&mut core, &mut sub, key(1)).unwrap();
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
fn an_explicit_release_keeps_volatile_work_live_without_publishing_it() {
    let dir = Dir::new("lease-release-volatile");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap(); // volatile: the durable authority is still the anchor
    core.release(&mut sub, key(1)).unwrap();
    assert_eq!((sub.lease(), on_disk(&dir)), (0, anchor)); // a release publishes nothing
                                                           // Rebinding the exact bytes it gave back resumes the live lineage; nothing durable was needed or written.
    let committed = turn(&mut core, &mut sub, key(1)).unwrap();
    assert_eq!((core.fault(), on_disk(&dir)), (None, anchor));
    assert_eq!(committed.live_generation, sub.generation());
    // A move while released is outside the lineage: refused before any claim, and the runtime fail-stops.
    core.release(&mut sub, key(1)).unwrap();
    assert_eq!(sub.unmanaged_learn(), 0);
    sub.clear_calls();
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
    assert_eq!((sub.calls(), sub.lease()), (vec!["shape", "state_digest"], 0));
    // A restart knows only acknowledged state: the released live identity was never persisted.
    drop(core);
    let mut fresh = Fake::new(0.1);
    let mut restarted = dir.core();
    turn(&mut restarted, &mut fresh, key(1)).unwrap();
    assert_eq!(restarted.fault(), None);
}

#[test]
fn after_a_release_the_durable_state_may_be_rebound_and_the_volatile_work_is_abandoned() {
    let dir = Dir::new("lease-release-durable");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    core.release(&mut sub, key(1)).unwrap();
    // The operator binds the durable state instead (as a restart would): accepted; the released work is not.
    let mut durable = Fake::new(0.1);
    for _ in 0..2 {
        turn(&mut core, &mut durable, key(2)).unwrap();
    }
    core.release(&mut durable, key(2)).unwrap(); // now the released identity is two turns past the anchor
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into()); // one turn past it
    assert_eq!((sub.lease(), on_disk(&dir)), (0, anchor));
}

#[test]
fn a_state_with_an_open_unmanaged_transaction_is_not_adopted() {
    let dir = Dir::new("lease-busy");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let (mut token, mut source) = (0u64, 0u64);
    assert_eq!(sub.txn_begin_at(0, &mut source, &mut token), 0); // an unmanaged transaction, left open
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

// -- the durability lane: explicit, off the HOT lane, bounded ----------------------------------------------------

#[test]
fn a_durable_request_captures_at_the_live_generation_and_returns_without_waiting() {
    let dir = Dir::new("durable-request");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    core.testing_durable_hold(true); // the worker takes nothing: the request must not care
    sub.clear_calls();
    let t = core.durable_request(&mut sub, key(1)).unwrap();
    assert_eq!((t.status(), t.generation, t.sequence, t.identity), (Status::Pending, sub.generation(), 1, [0u8; 32]));
    assert_eq!(sub.calls(), ["leased_capture"]); // one synchronous crossing: the unsealed capture
    let r = core.lane().report();
    assert_eq!((r.pending, r.last_requested, r.live_generation), (1, t, sub.generation()));
    assert_eq!(on_disk(&dir), anchor); // nothing durable yet
                                       // A bounded wait times out PENDING; an unknown sequence is refused.
    assert_eq!(core.lane().wait(1, Some(Duration::from_millis(20))).unwrap().status(), Status::Pending);
    assert_eq!(core.lane().wait(9, Some(Duration::ZERO)).unwrap_err(), Rt::Invalid.into());
    // The HOT lane is unaffected by the waiting job: a turn moves the live state on.
    turn(&mut core, &mut sub, key(1)).unwrap();
    core.testing_durable_hold(false);
    let done = core.lane().wait(1, None).unwrap();
    // Acknowledged: continuity names exactly the captured state (the generation it was taken at), not the newer
    // live one; without a checkpoint store only the identity is durable.
    assert_eq!((done.status(), done.recoverable, done.generation), (Status::Acknowledged, 0, t.generation));
    assert_ne!(done.identity, sub.identity());
    assert_eq!(on_disk(&dir).cognition(), Cognition::Anchored(done.identity));
    assert_eq!(core.snapshot().unwrap(), on_disk(&dir));
    assert_eq!(core.lane().report().last_acknowledged, done);
}

#[test]
fn a_waiting_request_is_superseded_never_silently_acknowledged() {
    let dir = Dir::new("durable-coalesce");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    core.testing_durable_hold(true);
    core.counters(true);
    let mut tickets = Vec::new();
    for _ in 0..3 {
        turn(&mut core, &mut sub, key(1)).unwrap();
        tickets.push(core.durable_request(&mut sub, key(1)).unwrap());
    }
    // Capacity one waiting job: the first two were replaced, immediately and visibly.
    for t in &tickets[..2] {
        assert_eq!(core.lane().wait(t.sequence, Some(Duration::ZERO)).unwrap().status(), Status::Superseded);
    }
    core.testing_durable_hold(false);
    let last = core.lane().wait(tickets[2].sequence, None).unwrap();
    assert_eq!((last.status(), last.identity), (Status::Acknowledged, sub.identity()));
    let c = core.counters(false);
    assert_eq!((c.durable_requests, c.durable_superseded, c.durable_acknowledged, c.publications), (3, 2, 1, 1));
}

#[test]
fn the_hot_lane_never_waits_for_a_stalled_durability_worker() {
    let dir = Dir::new("durable-stalled");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    core.testing_durable_hold(true);
    let pending = core.durable_request(&mut sub, key(1)).unwrap();
    let start = Instant::now();
    for n in 0..120 {
        let c = turn(&mut core, &mut sub, key(1)).unwrap();
        assert_eq!(c.live_generation, sub.generation());
        let (answer, g) = query(&mut core, &mut sub, key(1), 2).unwrap();
        assert_eq!((answer, g), (sub.answer(&query_rows(2)), sub.generation()), "transition {n}");
        if n % 30 == 0 {
            core.durable_request(&mut sub, key(1)).unwrap(); // supersedes, never waits
        }
    }
    assert!(start.elapsed() < Duration::from_secs(20)); // bounded by the stand-in's work, not by the stalled lane
    assert_eq!((on_disk(&dir), core.snapshot().unwrap(), core.fault()), (anchor, anchor, None));
    assert_eq!(core.lane().wait(pending.sequence, Some(Duration::ZERO)).unwrap().status(), Status::Superseded);
    core.testing_durable_hold(false);
    let last = core.lane().report().last_requested;
    assert_eq!(core.lane().wait(last.sequence, None).unwrap().status(), Status::Acknowledged);
}

#[test]
fn a_republication_of_the_durable_state_writes_nothing() {
    let dir = Dir::new("durable-idempotent");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    let first = publish(&mut core, &mut sub, key(1)).unwrap();
    let durable = on_disk(&dir);
    let again = publish(&mut core, &mut sub, key(1)).unwrap();
    assert_eq!((again.status(), again.identity), (Status::Acknowledged, first.identity));
    assert_eq!(on_disk(&dir), durable); // no new generation
}

#[test]
fn close_abandons_a_waiting_request_and_restart_sees_the_previous_authority() {
    let dir = Dir::new("durable-close");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    core.testing_durable_hold(true);
    let t = core.durable_request(&mut sub, key(1)).unwrap();
    let lane = core.lane();
    core.close();
    let abandoned = lane.wait(t.sequence, Some(Duration::ZERO)).unwrap();
    assert_eq!((abandoned.status(), abandoned.code), (Status::Abandoned, Rt::Closed as i32));
    assert_eq!(on_disk(&dir), anchor);
    core.open().unwrap();
    assert_eq!(core.snapshot().unwrap(), anchor);
    assert_eq!(core.durable_request(&mut sub, key(1)).unwrap_err().error, Code::StateMismatch.into());
}

#[test]
fn a_composite_without_a_checkpoint_store_is_never_captured_blind() {
    let dir = Dir::new("durable-composite");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    core.anchor(&mut sub, key(1)).unwrap();
    core.release(&mut sub, key(1)).unwrap();
    // A composite's envelope size depends on its memory capacity, which only a provisioned store states: refused,
    // never guessed.
    let composite = Key { kind: crate::substrate::KIND_COMPOSITE, ..key(1) };
    assert_eq!(core.durable_request(&mut sub, composite).unwrap_err().error, Rt::CheckpointInvalid.into());
}

// -- K1 Recovery R0: the bounded checkpoint store, written only by explicit durability --------------------------

mod recovery {
    use super::*;
    use crate::checkpoint::{self, Disposition, HEADER, SLOT_NAMES};

    pub struct Ck(pub PathBuf);

    impl Ck {
        pub fn new(name: &str) -> Ck {
            let p = std::env::temp_dir().join(format!("elpis-k1ck-{}-{}", std::process::id(), name));
            let _ = std::fs::remove_dir_all(&p);
            checkpoint::Store::provision(&p, ENVELOPE).unwrap();
            Ck(p)
        }

        pub fn slot_bytes(&self, i: usize) -> Vec<u8> {
            std::fs::read(self.0.join(SLOT_NAMES[i])).unwrap()
        }

        fn write_slot(&self, i: usize, bytes: &[u8]) {
            std::fs::write(self.0.join(SLOT_NAMES[i]), bytes).unwrap();
        }
    }

    impl Drop for Ck {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    pub fn core(dir: &Dir, ck: &Ck) -> Core {
        let mut c = dir.core();
        c.checkpoint_attach(&ck.0).unwrap();
        c
    }

    fn resume(core: &mut Core) -> (Disposition, Option<Vec<u8>>) {
        let r = core.recover().unwrap();
        (r.disposition, r.envelope)
    }

    #[test]
    fn provisioning_is_fixed_and_never_overwrites() {
        let ck = Ck::new("provision");
        assert_eq!(ck.slot_bytes(0), vec![0u8; HEADER + ENVELOPE]);
        assert_eq!(ck.slot_bytes(1).len(), HEADER + ENVELOPE);
        assert_eq!(checkpoint::Store::provision(&ck.0, ENVELOPE).unwrap_err(), Rt::CheckpointInvalid);
        assert_eq!(
            checkpoint::Store::provision(std::path::Path::new("relative"), ENVELOPE).unwrap_err(),
            Rt::CheckpointInvalid
        );
        std::fs::write(ck.0.join("stray"), b"x").unwrap(); // anything but the two slots refuses the store
        assert!(checkpoint::Store::open(&ck.0).is_err());
        std::fs::remove_file(ck.0.join("stray")).unwrap();
        ck.write_slot(1, &[0u8; 10]); // unequal sizes refuse it too
        assert!(checkpoint::Store::open(&ck.0).is_err());
    }

    #[test]
    fn volatile_turns_never_touch_the_slots_and_durable_publications_keep_the_state_resumable() {
        let (dir, ck) = (Dir::new("ck-learn"), Ck::new("learn"));
        let mut sub = Fake::new(0.1);
        let mut c = core(&dir, &ck);
        c.anchor(&mut sub, key(1)).unwrap();
        let (d, envelope) = resume(&mut c);
        assert_eq!(d, Disposition::Resumable);
        assert_eq!(checkpoint::envelope_identity(&envelope.unwrap()), Some(sub.identity()));
        for n in 0..5 {
            let slots = (ck.slot_bytes(0), ck.slot_bytes(1));
            for _ in 0..3 {
                turn(&mut c, &mut sub, key(1)).unwrap();
            }
            assert_eq!((ck.slot_bytes(0), ck.slot_bytes(1)), slots, "volatile turns wrote a slot ({n})");
            let t = publish(&mut c, &mut sub, key(1)).unwrap();
            assert_eq!((t.status(), t.recoverable, t.identity), (Status::Acknowledged, 1, sub.identity()));
            let (d, envelope) = resume(&mut c);
            assert_eq!(d, Disposition::Resumable, "publication {n}");
            assert_eq!(envelope.unwrap(), sub.envelope());
        }
        // Physically bounded: exactly two slot files of the provisioned size, whatever the number of publications.
        let mut names: Vec<_> = std::fs::read_dir(&ck.0).unwrap().map(|e| e.unwrap().file_name()).collect();
        names.sort();
        assert_eq!(names, SLOT_NAMES.map(std::ffi::OsString::from));
        assert!(SLOT_NAMES
            .iter()
            .all(|n| std::fs::metadata(ck.0.join(n)).unwrap().len() == (HEADER + ENVELOPE) as u64));
    }

    #[test]
    fn restart_resumes_exactly_the_last_acknowledged_state_never_a_later_volatile_one() {
        let (dir, ck) = (Dir::new("ck-restart"), Ck::new("restart"));
        let mut sub = Fake::new(0.1);
        let acknowledged = {
            let mut c = core(&dir, &ck);
            c.anchor(&mut sub, key(1)).unwrap();
            turn(&mut c, &mut sub, key(1)).unwrap();
            let t = publish(&mut c, &mut sub, key(1)).unwrap();
            turn(&mut c, &mut sub, key(1)).unwrap(); // volatile, then the process ends
            turn(&mut c, &mut sub, key(1)).unwrap();
            t
        };
        let mut c = core(&dir, &ck);
        let r = c.recover().unwrap();
        assert_eq!((r.disposition, r.authorized), (Disposition::Resumable, acknowledged.identity));
        assert_eq!(checkpoint::envelope_identity(r.envelope.as_ref().unwrap()), Some(acknowledged.identity));
        assert_ne!(sub.identity(), acknowledged.identity); // the volatile state is not what restart resumes
    }

    #[test]
    fn checkpoint_bytes_never_authorize_themselves() {
        let (dir, ck) = (Dir::new("ck-self"), Ck::new("self"));
        let mut sub = Fake::new(0.1);
        {
            // Unanchored: a valid slot (written straight into the store) is never a lineage.
            let mut store = checkpoint::Store::open(&ck.0).unwrap();
            let envelope = envelope_of(&sub.state().w, 7);
            store.envelope_buffer().copy_from_slice(&envelope);
            store.write(None).unwrap();
            drop(store);
            let mut c = dir.core();
            c.checkpoint_attach(&ck.0).unwrap();
            let r = c.recover().unwrap();
            assert_eq!(r.disposition, Disposition::NothingToRecover);
        }
        let mut c = core(&dir, &ck);
        c.anchor(&mut sub, key(1)).unwrap();
        turn(&mut c, &mut sub, key(1)).unwrap();
        publish(&mut c, &mut sub, key(1)).unwrap();
        drop(c);
        // A checksum-valid slot of another identity replaces the authorized one: not resumable.
        let other = Fake::new(0.7);
        let forged = envelope_of(&other.state().w, 99);
        let held = (0..2).find(|&i| ck.slot_bytes(i)[32..64] == sub.identity()).unwrap();
        let raw = ck.slot_bytes(held);
        let generation = u64::from_le_bytes(raw[16..24].try_into().unwrap());
        let mut slot = vec![0u8; HEADER];
        slot[..8].copy_from_slice(b"ELPK1CK\x01");
        slot[8..10].copy_from_slice(&1u16.to_le_bytes());
        slot[16..24].copy_from_slice(&generation.to_le_bytes());
        slot[24..32].copy_from_slice(&(ENVELOPE as u64).to_le_bytes());
        slot[32..64].copy_from_slice(&forged[ENVELOPE - 32..]);
        let sum = elpis_continuity::sha256::digest(&[b"elpis.k1-checkpoint.slot.v1\0", &slot[..96], &forged]);
        slot[96..128].copy_from_slice(&sum);
        slot.extend_from_slice(&forged);
        ck.write_slot(held, &slot);
        let mut c = core(&dir, &ck);
        let r = c.recover().unwrap();
        assert_eq!((r.disposition, r.envelope), (Disposition::CheckpointMissing, None));
        assert_eq!(r.authorized, sub.identity());
    }

    #[test]
    fn a_torn_or_corrupted_slot_is_not_a_checkpoint() {
        let (dir, ck) = (Dir::new("ck-torn"), Ck::new("torn"));
        let mut sub = Fake::new(0.1);
        let mut c = core(&dir, &ck);
        c.anchor(&mut sub, key(1)).unwrap();
        turn(&mut c, &mut sub, key(1)).unwrap();
        publish(&mut c, &mut sub, key(1)).unwrap();
        drop(c);
        let held = (0..2).find(|&i| ck.slot_bytes(i)[32..64] == sub.identity()).unwrap();
        let other = 1 - held;
        let mut torn = ck.slot_bytes(other);
        torn[HEADER + 20] ^= 0xff; // the older slot, corrupted: never a candidate
        ck.write_slot(other, &torn);
        let mut c = core(&dir, &ck);
        assert_eq!(resume(&mut c).0, Disposition::Resumable);
        drop(c);
        let mut torn = ck.slot_bytes(held);
        torn[HEADER + 20] ^= 0xff; // the authorized slot itself: the authorized state is not recoverable here
        ck.write_slot(held, &torn);
        let mut c = core(&dir, &ck);
        assert_eq!(resume(&mut c).0, Disposition::CheckpointMissing);
    }

    #[test]
    fn a_state_of_another_shape_is_refused_never_resized_for() {
        let (dir, ck) =
            (Dir::new("ck-shape"), Ck(std::env::temp_dir().join(format!("elpis-k1ck-{}-shape", std::process::id()))));
        let _ = std::fs::remove_dir_all(&ck.0);
        checkpoint::Store::provision(&ck.0, ENVELOPE + 8).unwrap();
        let mut sub = Fake::new(0.1);
        let mut c = core(&dir, &ck);
        assert_eq!(c.anchor(&mut sub, key(1)).unwrap_err().error, Rt::CheckpointShape.into());
        assert_eq!((anchored(&c), sub.lease()), (None, 0)); // nothing anchored, the claim given back
        assert_eq!(std::fs::metadata(ck.0.join(SLOT_NAMES[0])).unwrap().len(), (HEADER + ENVELOPE + 8) as u64);
        // A smaller store refuses the capture itself: still CHECKPOINT_SHAPE, never written short.
        let small = Ck(std::env::temp_dir().join(format!("elpis-k1ck-{}-shape-small", std::process::id())));
        let _ = std::fs::remove_dir_all(&small.0);
        checkpoint::Store::provision(&small.0, ENVELOPE - 8).unwrap();
        let dir2 = Dir::new("ck-shape-small");
        let mut c = core(&dir2, &small);
        assert_eq!(c.anchor(&mut sub, key(1)).unwrap_err().error, Rt::CheckpointShape.into());
    }

    #[cfg(feature = "testing")]
    pub mod crashes {
        use super::*;
        use crate::durable::{Plan, Step};

        /// D1 durable (anchor, a turn, an acknowledged publication), then two volatile turns and a request for D3
        /// under `plan`. Returns (D1, D3, the request's final ticket).
        fn interrupted(dir: &Dir, ck: &Ck, sub: &mut Fake, plan: Plan) -> (Digest, Digest, Ticket) {
            let mut c = core(dir, ck);
            c.anchor(sub, key(1)).unwrap();
            turn(&mut c, sub, key(1)).unwrap();
            let d1 = publish(&mut c, sub, key(1)).unwrap().identity;
            turn(&mut c, sub, key(1)).unwrap();
            turn(&mut c, sub, key(1)).unwrap();
            let d3 = sub.identity();
            c.testing_durable_plan(Some(plan));
            let t = publish(&mut c, sub, key(1)).unwrap();
            (d1, d3, t) // a died process ends here: the core and the in-memory state are gone
        }

        #[test]
        fn every_durable_step_resolves_to_an_exact_state_or_an_explicit_disposition() {
            use Disposition::*;
            // (step, torn bytes, the request's status, what restart decides, whether restart sees D3)
            let cases = [
                (Step::Taken, 0, Status::Abandoned, Resumable, false),
                (Step::Sealed, 0, Status::Abandoned, Resumable, false),
                (Step::SlotTorn, 77, Status::Abandoned, Resumable, false), // a torn slot is not a checkpoint
                (Step::SlotWritten, 0, Status::Abandoned, CandidateUnresolved, false),
                (Step::SlotSynced, 0, Status::Abandoned, CandidateUnresolved, false),
                (Step::RecordTorn, 50, Status::Abandoned, CandidateUnresolved, false),
                (Step::RecordWritten, 0, Status::Abandoned, Resumable, true), // an unsynced record survives a crash
                (Step::RecordSynced, 0, Status::Abandoned, Resumable, true),
                (Step::Acknowledged, 0, Status::Acknowledged, Resumable, true),
            ];
            for (step, arg, status, expected, newer) in cases {
                let name = format!("{step:?}");
                let (dir, ck) = (Dir::new(&format!("ck-die-{name}")), Ck::new(&format!("die-{name}")));
                let mut sub = Fake::new(0.1);
                let (d1, d3, t) = interrupted(&dir, &ck, &mut sub, Plan::At { step, park: false, arg });
                assert_eq!(t.status(), status, "{name}");
                let mut c = core(&dir, &ck);
                let r = c.recover().unwrap();
                assert_eq!(r.disposition, expected, "{name}");
                let want = if newer { d3 } else { d1 };
                assert_eq!(r.authorized, want, "{name}");
                match expected {
                    Resumable => {
                        // Exactly the authorized state, from its complete checkpoint; never the other one.
                        assert_eq!(checkpoint::envelope_identity(&r.envelope.unwrap()), Some(want), "{name}");
                    }
                    _ => {
                        // Continuity holds D1, a complete D3 exists: never rolled forward or back silently.
                        assert_eq!((r.candidate, r.authorized_present, r.envelope), (d3, true, None), "{name}");
                    }
                }
            }
        }

        #[test]
        fn a_simulated_death_on_the_lane_stops_the_hot_lane_too() {
            let (dir, ck) = (Dir::new("ck-death-hot"), Ck::new("death-hot"));
            let mut sub = Fake::new(0.1);
            let mut c = core(&dir, &ck);
            c.anchor(&mut sub, key(1)).unwrap();
            c.testing_durable_plan(Some(Plan::At { step: Step::Sealed, park: false, arg: 0 }));
            publish(&mut c, &mut sub, key(1)).unwrap();
            assert_eq!(c.fault(), Some(Code::TestingProcessDeath)); // the process is "gone"
            assert_eq!(turn(&mut c, &mut sub, key(1)).unwrap_err(), Code::TestingProcessDeath.into());
            assert_eq!(query(&mut c, &mut sub, key(1), 1).unwrap_err(), Code::TestingProcessDeath.into());
        }

        #[test]
        fn io_failures_are_definitive_and_leave_the_previous_authority() {
            for (plan, code) in [
                (Plan::SlotWriteFail, Rt::CheckpointIo as i32),
                (Plan::SlotTornFail(90), Rt::CheckpointIo as i32),
                (Plan::SlotSyncFail, Rt::CheckpointIo as i32),
                (Plan::RecordWriteFail, Code::PublicationRefused as i32),
                (Plan::RecordTornFail(60), Code::PublicationRefused as i32),
            ] {
                let name = format!("{plan:?}");
                let (dir, ck) = (Dir::new(&format!("ck-fail-{name}")), Ck::new(&format!("fail-{name}")));
                let mut sub = Fake::new(0.1);
                let (d1, _, t) = interrupted(&dir, &ck, &mut sub, plan);
                assert_eq!((t.status(), t.code), (Status::Failed, code), "{name}");
                // Nothing durable changed and the failed write was withdrawn: restart resumes D1 exactly.
                let mut c = core(&dir, &ck);
                let r = c.recover().unwrap();
                assert_eq!((r.disposition, r.authorized), (Disposition::Resumable, d1), "{name}");
            }
        }

        #[test]
        fn a_failed_request_never_stops_the_hot_lane_and_the_next_request_succeeds() {
            let (dir, ck) = (Dir::new("ck-retry"), Ck::new("retry"));
            let mut sub = Fake::new(0.1);
            let mut c = core(&dir, &ck);
            c.anchor(&mut sub, key(1)).unwrap();
            let anchor = on_disk(&dir);
            for plan in [Plan::SlotWriteFail, Plan::SlotTornFail(33), Plan::SlotSyncFail, Plan::RecordWriteFail] {
                turn(&mut c, &mut sub, key(1)).unwrap();
                c.testing_durable_plan(Some(plan));
                let failed = publish(&mut c, &mut sub, key(1)).unwrap();
                assert_eq!(failed.status(), Status::Failed, "{plan:?}");
                assert_eq!((on_disk(&dir), c.snapshot().unwrap(), c.fault()), (anchor, anchor, None), "{plan:?}");
                // The failed write left no unresolved candidate (withdrawn): durability is not stuck.
                assert_eq!(c.recover().unwrap().disposition, Disposition::Resumable, "{plan:?}");
                let (answer, _) = query(&mut c, &mut sub, key(1), 1).unwrap();
                assert_eq!(answer, sub.answer(&query_rows(1)));
            }
            let ok = publish(&mut c, &mut sub, key(1)).unwrap();
            assert_eq!((ok.status(), ok.recoverable, ok.identity), (Status::Acknowledged, 1, sub.identity()));
            assert_eq!(c.recover().unwrap().disposition, Disposition::Resumable);
        }

        #[test]
        fn an_uncertain_publication_faults_the_lane_until_restart_and_the_hot_lane_continues() {
            for (plan, durable_new) in [(Plan::RecordSyncLost, false), (Plan::RecordSyncDurable, true)] {
                let name = format!("{plan:?}");
                let (dir, ck) = (Dir::new(&format!("ck-uncertain-{name}")), Ck::new(&format!("uncertain-{name}")));
                let mut sub = Fake::new(0.1);
                let mut c = core(&dir, &ck);
                c.anchor(&mut sub, key(1)).unwrap();
                let d0 = sub.identity();
                turn(&mut c, &mut sub, key(1)).unwrap();
                let d1 = sub.identity();
                c.testing_durable_plan(Some(plan));
                let t = publish(&mut c, &mut sub, key(1)).unwrap();
                assert_eq!((t.status(), t.code), (Status::Uncertain, Code::PublicationUncertain as i32), "{name}");
                // The lane is faulted (sticky); its durable view is unknown; the HOT lane is not stopped.
                assert_eq!(c.snapshot().unwrap_err(), Code::PublicationUncertain.into());
                let r = c.durable_request(&mut sub, key(1)).unwrap_err();
                assert_eq!(r.error, Code::PublicationUncertain.into());
                assert_eq!(c.fault(), None);
                turn(&mut c, &mut sub, key(1)).unwrap();
                query(&mut c, &mut sub, key(1), 2).unwrap();
                assert_eq!(c.lane().report().fault, Code::PublicationUncertain as i32);
                drop(c);
                // Restart resolves to exactly one complete record.
                let mut c = core(&dir, &ck);
                let r = c.recover().unwrap();
                if durable_new {
                    assert_eq!((r.disposition, r.authorized), (Disposition::Resumable, d1), "{name}");
                } else {
                    assert_eq!((r.disposition, r.authorized, r.candidate), (Disposition::CandidateUnresolved, d0, d1));
                }
            }
        }

        #[test]
        fn an_unresolved_candidate_waits_for_the_operator() {
            let (dir, ck) = (Dir::new("ck-unresolved"), Ck::new("unresolved"));
            let mut sub = Fake::new(0.1);
            let (d1, d3, _) =
                interrupted(&dir, &ck, &mut sub, Plan::At { step: Step::SlotSynced, park: false, arg: 0 });
            let mut c = core(&dir, &ck);
            let r = c.recover().unwrap();
            assert_eq!((r.disposition, r.authorized, r.candidate), (Disposition::CandidateUnresolved, d1, d3));
            assert_eq!(c.checkpoint_discard(&d1).unwrap_err().error, Rt::CheckpointInvalid.into());
            c.checkpoint_discard(&d3).unwrap();
            assert_eq!(c.recover().unwrap().disposition, Disposition::Resumable);
        }

        #[test]
        fn an_operator_discards_or_adopts_an_unresolved_candidate_explicitly() {
            for adopt in [false, true] {
                let (dir, ck) = (Dir::new(&format!("ck-resolve-{adopt}")), Ck::new(&format!("resolve-{adopt}")));
                let mut sub = Fake::new(0.1);
                let plan = Plan::At { step: Step::SlotSynced, park: false, arg: 0 };
                let (d1, d3, _) = interrupted(&dir, &ck, &mut sub, plan);
                let mut c = core(&dir, &ck);
                if adopt {
                    let s = c.checkpoint_adopt(&d3).unwrap();
                    assert_eq!(s.cognition(), Cognition::Anchored(d3));
                } else {
                    c.checkpoint_discard(&d3).unwrap();
                }
                let r = c.recover().unwrap();
                let want = if adopt { d3 } else { d1 };
                assert_eq!((r.disposition, r.authorized), (Disposition::Resumable, want));
                assert_eq!(checkpoint::envelope_identity(&r.envelope.unwrap()), Some(want));
            }
        }

        #[test]
        fn operator_reconciliation_refuses_while_a_publication_is_in_flight() {
            let (dir, ck) = (Dir::new("ck-busy"), Ck::new("busy"));
            let mut sub = Fake::new(0.1);
            let mut c = core(&dir, &ck);
            c.anchor(&mut sub, key(1)).unwrap();
            turn(&mut c, &mut sub, key(1)).unwrap();
            c.testing_durable_hold(true);
            c.durable_request(&mut sub, key(1)).unwrap();
            assert_eq!(c.checkpoint_discard(&[1u8; 32]).unwrap_err().error, Rt::DurableBusy.into());
            c.release(&mut sub, key(1)).unwrap();
            assert_eq!(c.checkpoint_adopt(&[1u8; 32]).unwrap_err().error, Rt::DurableBusy.into());
            c.testing_durable_hold(false);
        }

        #[test]
        fn the_hot_lane_performs_no_continuity_or_checkpoint_io() {
            let (dir, ck) = (Dir::new("ck-io"), Ck::new("io"));
            let mut sub = Fake::new(0.1);
            let mut c = core(&dir, &ck);
            c.anchor(&mut sub, key(1)).unwrap();
            turn(&mut c, &mut sub, key(1)).unwrap();
            c.testing_continuity_counters(true);
            c.testing_checkpoint_counters(true);
            for _ in 0..100 {
                turn(&mut c, &mut sub, key(1)).unwrap();
                query(&mut c, &mut sub, key(1), 2).unwrap();
            }
            let (io, ck_io) = (c.testing_continuity_counters(false), c.testing_checkpoint_counters(false));
            assert_eq!(io, Default::default(), "continuity I/O on the HOT lane");
            assert_eq!(ck_io, Default::default(), "checkpoint I/O on the HOT lane");
            // One explicit durable publication: one slot write + sync, one record write + sync.
            publish(&mut c, &mut sub, key(1)).unwrap();
            let (io, ck_io) = (c.testing_continuity_counters(false), c.testing_checkpoint_counters(false));
            assert_eq!((io.pwrites, io.pwrite_bytes, io.data_syncs, io.opens, io.renames), (1, 176, 1, 0, 0));
            assert_eq!((ck_io.pwrites, ck_io.pwrite_bytes, ck_io.data_syncs), (1, (HEADER + ENVELOPE) as u64, 1));
        }
    }
}

#[cfg(feature = "testing")]
mod faults {
    use super::*;

    const DIE: u32 = 1;
    const WRITE_FAIL: u32 = 2;

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
    fn process_death_during_an_evolution_publication_fail_stops() {
        let dir = Dir::new("death");
        let mut core = dir.core();
        let authority = core.evolution_authority().unwrap().evolution();
        core.testing_arm(1, DIE, 1).unwrap(); // after the record is written, before it is synced
        assert_eq!(core.evolution_reserve(&authority, &[1u8; 32]).unwrap_err(), Code::TestingProcessDeath.into());
        assert_eq!(core.fault(), Some(Code::TestingProcessDeath));
    }

    #[test]
    fn a_durable_publication_without_a_store_writes_one_record_and_syncs_once() {
        let dir = Dir::new("io");
        let mut core = dir.core();
        let mut sub = Fake::new(0.1);
        core.anchor(&mut sub, key(1)).unwrap();
        turn(&mut core, &mut sub, key(1)).unwrap();
        core.testing_continuity_counters(true);
        turn(&mut core, &mut sub, key(1)).unwrap();
        assert_eq!(core.testing_continuity_counters(false), Default::default()); // the turn: nothing
        publish(&mut core, &mut sub, key(1)).unwrap();
        let io = core.testing_continuity_counters(false);
        assert_eq!((io.opens, io.preads, io.pwrites, io.pwrite_bytes, io.data_syncs), (0, 0, 1, 176, 1));
        assert_eq!((io.dir_syncs, io.renames, io.unlinks), (0, 0, 0));
    }
}
