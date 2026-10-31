//! RuntimeCore laws against a deterministic in-memory stand-in for one native K1 state.
//!
//! The stand-in follows the refusal contract of `ecsg_k1.h` (CAPACITY/INVALID keep the transaction open,
//! NONFINITE from the schedule and STALE discard it, a commit installs the complete candidate or nothing) and its
//! live coordinate (a generation advanced by exactly one per committed transition, reported under the lease) but
//! computes nothing of K1's mathematics; the real native state is exercised by the C ABI tests and the Python
//! integration lane. The suite runs over the volatile ECS_C session (ABI v8) and reads it back through the runtime;
//! no file, directory or feature is involved.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::authority::{Cognition, EvolutionState, SessionTemplate, Snapshot, StateFormat};
use crate::code::{Code, Error, Rt};
use crate::core::{Core, Counters, Stimulus, VolatileCommit};
use crate::fuel::{self, Budget, CEILING};
use crate::substrate::{
    features, Digest, Experience, K1Ops, Key, Observation, ResidentRelease, ScheduleOutcome, Transition, TxnAbort,
    K1_BUSY, K1_CAPACITY, K1_FMS_NOT_RESIDENT, K1_INVALID, K1_LEASED, K1_NONFINITE, K1_STALE,
};
use elpis_ecs_c_session::model as m;

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
    /// The memory candidate of a composite stand-in (the committed memory, plus this transaction's observations).
    memory: Vec<Digest>,
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
    /// Residency of an FMS-resident stand-in: `fms` states have a tier (WARM or demoted), residency holds, and a
    /// count of the materializations an operation performed (the synchronous promotion a HOT path must never do).
    fms: bool,
    warm: bool,
    holds: u32,
    promotions: u64,
    /// A composite stand-in: its retained state is (W, epoch) plus a memory of recorded observation keys, under one
    /// identity (as ELPISEC1 binds the K1 trailer and the memory).
    composite: bool,
    memory: Vec<Digest>,
    /// A faulty composite whose commit reports K1 learning (an epoch step) it never ran.
    learn_on_commit: bool,
}

impl State {
    /// An FMS operation pin: without a hold, a demoted state is materialized synchronously (ABI v1 behavior); under a
    /// hold it is resident by construction (the hold pins it).
    fn pin(&mut self) {
        if self.fms && !self.warm {
            assert_eq!(self.holds, 0, "a held state was demoted");
            self.promotions += 1;
            self.warm = true;
        }
    }
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
    let trailer = elpis_ecs_c_session::sha256::digest(&[&e[..n - 32]]);
    e[n - 32..].copy_from_slice(&trailer);
}

impl State {
    /// The retained-state identity: the envelope's SHA-256 trailer (K1's law); a composite's binds the K1 trailer
    /// and every memory key together.
    fn identity(&self) -> Digest {
        let e = envelope_of(&self.w, self.epoch);
        let mut out = [0u8; 32];
        out.copy_from_slice(&e[ENVELOPE - 32..]);
        if self.composite {
            let mut parts: Vec<&[u8]> = vec![b"composite", &out];
            parts.extend(self.memory.iter().map(|k| &k[..]));
            return elpis_ecs_c_session::sha256::digest(&parts);
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
            fms: false,
            warm: true,
            holds: 0,
            promotions: 0,
            composite: false,
            memory: Vec::new(),
            learn_on_commit: false,
        })))
    }

    /// A composite stand-in (K1 plus a memory; `ELPIS_RUNTIME_SUBSTRATE_COMPOSITE`).
    fn composite(seed: f64) -> Fake {
        let f = Fake::new(seed);
        f.state().composite = true;
        f
    }

    fn memory(&self) -> Vec<Digest> {
        self.state().memory.clone()
    }

    /// An FMS-resident stand-in (WARM until demoted).
    fn resident(seed: f64) -> Fake {
        let f = Fake::new(seed);
        f.state().fms = true;
        f
    }

    /// FMS demotion: refused (false) while a hold pins the state.
    fn demote(&self) -> bool {
        let mut s = self.state();
        if s.holds > 0 {
            return false;
        }
        s.warm = false;
        true
    }

    /// The explicit WARM-lane materialization (never a HOT operation's).
    fn materialize(&self) {
        let mut s = self.state();
        s.promotions += 1;
        s.warm = true;
    }

    fn holds(&self) -> u32 {
        self.state().holds
    }

    fn promotions(&self) -> u64 {
        self.state().promotions
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

struct FakeResident(Arc<Mutex<State>>);

impl ResidentRelease for FakeResident {
    fn resident_release(&mut self) -> i32 {
        let mut s = self.0.lock().unwrap();
        if s.holds == 0 {
            return K1_INVALID;
        }
        s.holds -= 1;
        0
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
        s.pin();
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
        s.pin();
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
        s.pin();
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
        s.pin();
        if s.lease != lease {
            return K1_LEASED;
        }
        if s.txn.is_some() {
            return K1_BUSY;
        }
        *token = s.next_token;
        *source_generation = s.generation;
        s.next_token += 1;
        let txn = Txn {
            token: *token,
            lease,
            generation: s.generation,
            w: s.w.clone(),
            epoch: s.epoch,
            memory: s.memory.clone(),
        };
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
        s.epoch = txn.epoch + u64::from(s.learn_on_commit);
        s.memory = txn.memory;
        s.generation += 1;
        out.epoch_after = s.epoch;
        out.generation_after = s.generation;
        out.steps = out.epoch_after - out.epoch_before;
        0
    }

    fn txn_abort(&mut self, token: u64) -> i32 {
        self.state().abort(token)
    }

    fn retain_abort(&self) -> Box<dyn TxnAbort> {
        Box::new(FakeAbort(Arc::clone(&self.0)))
    }

    fn resident_hold(&mut self) -> i32 {
        let mut s = self.state();
        if !s.fms {
            return 0;
        }
        s.calls.push("resident_hold");
        if !s.warm {
            return K1_FMS_NOT_RESIDENT; // never promoted here
        }
        s.holds += 1;
        0
    }

    fn retain_resident(&self) -> Option<Box<dyn ResidentRelease>> {
        self.state().fms.then(|| Box::new(FakeResident(Arc::clone(&self.0))) as Box<dyn ResidentRelease>)
    }

    fn format(&self) -> StateFormat {
        if self.state().composite {
            StateFormat::Composite
        } else {
            StateFormat::K1
        }
    }

    /// The stand-in's OBSERVE: `observation` points at the 32-byte relation key the test chose (opaque to
    /// RuntimeCore); a key starting 0xFF is refused by the "memory" (composite REFUSED, its own status -7).
    fn txn_observe(
        &mut self,
        lease: u64,
        token: u64,
        observation: Observation,
        key: &mut Digest,
        memory_status: &mut i32,
    ) -> i32 {
        let mut s = self.state();
        s.calls.push("txn_observe");
        if !s.composite {
            return K1_INVALID;
        }
        if s.lease != lease {
            return K1_LEASED;
        }
        let txn = match s.txn.as_mut() {
            Some(t) if t.token == token => t,
            _ => return K1_INVALID,
        };
        // SAFETY: the tests pass a pointer to a live Digest.
        let observed = unsafe { *(observation.0 as *const Digest) };
        if observed[4] == 0xFF {
            *memory_status = -7;
            return COMPOSITE_REFUSED;
        }
        txn.memory.push(observed);
        *key = observed;
        0
    }
}

/// `ELPIS_ECS_COMPOSITE_REFUSED`: the memory refused (its own status reported alongside).
const COMPOSITE_REFUSED: i32 = -10;

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

/// The suite's runtime: sessions of capacity 16, a fixed nonce seed (deterministic sessions).
fn new_core() -> Core {
    Core::new(SessionTemplate { capacity: 16, nonce_seed: [7; 32], operator_label: [1; 32], runtime_label: [2; 32] })
        .expect("valid template")
}

/// The operator founds the suite's source in the open session: one ACTIVE source whose memory source authority is 0
/// (the suite's observations record it), at the bound the managed turn's ceiling allows.
fn found_test_source(core: &mut Core) {
    let op = core.authority_entity(0).unwrap();
    let id = core.authority_next_entity_id(m::EntityKind::Source as u8, &[3; 32]).unwrap();
    let seq = op.sender_watermark;
    let found = m::Event::entity_found(&op.id, seq + 1, &id, m::EntityKind::Source, &[3; 32], &[4; 32], 1, 256, 0);
    core.authority_transition(&found).unwrap();
    let active = m::Event::entity_lifecycle(&op.id, seq + 2, &id, m::Lifecycle::Founded, m::Lifecycle::Active);
    core.authority_transition(&active).unwrap();
}

/// Open `core` (a new session) and found the suite's source.
fn open(core: &mut Core) -> Snapshot {
    core.open().expect("open");
    found_test_source(core);
    core.snapshot().unwrap()
}

/// What the suite once named a deployment directory: v8 has none. `Dir::new(name).core()` is an open runtime over a
/// fresh session with the suite's source.
struct Dir;

impl Dir {
    fn new(_name: &str) -> Dir {
        Dir
    }

    fn core(&self) -> Core {
        let mut c = new_core();
        open(&mut c);
        c
    }
}

fn anchored(core: &Core) -> Option<Digest> {
    match core.snapshot().unwrap().cognition() {
        Cognition::Unanchored => None,
        Cognition::Anchored(d) => Some(d),
    }
}

#[test]
fn features_are_k1s() {
    assert_eq!(
        (features(0), features(1), features(2), features(6), features(64), features(65)),
        (0, 3, 9, 83, 47904, 0)
    );
}

#[test]
fn a_closed_runtime_refuses_everything() {
    let mut core = new_core();
    let mut sub = Fake::new(0.1);
    assert_eq!(core.anchor(&mut sub, key(1)).unwrap_err().error, Rt::Closed.into());
    assert_eq!(core.evolution_authority().unwrap_err(), Rt::Closed.into());
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Rt::Closed.into());
    assert_eq!(core.snapshot().unwrap_err(), Rt::Closed.into());
    assert_eq!(core.authority_entity(0).unwrap_err(), Rt::Closed.into());
    assert!(sub.calls().is_empty());
    // No session exists while closed: the status names none.
    let status = core.authority_status();
    assert_eq!((status.instance, status.volatile_bytes, status.sessions), (0, 0, 0));
    // A capacity outside the session's bounds is refused at create, before any session exists.
    for capacity in [0, 1, 4097] {
        let t = SessionTemplate { capacity, nonce_seed: [0; 32], operator_label: [1; 32], runtime_label: [2; 32] };
        assert_eq!(Core::new(t).err(), Some(Rt::Invalid.into()));
    }
}

#[test]
fn anchor_is_explicit_once_in_the_session_reads_the_state_only_and_binds() {
    let dir = Dir::new("anchor");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let before = sub.retained();
    let initial = core.snapshot().unwrap().generation();
    let s = core.anchor(&mut sub, key(1)).unwrap();
    assert_eq!(s.generation(), initial + 1);
    assert_eq!(anchored(&core), Some(sub.identity()));
    assert_eq!(core.snapshot().unwrap(), s); // the session records the anchor, in memory
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
    // The session's authority did not move: the commit is the live coordinate's alone.
    assert_eq!(core.snapshot().unwrap(), anchor);
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
fn a_mismatched_state_fail_stops_before_any_mutation() {
    let dir = Dir::new("mismatch");
    let mut sub = Fake::new(0.1);
    let mut core = dir.core();
    core.anchor(&mut sub, key(1)).unwrap();
    core.release(&mut sub, key(1)).unwrap(); // the lineage is unbound: the next state is verified against it
    let mut other = Fake::new(0.2);
    let durable = core.snapshot().unwrap();
    let before = other.retained();
    assert_eq!(turn(&mut core, &mut other, key(2)).unwrap_err(), Code::StateMismatch.into());
    // Read-only crossings only: the shape (fuel admission) and the identity; never claimed.
    assert_eq!((other.retained(), other.calls(), other.lease()), (before, vec!["shape", "state_digest"], 0));
    assert_eq!(core.fault(), Some(Code::StateMismatch));
    // Fail-stopped for every lineage-dependent operation, the matching state included; nothing synthesized.
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
    assert_eq!(core.evolution_authority().unwrap_err(), Code::StateMismatch.into());
    assert_eq!(core.snapshot().unwrap(), durable);
    // The fail-stop ends with the session: close + open is a new session (unanchored, nothing resumed), in which the
    // state may be anchored afresh.
    core.close();
    open(&mut core);
    assert_eq!((anchored(&core), core.fault()), (None, None));
    core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
}

#[test]
fn an_open_runtime_refuses_another_state_without_a_native_call() {
    let dir = Dir::new("switch");
    let mut core = dir.core();
    let (mut first, mut second) = (Fake::new(0.1), Fake::new(0.1)); // identical bytes, different owner
    core.anchor(&mut first, key(1)).unwrap();
    assert_eq!(turn(&mut core, &mut second, key(2)).unwrap_err(), Rt::SubstrateSwitch.into());
    assert_eq!(core.live_identity(&mut second, key(2)).unwrap_err().error, Rt::SubstrateSwitch.into());
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
    assert_eq!(core.live_identity(&mut sub, key(1)).unwrap_err().error, Rt::TurnOpen.into());
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
fn evolution_reserves_executes_once_and_finalizes_in_the_session() {
    let dir = Dir::new("evolution");
    let mut core = dir.core();
    let authority = core.evolution_authority().unwrap().evolution();
    let assertion = [7u8; 32];
    let pending = core.evolution_reserve(&authority, &assertion).unwrap().evolution();
    assert_eq!(pending.assertion(), Some(assertion));
    // The reservation is in the session when reserve returns (before the boundary may execute).
    assert_eq!(core.snapshot().unwrap().evolution().assertion(), Some(assertion));
    assert_eq!(core.evolution_reserve(&authority, &assertion).unwrap_err(), Rt::EvolutionInFlight.into());
    let done = core.evolution_finalize(&[9u8; 32]).unwrap().evolution();
    assert_eq!((done.revision(), done.head(), done.assertion()), (1, [9u8; 32], None));
    assert_eq!(core.snapshot().unwrap().evolution(), done);
    assert_eq!(core.evolution_finalize(&[9u8; 32]).unwrap_err(), Rt::EvolutionNotInFlight.into());
    // A stale observation cannot reserve: refused, nothing changed and nothing executed (no fail-stop in v8; the
    // caller reads the authority again).
    assert_eq!(core.evolution_reserve(&authority, &assertion).unwrap_err(), Code::AuthorityMismatch.into());
    assert_eq!((core.fault(), core.snapshot().unwrap().evolution()), (None, done));
    let current = core.evolution_authority().unwrap().evolution();
    core.evolution_reserve(&current, &[8u8; 32]).unwrap();
    assert_eq!(core.evolution_finalize(&[6u8; 32]).unwrap().evolution().revision(), 2);
}

#[test]
fn an_abandoned_attempt_stays_pending_for_the_rest_of_the_session_and_a_new_session_starts_idle() {
    let dir = Dir::new("abandon");
    let assertion = [3u8; 32];
    let mut core = dir.core();
    let authority = core.evolution_authority().unwrap().evolution();
    core.evolution_reserve(&authority, &assertion).unwrap();
    core.evolution_abandon().unwrap();
    assert_eq!(core.fault(), Some(Code::EvolutionPending));
    assert_eq!(core.evolution_authority().unwrap_err(), Code::EvolutionPending.into());
    assert_eq!(core.evolution_abandon().unwrap_err(), Rt::EvolutionNotInFlight.into());
    // The session still holds the pending reservation (readable while fail-stopped); nothing finalizes it.
    assert_eq!(core.snapshot().unwrap().evolution().assertion(), Some(assertion));
    // No reconciliation exists: the session ends, and the next one starts at genesis (no attempt survives it).
    core.close();
    open(&mut core);
    assert_eq!(core.fault(), None);
    let idle = core.evolution_authority().unwrap().evolution();
    assert_eq!((idle.revision(), idle.assertion()), (0, None));
}

#[test]
fn volatile_cognition_never_makes_a_pending_evolution_attempt_retryable() {
    // An attempt reserved, then abandoned (it may have had an external effect). Volatile turns never clear, finalize or
    // retry it: the session carries the pending authority unchanged until it ends.
    let dir = Dir::new("evolution-volatile");
    let mut sub = Fake::new(0.1);
    let assertion = [4u8; 32];
    let mut core = dir.core();
    core.anchor(&mut sub, key(1)).unwrap();
    let authority = core.evolution_authority().unwrap().evolution();
    core.evolution_reserve(&authority, &assertion).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap(); // cognition stays HOT while the attempt is in flight
    core.evolution_abandon().unwrap();
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::EvolutionPending.into());
    let pending = core.snapshot().unwrap().evolution();
    assert_eq!(pending.assertion(), Some(assertion));
    assert_eq!(core.evolution_reserve(&pending, &[8u8; 32]).unwrap_err(), Code::EvolutionPending.into());
    assert_eq!(core.evolution_finalize(&[8u8; 32]).unwrap_err(), Rt::EvolutionNotInFlight.into());
    assert_eq!(core.snapshot().unwrap().evolution(), pending);
}

// -- turn lifecycle: an open native transaction is never forgotten ------------------------------------------------

/// What an implicit abort must leave untouched: the retained K1 state, its generation and digest, and the session.
struct Witness {
    retained: (Vec<u64>, u64),
    generation: u64,
    identity: Digest,
    durable: Snapshot,
}

fn witness(core: &Core, sub: &Fake) -> Witness {
    Witness {
        retained: sub.retained(),
        generation: sub.generation(),
        identity: sub.identity(),
        durable: core.snapshot().unwrap(),
    }
}

fn unchanged(w: &Witness, _dir: &Dir, sub: &Fake) {
    assert_eq!((sub.retained(), sub.generation(), sub.identity()), (w.retained.clone(), w.generation, w.identity));
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
    // A new open is a new session: it knows no lineage, so nothing of the last one is adopted.
    core.open().unwrap();
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::Unanchored.into());
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
    // An evolution attempt is reserved and abandoned: the runtime fail-stops, turn open.
    let authority = core.evolution_authority().unwrap().evolution();
    core.evolution_reserve(&authority, &[1u8; 32]).unwrap();
    core.evolution_abandon().unwrap();
    let durable = core.snapshot().unwrap();
    assert_eq!(core.fault(), Some(Code::EvolutionPending));
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
    let mut sub = Fake::new(0.1);
    let mut core = new_core();
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
    assert_eq!(new_core().counters(false), Counters::default());
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
    let mut core = dir.core();
    core.anchor(&mut sub, key(1)).unwrap();
    core.release(&mut sub, key(1)).unwrap();
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
        assert_eq!(
            (c.k1_reserves, c.k1_txn_begins, c.k1_run_schedules, c.k1_commits, c.session_transitions),
            (0, 0, 0, 0, 0)
        );
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
    assert_eq!((core.fault(), core.counters(false).session_transitions), (Some(Code::StateMismatch), 0));
    assert_eq!(core.live_identity(&mut sub, key(1)).unwrap_err().error, Code::StateMismatch.into());
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
fn an_explicit_release_keeps_volatile_work_live_for_the_session() {
    let dir = Dir::new("lease-release-volatile");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap(); // volatile: the session's identity is still the anchor
    core.release(&mut sub, key(1)).unwrap();
    assert_eq!((sub.lease(), core.snapshot().unwrap()), (0, anchor)); // a release records nothing
                                                                      // Rebinding the exact bytes it gave back resumes the live lineage.
    let committed = turn(&mut core, &mut sub, key(1)).unwrap();
    assert_eq!((core.fault(), core.snapshot().unwrap()), (None, anchor));
    assert_eq!(committed.live_generation, sub.generation());
    // A move while released is outside the lineage: refused before any claim, and the runtime fail-stops.
    core.release(&mut sub, key(1)).unwrap();
    assert_eq!(sub.unmanaged_learn(), 0);
    sub.clear_calls();
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into());
    assert_eq!((sub.calls(), sub.lease()), (vec!["shape", "state_digest"], 0));
    // A new session knows nothing of the last one: unanchored, the released identity gone with it.
    core.close();
    open(&mut core);
    let mut fresh = Fake::new(0.1);
    assert_eq!(turn(&mut core, &mut fresh, key(1)).unwrap_err(), Code::Unanchored.into());
    assert!(fresh.calls().is_empty());
}

#[test]
fn after_a_release_the_anchored_state_may_be_rebound_and_the_volatile_work_is_abandoned() {
    let dir = Dir::new("lease-release-anchored");
    let mut core = dir.core();
    let mut sub = Fake::new(0.1);
    let anchor = core.anchor(&mut sub, key(1)).unwrap();
    turn(&mut core, &mut sub, key(1)).unwrap();
    core.release(&mut sub, key(1)).unwrap();
    // The caller binds a copy of the anchored state instead (an in-memory rollback to the session's identity):
    // accepted; the released work is not.
    let mut kept = Fake::new(0.1);
    for _ in 0..2 {
        turn(&mut core, &mut kept, key(2)).unwrap();
    }
    core.release(&mut kept, key(2)).unwrap(); // now the released identity is two turns past the anchor
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Code::StateMismatch.into()); // one turn past it
    assert_eq!((sub.lease(), core.snapshot().unwrap()), (0, anchor));
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
fn a_demoted_fms_state_is_refused_before_anything_touches_it_and_binds_after_explicit_materialization() {
    let dir = Dir::new("resident-demoted");
    let mut sub = Fake::resident(0.1);
    let mut core = dir.core();
    core.anchor(&mut sub, key(1)).unwrap();
    assert_eq!(sub.holds(), 1); // anchoring binds: the state is held resident from then on
    assert!(!sub.demote(), "a bound state was demoted");
    turn(&mut core, &mut sub, key(1)).unwrap();
    core.release(&mut sub, key(1)).unwrap();
    assert_eq!(sub.holds(), 0); // unbound: demotable again
    assert!(sub.demote());
    sub.clear_calls();
    let touched = |c: Counters| (c.k1_state_digests, c.k1_lease_claims, c.k1_identities, c.k1_queries, c.k1_txn_begins);
    let before = (touched(core.counters(false)), sub.lease(), sub.promotions());
    // The HOT operation is refused before any identity read, claim or pin: nothing is promoted or waited for.
    let r = query(&mut core, &mut sub, key(1), 1).unwrap_err();
    assert_eq!(r, Rt::NotResident.into());
    assert_eq!(turn(&mut core, &mut sub, key(1)).unwrap_err(), Rt::NotResident.into());
    // Only the pin-free shape read (fuel admission) and the refused holds: no identity read, claim or pinned call.
    assert!(sub.calls().iter().all(|c| ["shape", "resident_hold"].contains(c)), "{:?}", sub.calls());
    assert_eq!(sub.calls().iter().filter(|c| **c == "resident_hold").count(), 2);
    assert_eq!((touched(core.counters(false)), sub.lease(), sub.promotions()), before);
    assert_eq!((core.fault(), sub.holds()), (None, 0)); // no fail-stop, no hold left behind
                                                        // The explicit WARM-lane materialization, then the same HOT operation binds and runs without any promotion.
    sub.materialize();
    let promotions = sub.promotions();
    let (answer, _) = query(&mut core, &mut sub, key(1), 1).unwrap();
    assert_eq!(answer, sub.answer(&query_rows(1)));
    turn(&mut core, &mut sub, key(1)).unwrap();
    assert_eq!((sub.promotions(), sub.holds()), (promotions, 1));
}

#[test]
fn the_residency_hold_lives_exactly_as_long_as_the_binding() {
    let dir = Dir::new("resident-lifetime");
    let mut sub = Fake::resident(0.1);
    {
        let mut core = dir.core();
        core.anchor(&mut sub, key(1)).unwrap();
        assert_eq!(sub.holds(), 1);
        core.close();
        assert_eq!(sub.holds(), 0); // close unbinds
        core.open().unwrap(); // a new session: the state is anchored afresh
        core.anchor(&mut sub, key(1)).unwrap(); // binding takes exactly one hold again
        query(&mut core, &mut sub, key(1), 1).unwrap();
        query(&mut core, &mut sub, key(1), 1).unwrap();
        assert_eq!(sub.holds(), 1);
        core.release(&mut sub, key(1)).unwrap();
        core.release(&mut sub, key(1)).unwrap(); // releasing an unbound state drops nothing more
        assert_eq!(sub.holds(), 0);
        query(&mut core, &mut sub, key(1), 1).unwrap();
        assert_eq!(sub.holds(), 1);
    } // destruction unbinds
    assert_eq!(sub.holds(), 0);
    // A bind that fails after the hold (a state that is not the lineage) gives the hold back.
    let mut other = Fake::resident(0.7);
    let mut core = dir.core();
    let mut lineage = Fake::new(0.1);
    core.anchor(&mut lineage, key(4)).unwrap();
    core.release(&mut lineage, key(4)).unwrap();
    assert_eq!(query(&mut core, &mut other, key(2), 1).unwrap_err(), Code::StateMismatch.into());
    assert_eq!((other.holds(), other.lease()), (0, 0));
    // A RAM state takes no hold at all.
    let dir = Dir::new("resident-ram");
    let mut ram = Fake::new(0.1);
    let mut core = dir.core();
    core.anchor(&mut ram, key(3)).unwrap();
    query(&mut core, &mut ram, key(3), 1).unwrap();
    assert!(!ram.calls().contains(&"resident_hold"));
}

// -- the volatile session authority (ABI v8) ---------------------------------------------------------------------

mod session {
    use super::*;

    fn composite_key(owner: u64) -> Key {
        Key { kind: 3, handle: 0x2000 + owner as usize, id: 0, owner, dim: DIM }
    }

    /// A stand-in observation whose head reads as an `elpis_ecs_mem_observation` of ABI v1 recording memory source
    /// authority `source` (bytes 24..28); `byte` tells them apart.
    fn observation(source: u32, byte: u8) -> Digest {
        let mut d = [0u8; 32];
        d[0] = 1;
        d[4] = byte;
        d[24..28].copy_from_slice(&source.to_le_bytes());
        d[31] = 0x5a;
        d
    }

    fn observe(core: &mut Core, sub: &mut Fake, k: Key, d: &Digest) -> Result<Digest, Error> {
        core.turn_observe(sub, k, Observation(d as *const Digest as *const _)).0.map_err(|r| r.error)
    }

    /// The operator's identity and next sender sequence, read from the open session.
    fn operator(core: &Core) -> (Digest, u64) {
        let op = core.authority_entity(0).unwrap();
        (op.id, op.sender_watermark + 1)
    }

    /// The operator founds a source and activates it, through the runtime.
    fn found_source(core: &mut Core, label: u8, source_authority: u32, limit: u32) -> Digest {
        let label = [label; 32];
        let id = core.authority_next_entity_id(m::EntityKind::Source as u8, &label).unwrap();
        let (op, seq) = operator(core);
        let kind = m::EntityKind::Source;
        let found = m::Event::entity_found(&op, seq, &id, kind, &label, &[4; 32], 1, limit, source_authority);
        core.authority_transition(&found).unwrap();
        lifecycle(core, &id, m::Lifecycle::Founded, m::Lifecycle::Active).unwrap();
        id
    }

    fn lifecycle(core: &mut Core, id: &Digest, from: m::Lifecycle, to: m::Lifecycle) -> Result<u64, Error> {
        let (op, seq) = operator(core);
        let event = m::Event::entity_lifecycle(&op, seq, id, from, to);
        core.authority_transition(&event).map(|r| r.generation).map_err(|r| r.error)
    }

    fn template(seed: u8) -> SessionTemplate {
        SessionTemplate { capacity: 16, nonce_seed: [seed; 32], operator_label: [1; 32], runtime_label: [2; 32] }
    }

    #[test]
    fn each_open_is_a_new_session_at_genesis_and_close_ends_it() {
        let mut core = Core::new(template(7)).unwrap();
        let first = core.open().unwrap();
        assert_eq!((first.generation(), first.cognition()), (1, Cognition::Unanchored));
        let s1 = core.authority_status();
        assert_eq!((s1.backend, s1.sessions, s1.generation, s1.entity_count), (3, 1, 1, 2));
        assert_ne!(s1.instance, 0);
        assert_eq!(s1.volatile_bytes, elpis_ecs_c_session::Session::volatile_bytes_for(16) as u64);
        // The session evolves in memory: a source founded, the lineage anchored.
        let mut sub = Fake::new(0.1);
        found_source(&mut core, 9, 0, 8);
        core.anchor(&mut sub, key(1)).unwrap();
        turn(&mut core, &mut sub, key(1)).unwrap();
        assert_eq!(core.authority_status().generation, 4);
        assert_eq!(core.open().unwrap_err(), Code::Open.into()); // open twice: refused, the session kept
                                                                 // Close ends it: nothing of it is kept, and the status names no session.
        core.close();
        let closed = core.authority_status();
        assert_eq!((closed.instance, closed.volatile_bytes, closed.sessions, closed.entity_count), (0, 0, 1, 0));
        // The next open is another session: another lineage and instance, at genesis, unanchored, no source.
        let second = core.open().unwrap();
        assert_eq!((second.generation(), second.cognition()), (1, Cognition::Unanchored));
        let s2 = core.authority_status();
        assert_eq!((s2.sessions, s2.entity_count, s2.generation), (2, 2, 1));
        assert_ne!(s2.session, s1.session);
        assert_ne!(s2.instance, s1.instance);
        assert_eq!(core.authority_entity(2).unwrap_err(), Rt::Invalid.into());
        // Its sources are its own: an observation of the last session's source is refused.
        let (mut comp, k) = (Fake::composite(0.2), composite_key(7));
        core.anchor(&mut comp, k).unwrap();
        core.turn_begin_observe(&mut comp, k, &CEILING).unwrap();
        assert_eq!(observe(&mut core, &mut comp, k, &observation(0, 1)).unwrap_err(), Rt::ObservationRefused.into());
    }

    #[test]
    fn a_nonce_seed_makes_sessions_reproducible_across_runtimes_and_distinct_within_one() {
        let (mut a, mut b, mut c) =
            (Core::new(template(7)).unwrap(), Core::new(template(7)).unwrap(), Core::new(template(8)).unwrap());
        let lineage = |core: &mut Core| {
            core.open().unwrap();
            let s = core.authority_status();
            core.close();
            (s.session, s.state_root)
        };
        let (a1, b1, c1) = (lineage(&mut a), lineage(&mut b), lineage(&mut c));
        assert_eq!(a1, b1); // configured alike: the same first session (differential testing)
        assert_ne!(a1, c1);
        let (a2, b2) = (lineage(&mut a), lineage(&mut b));
        assert_eq!(a2, b2);
        assert_ne!(a1, a2); // one runtime's successive sessions differ
                            // A zero seed draws every session's nonce from the kernel: two runtimes never share one.
        let (mut x, mut y) = (Core::new(template(0)).unwrap(), Core::new(template(0)).unwrap());
        assert_ne!(lineage(&mut x), lineage(&mut y));
    }

    #[test]
    fn typed_observations_are_admitted_against_the_sessions_active_sources_before_the_native_call() {
        let dir = Dir::new("session-admission");
        let mut core = dir.core();
        let (mut sub, k) = (Fake::composite(0.2), composite_key(7));
        core.anchor(&mut sub, k).unwrap();
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        sub.clear_calls();
        let a = observation(0, 1);
        assert_eq!(observe(&mut core, &mut sub, k, &a).unwrap(), a);
        // A source the session does not know, and a record of another memory ABI: refused before the native call,
        // recoverably (the turn stays open, its candidate unchanged).
        let mut foreign = observation(0, 3);
        foreign[0] = 2;
        assert_eq!(observe(&mut core, &mut sub, k, &observation(9, 2)).unwrap_err(), Rt::ObservationRefused.into());
        assert_eq!(observe(&mut core, &mut sub, k, &foreign).unwrap_err(), Rt::Invalid.into());
        assert_eq!(sub.calls(), ["txn_observe"]);
        assert!(core.turn_is_open() && core.fault().is_none());
        core.turn_commit(&mut sub, k).unwrap();
        assert_eq!(sub.memory(), vec![a]);
    }

    #[test]
    fn a_sources_per_turn_bound_and_its_lifecycle_follow_the_operators_events() {
        let dir = Dir::new("session-bound");
        let mut core = dir.core();
        let (mut sub, k) = (Fake::composite(0.2), composite_key(7));
        core.anchor(&mut sub, k).unwrap();
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        assert_eq!(observe(&mut core, &mut sub, k, &observation(5, 1)).unwrap_err(), Rt::ObservationRefused.into());
        // An operator event while a managed turn is open is refused: its admission must not change under it.
        let (op, seq) = operator(&core);
        let label = [9; 32];
        let id = core.authority_next_entity_id(m::EntityKind::Source as u8, &label).unwrap();
        let found = m::Event::entity_found(&op, seq, &id, m::EntityKind::Source, &label, &[4; 32], 1, 2, 5);
        let generation = core.authority_status().generation;
        assert_eq!(core.authority_transition(&found).unwrap_err().error, Rt::TurnOpen.into());
        assert_eq!(core.authority_status().generation, generation);
        core.turn_abort(k).unwrap();
        assert_eq!(found_source(&mut core, 9, 5, 2), id);
        // Source 5 admits two observations per turn; source 0 keeps its own bound.
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        for byte in [1, 2] {
            observe(&mut core, &mut sub, k, &observation(5, byte)).unwrap();
        }
        assert_eq!(observe(&mut core, &mut sub, k, &observation(5, 3)).unwrap_err(), Rt::ObservationRefused.into());
        observe(&mut core, &mut sub, k, &observation(0, 4)).unwrap();
        core.turn_commit(&mut sub, k).unwrap();
        // The bound is per turn.
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        observe(&mut core, &mut sub, k, &observation(5, 5)).unwrap();
        core.turn_commit(&mut sub, k).unwrap();
        assert_eq!(sub.memory().len(), 4);
        // DORMANT: its observations are refused until it is ACTIVE again.
        lifecycle(&mut core, &id, m::Lifecycle::Active, m::Lifecycle::Dormant).unwrap();
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        assert_eq!(observe(&mut core, &mut sub, k, &observation(5, 6)).unwrap_err(), Rt::ObservationRefused.into());
        core.turn_abort(k).unwrap();
        lifecycle(&mut core, &id, m::Lifecycle::Dormant, m::Lifecycle::Active).unwrap();
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        observe(&mut core, &mut sub, k, &observation(5, 6)).unwrap();
        core.turn_commit(&mut sub, k).unwrap();
        assert_eq!(sub.memory().len(), 5);
        // A refused operator event (a stale sequence) changes nothing and leaves the exact session code.
        let stale = m::Event::entity_lifecycle(&op, seq, &id, m::Lifecycle::Active, m::Lifecycle::Dormant);
        let generation = core.authority_status().generation;
        assert_eq!(core.authority_transition(&stale).unwrap_err().error, Rt::Authority.into());
        assert_eq!(core.authority_status().generation, generation);
        assert_eq!(core.authority_status().ecs_c_code, elpis_ecs_c_session::Code::Duplicate as i32);
        assert_eq!(core.fault(), None);
    }

    #[test]
    fn operator_events_are_the_operators_and_persistence_only_kinds_are_retired() {
        let dir = Dir::new("session-op");
        let mut core = dir.core();
        let generation = core.authority_status().generation;
        let (op, seq) = operator(&core);
        let rt = core.authority_entity(1).unwrap().id;
        let e1 = EvolutionState::GENESIS;
        // RuntimeCore's own events are refused before the session sees them, whoever they claim to be from.
        for event in [
            m::Event::anchor(&rt, 1, StateFormat::K1, &[3; 32]),
            m::Event::publish(&rt, 1, StateFormat::K1, &[3; 32], &[4; 32], 0, &[0; 32]),
            m::Event::import_composite(&rt, 1, &[3; 32], &[4; 32]),
            m::Event::evolution_reserve(&rt, 1, &e1, &[5; 32]),
            m::Event::evolution_finalize(&rt, 1, &e1, &[5; 32]),
        ] {
            assert_eq!(core.authority_transition(&event).unwrap_err().error, Rt::Invalid.into(), "{:?}", event.kind);
        }
        // The operations of durable storage are retired: refused RUNTIME_RETIRED, nothing changed, no fail-stop.
        for event in [
            m::Event::import_legacy(&op, seq, &[0; m::LEGACY_RECORD_BYTES]),
            m::Event::reconcile_rollback(&op, seq, 1, &[6; 32], 2),
            m::Event::reconcile_unauthenticated(&op, seq, 0, 2, &[6; 32]),
            m::Event::reconcile_repin(&op, seq, [0, 0], &[6; 32]),
        ] {
            assert_eq!(core.authority_transition(&event).unwrap_err().error, Rt::Retired.into(), "{:?}", event.kind);
            assert_eq!(core.authority_status().ecs_c_code, elpis_ecs_c_session::Code::PersistenceRetired as i32);
        }
        assert_eq!((core.authority_status().generation, core.fault()), (generation, None));
        // A closed runtime refuses them too.
        core.close();
        let event = m::Event::entity_reclaim(&op, seq, &[2; 32]);
        assert_eq!(core.authority_transition(&event).unwrap_err().error, Rt::Closed.into());
    }

    #[test]
    fn the_session_reads_are_exactly_its_rows_and_change_nothing() {
        let mut core = new_core();
        assert_eq!(core.authority_entity(0).unwrap_err(), Rt::Closed.into());
        open(&mut core);
        let found = found_source(&mut core, 9, 4242, 2);
        let status = core.authority_status();
        let n = status.entity_count; // operator, runtime, the suite's source, this one
        assert_eq!(n, 4);
        let rows: Vec<_> = (0..n).map(|i| core.authority_entity(i).unwrap()).collect();
        assert_eq!(core.authority_entity(n).unwrap_err(), Rt::Invalid.into());
        assert_eq!(rows[3].id, found);
        assert_eq!(
            (rows[0].kind, rows[1].kind, rows[2].kind),
            (m::EntityKind::Operator, m::EntityKind::Runtime, m::EntityKind::Source)
        );
        for kind in [m::EntityKind::Operator, m::EntityKind::Runtime, m::EntityKind::Source] {
            let a = core.authority_next_entity_id(kind as u8, &[7; 32]).unwrap();
            assert_eq!(a, core.authority_next_entity_id(kind as u8, &[7; 32]).unwrap());
        }
        assert_eq!(core.authority_next_entity_id(0, &[7; 32]).unwrap_err(), Rt::Invalid.into());
        assert_eq!(core.authority_next_entity_id(4, &[7; 32]).unwrap_err(), Rt::Invalid.into());
        // Reads change nothing.
        assert_eq!(core.authority_status(), status);
    }

    #[test]
    fn a_session_at_capacity_refuses_a_founding_and_stays_usable() {
        let mut core = Core::new(SessionTemplate { capacity: 3, ..template(5) }).unwrap();
        core.open().unwrap();
        let generation = core.authority_status().generation;
        found_source(&mut core, 1, 0, 4); // the third and last row
        let label = [2; 32];
        let id = core.authority_next_entity_id(m::EntityKind::Source as u8, &label).unwrap();
        let (op, seq) = operator(&core);
        let found = m::Event::entity_found(&op, seq, &id, m::EntityKind::Source, &label, &[4; 32], 1, 4, 1);
        let before = core.authority_status();
        assert_eq!(before.generation, generation + 2);
        assert_eq!(core.authority_transition(&found).unwrap_err().error, Rt::Authority.into());
        let after = core.authority_status();
        assert_eq!(after.ecs_c_code, elpis_ecs_c_session::Code::Capacity as i32);
        assert_eq!((after.generation, after.entity_count, after.state_root), (before.generation, 3, before.state_root));
        // Still usable: the lineage anchors and learns in the full session.
        let mut sub = Fake::new(0.1);
        core.anchor(&mut sub, key(1)).unwrap();
        turn(&mut core, &mut sub, key(1)).unwrap();
        assert_eq!(core.fault(), None);
    }
}

mod observe_only {
    use super::*;

    fn composite_key(owner: u64) -> Key {
        Key { kind: 3, handle: 0x2000 + owner as usize, id: 0, owner, dim: DIM }
    }

    /// A stand-in observation: 32 bytes whose head reads as an `elpis_ecs_mem_observation` of ABI v1 recording
    /// source authority 0 (bytes 24..28), the source the ECS_C fixture provisions; `byte` tells them apart.
    fn observation(byte: u8) -> Digest {
        let mut d = [0u8; 32];
        d[0] = 1;
        d[4] = byte;
        d[31] = 0x5a;
        d
    }

    fn observe(core: &mut Core, sub: &mut Fake, k: Key, d: &Digest) -> Result<Digest, (Error, i32)> {
        let (result, memory_status) = core.turn_observe(sub, k, Observation(d as *const Digest as *const _));
        result.map_err(|r| (r.error, memory_status))
    }

    fn anchored_composite(name: &str) -> (Dir, Core, Fake, Key, Snapshot) {
        let dir = Dir::new(name);
        let mut core = dir.core();
        let mut sub = Fake::composite(0.2);
        let k = composite_key(7);
        let anchor = core.anchor(&mut sub, k).unwrap();
        sub.clear_calls();
        core.counters(true);
        (dir, core, sub, k, anchor)
    }

    #[test]
    fn an_observe_only_turn_records_evidence_without_any_k1_schedule_and_commits_volatile() {
        let (_dir, mut core, mut sub, k, anchor) = anchored_composite("observe-only");
        let (k1_before, generation) = (sub.retained(), sub.generation());
        let a = observation(1);
        assert_eq!(core.turn_begin_observe(&mut sub, k, &CEILING).unwrap(), generation);
        assert_eq!(observe(&mut core, &mut sub, k, &a).unwrap(), a);
        let c = core.turn_commit(&mut sub, k).unwrap();
        // The live coordinate advanced by exactly one; K1 reports no step and no epoch change.
        assert_eq!((c.transition.generation_before, c.transition.generation_after), (generation, generation + 1));
        assert_eq!((c.transition.steps, c.transition.epoch_before, c.transition.epoch_after), (0, 0, 0));
        assert_eq!((c.live_generation, core.live_generation()), (generation + 1, Some(generation + 1)));
        // K1's retained values are byte-identical; only the memory moved.
        assert_eq!(sub.retained(), k1_before);
        assert_eq!(sub.memory(), vec![a]);
        // Volatile: the session's authority did not move.
        assert_eq!(core.snapshot().unwrap(), anchor);
        assert_ne!(sub.identity(), anchored(&core).unwrap());
        // The first turn reads the state's shape once (cold path, cached); then begin, one observe, commit.
        assert_eq!(sub.calls(), ["shape", "txn_begin_at", "txn_observe", "txn_commit"]);
        // A warm observe-only turn: no schedule, no digest, no capture.
        sub.clear_calls();
        core.counters(true);
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        observe(&mut core, &mut sub, k, &observation(2)).unwrap();
        core.turn_commit(&mut sub, k).unwrap();
        assert_eq!(sub.calls(), ["txn_begin_at", "txn_observe", "txn_commit"]);
        let expected = Counters { k1_txn_begins: 1, k1_observes: 1, k1_commits: 1, ..Counters::default() };
        assert_eq!(core.counters(false), expected);
        assert_eq!((sub.retained(), sub.memory()), (k1_before, vec![a, observation(2)]));
        assert_eq!(core.fault(), None);
    }

    #[test]
    fn a_k1_or_fms_substrate_is_refused_before_any_native_call() {
        for (name, mut sub, k) in [
            ("observe-k1", Fake::new(0.1), key(1)),
            ("observe-fms", Fake::resident(0.1), Key { kind: 2, handle: 0x3000, id: 9, owner: 3, dim: DIM }),
        ] {
            let dir = Dir::new(name);
            let mut core = dir.core();
            core.anchor(&mut sub, k).unwrap();
            sub.clear_calls();
            assert_eq!(core.turn_begin_observe(&mut sub, k, &CEILING).unwrap_err().error, Rt::Invalid.into());
            assert!(sub.calls().is_empty() && !core.turn_is_open() && !sub.txn_open(), "{name}");
            // A K1 lineage has no memory in a LEARN turn either.
            let i = input(2, 1);
            let mut s3 = vec![0.0; features(DIM)];
            core.turn_begin(&mut sub, k, &stimulus(&i), &CEILING, &mut s3).unwrap();
            assert_eq!(observe(&mut core, &mut sub, k, &observation(2)).unwrap_err().0, Rt::Invalid.into());
            core.turn_abort(k).unwrap();
        }
    }

    #[test]
    fn a_commit_with_nothing_observed_is_refused_and_the_turn_stays_open() {
        let (_dir, mut core, mut sub, k, _anchor) = anchored_composite("observe-empty");
        let generation = sub.generation();
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        assert_eq!(core.turn_commit(&mut sub, k).unwrap_err().error, Rt::Invalid.into());
        assert!(core.turn_is_open() && sub.txn_open());
        // A refused observation does not count either.
        assert_eq!(observe(&mut core, &mut sub, k, &observation(0xFF)).unwrap_err(), (Rt::EcsRefused.into(), -7));
        assert_eq!(core.turn_commit(&mut sub, k).unwrap_err().error, Rt::Invalid.into());
        core.turn_abort(k).unwrap();
        assert!(!core.turn_is_open() && !sub.txn_open());
        assert_eq!((sub.generation(), sub.memory()), (generation, vec![]));
    }

    #[test]
    fn the_observation_bound_is_fuel_checked_before_the_native_call_and_recoverable() {
        let (_dir, mut core, mut sub, k, _anchor) = anchored_composite("observe-fuel");
        let two = Budget { max_observations: 2, ..CEILING };
        core.turn_begin_observe(&mut sub, k, &two).unwrap();
        let (a, b, c) = (observation(1), observation(2), observation(3));
        observe(&mut core, &mut sub, k, &a).unwrap();
        observe(&mut core, &mut sub, k, &b).unwrap();
        assert_eq!(observe(&mut core, &mut sub, k, &c).unwrap_err().0, Rt::Fuel.into());
        assert_eq!(core.counters(false).k1_observes, 2); // the refused one never crossed
        assert!(core.turn_is_open());
        core.turn_commit(&mut sub, k).unwrap();
        assert_eq!(sub.memory(), vec![a, b]);
        // The bound holds in a LEARN turn too.
        let one = Budget { max_observations: 1, ..CEILING };
        let i = input(2, 1);
        let mut s3 = vec![0.0; features(DIM)];
        core.turn_begin(&mut sub, k, &stimulus(&i), &one, &mut s3).unwrap();
        observe(&mut core, &mut sub, k, &c).unwrap();
        assert_eq!(observe(&mut core, &mut sub, k, &observation(4)).unwrap_err().0, Rt::Fuel.into());
        core.turn_commit(&mut sub, k).unwrap();
        assert_eq!(sub.memory(), vec![a, b, c]);
        // A budget outside the ceiling is refused before anything is begun.
        sub.clear_calls();
        for bad in [Budget { max_observations: 0, ..CEILING }, Budget { max_observations: 257, ..CEILING }] {
            assert_eq!(core.turn_begin_observe(&mut sub, k, &bad).unwrap_err().error, Rt::Fuel.into());
        }
        assert!(!sub.calls().contains(&"txn_begin_at") && !core.turn_is_open());
        assert_eq!(fuel::CEILING.max_observations, fuel::MAX_TURN_OBSERVATIONS);
    }

    #[test]
    fn a_refused_observation_leaves_the_candidate_and_an_abort_installs_nothing() {
        let (_dir, mut core, mut sub, k, _anchor) = anchored_composite("observe-abort");
        let generation = sub.generation();
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        observe(&mut core, &mut sub, k, &observation(1)).unwrap();
        assert_eq!(observe(&mut core, &mut sub, k, &observation(0xFF)).unwrap_err(), (Rt::EcsRefused.into(), -7));
        assert!(core.turn_is_open());
        core.turn_abort(k).unwrap();
        assert_eq!((sub.generation(), sub.memory(), core.live_generation()), (generation, vec![], Some(generation)));
        // The next turn starts from the committed state, not from the aborted candidate.
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        observe(&mut core, &mut sub, k, &observation(2)).unwrap();
        core.turn_commit(&mut sub, k).unwrap();
        assert_eq!(sub.memory(), vec![observation(2)]);
    }

    #[test]
    fn an_observe_only_commit_that_reports_k1_learning_fail_stops() {
        let (_dir, mut core, mut sub, k, anchor) = anchored_composite("observe-learned");
        sub.state().learn_on_commit = true;
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        observe(&mut core, &mut sub, k, &observation(1)).unwrap();
        let refused = core.turn_commit(&mut sub, k).unwrap_err();
        assert_eq!(refused.error, Code::StateMismatch.into());
        assert_eq!(core.fault(), Some(Code::StateMismatch));
        // Nothing is built on it: every HOT operation is refused; the session's authority is unchanged.
        assert!(core.turn_begin_observe(&mut sub, k, &CEILING).is_err());
        assert!(core.live_identity(&mut sub, k).is_err());
        assert_eq!(core.snapshot().unwrap(), anchor);
    }

    #[test]
    fn observe_only_turns_keep_the_lifecycle_refusals() {
        let mut core = new_core();
        let mut sub = Fake::composite(0.2);
        let k = composite_key(7);
        assert_eq!(core.turn_begin_observe(&mut sub, k, &CEILING).unwrap_err().error, Rt::Closed.into());
        open(&mut core);
        assert_eq!(core.turn_begin_observe(&mut sub, k, &CEILING).unwrap_err().error, Code::Unanchored.into());
        assert!(sub.calls().is_empty());
        core.anchor(&mut sub, k).unwrap();
        core.turn_begin_observe(&mut sub, k, &CEILING).unwrap();
        assert_eq!(core.turn_begin_observe(&mut sub, k, &CEILING).unwrap_err().error, Rt::TurnOpen.into());
        let mut other = Fake::composite(0.3);
        assert_eq!(
            observe(&mut core, &mut other, composite_key(8), &observation(1)).unwrap_err().0,
            Rt::TurnNotOpen.into()
        );
        assert_eq!(core.live_identity(&mut sub, k).unwrap_err().error, Rt::TurnOpen.into());
        // Close ends the turn natively: nothing installed.
        observe(&mut core, &mut sub, k, &observation(1)).unwrap();
        core.close();
        assert!(!sub.txn_open() && sub.memory().is_empty());
        // The next session is unanchored: nothing of the last one is resumed.
        open(&mut core);
        assert_eq!(
            core.turn_begin_observe(&mut other, composite_key(8), &CEILING).unwrap_err().error,
            Code::Unanchored.into()
        );
    }
}
