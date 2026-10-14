//! The durability lane (WARM): explicit durable publication of the live state, off the HOT lane
//! (docs/RUNTIME_CORE.md, "Live and durable authority").
//!
//! ```text
//!   HOT (caller thread, under the runtime handle's lock)       WARM (one worker thread per open runtime)
//!   ---------------------------------------------------        -----------------------------------------
//!   query / turn_begin / turn_commit: no I/O, no lane          wait for a job and the stores
//!   durable_request: leased_capture (unsealed, ~memcpy)  --->  seal (envelope_seal) and verify the identity
//!                    + enqueue (capacity 1, coalescing)        checkpoint slot write + fdatasync (two fixed slots)
//!                    returns PENDING at once                   continuity publication (one record + fdatasync)
//!   durable_wait / durable_status: lane lock only        <---  ticket: ACKNOWLEDGED / FAILED / UNCERTAIN
//! ```
//!
//! The lane owns the continuity store and the checkpoint store. The HOT lane never touches either: a managed
//! commit installs the native candidate and advances the live coordinate, nothing else. Durable publication
//! happens only when a caller asks for it (or by an explicit control operation: anchor, import, adopt, evolution).
//!
//! Handoff. A request captures the authoritative state's envelope UNSEALED (`leased_capture`) into a lane-owned
//! buffer while the caller holds the runtime handle's lock: the K1 guard admits one entrant, so the capture must be
//! serialized with the HOT operations on the state, and it is the only synchronous step. The buffer is then owned
//! by the job alone: no mutable pointer into the live state ever reaches the worker. The worker seals, hashes,
//! writes, syncs and publishes without the handle's lock.
//!
//! Boundedness. At most one job runs and at most one waits. A request that finds a job waiting replaces it (the
//! newer capture descends from the older one in the same lineage); the replaced ticket is SUPERSEDED, never
//! silently acknowledged. A request never waits for the worker; at most two capture buffers exist. A stalled worker
//! therefore never blocks the HOT lane.
//!
//! Acknowledgment. ACKNOWLEDGED means continuity's record naming the captured identity is synced, and, when
//! `recoverable` is 1, a verified checkpoint slot holding that complete envelope was synced before it: restart
//! resumes exactly that state (K1 Recovery R0). Without a checkpoint store only the identity is durable
//! (`recoverable` 0): the caller keeps the bytes. FAILED means nothing durable changed (the previous authority
//! stands). UNCERTAIN means the continuity publication's outcome is unknown: the lane faults (sticky until reopen);
//! restart resolves to exactly one complete record. A crash at any point loses at most the volatile work since the
//! last acknowledgment; restart never adopts an unacknowledged state (checkpoint bytes never authorize themselves).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use elpis_continuity::{Code, Cognition, Digest, Snapshot, StateFormat, Store};

use crate::checkpoint;
use crate::code::{Error, Rt};
use crate::substrate::Sealer;

/// What happened to one durable publication request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Status {
    /// No such request.
    None = 0,
    /// Captured; waiting for, or in, the worker.
    Pending = 1,
    /// Durable: continuity names the identity (and, when `recoverable`, a synced checkpoint slot holds it).
    Acknowledged = 2,
    /// Nothing durable changed: the previous durable authority stands (`code` says why).
    Failed = 3,
    /// The continuity publication's outcome is unknown; the lane is faulted until reopen; restart resolves it.
    Uncertain = 4,
    /// Replaced, before the worker took it, by a newer request of the same lineage.
    Superseded = 5,
    /// The runtime closed (or the process was declared dead) before the worker took it.
    Abandoned = 6,
    /// Resolved so long ago that the lane keeps no record of it (see the last acknowledgment instead).
    Unknown = 7,
}

/// `elpis_runtime_durable_ticket`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ticket {
    /// The request's sequence (monotonic per runtime handle, from 1).
    pub sequence: u64,
    /// The live generation the capture was taken at.
    pub generation: u64,
    pub status: u32,
    /// 1: a verified checkpoint slot holds the acknowledged envelope (restart can resume it).
    pub recoverable: u32,
    /// The refusal or fault code behind FAILED / UNCERTAIN / ABANDONED (0 otherwise).
    pub code: i32,
    pub reserved: u32,
    /// The captured state's retained-state identity (zero until the worker sealed it).
    pub identity: Digest,
}

const _: () = assert!(std::mem::size_of::<Ticket>() == 64);

impl Ticket {
    pub fn status(&self) -> Status {
        match self.status {
            1 => Status::Pending,
            2 => Status::Acknowledged,
            3 => Status::Failed,
            4 => Status::Uncertain,
            5 => Status::Superseded,
            6 => Status::Abandoned,
            7 => Status::Unknown,
            _ => Status::None,
        }
    }

    fn resolve(mut self, status: Status, code: i32) -> Ticket {
        self.status = status as u32;
        self.code = code;
        self
    }
}

/// The lane's own counters (diagnostics; never authority).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub requests: u64,
    pub superseded: u64,
    pub acknowledged: u64,
    pub failed: u64,
    pub uncertain: u64,
    pub publications: u64,
}

/// The two stores the lane owns. Held by exactly one party at a time: the worker (during a job) or a control
/// operation (anchor, evolution, recovery, ...). The HOT lane never holds them.
pub struct Stores {
    pub continuity: Store,
    pub checkpoint: Option<checkpoint::Store>,
}

/// One captured envelope on its way to durability. Owned by the lane alone from the moment it is enqueued.
pub struct Job {
    pub ticket: Ticket,
    pub format: StateFormat,
    pub envelope: Vec<u8>,
    pub seal: Sealer,
}

/// Testing builds only: a planned fault of the next job (docs/RUNTIME_CORE.md, "Durability crash matrix").
#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// The process dies (simulated) at a step of the next job, or parks there (blocks until the process is
    /// killed, or the runtime closes) with exactly the on-disk effect of that step.
    At {
        step: Step,
        park: bool,
        arg: u64,
    },
    /// An I/O failure: the checkpoint write fails / is torn (`arg` bytes) / its sync fails, or the continuity
    /// record write fails / is torn / its sync fails with the bytes lost / its sync fails although durable.
    SlotWriteFail,
    SlotTornFail(u64),
    SlotSyncFail,
    RecordWriteFail,
    RecordTornFail(u64),
    RecordSyncLost,
    RecordSyncDurable,
}

/// The named steps of a durable publication (testing builds).
#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Step {
    /// The worker took the job; nothing sealed or written.
    Taken = 1,
    /// Sealed and verified; nothing written.
    Sealed = 2,
    /// `arg` bytes of the checkpoint slot written.
    SlotTorn = 3,
    /// The complete checkpoint slot written, not synced.
    SlotWritten = 4,
    /// The checkpoint slot synced; continuity not yet published.
    SlotSynced = 5,
    /// `arg` bytes of the continuity record written.
    RecordTorn = 6,
    /// The complete continuity record written, not synced.
    RecordWritten = 7,
    /// The continuity record synced; the acknowledgment not yet recorded.
    RecordSynced = 8,
    /// Acknowledged.
    Acknowledged = 9,
}

#[cfg(feature = "testing")]
impl Step {
    pub fn from_u32(v: u32) -> Option<Step> {
        use Step::*;
        [Taken, Sealed, SlotTorn, SlotWritten, SlotSynced, RecordTorn, RecordWritten, RecordSynced, Acknowledged]
            .into_iter()
            .find(|s| *s as u32 == v)
    }
}

const RING: usize = 16;

struct Shared {
    /// The runtime is open (stores opened, worker running).
    open: bool,
    /// The stores, when nobody holds them (None while held, and before the first open).
    stores: Option<Stores>,
    pending: Option<Job>,
    running: Option<Ticket>,
    resolved: [Ticket; RING],
    ring_next: usize,
    last_requested: Ticket,
    last_acknowledged: Ticket,
    /// The current durable authority as last read or published (None when closed or unknown).
    durable: Option<Snapshot>,
    /// Sticky lane fault (an uncertain publication, a simulated process death) until reopen.
    fault: Option<Code>,
    next_sequence: u64,
    /// The attached checkpoint store's envelope capacity.
    capacity: Option<usize>,
    /// Spare capture buffers (at most two exist: one running, one pending).
    pool: Vec<Vec<u8>>,
    shutdown: bool,
    counters: Counters,
    /// Testing: the worker takes no new job while held (the HOT lane must not care).
    hold: bool,
    #[cfg(feature = "testing")]
    plan: Option<Plan>,
    /// Testing: the step the worker is parked at (0: none).
    parked: u32,
}

/// The lane: shared by the runtime handle (HOT, control operations) and its worker.
pub struct Lane {
    shared: Mutex<Shared>,
    cv: Condvar,
    /// Diagnostics mirror of the live generation, owned and written by the HOT state machine (an atomic store; the
    /// HOT lane never touches the lane itself); status reads it without the handle's lock.
    live_generation: Arc<AtomicU64>,
    /// Testing: a simulated process death happened; every HOT operation refuses from now on.
    dead: AtomicBool,
}

/// The durability state a status read reports (`elpis_runtime_durability`, without the continuity record).
#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    pub last_requested: Ticket,
    pub last_acknowledged: Ticket,
    pub running: Ticket,
    pub live_generation: u64,
    pub pending: u32,
    pub fault: i32,
    pub parked: u32,
    pub capacity: u64,
    pub durable: Option<Snapshot>,
}

fn lock(m: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    // A panic aborts the process (panic = "abort" in release); in tests the lock is never poisoned by a live owner.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Lane {
    pub fn new(continuity: Store, live_generation: Arc<AtomicU64>) -> Lane {
        Lane {
            shared: Mutex::new(Shared {
                open: false,
                stores: Some(Stores { continuity, checkpoint: None }),
                pending: None,
                running: None,
                resolved: [Ticket::default(); RING],
                ring_next: 0,
                last_requested: Ticket::default(),
                last_acknowledged: Ticket::default(),
                durable: None,
                fault: None,
                next_sequence: 1,
                capacity: None,
                pool: Vec::new(),
                shutdown: false,
                counters: Counters::default(),
                hold: false,
                #[cfg(feature = "testing")]
                plan: None,
                parked: 0,
            }),
            cv: Condvar::new(),
            live_generation,
            dead: AtomicBool::new(false),
        }
    }

    // -- control operations -------------------------------------------------------------------------------------

    /// Run `f` with exclusive use of the stores (waits while the worker runs a job; never while a job is merely
    /// pending). Afterwards the durable cache is refreshed from continuity. Control operations only: the HOT lane
    /// never calls this.
    pub fn with_stores<R>(&self, f: impl FnOnce(&mut Stores) -> R) -> Result<R, Error> {
        let mut stores = {
            let mut g = lock(&self.shared);
            loop {
                if let Some(s) = g.stores.take() {
                    break s;
                }
                if !g.open && g.running.is_none() {
                    return Err(Rt::Closed.into()); // a simulated process death took them with it
                }
                g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
        };
        let r = f(&mut stores);
        let mut g = lock(&self.shared);
        g.durable = stores.continuity.snapshot().ok();
        g.capacity = stores.checkpoint.as_ref().map(|c| c.capacity());
        g.stores = Some(stores);
        drop(g);
        self.cv.notify_all();
        Ok(r)
    }

    /// Open the stores (continuity resolves the current authority) and the lane. The caller spawns the worker.
    pub fn open(&self) -> Result<Snapshot, Error> {
        let snapshot = self.with_stores(|s| s.continuity.open())??;
        let mut g = lock(&self.shared);
        g.open = true;
        g.shutdown = false;
        g.fault = None;
        g.parked = 0;
        g.durable = Some(snapshot);
        self.dead.store(false, Ordering::Release);
        Ok(snapshot)
    }

    /// Ask the worker to stop. Abandons a pending job; a running job completes (its I/O cannot be interrupted)
    /// unless the worker is parked (testing), in which case it is a simulated process death.
    pub fn shutdown(&self) {
        let mut g = lock(&self.shared);
        g.shutdown = true;
        if let Some(job) = g.pending.take() {
            let t = job.ticket.resolve(Status::Abandoned, Rt::Closed as i32);
            Self::record(&mut g, t);
            Self::recycle(&mut g, job.envelope);
        }
        drop(g);
        self.cv.notify_all();
    }

    /// After the worker joined: close the stores and the lane.
    pub fn close(&self) {
        let mut g = lock(&self.shared);
        if let Some(s) = g.stores.as_mut() {
            s.continuity.close();
            s.checkpoint = None;
        }
        g.open = false;
        g.durable = None;
        g.capacity = None;
        g.hold = false;
        #[cfg(feature = "testing")]
        {
            g.plan = None;
        }
        g.parked = 0;
        drop(g);
        self.cv.notify_all();
    }

    /// The cached durable authority (no I/O): what the last open or publication established. Refused with the
    /// lane fault when the durable outcome is unknown (an uncertain publication, a simulated death).
    pub fn durable(&self) -> Result<Snapshot, Error> {
        let g = lock(&self.shared);
        if !g.open {
            return Err(Rt::Closed.into());
        }
        if let Some(code) = g.fault {
            return Err(code.into());
        }
        g.durable.ok_or(Error::Continuity(Code::Uninitialized))
    }

    pub fn fault(&self) -> Option<Code> {
        lock(&self.shared).fault
    }

    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::Acquire)
    }

    /// The attached checkpoint store's envelope capacity, if any.
    pub fn capacity(&self) -> Option<usize> {
        lock(&self.shared).capacity
    }

    /// Whether a job is pending or running (control operations that change the lineage refuse while one is).
    pub fn busy(&self) -> bool {
        let g = lock(&self.shared);
        g.pending.is_some() || g.running.is_some()
    }

    pub fn counters(&self, reset: bool) -> Counters {
        let mut g = lock(&self.shared);
        let c = g.counters;
        if reset {
            g.counters = Counters::default();
        }
        c
    }

    /// Count a control operation's continuity publication.
    pub fn count_publication(&self) {
        lock(&self.shared).counters.publications += 1;
    }

    // -- the HOT side of a request: never waits for the worker ----------------------------------------------------

    /// A buffer of exactly `size` bytes for a capture (from the pool when it fits; allocated otherwise, on the
    /// first request of a shape only).
    pub fn buffer(&self, size: usize) -> Vec<u8> {
        let mut g = lock(&self.shared);
        while let Some(b) = g.pool.pop() {
            if b.len() == size {
                return b;
            }
        }
        drop(g);
        vec![0u8; size]
    }

    pub fn give_back(&self, buffer: Vec<u8>) {
        let mut g = lock(&self.shared);
        Self::recycle(&mut g, buffer);
    }

    fn recycle(g: &mut Shared, buffer: Vec<u8>) {
        if g.pool.len() < 2 {
            g.pool.push(buffer);
        }
    }

    /// Hand a captured envelope to the worker. Capacity one waiting job: a waiting job is replaced (SUPERSEDED).
    /// Returns the new ticket (PENDING). Refused with the lane fault while faulted, `RUNTIME_CLOSED` while closed.
    pub fn enqueue(
        &self,
        generation: u64,
        format: StateFormat,
        envelope: Vec<u8>,
        seal: Sealer,
    ) -> Result<Ticket, Error> {
        let mut g = lock(&self.shared);
        if !g.open || g.shutdown {
            Self::recycle(&mut g, envelope);
            return Err(Rt::Closed.into());
        }
        if let Some(code) = g.fault {
            Self::recycle(&mut g, envelope);
            return Err(code.into());
        }
        let ticket =
            Ticket { sequence: g.next_sequence, generation, status: Status::Pending as u32, ..Ticket::default() };
        g.next_sequence += 1;
        g.counters.requests += 1;
        if let Some(old) = g.pending.take() {
            let t = old.ticket.resolve(Status::Superseded, 0);
            g.counters.superseded += 1;
            Self::record(&mut g, t);
            Self::recycle(&mut g, old.envelope);
        }
        g.pending = Some(Job { ticket, format, envelope, seal });
        g.last_requested = ticket;
        drop(g);
        self.cv.notify_all();
        Ok(ticket)
    }

    fn record(g: &mut Shared, t: Ticket) {
        let i = g.ring_next;
        g.resolved[i] = t;
        g.ring_next = (i + 1) % RING;
        if g.last_requested.sequence == t.sequence {
            g.last_requested = t;
        }
        if t.status() == Status::Acknowledged {
            g.last_acknowledged = t;
        }
    }

    fn ticket_of(g: &Shared, sequence: u64) -> Option<Ticket> {
        if sequence == 0 || sequence >= g.next_sequence {
            return None;
        }
        if let Some(job) = &g.pending {
            if job.ticket.sequence == sequence {
                return Some(job.ticket);
            }
        }
        if let Some(t) = g.running {
            if t.sequence == sequence {
                return Some(t);
            }
        }
        Some(g.resolved.iter().find(|t| t.sequence == sequence).copied().unwrap_or(Ticket {
            sequence,
            status: Status::Unknown as u32,
            ..Ticket::default()
        }))
    }

    /// Wait (the caller's explicit choice; never on the HOT lane) until request `sequence` resolves or `timeout`
    /// elapses. Returns the ticket as it stands (PENDING after a timeout). `RUNTIME_INVALID` for a sequence never
    /// issued by this handle.
    pub fn wait(&self, sequence: u64, timeout: Option<Duration>) -> Result<Ticket, Error> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut g = lock(&self.shared);
        loop {
            let t = Self::ticket_of(&g, sequence).ok_or(Error::Runtime(Rt::Invalid))?;
            if t.status() != Status::Pending {
                return Ok(t);
            }
            if !g.open && g.running.is_none() && g.pending.is_none() {
                return Ok(t.resolve(Status::Abandoned, Rt::Closed as i32));
            }
            match deadline {
                None => g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner()),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return Ok(t);
                    }
                    g = self.cv.wait_timeout(g, d - now).unwrap_or_else(|e| e.into_inner()).0;
                }
            }
        }
    }

    pub fn report(&self) -> Report {
        let g = lock(&self.shared);
        Report {
            last_requested: g.last_requested,
            last_acknowledged: g.last_acknowledged,
            running: g.running.unwrap_or_default(),
            live_generation: self.live_generation.load(Ordering::Relaxed),
            pending: g.pending.is_some() as u32,
            fault: g.fault.map_or(0, |c| c as i32),
            parked: g.parked,
            capacity: g.capacity.unwrap_or(0) as u64,
            durable: if g.fault.is_some() { None } else { g.durable },
        }
    }

    /// Testing: hold (or release) the worker before it takes a new job.
    pub fn hold(&self, hold: bool) {
        lock(&self.shared).hold = hold;
        self.cv.notify_all();
    }

    #[cfg(feature = "testing")]
    pub fn plan(&self, plan: Option<Plan>) {
        lock(&self.shared).plan = plan;
    }

    // -- the worker -----------------------------------------------------------------------------------------------

    /// The worker loop: one job at a time, without the runtime handle's lock. Returns at shutdown.
    pub fn work(&self) {
        loop {
            let (job, mut stores) = {
                let mut g = lock(&self.shared);
                loop {
                    if g.shutdown {
                        return;
                    }
                    if !g.hold && g.pending.is_some() && g.stores.is_some() {
                        break;
                    }
                    g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
                }
                let job = g.pending.take().expect("pending");
                g.running = Some(job.ticket);
                (job, g.stores.take().expect("stores"))
            };
            #[cfg(feature = "testing")]
            let plan = lock(&self.shared).plan.take();
            #[cfg(not(feature = "testing"))]
            let plan = ();
            let mut job = job;
            let outcome = persist(&mut stores, &mut job, plan, self);
            let mut g = lock(&self.shared);
            g.running = None;
            let (ticket, stores) = match outcome {
                Outcome::Done(t) => (t, Some(stores)),
                Outcome::Died(t) => {
                    // A simulated process death: descriptors close, nothing more is written, every HOT operation
                    // refuses. Restart (close + open) reads whatever the disk holds.
                    g.fault = Some(Code::TestingProcessDeath);
                    self.dead.store(true, Ordering::Release);
                    stores.continuity.close();
                    (t, Some(Stores { checkpoint: None, ..stores }))
                }
            };
            match ticket.status() {
                Status::Acknowledged => g.counters.acknowledged += 1,
                Status::Uncertain => {
                    g.counters.uncertain += 1;
                    g.fault.get_or_insert(Code::PublicationUncertain);
                }
                Status::Failed => g.counters.failed += 1,
                _ => {}
            }
            if let Some(s) = stores {
                g.durable = s.continuity.snapshot().ok();
                g.capacity = s.checkpoint.as_ref().map(|c| c.capacity());
                g.stores = Some(s);
            }
            Self::record(&mut g, ticket);
            Self::recycle(&mut g, std::mem::take(&mut job.envelope));
            drop(g);
            self.cv.notify_all();
        }
    }

    /// Testing: park the worker at `step` (with exactly that step's on-disk effect) until shutdown. The process is
    /// expected to be killed meanwhile; a shutdown instead is a simulated death.
    #[cfg(feature = "testing")]
    fn park(&self, step: Step) {
        let mut g = lock(&self.shared);
        g.parked = step as u32;
        self.cv.notify_all();
        while !g.shutdown {
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }
}

enum Outcome {
    Done(Ticket),
    Died(Ticket),
}

#[cfg(feature = "testing")]
type PlanArg = Option<Plan>;
#[cfg(not(feature = "testing"))]
type PlanArg = ();

/// One durable publication: seal -> verify -> checkpoint slot (write + sync) -> continuity record (write + sync).
/// Runs on the worker with the stores held and without the runtime handle's lock.
#[allow(unused_variables)]
fn persist(stores: &mut Stores, job: &mut Job, plan: PlanArg, lane: &Lane) -> Outcome {
    let ticket = job.ticket;
    let failed = |code: i32| Outcome::Done(ticket.resolve(Status::Failed, code));

    // A testing step: die (simulated) or park with exactly this step's on-disk effect.
    #[cfg(feature = "testing")]
    macro_rules! step {
        ($s:expr, $t:expr) => {
            if let Some(Plan::At { step, park, .. }) = plan {
                if step == $s {
                    if park {
                        lane.park(step);
                    }
                    return Outcome::Died($t.resolve(Status::Abandoned, Code::TestingProcessDeath as i32));
                }
            }
        };
    }
    #[cfg(not(feature = "testing"))]
    macro_rules! step {
        ($s:expr, $t:expr) => {};
    }

    step!(Step::Taken, ticket);

    // Seal the immutable capture and verify it: the identity is the envelope's own trailer.
    let len = job.envelope.len();
    // SAFETY: the sealer is the substrate library's pure `envelope_seal`; the buffer is owned by this job alone.
    if unsafe { (job.seal)(job.envelope.as_mut_ptr(), len) } != 0 {
        return failed(Rt::EcsState as i32);
    }
    let identity = match checkpoint::envelope_identity(&job.envelope) {
        Some(d) if checkpoint::envelope_format(&job.envelope) == Some(job.format) => d,
        _ => return failed(Rt::CheckpointShape as i32),
    };
    let ticket = Ticket { identity, ..ticket };
    let failed = |code: i32| Outcome::Done(ticket.resolve(Status::Failed, code));
    step!(Step::Sealed, ticket);

    let authority = match stores.continuity.snapshot() {
        Ok(s) => s,
        Err(code) => return failed(code as i32),
    };
    let durable = match authority.cognition() {
        Cognition::Unanchored => return failed(Code::Unanchored as i32),
        Cognition::Anchored(d) => d,
    };
    if authority.format() != job.format {
        return failed(Code::StateMismatch as i32);
    }
    let recoverable = stores.checkpoint.as_ref().is_some_and(|c| c.holding(&identity).is_some());
    if identity == durable && (stores.checkpoint.is_none() || recoverable) {
        // Already the durable authority: nothing to write.
        return Outcome::Done(Ticket { recoverable: recoverable as u32, ..ticket.resolve(Status::Acknowledged, 0) });
    }

    // The checkpoint slot first: the complete envelope is durable before continuity names it.
    let mut written: Option<usize> = None;
    if let Some(store) = stores.checkpoint.as_mut() {
        if store.unresolved(&durable).is_some() {
            return failed(Rt::CheckpointUnresolved as i32);
        }
        if store.capacity() != len {
            return failed(Rt::CheckpointShape as i32);
        }
        if store.holding(&identity).is_none() {
            #[cfg(feature = "testing")]
            {
                use checkpoint::probe::Fault;
                store.probe.plan = match plan {
                    Some(Plan::At { step: Step::SlotTorn, arg, .. }) => Some(Fault::TornDie(arg as usize)),
                    Some(Plan::At { step: Step::SlotWritten, .. }) => Some(Fault::DieBeforeSync),
                    Some(Plan::SlotWriteFail) => Some(Fault::WriteFail),
                    Some(Plan::SlotTornFail(n)) => Some(Fault::TornFail(n as usize)),
                    Some(Plan::SlotSyncFail) => Some(Fault::SyncFail),
                    _ => None,
                };
            }
            match store.write_envelope(&job.envelope, Some(&durable)) {
                Ok((slot, _)) => written = Some(slot),
                Err(rt) => {
                    #[cfg(feature = "testing")]
                    if std::mem::take(&mut store.probe.died) {
                        if let Some(Plan::At { step, park: true, .. }) = plan {
                            lane.park(step);
                        }
                        return Outcome::Died(ticket.resolve(Status::Abandoned, Code::TestingProcessDeath as i32));
                    }
                    // Nothing was published: withdraw whatever the failed write left (best effort; a slot that
                    // survives is an explicit unresolved candidate for the operator, never an authority).
                    let target = (0..2).find(|&i| store.slots()[i].is_none());
                    if let Some(i) = target {
                        let _ = store.retract(i);
                    }
                    return failed(rt as i32);
                }
            }
        }
    }
    step!(Step::SlotSynced, ticket);

    // The continuity publication: one record, one sync. The lineage's previous durable identity -> this one.
    #[cfg(feature = "testing")]
    {
        let arm = match plan {
            Some(Plan::At { step: Step::RecordTorn, arg, .. }) => Some((4u32, arg)),
            Some(Plan::At { step: Step::RecordWritten, .. }) => Some((1, 1)),
            Some(Plan::At { step: Step::RecordSynced, .. }) => Some((1, 2)),
            Some(Plan::RecordWriteFail) => Some((2, 0)),
            Some(Plan::RecordTornFail(n)) => Some((3, n)),
            Some(Plan::RecordSyncLost) => Some((5, 0)),
            Some(Plan::RecordSyncDurable) => Some((6, 0)),
            _ => None,
        };
        if let Some((action, arg)) = arm {
            let _ = stores.continuity.testing_arm(1, action, arg);
        }
    }
    let published = stores.continuity.commit_cognition_as(job.format, Some(&durable), Some(&identity));
    match published {
        Ok(_) => {
            lane.count_publication();
            let t =
                Ticket { recoverable: stores.checkpoint.is_some() as u32, ..ticket.resolve(Status::Acknowledged, 0) };
            #[cfg(feature = "testing")]
            if let Some(Plan::At { step: Step::Acknowledged, park, .. }) = plan {
                // Dies after the acknowledgment: the ticket stays ACKNOWLEDGED (the record is synced).
                if park {
                    lane.park(Step::Acknowledged);
                }
                return Outcome::Died(t);
            }
            Outcome::Done(t)
        }
        Err(Code::PublicationRefused) => {
            // Definitive: the record was not written (or torn, which restart skips): the old authority stands.
            if let (Some(slot), Some(store)) = (written, stores.checkpoint.as_mut()) {
                let _ = store.retract(slot);
            }
            failed(Code::PublicationRefused as i32)
        }
        Err(Code::PublicationUncertain) => {
            Outcome::Done(ticket.resolve(Status::Uncertain, Code::PublicationUncertain as i32))
        }
        Err(Code::TestingProcessDeath) => {
            #[cfg(feature = "testing")]
            if let Some(Plan::At { step, park: true, .. }) = plan {
                lane.park(step);
            }
            Outcome::Died(ticket.resolve(Status::Abandoned, Code::TestingProcessDeath as i32))
        }
        Err(code) => {
            if let (Some(slot), Some(store)) = (written, stores.checkpoint.as_mut()) {
                let _ = store.retract(slot);
            }
            failed(code as i32)
        }
    }
}
