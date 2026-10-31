//! The volatile session (ECSC_VOLATILE_SESSION_R0): the ECS_C authority as bounded, RAM-only state.
//!
//! A session is created at a fresh genesis (the founding operator and runtime, ACTIVE), evolves only by
//! [`Session::transition`], and ends when it is dropped or its process ends. It keeps no history, writes nothing and
//! recovers nothing: a new session is a new genesis, never a resumption.
//!
//! **Staged, all-or-nothing transitions.** A transition validates the caller's coordinate, stages a copy of the live
//! state into a preallocated buffer, applies the one transition law to the copy ([`model::apply_into`]), encodes it
//! into a preallocated buffer to compute its state root, and only then publishes it by swapping the two buffers. Every
//! fallible step precedes the swap, so a refused transition leaves the live state, its coordinate and every counter
//! but the refusal count exactly as they were.
//!
//! **Bounded memory.** Every byte a session holds is reserved at creation for its capacity (fallibly: `NoMem`, never
//! an abort) and none is allocated afterwards: [`Session::volatile_bytes`] is exact and constant for a session's life.
//!
//! **Identity.** A session's lineage is derived from its nonce (the caller's, or one drawn from the kernel's random
//! source when the caller supplies none) and its capacity; a process-unique instance number distinguishes sessions
//! even if a nonce were reused. A [`Coordinate`] names both, so a coordinate, receipt or handle of another session is
//! refused (`SessionMismatch`) rather than mistaken for this one's.
//!
//! **Persistence retired.** The commit digest that chains receipts is computed in memory from the state root and the
//! receipt (the durable authority used its record checksum). Legacy import and the three reconciliations, which
//! existed only to adopt or repair durable records, are refused `PersistenceRetired`.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::error::Code;
use crate::model::{
    apply_into, genesis_with, state_bytes, state_root, Digest, Entity, EntityKind, Event, EventKind, Prior, State,
    MAX_CAPACITY, MIN_CAPACITY, ZERO,
};
use crate::sha256;

pub mod domain {
    /// Stands in for a deployment in a session's genesis: a session belongs to no deployment.
    pub const DEPLOYMENT: &[u8] = b"elpis.ecs_c.session.deployment.v1\x00";
    /// The in-memory digest of one committed generation (the predecessor digest of the next receipt).
    pub const COMMIT: &[u8] = b"elpis.ecs_c.session.commit.v1\x00";
}

/// The process-unique instance number of the next session (never 0, never reused in a process).
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

#[cfg(not(test))]
extern "C" {
    fn getrandom(buf: *mut u8, len: usize, flags: u32) -> isize;
}

/// Fill `out` from the kernel's random source (getrandom(2): a system call, no file).
fn draw_nonce(out: &mut Digest) -> Result<(), Code> {
    #[cfg(test)]
    {
        use std::sync::atomic::AtomicU64 as A;
        static DRAWN: A = A::new(0x5eed);
        let n = DRAWN.fetch_add(1, Ordering::Relaxed);
        *out = sha256::digest(&[b"test-nonce", &n.to_le_bytes()]);
        Ok(())
    }
    #[cfg(not(test))]
    {
        let mut filled = 0usize;
        while filled < out.len() {
            // SAFETY: writes at most out.len() - filled bytes into `out`.
            let n = unsafe { getrandom(out.as_mut_ptr().add(filled), out.len() - filled, 0) };
            if n <= 0 {
                return Err(Code::EntropyUnavailable);
            }
            filled += n as usize;
        }
        if *out == ZERO {
            return Err(Code::EntropyUnavailable);
        }
        Ok(())
    }
}

/// What a session is created with. A zero `nonce` asks the session to draw one from the kernel's random source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionConfig {
    pub capacity: u32,
    pub nonce: Digest,
    pub operator_label: Digest,
    pub runtime_label: Digest,
}

/// The coordinate a caller holds: compare-and-publish names the session (lineage and instance), its generation and
/// its state root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coordinate {
    pub session: Digest,
    pub instance: u64,
    pub generation: u64,
    pub state_root: Digest,
}

/// What one accepted transition returns. The caller keeps receipts; the session keeps only the chain head and count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionReceipt {
    pub generation: u64,
    pub predecessor_generation: u64,
    pub receipt: Digest,
    pub commit: Digest,
    pub state_root: Digest,
    pub receipt_head: Digest,
    pub receipt_count: u64,
    pub founded: Option<Digest>,
}

/// The in-memory digest of a committed generation.
pub fn commit_digest(lineage: &Digest, generation: u64, state_root: &Digest, receipt: &Digest) -> Digest {
    sha256::digest(&[domain::COMMIT, lineage, &generation.to_le_bytes(), state_root, receipt])
}

/// Copy `src` into `dst`, reusing `dst`'s entity buffer (no allocation when it has room for `src`'s rows).
fn stage(dst: &mut State, src: &State) {
    let mut entities = core::mem::take(&mut dst.entities);
    entities.clear();
    entities.extend_from_slice(&src.entities);
    *dst = State {
        cognition: src.cognition,
        publications: src.publications,
        last_publication_generation: src.last_publication_generation,
        evolution: src.evolution,
        receipt_head: src.receipt_head,
        receipt_count: src.receipt_count,
        evidence_head: src.evidence_head,
        evidence_count: src.evidence_count,
        reconcile_head: src.reconcile_head,
        reconcile_count: src.reconcile_count,
        reclaimed_head: src.reclaimed_head,
        reclaimed_count: src.reclaimed_count,
        next_founding_index: src.next_founding_index,
        legacy: src.legacy,
        entities,
    };
}

fn reserve<T>(n: usize) -> Result<Vec<T>, Code> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Code::NoMem)?;
    Ok(v)
}

/// The ECS_C authority of one volatile session.
pub struct Session {
    capacity: u32,
    lineage: Digest,
    instance: u64,
    generation: u64,
    state_root: Digest,
    commit: Digest,
    live: State,
    staging: State,
    encoded: Vec<u8>,
    transitions: u64,
    refusals: u64,
}

impl Session {
    /// Exact bytes a session of `capacity` holds for its whole life (its handle, the live and staging entity
    /// buffers and the encoding buffer), or 0 outside 2..=4096.
    pub const fn volatile_bytes_for(capacity: u32) -> usize {
        if capacity < MIN_CAPACITY || capacity > MAX_CAPACITY {
            return 0;
        }
        core::mem::size_of::<Session>() + 2 * capacity as usize * core::mem::size_of::<Entity>() + state_bytes(capacity)
    }

    /// A fresh session at genesis (generation 1). `Invalid` for a capacity outside 2..=4096; `NoMem` when its memory
    /// cannot be reserved; `EntropyUnavailable` when a nonce must be drawn and cannot be.
    pub fn create(config: &SessionConfig) -> Result<Session, Code> {
        if !(MIN_CAPACITY..=MAX_CAPACITY).contains(&config.capacity) {
            return Err(Code::Invalid);
        }
        let mut nonce = config.nonce;
        if nonce == ZERO {
            draw_nonce(&mut nonce)?;
        }
        let rows = config.capacity as usize;
        let live_rows = reserve::<Entity>(rows)?;
        let staging_rows = reserve::<Entity>(rows)?;
        let mut encoded = reserve::<u8>(state_bytes(config.capacity))?;
        encoded.resize(state_bytes(config.capacity), 0);
        let deployment = sha256::digest(&[domain::DEPLOYMENT]);
        let event = Event::genesis(&nonce, &config.operator_label, &config.runtime_label);
        let (lineage, live, receipt) = genesis_with(&deployment, config.capacity, &event, live_rows)?;
        live.encode(config.capacity, &mut encoded);
        let root = state_root(&lineage, 1, config.capacity, &encoded);
        let mut staging = State { entities: staging_rows, ..stage_template(&live) };
        stage(&mut staging, &live);
        Ok(Session {
            capacity: config.capacity,
            lineage,
            instance: NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed),
            generation: 1,
            state_root: root,
            commit: commit_digest(&lineage, 1, &root, &receipt),
            live,
            staging,
            encoded,
            transitions: 0,
            refusals: 0,
        })
    }

    pub fn coordinate(&self) -> Coordinate {
        Coordinate {
            session: self.lineage,
            instance: self.instance,
            generation: self.generation,
            state_root: self.state_root,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The session identity (the lineage of its genesis).
    pub fn lineage(&self) -> &Digest {
        &self.lineage
    }

    pub fn instance(&self) -> u64 {
        self.instance
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn state_root(&self) -> &Digest {
        &self.state_root
    }

    /// The in-memory digest of the current generation.
    pub fn commit(&self) -> &Digest {
        &self.commit
    }

    /// The live materialized state (read only).
    pub fn state(&self) -> &State {
        &self.live
    }

    /// Accepted and refused transitions so far (telemetry: not part of the state).
    pub fn counters(&self) -> (u64, u64) {
        (self.transitions, self.refusals)
    }

    /// Exact bytes this session holds (constant for its life).
    pub fn volatile_bytes(&self) -> usize {
        Session::volatile_bytes_for(self.capacity)
    }

    /// Typed-evidence admission (pure): the per-turn bound, or why the evidence is not admissible now.
    pub fn check_evidence(&self, source: &Digest, version: u64, scope: &Digest) -> Result<u32, Code> {
        self.live.check_evidence(source, version, scope)
    }

    /// The identity the next ENTITY_FOUND of `kind` and `label` founds (the receiver that event must name).
    pub fn next_entity_id(&self, kind: EntityKind, label: &Digest) -> Digest {
        self.live.next_entity_id(&self.lineage, kind, label)
    }

    /// Apply one event, all or nothing, at the caller's coordinate.
    pub fn transition(&mut self, expected: &Coordinate, event: &Event) -> Result<SessionReceipt, Code> {
        let result = self.transition_inner(expected, event);
        match result {
            Ok(_) => self.transitions = self.transitions.saturating_add(1),
            Err(_) => self.refusals = self.refusals.saturating_add(1),
        }
        result
    }

    fn transition_inner(&mut self, expected: &Coordinate, event: &Event) -> Result<SessionReceipt, Code> {
        match event.kind {
            EventKind::Genesis => return Err(Code::Invalid),
            EventKind::ImportLegacy
            | EventKind::ReconcileRollback
            | EventKind::ReconcileUnauthenticated
            | EventKind::ReconcileRepin => return Err(Code::PersistenceRetired),
            _ => {}
        }
        if expected.session != self.lineage || expected.instance != self.instance {
            return Err(Code::SessionMismatch);
        }
        if expected.generation != self.generation || expected.state_root != self.state_root {
            return Err(Code::Stale);
        }
        // Stage: everything fallible happens to the copy.
        stage(&mut self.staging, &self.live);
        let prior = Prior {
            lineage: self.lineage,
            generation: self.generation,
            record_digest: self.commit,
            state_root: self.state_root,
            state: &self.live,
        };
        let (generation, receipt, founded) = apply_into(&prior, self.capacity, event, &mut self.staging)?;
        self.staging.encode(self.capacity, &mut self.encoded);
        let root = state_root(&self.lineage, generation, self.capacity, &self.encoded);
        let commit = commit_digest(&self.lineage, generation, &root, &receipt);
        // Publish: nothing below can fail.
        core::mem::swap(&mut self.live, &mut self.staging);
        let predecessor_generation = self.generation;
        self.generation = generation;
        self.state_root = root;
        self.commit = commit;
        Ok(SessionReceipt {
            generation,
            predecessor_generation,
            receipt,
            commit,
            state_root: root,
            receipt_head: self.live.receipt_head,
            receipt_count: self.live.receipt_count,
            founded,
        })
    }
}

/// The scalar fields of `src` around an empty entity vector (filled by [`stage`]).
fn stage_template(src: &State) -> State {
    State {
        cognition: src.cognition,
        publications: src.publications,
        last_publication_generation: src.last_publication_generation,
        evolution: src.evolution,
        receipt_head: src.receipt_head,
        receipt_count: src.receipt_count,
        evidence_head: src.evidence_head,
        evidence_count: src.evidence_count,
        reconcile_head: src.reconcile_head,
        reconcile_count: src.reconcile_count,
        reclaimed_head: src.reclaimed_head,
        reclaimed_count: src.reclaimed_count,
        next_founding_index: src.next_founding_index,
        legacy: src.legacy,
        entities: Vec::new(),
    }
}

impl Drop for Session {
    /// The session's memory is released at once; the encoding buffer, which holds the last state image, is cleared
    /// first so the freed memory does not carry it.
    fn drop(&mut self) {
        for b in self.encoded.iter_mut() {
            // SAFETY: a valid, aligned, exclusively borrowed byte.
            unsafe { core::ptr::write_volatile(b, 0) };
        }
        self.live.entities.clear();
        self.staging.entities.clear();
    }
}
