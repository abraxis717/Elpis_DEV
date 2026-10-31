//! The volatile session, attacked (std test harness).
//!
//! * The differential: every random event a session accepts or refuses is judged by the pure law
//!   ([`model::apply`], the one the retiring durable authority applies), outcome for outcome and state for state;
//!   only the persistence-only kinds, which a session must refuse `PersistenceRetired`, differ.
//! * Every refusal is atomic: the live state, its coordinate and the accepted-transition count are unchanged.
//! * Transitions never allocate (a thread-local counting allocator), and a session's footprint is constant.
//! * Sessions share nothing and carry nothing: another session's coordinates are refused, and a new session is a
//!   fresh genesis.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::vec::Vec;

use crate::error::Code;
use crate::model::*;
use crate::register::{EvolutionState, StateFormat};
use crate::session::*;

// -- a counting allocator, per thread (tests run concurrently) ----------------------------------------------

struct Counting;

std::thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

fn note() {
    let _ = TRACKING.try_with(|t| {
        if t.get() {
            let _ = ALLOCATIONS.try_with(|a| a.set(a.get() + 1));
        }
    });
}

// SAFETY: forwards to the system allocator; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        note();
        System.realloc(p, layout, size)
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        note();
        System.dealloc(p, layout)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn counted<R>(f: impl FnOnce() -> R) -> (R, u64) {
    ALLOCATIONS.with(|a| a.set(0));
    TRACKING.with(|t| t.set(true));
    let r = f();
    TRACKING.with(|t| t.set(false));
    (r, ALLOCATIONS.with(|a| a.get()))
}

// -- helpers ------------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn digest(&mut self) -> Digest {
        let mut d = [0u8; 32];
        for chunk in d.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        d
    }
}

fn label(b: u8) -> Digest {
    [b; 32]
}

fn config(capacity: u32, nonce: u8) -> SessionConfig {
    SessionConfig { capacity, nonce: [nonce; 32], operator_label: label(1), runtime_label: label(2) }
}

fn session(capacity: u32, nonce: u8) -> Session {
    Session::create(&config(capacity, nonce)).expect("create")
}

fn operator(s: &Session) -> Digest {
    s.state().entities[0].id
}

fn runtime(s: &Session) -> Digest {
    s.state().entities[1].id
}

fn next_seq(s: &Session, who: &Digest) -> u64 {
    s.state().entity(who).map_or(1, |e| e.sender_watermark + 1)
}

fn send(s: &mut Session, e: &Event) -> Result<SessionReceipt, Code> {
    let c = s.coordinate();
    s.transition(&c, e)
}

fn found(s: &mut Session, kind: EntityKind, lab: u8, authority: u32, scope: u8) -> Digest {
    let op = operator(s);
    let id = s.next_entity_id(kind, &label(lab));
    let source = kind == EntityKind::Source;
    let e = Event::entity_found(
        &op,
        next_seq(s, &op),
        &id,
        kind,
        &label(lab),
        &if source { label(scope) } else { ZERO },
        source as u64,
        if source { 8 } else { 0 },
        if source { authority } else { 0 },
    );
    send(s, &e).expect("found");
    id
}

fn lifecycle(s: &mut Session, id: &Digest, from: Lifecycle, to: Lifecycle) -> Result<SessionReceipt, Code> {
    let op = operator(s);
    let e = Event::entity_lifecycle(&op, next_seq(s, &op), id, from, to);
    send(s, &e)
}

/// The prior the pure law sees for a session's current state.
fn prior(s: &Session) -> Prior<'_> {
    Prior {
        lineage: *s.lineage(),
        generation: s.generation(),
        record_digest: *s.commit(),
        state_root: *s.state_root(),
        state: s.state(),
    }
}

fn persistence_only(kind: EventKind) -> bool {
    matches!(
        kind,
        EventKind::ImportLegacy
            | EventKind::ReconcileRollback
            | EventKind::ReconcileUnauthenticated
            | EventKind::ReconcileRepin
    )
}

/// A random event over a session's current entities: well-formed or not, authorized or not, in or out of order.
/// Mostly the sender a kind authorizes and a plausible target (so every kind is often accepted), with a steady
/// fraction of foreign senders, unknown targets, wrong sequences and wrong expectations.
fn random_event(r: &mut Rng, s: &Session) -> Event {
    let ids: Vec<Digest> = s.state().entities.iter().map(|e| e.id).collect();
    let pick = |r: &mut Rng| -> Digest {
        match r.below(10) {
            0 => r.digest(),
            1 => ids[r.below(2) as usize], // a founder, rarely
            _ if ids.len() > 2 => ids[2 + r.below(ids.len() as u64 - 2) as usize],
            _ => ids[r.below(ids.len() as u64) as usize],
        }
    };
    let kind_slot = r.below(16);
    // An ACTIVE entity of the kind the event's authorization names (the founders may have gone DORMANT).
    let active = |kind: EntityKind, fallback: Digest| {
        s.state()
            .entities
            .iter()
            .find(|e| e.kind == kind && e.lifecycle == Lifecycle::Active)
            .map_or(fallback, |e| e.id)
    };
    let authorized = match kind_slot {
        7..=9 => active(EntityKind::Runtime, runtime(s)),
        10 | 11 if r.below(2) == 0 => active(EntityKind::Runtime, runtime(s)),
        _ => active(EntityKind::Operator, operator(s)),
    };
    let sender = if r.below(7) == 0 { pick(r) } else { authorized };
    let seq = match r.below(12) {
        0 => next_seq(s, &sender) + 1,
        1 => next_seq(s, &sender).saturating_sub(1),
        _ => next_seq(s, &sender),
    };
    let lifecycles = [Lifecycle::Founded, Lifecycle::Active, Lifecycle::Dormant, Lifecycle::Terminated];
    let identity = |r: &mut Rng| if r.below(8) == 0 { ZERO } else { r.digest() };
    let cognition = match s.state().cognition {
        Cognition::Anchored { identity, .. } => identity,
        Cognition::Unanchored => ZERO,
    };
    match kind_slot {
        0..=2 => {
            let kind = [EntityKind::Source, EntityKind::Source, EntityKind::Operator, EntityKind::Runtime]
                [r.below(4) as usize];
            let lab = label(r.below(250) as u8);
            let id = if r.below(6) == 0 { r.digest() } else { s.next_entity_id(kind, &lab) };
            let source = kind == EntityKind::Source;
            Event::entity_found(
                &sender,
                seq,
                &id,
                kind,
                &lab,
                &if source { label(1 + r.below(3) as u8) } else { ZERO },
                if source { 1 + r.below(2) } else { 0 },
                if source { 1 + r.below(16) as u32 } else { 0 },
                if source { r.below(6) as u32 } else { 0 },
            )
        }
        3..=5 => {
            let target = pick(r);
            let current = s.state().entity(&target).map(|e| e.lifecycle);
            let from = match current {
                Some(l) if r.below(4) != 0 => l,
                _ => lifecycles[r.below(4) as usize],
            };
            Event::entity_lifecycle(&sender, seq, &target, from, lifecycles[r.below(4) as usize])
        }
        6 => {
            let target = pick(r);
            let current = s.state().entity(&target).map_or(1, |e| e.source_version);
            Event::source_version(
                &sender,
                seq,
                &target,
                if r.below(5) == 0 { current + 1 } else { current },
                current + 1 + r.below(2),
                1 + r.below(16) as u32,
                &label(1 + r.below(3) as u8),
            )
        }
        7 => Event::anchor(
            &sender,
            seq,
            if r.below(4) != 0 { StateFormat::K1 } else { StateFormat::Composite },
            &identity(r),
        ),
        8 => {
            let count = r.below(3);
            Event::publish(
                &sender,
                seq,
                if r.below(2) == 0 { StateFormat::K1 } else { StateFormat::Composite },
                &if r.below(5) == 0 { r.digest() } else { cognition },
                &identity(r),
                count,
                &if count == 0 { ZERO } else { r.digest() },
            )
        }
        9 => Event::import_composite(&sender, seq, &if r.below(3) == 0 { r.digest() } else { cognition }, &identity(r)),
        10 => {
            let expected = if r.below(5) == 0 { EvolutionState::GENESIS } else { s.state().evolution };
            Event::evolution_reserve(&sender, seq, &expected, &r.digest())
        }
        11 => {
            let expected = if r.below(5) == 0 { EvolutionState::GENESIS } else { s.state().evolution };
            Event::evolution_finalize(&sender, seq, &expected, &identity(r))
        }
        12 => Event::entity_reclaim(&sender, seq, &pick(r)),
        13 => {
            let mut record = [0u8; LEGACY_RECORD_BYTES];
            for b in record.iter_mut() {
                *b = r.below(256) as u8;
            }
            Event::import_legacy(&sender, seq, &record)
        }
        14 => Event::reconcile_rollback(&sender, seq, s.generation() + 1 + r.below(3), &r.digest(), r.below(3) as u8),
        _ => {
            if r.below(2) == 0 {
                Event::reconcile_unauthenticated(
                    &sender,
                    seq,
                    r.below(2) as u8,
                    SLOT_CLASS_UNAUTHENTICATED,
                    &r.digest(),
                )
            } else {
                Event::reconcile_repin(&sender, seq, [r.below(5) as u8, r.below(5) as u8], s.commit())
            }
        }
    }
}

// -- tests --------------------------------------------------------------------------------------------------

#[test]
fn a_session_starts_at_a_fresh_bounded_genesis() {
    let s = session(8, 7);
    let st = s.state();
    assert_eq!(s.generation(), 1);
    assert_eq!(st.entities.len(), 2);
    assert!(st.entities.iter().all(|e| e.lifecycle == Lifecycle::Active));
    assert_eq!((st.entities[0].kind, st.entities[1].kind), (EntityKind::Operator, EntityKind::Runtime));
    assert_eq!(st.cognition, Cognition::Unanchored);
    assert_eq!(st.evolution, EvolutionState::GENESIS);
    assert_eq!((st.receipt_count, st.publications, st.evidence_count, st.reclaimed_count), (1, 0, 0, 0));
    let c = s.coordinate();
    assert_eq!((c.session, c.generation, c.state_root), (*s.lineage(), 1, *s.state_root()));
    assert!(c.instance >= 1);
    assert_eq!(s.volatile_bytes(), Session::volatile_bytes_for(8));
    assert!(s.volatile_bytes() > state_bytes(8));
    for bad in [0, 1, MAX_CAPACITY + 1] {
        assert_eq!(Session::create(&config(bad, 7)).err(), Some(Code::Invalid));
        assert_eq!(Session::volatile_bytes_for(bad), 0);
    }
    // A zero nonce is drawn: two such sessions are distinct. The same nonce gives the same lineage
    // (deterministic), and still two distinct instances.
    let drawn_a = Session::create(&SessionConfig { nonce: ZERO, ..config(8, 0) }).unwrap();
    let drawn_b = Session::create(&SessionConfig { nonce: ZERO, ..config(8, 0) }).unwrap();
    assert_ne!(drawn_a.lineage(), drawn_b.lineage());
    let twin = session(8, 7);
    assert_eq!(twin.lineage(), s.lineage());
    assert_eq!(twin.state_root(), s.state_root());
    assert_ne!(twin.instance(), s.instance());
}

#[test]
fn the_session_applies_exactly_the_pure_law_and_refuses_atomically() {
    let mut accepted = [0u32; 15];
    let mut refused = [0u32; 15];
    let mut codes = std::collections::BTreeSet::new();
    for seed in 1..=6u64 {
        let mut r = Rng(seed * 0x1234_5678);
        let mut s = session(12, seed as u8);
        for _ in 0..6000 {
            let e = random_event(&mut r, &s);
            let expected = apply(&prior(&s), s.capacity(), &e);
            let before_state = s.state().clone();
            let before = s.coordinate();
            let (before_ok, before_refused) = s.counters();
            let got = send(&mut s, &e);
            match (&expected, &got) {
                _ if persistence_only(e.kind) => {
                    assert_eq!(got.err(), Some(Code::PersistenceRetired), "{:?}", e.kind);
                }
                (Ok(a), Ok(receipt)) => {
                    assert_eq!(s.state(), &a.state);
                    assert_eq!(
                        (receipt.generation, receipt.receipt, receipt.founded),
                        (a.generation, a.receipt, a.founded)
                    );
                    assert_eq!(receipt.predecessor_generation, before.generation);
                    let mut buf = std::vec![0u8; state_bytes(s.capacity())];
                    a.state.encode(s.capacity(), &mut buf);
                    assert_eq!(*s.state_root(), state_root(s.lineage(), a.generation, s.capacity(), &buf));
                    assert_eq!(receipt.commit, commit_digest(s.lineage(), a.generation, s.state_root(), &a.receipt));
                    assert_eq!(receipt.receipt_head, receipt_chain(&before_state.receipt_head, &a.receipt));
                    accepted[e.kind as usize] += 1;
                    continue;
                }
                (Err(want), Err(have)) => assert_eq!(want, have, "{:?}", e.kind),
                _ => panic!("{:?}: law {:?}, session {:?}", e.kind, expected.as_ref().err(), got.as_ref().err()),
            }
            // Refused: nothing moved but the refusal count.
            codes.insert(got.err().unwrap() as i32);
            refused[e.kind as usize] += 1;
            assert_eq!(s.state(), &before_state);
            assert_eq!(s.coordinate(), before);
            assert_eq!(s.counters(), (before_ok, before_refused + 1));
        }
    }
    // Not vacuous: every non-persistence kind was accepted somewhere, every kind refused, and many distinct codes
    // occurred (including the volatile session's own).
    for kind in [2usize, 3, 4, 5, 6, 7, 8, 9, 14] {
        assert!(accepted[kind] > 0, "kind {kind} never accepted");
    }
    for (kind, n) in refused.iter().enumerate().take(15).skip(2) {
        assert!(*n > 0, "kind {kind} never refused");
    }
    assert!(codes.len() >= 12, "{codes:?}");
    assert!(codes.contains(&(Code::PersistenceRetired as i32)));
    let (a, f): (u32, u32) = (accepted.iter().sum(), refused.iter().sum());
    std::eprintln!(
        "differential: {} events ({a} accepted, {f} refused), {} distinct refusal codes {codes:?}; accepted by kind \
         {:?}; refused by kind {:?}",
        a + f,
        codes.len(),
        &accepted[2..],
        &refused[2..]
    );
}

#[test]
fn persistence_only_operations_are_retired_even_when_well_formed() {
    let mut s = session(8, 3);
    let op = operator(&s);
    let watermark = s.state().entity(&op).unwrap().sender_watermark;
    let snapshot = crate::register::Snapshot::GENESIS.encode();
    let events = [
        Event::import_legacy(&op, watermark + 1, &snapshot),
        Event::reconcile_rollback(&op, watermark + 1, s.generation() + 5, &[9; 32], 0),
        Event::reconcile_unauthenticated(&op, watermark + 1, 0, SLOT_CLASS_UNAUTHENTICATED, &[9; 32]),
        Event::reconcile_repin(&op, watermark + 1, [0, 0], s.commit()),
    ];
    for e in events {
        // The pure law (the durable authority's) would consider these; a session never does.
        let before = s.coordinate();
        assert_eq!(send(&mut s, &e).err(), Some(Code::PersistenceRetired), "{:?}", e.kind);
        assert_eq!(s.coordinate(), before);
        assert_eq!(s.state().entity(&op).unwrap().sender_watermark, watermark, "no sequence consumed");
    }
    // GENESIS is creation only.
    let genesis = Event::genesis(&[1; 32], &label(1), &label(2));
    assert_eq!(send(&mut s, &genesis).err(), Some(Code::Invalid));
    assert_eq!((s.state().reconcile_count, s.state().legacy), (0, None));
}

#[test]
fn coordinates_of_another_session_or_generation_are_refused() {
    let mut a = session(8, 1);
    let mut b = session(8, 2);
    let twin = session(8, 1); // same nonce as `a`: same lineage, another instance
    let op = operator(&b);
    let e = Event::entity_lifecycle(&op, 1, &runtime(&b), Lifecycle::Active, Lifecycle::Dormant);
    assert_eq!(b.transition(&a.coordinate(), &e).err(), Some(Code::SessionMismatch));
    let mut twin_coordinate = a.coordinate();
    twin_coordinate.instance = twin.instance();
    assert_eq!(
        a.transition(&twin_coordinate, &Event::entity_reclaim(&operator(&a), 1, &ZERO)).err(),
        Some(Code::SessionMismatch)
    );
    let mut stale = b.coordinate();
    stale.generation -= 1;
    assert_eq!(b.transition(&stale, &e).err(), Some(Code::Stale));
    let mut forged = b.coordinate();
    forged.state_root[0] ^= 1;
    assert_eq!(b.transition(&forged, &e).err(), Some(Code::Stale));
    let held = b.coordinate();
    send(&mut b, &e).expect("current coordinate");
    // The coordinate held before that transition is stale now.
    let again = Event::entity_lifecycle(&op, 2, &runtime(&b), Lifecycle::Dormant, Lifecycle::Active);
    assert_eq!(b.transition(&held, &again).err(), Some(Code::Stale));
    assert_eq!(b.counters(), (1, 4)); // three refusals before the accepted transition, one after
}

#[test]
fn capacity_is_refused_before_any_change_and_reclaimed_rows_are_reusable() {
    let mut s = session(5, 4);
    let bytes = s.volatile_bytes();
    let a = found(&mut s, EntityKind::Source, 10, 7, 1);
    found(&mut s, EntityKind::Source, 11, 8, 1);
    found(&mut s, EntityKind::Source, 12, 9, 1);
    assert_eq!(s.state().entities.len(), 5);
    let op = operator(&s);
    let id = s.next_entity_id(EntityKind::Source, &label(13));
    let full = Event::entity_found(&op, next_seq(&s, &op), &id, EntityKind::Source, &label(13), &label(1), 1, 8, 10);
    let before = (s.state().clone(), s.coordinate());
    assert_eq!(send(&mut s, &full).err(), Some(Code::Capacity));
    assert_eq!((s.state().clone(), s.coordinate()), before);
    // TERMINATED, then reclaimed: a row and its 32-bit source_authority (7) are free again; identities never reused.
    lifecycle(&mut s, &a, Lifecycle::Founded, Lifecycle::Terminated).unwrap();
    {
        let e_ = Event::entity_reclaim(&op, next_seq(&s, &op), &a);
        send(&mut s, &e_)
    }
    .unwrap();
    let reused = found(&mut s, EntityKind::Source, 13, 7, 1);
    assert_ne!(reused, a);
    assert_eq!(s.state().source_for_authority(7).unwrap().id, reused);
    assert_eq!((s.state().reclaimed_count, s.volatile_bytes()), (1, bytes));
}

#[test]
fn evidence_admission_follows_source_versions_scopes_and_lifecycles() {
    let mut s = session(8, 5);
    let src = found(&mut s, EntityKind::Source, 20, 3, 1);
    assert_eq!(s.check_evidence(&src, 1, &label(1)).err(), Some(Code::Lifecycle)); // FOUNDED
    lifecycle(&mut s, &src, Lifecycle::Founded, Lifecycle::Active).unwrap();
    assert_eq!(s.check_evidence(&src, 1, &label(1)), Ok(8));
    assert_eq!(s.check_evidence(&src, 2, &label(1)).err(), Some(Code::SourceStale));
    assert_eq!(s.check_evidence(&src, 1, &label(2)).err(), Some(Code::ScopeConflict));
    assert_eq!(s.check_evidence(&operator(&s), 0, &ZERO).err(), Some(Code::Unauthorized));
    assert_eq!(s.check_evidence(&[0xee; 32], 1, &label(1)).err(), Some(Code::UnknownEntity));
    let op = operator(&s);
    {
        let e_ = Event::source_version(&op, next_seq(&s, &op), &src, 1, 2, 4, &label(2));
        send(&mut s, &e_)
    }
    .unwrap();
    assert_eq!(s.check_evidence(&src, 1, &label(1)).err(), Some(Code::SourceStale)); // revision: v1 never again
    assert_eq!(s.check_evidence(&src, 2, &label(2)), Ok(4));
    lifecycle(&mut s, &src, Lifecycle::Active, Lifecycle::Dormant).unwrap();
    assert_eq!(s.check_evidence(&src, 2, &label(2)).err(), Some(Code::Lifecycle));
}

#[test]
fn in_session_cognition_and_evolution_coordinates_evolve_in_memory() {
    let mut s = session(4, 6);
    let rt = runtime(&s);
    let k1 = [0x11; 32];
    {
        let e_ = Event::anchor(&rt, next_seq(&s, &rt), StateFormat::K1, &k1);
        send(&mut s, &e_)
    }
    .unwrap();
    let after = [0x22; 32];
    let evidence = [0x33; 32];
    {
        let e_ = Event::publish(&rt, next_seq(&s, &rt), StateFormat::K1, &k1, &after, 4, &evidence);
        send(&mut s, &e_)
    }
    .unwrap();
    assert_eq!(s.state().cognition, Cognition::Anchored { format: StateFormat::K1, identity: after });
    assert_eq!((s.state().publications, s.state().evidence_count), (2, 4));
    let composite = [0x44; 32];
    {
        let e_ = Event::import_composite(&rt, next_seq(&s, &rt), &after, &composite);
        send(&mut s, &e_)
    }
    .unwrap();
    let expected = s.state().evolution;
    let e = Event::evolution_reserve(&rt, next_seq(&s, &rt), &expected, &[0x55; 32]);
    send(&mut s, &e).unwrap();
    let pending = s.state().evolution;
    assert!(pending.assertion().is_some());
    let e = Event::evolution_finalize(&rt, next_seq(&s, &rt), &pending, &[0x66; 32]);
    send(&mut s, &e).unwrap();
    assert_eq!((s.state().evolution.revision(), s.state().evolution.head()), (1, [0x66; 32]));
    assert_eq!(s.generation(), 6);
}

#[test]
fn transitions_never_allocate_and_the_footprint_is_constant() {
    let mut s = session(16, 8);
    let bytes = s.volatile_bytes();
    let mut r = Rng(99);
    let (mut accepted, mut refused) = (0u64, 0u64);
    for _ in 0..20_000 {
        // Building the event is the test's own allocation; only the transition itself is counted.
        let e = random_event(&mut r, &s);
        let c = s.coordinate();
        let (result, allocations) = counted(|| s.transition(&c, &e));
        assert_eq!(allocations, 0, "{:?} allocated ({:?})", e.kind, result.err());
        if result.is_ok() {
            accepted += 1;
        } else {
            refused += 1;
        }
        assert!(s.state().entities.len() <= 16);
    }
    assert_eq!(s.volatile_bytes(), bytes);
    assert!(accepted > 2_000 && refused > 2_000, "{accepted} accepted, {refused} refused");
    std::eprintln!(
        "allocation-free: 20000 transitions ({accepted} accepted, {refused} refused), 0 allocations; \
                    footprint {bytes} bytes constant"
    );
    // Creation reserves everything once: exactly three buffers (live rows, staging rows, the encoding).
    let (created, allocations) = counted(|| Session::create(&config(16, 9)).unwrap());
    assert_eq!(allocations, 3);
    drop(created);
}

#[test]
fn a_long_random_run_stays_bounded() {
    let mut s = session(32, 10);
    let bytes = s.volatile_bytes();
    let mut r = Rng(2026);
    for _ in 0..100_000 {
        let e = random_event(&mut r, &s);
        let _ = send(&mut s, &e);
        assert!(s.state().entities.len() <= 32);
    }
    let (ok, refused) = s.counters();
    assert_eq!(ok + refused, 100_000);
    assert_eq!(s.generation(), ok + 1);
    assert_eq!(s.state().receipt_count, ok + 1);
    assert_eq!(s.volatile_bytes(), bytes);
}

#[test]
fn a_new_session_carries_nothing_from_the_last() {
    let mut a = session(8, 11);
    let src = found(&mut a, EntityKind::Source, 30, 5, 1);
    lifecycle(&mut a, &src, Lifecycle::Founded, Lifecycle::Active).unwrap();
    let rt = runtime(&a);
    let e = Event::anchor(&rt, next_seq(&a, &rt), StateFormat::Composite, &[0x77; 32]);
    send(&mut a, &e).unwrap();
    let a_coordinate = a.coordinate();
    assert!(a.generation() > 1);
    drop(a);
    // Even with the same nonce, a new session is a genesis: nothing of the last carries over.
    for nonce in [11u8, 12] {
        let mut b = session(8, nonce);
        assert_eq!(b.generation(), 1);
        assert_eq!(b.state().entities.len(), 2);
        assert_eq!(b.state().cognition, Cognition::Unanchored);
        assert_eq!(b.state().entity(&src), None);
        assert_eq!(b.check_evidence(&src, 1, &label(1)).err(), Some(Code::UnknownEntity));
        let e = Event::entity_reclaim(&operator(&b), 1, &src);
        assert_eq!(b.transition(&a_coordinate, &e).err(), Some(Code::SessionMismatch));
    }
}

#[cfg(feature = "c-abi")]
#[test]
fn the_c_abi_creates_transitions_and_destroys() {
    use crate::ffi::*;
    unsafe {
        let cfg =
            CConfig { abi_version: 1, capacity: 4, nonce: [3; 32], operator_label: label(1), runtime_label: label(2) };
        let mut p = core::ptr::null_mut();
        assert_eq!(elpis_ecs_c_session_create(&cfg, &mut p), 0);
        let mut c: CCoordinate = core::mem::zeroed();
        assert_eq!(elpis_ecs_c_session_read_coordinate(p, &mut c), 0);
        let op = (*p).state().entities[0].id;
        let e = Event::entity_lifecycle(&op, 1, &(*p).state().entities[1].id, Lifecycle::Active, Lifecycle::Dormant);
        let raw = e.encode();
        let mut receipt: CReceipt = core::mem::zeroed();
        assert_eq!(elpis_ecs_c_session_transition(p, &c, &raw, &mut receipt), 0);
        assert_eq!((receipt.generation, receipt.predecessor_generation), (2, 1));
        // The same coordinate is stale now; a malformed event is Invalid; both leave the session as it was.
        assert_eq!(elpis_ecs_c_session_transition(p, &c, &raw, core::ptr::null_mut()), Code::Stale as i32);
        let mut bad = raw;
        bad[3] = 1;
        assert_eq!(elpis_ecs_c_session_transition(p, &c, &bad, core::ptr::null_mut()), Code::Invalid as i32);
        let mut status: CStatus = core::mem::zeroed();
        assert_eq!(elpis_ecs_c_session_read_status(p, &mut status), 0);
        assert_eq!((status.generation, status.transitions, status.refusals), (2, 1, 1));
        assert_eq!(status.volatile_bytes, elpis_ecs_c_session_bytes(4));
        elpis_ecs_c_session_destroy(p);
    }
}
