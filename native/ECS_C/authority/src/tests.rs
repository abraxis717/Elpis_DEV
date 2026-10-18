//! ECS_C authority qualification: the transition law, the crash matrix, the T1 adversary matrix and the
//! boundedness of files, bytes, recovery work and retained metadata over long sequences.
//!
//! Deterministic and bounded: fixed key material, private temporary directories, no sleeps, no threads.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use elpis_continuity::sha256;
use elpis_continuity::{EvolutionState, StateFormat, MAX_COUNTER};

use crate::error::Code;
use crate::legacy::read_continuity;
use crate::model::*;
use crate::record::*;
use crate::store::probe::{Fault, Probe};
use crate::store::*;

// -- fixtures ------------------------------------------------------------------------------------------------

const MASTER: [u8; 32] = [0x5a; 32];
const NONCE: [u8; 32] = [0xc3; 32];

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> TempDir {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("elpis-ecs-c-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }
    fn state(&self) -> PathBuf {
        self.0.join("state")
    }
    fn authority(&self) -> PathBuf {
        self.0.join("authority")
    }
    fn slot(&self, i: usize) -> PathBuf {
        self.state().join(STATE_FILES[i])
    }
    fn witness(&self, i: usize) -> PathBuf {
        self.authority().join(WITNESS_FILES[i])
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn d(label: &str) -> Digest {
    sha256::digest(&[label.as_bytes()])
}

fn provision_fixed(t: &TempDir, capacity: u32, master: &[u8; 32], nonce: &Digest) -> Result<Provisioned, Code> {
    provision_with(
        &mut Io::default(),
        &t.state(),
        &t.authority(),
        capacity,
        &d("operator"),
        &d("runtime"),
        master,
        nonce,
    )
}

fn open(t: &TempDir) -> Authority {
    let mut a = Authority::new(&t.state(), &t.authority()).unwrap();
    a.open().unwrap();
    a
}

fn reopen(t: &TempDir) -> (Result<Report, Refusal>, Authority) {
    let mut a = Authority::new(&t.state(), &t.authority()).unwrap();
    let r = a.open();
    (r, a)
}

fn fixture(capacity: u32) -> (TempDir, Authority) {
    let t = TempDir::new();
    provision_fixed(&t, capacity, &MASTER, &NONCE).unwrap();
    let a = open(&t);
    (t, a)
}

fn disposition(r: &Report) -> Disposition {
    Disposition::from_u32(r.disposition).unwrap()
}

fn refused(r: Result<Report, Refusal>) -> (Code, Option<Disposition>) {
    let refusal = r.unwrap_err();
    (refusal.code, refusal.report.map(|r| disposition(&r)))
}

fn cur(a: &Authority) -> (u64, Digest) {
    let c = a.current().unwrap();
    (c.generation, c.state_root)
}

fn op(a: &Authority) -> Digest {
    a.current().unwrap().state.entities[0].id
}

fn rt(a: &Authority) -> Digest {
    a.current().unwrap().state.entities[1].id
}

fn seq(a: &Authority, who: &Digest) -> u64 {
    a.current().unwrap().state.entity(who).unwrap().sender_watermark + 1
}

fn tx(a: &mut Authority, e: &Event) -> Result<Receipt, Code> {
    let (g, r) = cur(a);
    a.transition(g, &r, e)
}

/// `tx` on a local, with an event that may read the authority (the event is built before the mutable borrow).
macro_rules! send {
    ($a:ident, $e:expr) => {{
        let event: Event = *$e;
        tx(&mut $a, &event)
    }};
}

fn found(a: &mut Authority, kind: EntityKind, label: &str, scope: &Digest, version: u64, limit: u32) -> Digest {
    let o = op(a);
    let id = a.next_entity_id(kind, &d(label)).unwrap();
    let s = seq(a, &o);
    let r = tx(a, &Event::entity_found(&o, s, &id, kind, &d(label), scope, version, limit)).unwrap();
    assert_eq!(r.founded, Some(id));
    id
}

fn found_source(a: &mut Authority, label: &str) -> Digest {
    found(a, EntityKind::Source, label, &d("scope"), 1, 8)
}

fn lifecycle(a: &mut Authority, id: &Digest, from: Lifecycle, to: Lifecycle) -> Result<Receipt, Code> {
    let o = op(a);
    let s = seq(a, &o);
    tx(a, &Event::entity_lifecycle(&o, s, id, from, to))
}

fn active_source(a: &mut Authority, label: &str) -> Digest {
    let id = found_source(a, label);
    lifecycle(a, &id, Lifecycle::Founded, Lifecycle::Active).unwrap();
    id
}

fn identity(a: &Authority) -> Option<Digest> {
    match a.current().unwrap().state.cognition {
        Cognition::Anchored { identity, .. } => Some(identity),
        Cognition::Unanchored => None,
    }
}

fn anchor(a: &mut Authority, id: &Digest) -> Result<Receipt, Code> {
    let r = rt(a);
    let s = seq(a, &r);
    tx(a, &Event::anchor(&r, s, StateFormat::K1, id))
}

fn publish_next(a: &mut Authority, label: &str) -> Result<Receipt, Code> {
    let r = rt(a);
    let s = seq(a, &r);
    let before = identity(a).unwrap();
    tx(a, &Event::publish(&r, s, StateFormat::K1, &before, &d(label), 0, &ZERO))
}

/// A short history so both record slots and both witness slots hold valid content.
fn with_history(capacity: u32) -> (TempDir, Authority) {
    let (t, mut a) = fixture(capacity);
    anchor(&mut a, &d("k1-0")).unwrap();
    publish_next(&mut a, "k1-1").unwrap();
    publish_next(&mut a, "k1-2").unwrap();
    (t, a)
}

fn raw(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap()
}

fn put_raw(path: &Path, bytes: &[u8]) {
    let mode = fs::metadata(path).unwrap().permissions().mode();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}

/// The slot index holding the record with `digest`.
fn slot_of(t: &TempDir, digest: &Digest) -> usize {
    (0..2)
        .find(|i| {
            let b = raw(&t.slot(*i));
            b.len() >= 32 && b[b.len() - 32..] == digest[..]
        })
        .expect("a slot holds the record")
}

/// Recompute the unkeyed checksum of a modified record (anyone can; only the MAC needs the key).
fn reseal_unkeyed(b: &mut [u8]) {
    let at = b.len() - 32;
    let sum = sha256::digest(&[RECORD_CHECKSUM, &b[..at]]);
    b[at..].copy_from_slice(&sum);
}

fn files(t: &TempDir) -> BTreeMap<String, (u64, u32)> {
    let mut out = BTreeMap::new();
    for (dir, tag) in [(t.state(), "state"), (t.authority(), "authority")] {
        for e in fs::read_dir(&dir).unwrap() {
            let e = e.unwrap();
            let m = fs::symlink_metadata(e.path()).unwrap();
            out.insert(format!("{tag}/{}", e.file_name().to_string_lossy()), (m.len(), m.mode() & 0o777));
        }
    }
    out
}

type Damage = fn(&TempDir);
type Craft = fn(&Record, &Record) -> Vec<u8>;

fn keys() -> Keys {
    Keys::derive(&MASTER)
}

/// Seal a successor of `older` carrying `state` (possibly not what the law produces), with the real key.
fn seal_after(older: &Record, generation: u64, state: State, event: Event) -> Vec<u8> {
    let receipt =
        receipt(&older.lineage, generation, older.generation, &older.digest, &older.state_root, &event.encode());
    let (_, bytes) = Record::seal(
        &keys(),
        older.capacity,
        generation,
        older.generation,
        &older.lineage,
        &older.digest,
        &older.state_root,
        &receipt,
        state,
        event,
    );
    bytes
}

// -- layout and wire ------------------------------------------------------------------------------------------

#[test]
fn layout_constants_and_codes() {
    assert_eq!(record_bytes(0), 1344);
    assert_eq!(record_bytes(4), 1344 + 4 * 160);
    for capacity in [MIN_CAPACITY, 4, 256, MAX_CAPACITY] {
        assert_eq!(capacity_of(record_bytes(capacity) as u64), Some(capacity));
    }
    assert_eq!(capacity_of(record_bytes(1) as u64), None);
    assert_eq!(capacity_of(record_bytes(MAX_CAPACITY + 1) as u64), None);
    assert_eq!(capacity_of(record_bytes(4) as u64 + 1), None);
    assert_eq!(WITNESS_BYTES, 240);
    let mut names: Vec<&str> = Code::ALL.iter().map(|c| c.name()).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), Code::ALL.len());
    for c in Code::ALL {
        assert_eq!(Code::from_i32(c as i32), Some(c));
    }
    for s in Step::ALL {
        assert_eq!(Step::from_u32(s as u32), Some(s));
    }
    for d in Disposition::ALL {
        assert_eq!(Disposition::from_u32(d as u32), Some(d));
    }
}

#[test]
fn every_event_round_trips_and_malformed_events_are_refused() {
    let (s, x) = (d("sender"), d("x"));
    let ev = EvolutionState::pending(3, d("head"), d("assertion")).unwrap();
    let events = [
        Event::genesis(&x, &d("o"), &d("r")),
        Event::entity_found(&s, 1, &x, EntityKind::Source, &d("l"), &d("scope"), 1, 8),
        Event::entity_lifecycle(&s, 2, &x, Lifecycle::Founded, Lifecycle::Active),
        Event::source_version(&s, 3, &x, 1, 2, 9, &d("scope2")),
        Event::anchor(&s, 4, StateFormat::Composite, &x),
        Event::publish(&s, 5, StateFormat::K1, &x, &d("y"), 3, &d("evidence")),
        Event::import_composite(&s, 6, &x, &d("y")),
        Event::evolution_reserve(&s, 7, &EvolutionState::GENESIS, &x),
        Event::evolution_finalize(&s, 8, &ev, &x),
        Event::import_legacy(&s, 9, &[7u8; LEGACY_RECORD_BYTES]),
        Event::reconcile_rollback(&s, 10, 9, &x, 1),
        Event::reconcile_unauthenticated(&s, 11, 1, 2, &x),
        Event::reconcile_repin(&s, 12, [1, 0], &x),
        Event::entity_reclaim(&s, 13, &x),
    ];
    assert_eq!(events.len(), EventKind::ALL.len());
    for e in events {
        let wire = e.encode();
        assert_eq!(Event::decode(&wire), Ok(e));
        // Reserved header bytes, unknown kinds, payload beyond the kind's length.
        for at in [2usize, 7] {
            let mut bad = wire;
            bad[at] = 1;
            assert_eq!(Event::decode(&bad), Err(Code::Invalid));
        }
        if e.kind.payload_len() < PAYLOAD_BYTES {
            let mut bad = wire;
            bad[PAYLOAD_AT + e.kind.payload_len()] = 1;
            assert_eq!(Event::decode(&bad), Err(Code::Invalid));
        }
        if !e.kind.entity_scoped() {
            let mut bad = wire;
            bad[48] = 1;
            assert_eq!(Event::decode(&bad), Err(Code::Invalid));
        }
    }
    let mut unknown = events[1].encode();
    unknown[0] = 99;
    assert_eq!(Event::decode(&unknown), Err(Code::Invalid));
    assert_eq!(Event::decode(&[0u8; 100]), Err(Code::Invalid));
}

#[test]
fn hmac_is_the_authentication_and_the_checksum_only_identity() {
    // A different key changes the MAC and the deployment id, never the checksum of identical bytes.
    let (k1, k2) = (Keys::derive(&MASTER), Keys::derive(&[0x11; 32]));
    assert_ne!(k1.deployment, k2.deployment);
    let w = Witness {
        capacity: 4,
        generation: 9,
        lineage: d("l"),
        deployment: k1.deployment,
        record_digest: d("r"),
        state_root: d("s"),
    };
    let sealed = w.seal(&k1);
    assert_eq!(Witness::classify(&sealed, &k1), (SlotClass::Valid, Some(w)));
    assert_eq!(Witness::classify(&sealed, &k2).0, SlotClass::Unauthenticated);
    let mut torn = sealed;
    torn[40] ^= 1;
    assert_eq!(Witness::classify(&torn, &k1).0, SlotClass::Torn);
    assert_eq!(Witness::classify(&[0u8; WITNESS_BYTES], &k1).0, SlotClass::Empty);
}

// -- provisioning ---------------------------------------------------------------------------------------------

#[test]
fn genesis_is_deterministic_and_founds_the_operator_and_the_runtime() {
    let (t1, t2) = (TempDir::new(), TempDir::new());
    let p1 = provision_fixed(&t1, 4, &MASTER, &NONCE).unwrap();
    let p2 = provision_fixed(&t2, 4, &MASTER, &NONCE).unwrap();
    assert_eq!(p1, p2);
    assert_eq!(raw(&t1.slot(0)), raw(&t2.slot(0)));
    assert_eq!(raw(&t1.witness(0)), raw(&t2.witness(0)));
    // Another nonce is another lineage under the same key.
    let t3 = TempDir::new();
    let p3 = provision_fixed(&t3, 4, &MASTER, &[0x01; 32]).unwrap();
    assert_ne!(p3.lineage, p1.lineage);
    assert_eq!(p3.deployment, p1.deployment);

    let mut a = open(&t1);
    let report = a.report().unwrap();
    assert_eq!(disposition(&report), Disposition::Clean);
    assert!(report.replay_verified, "genesis is recomputed from its event");
    let c = a.current().unwrap();
    assert_eq!((c.generation, c.digest, c.lineage), (1, p1.record_digest, p1.lineage));
    let s = &c.state;
    assert_eq!(s.entities.len(), 2);
    assert_eq!((s.entities[0].kind, s.entities[0].lifecycle), (EntityKind::Operator, Lifecycle::Active));
    assert_eq!((s.entities[1].kind, s.entities[1].lifecycle), (EntityKind::Runtime, Lifecycle::Active));
    assert_eq!(s.entities[0].id, entity_id(&c.lineage, 0, EntityKind::Operator, &d("operator")));
    assert_eq!((s.cognition, s.evolution, s.next_founding_index), (Cognition::Unanchored, EvolutionState::GENESIS, 2));
    assert_eq!((s.receipt_count, s.receipt_head), (1, receipt_chain(&ZERO, &c.receipt)));
    a.close();
}

#[test]
fn provisioning_creates_exactly_the_fixed_file_set() {
    let t = TempDir::new();
    let mut io = Io::default();
    let p = provision_with(&mut io, &t.state(), &t.authority(), 4, &d("o"), &d("r"), &MASTER, &NONCE).unwrap();
    let r = record_bytes(4) as u64;
    assert_eq!(p.record_bytes, r);
    let expected: BTreeMap<String, (u64, u32)> = [
        ("authority/deployment.key", (32, 0o400)),
        ("authority/witness.a", (240, 0o600)),
        ("authority/witness.b", (240, 0o600)),
        ("state/authority.a", (r, 0o600)),
        ("state/authority.b", (r, 0o600)),
    ]
    .into_iter()
    .map(|(n, v)| (n.to_string(), v))
    .collect();
    assert_eq!(files(&t), expected);
    for dir in [t.state(), t.authority()] {
        assert_eq!(fs::metadata(dir).unwrap().mode() & 0o777, 0o700);
    }
    assert_eq!(io.counters.creates, 2 + 5, "two directories and five files, once");
    assert_eq!(io.counters.renames, 5);
}

#[test]
fn provisioning_refuses_to_overwrite_nest_admit_foreign_files_or_bad_capacity() {
    let t = TempDir::new();
    provision_fixed(&t, 4, &MASTER, &NONCE).unwrap();
    assert_eq!(provision_fixed(&t, 4, &MASTER, &NONCE), Err(Code::AlreadyProvisioned));

    for capacity in [0, 1, MAX_CAPACITY + 1] {
        assert_eq!(provision_fixed(&TempDir::new(), capacity, &MASTER, &NONCE), Err(Code::Invalid));
    }
    let nested = TempDir::new();
    let inner = nested.state().join("authority");
    let r = provision_with(&mut Io::default(), &nested.state(), &inner, 4, &ZERO, &ZERO, &MASTER, &NONCE);
    assert_eq!(r, Err(Code::Path));
    let r = provision_with(
        &mut Io::default(),
        &nested.authority().join("s"),
        &nested.authority(),
        4,
        &ZERO,
        &ZERO,
        &MASTER,
        &NONCE,
    );
    assert_eq!(r, Err(Code::Path));
    let r = provision_with(
        &mut Io::default(),
        Path::new("relative"),
        &nested.authority(),
        4,
        &ZERO,
        &ZERO,
        &MASTER,
        &NONCE,
    );
    assert_eq!(r, Err(Code::Path));

    let foreign = TempDir::new();
    fs::create_dir_all(foreign.state()).unwrap();
    fs::write(foreign.state().join("notes.txt"), b"x").unwrap();
    assert_eq!(provision_fixed(&foreign, 4, &MASTER, &NONCE), Err(Code::Corrupt));

    let open_dir = TempDir::new();
    fs::create_dir_all(open_dir.authority()).unwrap();
    fs::set_permissions(open_dir.authority(), fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(provision_fixed(&open_dir, 4, &MASTER, &NONCE), Err(Code::Unprotected));
}

#[test]
fn provisioning_interrupted_at_every_step_is_repeatable_or_published() {
    let provisioning = [
        Step::ProvisionFilesWritten,
        Step::ProvisionStateRenamed,
        Step::ProvisionStateSynced,
        Step::ProvisionKeyRenamed,
        Step::ProvisionWitnessBRenamed,
        Step::ProvisionAuthoritySynced,
        Step::ProvisionPublished,
        Step::ProvisionDone,
    ];
    for step in provisioning {
        let t = TempDir::new();
        let mut io = Io { probe: Probe { plan: Some((0, Fault::Die(step))), ..Probe::default() }, ..Io::default() };
        let r = provision_with(&mut io, &t.state(), &t.authority(), 4, &d("o"), &d("r"), &MASTER, &NONCE);
        assert_eq!(r, Err(Code::TestingProcessDeath), "{step:?}");
        let published = matches!(step, Step::ProvisionPublished | Step::ProvisionDone);
        let (opened, mut a) = reopen(&t);
        if published {
            assert_eq!(disposition(&opened.unwrap()), Disposition::Clean, "{step:?}");
            a.close();
            assert_eq!(provision_fixed(&t, 4, &MASTER, &NONCE), Err(Code::AlreadyProvisioned));
        } else {
            assert_eq!(refused(opened), (Code::ProvisionIncomplete, None), "{step:?}");
            // Never published: provisioning simply runs again over its own debris.
            provision_fixed(&t, 4, &MASTER, &NONCE).unwrap();
            let (opened, _a) = reopen(&t);
            assert_eq!(disposition(&opened.unwrap()), Disposition::Clean, "{step:?}");
        }
    }
}

// -- admission ------------------------------------------------------------------------------------------------

#[test]
fn open_refuses_an_unprotected_authority_directory_key_or_witness() {
    let cases: [(&str, Damage); 5] = [
        ("group-readable authority directory", |t| {
            fs::set_permissions(t.authority(), fs::Permissions::from_mode(0o750)).unwrap()
        }),
        ("writable key", |t| {
            fs::set_permissions(t.authority().join(KEY_FILE), fs::Permissions::from_mode(0o600)).unwrap()
        }),
        ("world-readable witness", |t| fs::set_permissions(t.witness(1), fs::Permissions::from_mode(0o644)).unwrap()),
        ("symlinked key", |t| {
            let key = t.authority().join(KEY_FILE);
            let elsewhere = t.0.join("key-elsewhere");
            fs::rename(&key, &elsewhere).unwrap();
            std::os::unix::fs::symlink(&elsewhere, &key).unwrap();
        }),
        ("symlinked authority directory", |t| {
            let moved = t.0.join("authority-real");
            fs::rename(t.authority(), &moved).unwrap();
            std::os::unix::fs::symlink(&moved, t.authority()).unwrap();
        }),
    ];
    for (name, damage) in cases {
        let t = TempDir::new();
        provision_fixed(&t, 4, &MASTER, &NONCE).unwrap();
        damage(&t);
        assert_eq!(refused(reopen(&t).0), (Code::Unprotected, None), "{name}");
    }
}

#[test]
fn open_refuses_foreign_missing_or_resized_files() {
    let cases: [(&str, Damage, Code); 6] = [
        ("extra file in the state directory", |t| fs::write(t.state().join("x"), b"x").unwrap(), Code::Corrupt),
        (
            "truncated slot",
            |t| fs::OpenOptions::new().write(true).open(t.slot(1)).unwrap().set_len(100).unwrap(),
            Code::Corrupt,
        ),
        (
            "grown slot pair",
            |t| {
                for i in 0..2 {
                    let f = fs::OpenOptions::new().write(true).open(t.slot(i)).unwrap();
                    f.set_len(record_bytes(4) as u64 + 1).unwrap();
                }
            },
            Code::Corrupt,
        ),
        (
            "symlinked slot",
            |t| {
                let elsewhere = t.0.join("slot-elsewhere");
                fs::rename(t.slot(1), &elsewhere).unwrap();
                std::os::unix::fs::symlink(&elsewhere, t.slot(1)).unwrap();
            },
            Code::Corrupt,
        ),
        ("missing witness", |t| fs::remove_file(t.witness(1)).unwrap(), Code::Corrupt),
        ("missing state directory", |t| fs::remove_dir_all(t.state()).unwrap(), Code::Corrupt),
    ];
    for (name, damage, code) in cases {
        let t = TempDir::new();
        provision_fixed(&t, 4, &MASTER, &NONCE).unwrap();
        damage(&t);
        assert_eq!(refused(reopen(&t).0), (code, None), "{name}");
    }
    assert_eq!(refused(reopen(&TempDir::new()).0), (Code::NotProvisioned, None));
}

#[test]
fn one_owner_at_a_time() {
    let (t, mut a) = fixture(4);
    assert_eq!(refused(reopen(&t).0), (Code::Locked, None));
    assert_eq!(a.open().unwrap_err().code, Code::Open);
    a.close();
    assert_eq!(a.current().unwrap_err(), Code::Closed);
    let _b = open(&t);
}

// -- the transition law ---------------------------------------------------------------------------------------

#[test]
fn entity_lifecycle_law() {
    let (_t, mut a) = fixture(8);
    let s = found_source(&mut a, "camera");
    assert_eq!(a.current().unwrap().state.entity(&s).unwrap().lifecycle, Lifecycle::Founded);
    assert_eq!(a.check_evidence(&s, 1, &d("scope")), Err(Code::Lifecycle), "a founded source admits nothing");
    let g = cur(&a).0;
    for (from, to) in [
        (Lifecycle::Founded, Lifecycle::Dormant),
        (Lifecycle::Active, Lifecycle::Dormant), // from mismatch
        (Lifecycle::Founded, Lifecycle::Founded),
    ] {
        assert_eq!(lifecycle(&mut a, &s, from, to).unwrap_err(), Code::Lifecycle, "{from:?}->{to:?}");
    }
    assert_eq!(cur(&a).0, g, "refusals write nothing");
    lifecycle(&mut a, &s, Lifecycle::Founded, Lifecycle::Active).unwrap();
    lifecycle(&mut a, &s, Lifecycle::Active, Lifecycle::Dormant).unwrap();
    assert_eq!(a.check_evidence(&s, 1, &d("scope")), Err(Code::Lifecycle));
    lifecycle(&mut a, &s, Lifecycle::Dormant, Lifecycle::Active).unwrap();
    assert_eq!(a.check_evidence(&s, 1, &d("scope")), Ok(8));
    lifecycle(&mut a, &s, Lifecycle::Active, Lifecycle::Terminated).unwrap();
    for to in [Lifecycle::Active, Lifecycle::Dormant, Lifecycle::Founded] {
        assert_eq!(lifecycle(&mut a, &s, Lifecycle::Terminated, to).unwrap_err(), Code::Lifecycle);
    }
    // The founders are never terminated, and the authority always keeps an ACTIVE operator.
    let (o, r) = (op(&a), rt(&a));
    assert_eq!(lifecycle(&mut a, &o, Lifecycle::Active, Lifecycle::Terminated).unwrap_err(), Code::Lifecycle);
    assert_eq!(lifecycle(&mut a, &r, Lifecycle::Active, Lifecycle::Terminated).unwrap_err(), Code::Lifecycle);
    assert_eq!(lifecycle(&mut a, &o, Lifecycle::Active, Lifecycle::Dormant).unwrap_err(), Code::Lifecycle);
    let second = found(&mut a, EntityKind::Operator, "second operator", &ZERO, 0, 0);
    lifecycle(&mut a, &second, Lifecycle::Founded, Lifecycle::Active).unwrap();
    lifecycle(&mut a, &o, Lifecycle::Active, Lifecycle::Dormant).unwrap();
    // A dormant operator sends nothing; the second operator can.
    let e = Event::entity_lifecycle(&o, seq(&a, &o), &o, Lifecycle::Dormant, Lifecycle::Active);
    assert_eq!(send!(a, &e).unwrap_err(), Code::Unauthorized);
    let e = Event::entity_lifecycle(&second, seq(&a, &second), &o, Lifecycle::Dormant, Lifecycle::Active);
    send!(a, &e).unwrap();
    // The runtime may go dormant; it then publishes nothing.
    lifecycle(&mut a, &r, Lifecycle::Active, Lifecycle::Dormant).unwrap();
    assert_eq!(anchor(&mut a, &d("k1")).unwrap_err(), Code::Unauthorized);
}

#[test]
fn sender_authorization_matrix() {
    let (_t, mut a) = fixture(8);
    let src = active_source(&mut a, "camera");
    let (o, r) = (op(&a), rt(&a));
    let ev = EvolutionState::GENESIS;
    let operator_only = |sender: &Digest, s: u64, a: &Authority| {
        let id = a.next_entity_id(EntityKind::Source, &d("x")).unwrap();
        vec![
            Event::entity_found(sender, s, &id, EntityKind::Source, &d("x"), &d("scope"), 1, 1),
            Event::entity_lifecycle(sender, s, &src, Lifecycle::Active, Lifecycle::Dormant),
            Event::source_version(sender, s, &src, 1, 2, 1, &d("scope")),
            Event::import_legacy(sender, s, &[0u8; LEGACY_RECORD_BYTES]),
            Event::entity_reclaim(sender, s, &src),
        ]
    };
    let runtime_only = |sender: &Digest, s: u64| {
        vec![
            Event::anchor(sender, s, StateFormat::K1, &d("k1")),
            Event::publish(sender, s, StateFormat::K1, &d("a"), &d("b"), 0, &ZERO),
            Event::import_composite(sender, s, &d("a"), &d("b")),
        ]
    };
    for e in operator_only(&r, seq(&a, &r), &a).iter().chain(operator_only(&src, 1, &a).iter()) {
        assert_eq!(send!(a, e).unwrap_err(), Code::Unauthorized, "{:?}", e.kind);
    }
    for e in runtime_only(&o, seq(&a, &o)).iter().chain(runtime_only(&src, 1).iter()) {
        assert_eq!(send!(a, e).unwrap_err(), Code::Unauthorized, "{:?}", e.kind);
    }
    // Evolution: the runtime or an operator, never a source.
    assert_eq!(send!(a, &Event::evolution_reserve(&src, 1, &ev, &d("x"))).unwrap_err(), Code::Unauthorized);
    send!(a, &Event::evolution_reserve(&o, seq(&a, &o), &ev, &d("x"))).unwrap();
    // Unknown senders, and reconciliations outside a reconcile mode.
    assert_eq!(send!(a, &Event::anchor(&d("nobody"), 1, StateFormat::K1, &d("k"))).unwrap_err(), Code::UnknownEntity);
    let e = Event::reconcile_rollback(&o, seq(&a, &o), 99, &d("x"), 0);
    assert_eq!(send!(a, &e).unwrap_err(), Code::NotReconcilable);
    // GENESIS is provisioning's alone.
    assert_eq!(send!(a, &Event::genesis(&d("n"), &d("o"), &d("r"))).unwrap_err(), Code::Invalid);
}

#[test]
fn sender_sequences_refuse_duplicates_and_gaps() {
    let (_t, mut a) = fixture(4);
    let r = rt(&a);
    assert_eq!(send!(a, &Event::anchor(&r, 2, StateFormat::K1, &d("k"))).unwrap_err(), Code::SequenceGap);
    assert_eq!(send!(a, &Event::anchor(&r, 0, StateFormat::K1, &d("k"))).unwrap_err(), Code::Duplicate);
    send!(a, &Event::anchor(&r, 1, StateFormat::K1, &d("k"))).unwrap();
    let e = Event::publish(&r, 1, StateFormat::K1, &d("k"), &d("k2"), 0, &ZERO);
    assert_eq!(send!(a, &e).unwrap_err(), Code::Duplicate, "a retried request that already landed");
    assert_eq!(a.current().unwrap().state.entity(&r).unwrap().sender_watermark, 1);
}

#[test]
fn compare_and_publish_refuses_stale_requests_before_any_write() {
    let (_t, mut a) = fixture(4);
    let (g, root) = cur(&a);
    let r = rt(&a);
    let e = Event::anchor(&r, 1, StateFormat::K1, &d("k"));
    a.testing_counters(true);
    assert_eq!(a.transition(g + 1, &root, &e).unwrap_err(), Code::Stale);
    assert_eq!(a.transition(g, &d("other root"), &e).unwrap_err(), Code::Stale);
    // A founding request names the identity the current state founds next; any other is stale.
    let o = op(&a);
    let e = Event::entity_found(&o, 1, &d("guess"), EntityKind::Source, &d("l"), &d("scope"), 1, 1);
    assert_eq!(send!(a, &e).unwrap_err(), Code::Stale);
    assert_eq!(a.testing_counters(false).pwrites, 0);
    assert_eq!(cur(&a), (g, root));
}

#[test]
fn capacity_refuses_before_any_write_and_reclamation_is_explicit_and_bounded() {
    let (_t, mut a) = fixture(4);
    let s1 = active_source(&mut a, "s1");
    let _s2 = active_source(&mut a, "s2");
    let o = op(&a);
    let id = a.next_entity_id(EntityKind::Source, &d("s3")).unwrap();
    let e = Event::entity_found(&o, seq(&a, &o), &id, EntityKind::Source, &d("s3"), &d("scope"), 1, 1);
    a.testing_counters(true);
    let g = cur(&a).0;
    assert_eq!(send!(a, &e).unwrap_err(), Code::Capacity);
    assert_eq!((a.testing_counters(false).pwrites, cur(&a).0), (0, g), "refused before any write");
    // Only a terminated entity is reclaimed, by the operator, explicitly; its row folds into a chain head.
    let reclaim = |a: &mut Authority, id: &Digest| {
        let o = op(a);
        let s = seq(a, &o);
        tx(a, &Event::entity_reclaim(&o, s, id))
    };
    assert_eq!(reclaim(&mut a, &s1).unwrap_err(), Code::Lifecycle);
    lifecycle(&mut a, &s1, Lifecycle::Active, Lifecycle::Terminated).unwrap();
    reclaim(&mut a, &s1).unwrap();
    assert!(a.current().unwrap().state.entity(&s1).is_none());
    assert_eq!(a.check_evidence(&s1, 1, &d("scope")), Err(Code::UnknownEntity));
    let s3 = found_source(&mut a, "s1"); // the same label founds a new identity: indices are never reused
    assert_ne!(s3, s1);
    let st = &a.current().unwrap().state;
    assert_eq!((st.entities.len(), st.reclaimed_count, st.next_founding_index), (4, 1, 5));
    assert_ne!(st.reclaimed_head, ZERO);
}

#[test]
fn source_versions_and_scopes_govern_evidence_admission() {
    let (_t, mut a) = fixture(4);
    let s = active_source(&mut a, "camera");
    let scope = d("scope");
    assert_eq!(a.check_evidence(&s, 1, &scope), Ok(8));
    assert_eq!(a.check_evidence(&s, 2, &scope), Err(Code::SourceStale));
    assert_eq!(a.check_evidence(&s, 1, &d("elsewhere")), Err(Code::ScopeConflict));
    assert_eq!(a.check_evidence(&op(&a), 0, &ZERO), Err(Code::Unauthorized));
    assert_eq!(a.check_evidence(&d("nobody"), 1, &scope), Err(Code::UnknownEntity));
    let o = op(&a);
    let bump = |a: &mut Authority, expected: u64, version: u64, limit: u32, scope: &Digest| {
        let s2 = seq(a, &o);
        tx(a, &Event::source_version(&o, s2, &s, expected, version, limit, scope))
    };
    assert_eq!(bump(&mut a, 2, 3, 8, &scope).unwrap_err(), Code::SourceStale);
    assert_eq!(bump(&mut a, 1, 1, 8, &scope).unwrap_err(), Code::Invalid, "versions only increase");
    assert_eq!(bump(&mut a, 1, 2, MAX_PER_TURN + 1, &scope).unwrap_err(), Code::Invalid);
    assert_eq!(bump(&mut a, 1, 2, 8, &ZERO).unwrap_err(), Code::Invalid);
    bump(&mut a, 1, 2, 16, &d("scope2")).unwrap();
    // The new version invalidates evidence from the old one.
    assert_eq!(a.check_evidence(&s, 1, &scope), Err(Code::SourceStale));
    assert_eq!(a.check_evidence(&s, 2, &d("scope2")), Ok(16));
    // Non-sources have no versions; invalid founding shapes are refused.
    let e = Event::source_version(&o, seq(&a, &o), &o, 0, 1, 1, &scope);
    assert_eq!(send!(a, &e).unwrap_err(), Code::Invalid);
    for (kind, scope, version, limit) in [
        (EntityKind::Source, ZERO, 1, 1),
        (EntityKind::Source, d("s"), 0, 1),
        (EntityKind::Source, d("s"), 1, 0),
        (EntityKind::Operator, d("s"), 0, 0),
        (EntityKind::Runtime, ZERO, 1, 0),
    ] {
        let id = a.next_entity_id(kind, &d("bad")).unwrap();
        let e = Event::entity_found(&o, seq(&a, &o), &id, kind, &d("bad"), &scope, version, limit);
        assert_eq!(send!(a, &e).unwrap_err(), Code::Invalid, "{kind:?}");
    }
}

#[test]
fn cognition_lineage_law() {
    let (_t, mut a) = fixture(4);
    let r = rt(&a);
    let s = |a: &Authority| seq(a, &r);
    let e = Event::publish(&r, s(&a), StateFormat::K1, &d("a"), &d("b"), 0, &ZERO);
    assert_eq!(send!(a, &e).unwrap_err(), Code::Unanchored);
    assert_eq!(send!(a, &Event::anchor(&r, s(&a), StateFormat::K1, &ZERO)).unwrap_err(), Code::Invalid);
    anchor(&mut a, &d("k1-0")).unwrap();
    assert_eq!(anchor(&mut a, &d("k1-x")).unwrap_err(), Code::AlreadyAnchored);
    let e = Event::publish(&r, s(&a), StateFormat::K1, &d("wrong"), &d("k1-1"), 0, &ZERO);
    assert_eq!(send!(a, &e).unwrap_err(), Code::StateMismatch);
    let e = Event::publish(&r, s(&a), StateFormat::Composite, &d("k1-0"), &d("k1-1"), 0, &ZERO);
    assert_eq!(send!(a, &e).unwrap_err(), Code::StateMismatch, "an identity of one format never stands for another");
    for (count, digest) in [(1, ZERO), (0, d("e"))] {
        let e = Event::publish(&r, s(&a), StateFormat::K1, &d("k1-0"), &d("k1-1"), count, &digest);
        assert_eq!(send!(a, &e).unwrap_err(), Code::Invalid);
    }
    let e = Event::publish(&r, s(&a), StateFormat::K1, &d("k1-0"), &d("k1-1"), 3, &d("evidence"));
    let g = send!(a, &e).unwrap().generation;
    let st = a.current().unwrap().state.clone();
    assert_eq!((st.publications, st.last_publication_generation, st.evidence_count), (2, g, 3));
    assert_ne!(st.evidence_head, ZERO);
    // The explicit legacy move of a K1 lineage to a composite one.
    let e = Event::import_composite(&r, s(&a), &d("wrong"), &d("c-0"));
    assert_eq!(send!(a, &e).unwrap_err(), Code::StateMismatch);
    send!(a, &Event::import_composite(&r, s(&a), &d("k1-1"), &d("c-0"))).unwrap();
    assert_eq!(
        a.current().unwrap().state.cognition,
        Cognition::Anchored { format: StateFormat::Composite, identity: d("c-0") }
    );
    let e = Event::import_composite(&r, s(&a), &d("c-0"), &d("c-1"));
    assert_eq!(send!(a, &e).unwrap_err(), Code::StateMismatch, "only a K1 lineage is imported");
    let e = Event::publish(&r, s(&a), StateFormat::K1, &d("c-0"), &d("c-1"), 0, &ZERO);
    assert_eq!(send!(a, &e).unwrap_err(), Code::StateMismatch);
    send!(a, &Event::publish(&r, s(&a), StateFormat::Composite, &d("c-0"), &d("c-1"), 0, &ZERO)).unwrap();
}

#[test]
fn evolution_law_and_digest_match_continuity() {
    let (_t, mut a) = fixture(4);
    let o = op(&a);
    let ev = |a: &Authority| a.current().unwrap().state.evolution;
    let wrong = EvolutionState::idle(1, d("h")).unwrap();
    let e = Event::evolution_reserve(&o, seq(&a, &o), &wrong, &d("assertion"));
    assert_eq!(send!(a, &e).unwrap_err(), Code::EvolutionMismatch);
    let e = Event::evolution_finalize(&o, seq(&a, &o), &EvolutionState::GENESIS, &d("result"));
    assert_eq!(send!(a, &e).unwrap_err(), Code::EvolutionNotPending);
    send!(a, &Event::evolution_reserve(&o, seq(&a, &o), &EvolutionState::GENESIS, &d("assertion"))).unwrap();
    let pending = EvolutionState::pending(0, ZERO, d("assertion")).unwrap();
    assert_eq!(ev(&a), pending);
    assert_eq!(ev(&a).digest(), pending.digest(), "elpis.continuity.evolution-authority.v2, unchanged");
    let e = Event::evolution_reserve(&o, seq(&a, &o), &pending, &d("again"));
    assert_eq!(send!(a, &e).unwrap_err(), Code::EvolutionPending);
    let e = Event::evolution_finalize(&o, seq(&a, &o), &pending, &ZERO);
    assert_eq!(send!(a, &e).unwrap_err(), Code::Invalid);
    send!(a, &Event::evolution_finalize(&o, seq(&a, &o), &pending, &d("result"))).unwrap();
    assert_eq!(ev(&a), EvolutionState::idle(1, d("result")).unwrap());
    // The same law and digest as a continuity store driven through the same steps.
    let t = TempDir::new();
    let mut c = elpis_continuity::Store::new(&t.0.join("continuity")).unwrap();
    c.open().unwrap();
    c.reserve_evolution(Some(&EvolutionState::GENESIS), Some(&d("assertion"))).unwrap();
    let snap = c.finalize_evolution(Some(&pending), Some(&d("result"))).unwrap();
    assert_eq!(snap.evolution(), ev(&a));
    assert_eq!(snap.evolution().digest(), ev(&a).digest());
}

#[test]
fn exhaustion_is_refused_before_any_write() {
    let (_t, a) = fixture(4);
    let c = a.current().unwrap();
    let (o, r) = (op(&a), rt(&a));
    let at = |generation: u64, state: &State, event: &Event| {
        let prior = Prior { lineage: c.lineage, generation, record_digest: c.digest, state_root: c.state_root, state };
        apply(&prior, 4, event)
    };
    let anchor = Event::anchor(&r, 1, StateFormat::K1, &d("k"));
    assert_eq!(at(MAX_COUNTER, &c.state, &anchor).unwrap_err(), Code::Exhausted);
    assert!(at(MAX_COUNTER - 1, &c.state, &anchor).is_ok());
    // A reservation needs room for its finalization.
    let reserve = Event::evolution_reserve(&o, 1, &EvolutionState::GENESIS, &d("a"));
    assert_eq!(at(MAX_COUNTER - 1, &c.state, &reserve).unwrap_err(), Code::Exhausted);
    assert!(at(MAX_COUNTER - 2, &c.state, &reserve).is_ok());
    // A sender at the largest sequence sends nothing more.
    let mut tired = c.state.clone();
    tired.entities[1].sender_watermark = MAX_COUNTER;
    let e = Event::anchor(&r, MAX_COUNTER + 1, StateFormat::K1, &d("k"));
    assert_eq!(at(c.generation, &tired, &e).unwrap_err(), Code::Exhausted);
}

#[test]
fn a_rollback_acceptance_only_moves_forward() {
    let (_t, a) = fixture(4);
    let c = a.current().unwrap();
    let o = op(&a);
    for abandoned in [0, c.generation - 1, c.generation, MAX_COUNTER] {
        let e = Event::reconcile_rollback(&o, 1, abandoned, &d("abandoned"), 0);
        assert_eq!(apply(&c.prior(), 4, &e).unwrap_err(), Code::Invalid, "abandoned {abandoned}");
    }
    let e = Event::reconcile_rollback(&o, 1, c.generation + 1, &d("abandoned"), 0);
    let applied = apply(&c.prior(), 4, &e).unwrap();
    assert_eq!(applied.generation, c.generation + 2);
    assert_eq!(applied.state.reconcile_count, 1);
    for (class, digest) in [(3, d("abandoned")), (0, ZERO)] {
        let e = Event::reconcile_rollback(&o, 1, c.generation + 1, &digest, class);
        assert_eq!(apply(&c.prior(), 4, &e).unwrap_err(), Code::Invalid);
    }
}

#[test]
fn legacy_import_reads_a_continuity_register_exactly_once() {
    let t = TempDir::new();
    let dir = t.0.join("continuity");
    let mut c = elpis_continuity::Store::new(&dir).unwrap();
    c.open().unwrap();
    c.anchor_cognition(Some(&d("k1-legacy"))).unwrap();
    let snap = c.reserve_evolution(Some(&EvolutionState::GENESIS), Some(&d("assertion"))).unwrap();
    assert_eq!(read_continuity(&dir), Err(Code::Locked), "read only under continuity's own lock");
    c.close();
    let record = read_continuity(&dir).unwrap();
    assert_eq!(elpis_continuity::Snapshot::decode(&record), Ok(Some(snap)));

    let (_t2, mut a) = fixture(4);
    let o = op(&a);
    let mut corrupt = record;
    corrupt[20] ^= 1;
    assert_eq!(send!(a, &Event::import_legacy(&o, seq(&a, &o), &corrupt)).unwrap_err(), Code::LegacyRefused);
    send!(a, &Event::import_legacy(&o, seq(&a, &o), &record)).unwrap();
    let st = a.current().unwrap().state.clone();
    assert_eq!(st.cognition, Cognition::Anchored { format: StateFormat::K1, identity: d("k1-legacy") });
    assert_eq!(st.evolution, snap.evolution());
    assert_eq!(st.evolution.digest(), snap.evolution().digest());
    let legacy = st.legacy.unwrap();
    assert_eq!((legacy.generation, legacy.record_digest, legacy.anchored), (snap.generation(), snap.digest(), true));
    assert_eq!(send!(a, &Event::import_legacy(&o, seq(&a, &o), &record)).unwrap_err(), Code::LegacyRefused);
    // The continuity directory is untouched, and an already-anchored authority imports nothing.
    assert_eq!(read_continuity(&dir).unwrap(), record);
    let (_t3, mut b) = fixture(4);
    anchor(&mut b, &d("k")).unwrap();
    let o = op(&b);
    assert_eq!(send!(b, &Event::import_legacy(&o, seq(&b, &o), &record)).unwrap_err(), Code::LegacyRefused);
    // A v3 (composite) lineage imports with its format.
    let t4 = TempDir::new();
    let dir4 = t4.0.join("continuity");
    let mut c4 = elpis_continuity::Store::new(&dir4).unwrap();
    c4.open().unwrap();
    c4.anchor_cognition_as(StateFormat::Composite, Some(&d("ec"))).unwrap();
    c4.close();
    let (_t5, mut e) = fixture(4);
    let o = op(&e);
    send!(e, &Event::import_legacy(&o, seq(&e, &o), &read_continuity(&dir4).unwrap())).unwrap();
    assert_eq!(
        e.current().unwrap().state.cognition,
        Cognition::Anchored { format: StateFormat::Composite, identity: d("ec") }
    );
    assert_eq!(read_continuity(&t5_missing()), Err(Code::LegacyRefused));
}

fn t5_missing() -> PathBuf {
    std::env::temp_dir().join("elpis-ecs-c-no-such-continuity-directory")
}

#[test]
fn receipts_chain_to_the_head_an_operator_can_verify() {
    let (t, mut a) = fixture(4);
    let mut head = a.current().unwrap().state.receipt_head;
    let mut receipts = vec![];
    anchor(&mut a, &d("k0")).unwrap();
    receipts.push(a.current().unwrap().receipt);
    for i in 1..20 {
        let r = publish_next(&mut a, &format!("k{i}")).unwrap();
        receipts.push(r.receipt);
        assert_eq!(r.receipt_count, i + 2);
    }
    for r in &receipts {
        head = receipt_chain(&head, r);
    }
    assert_eq!(head, a.current().unwrap().state.receipt_head);
    let before = a.current().unwrap().clone();
    a.close();
    let b = open(&t);
    assert_eq!(*b.current().unwrap(), before, "reopen restores the exact authority");
    assert!(b.report().unwrap().replay_verified);
}

// -- the crash matrix -----------------------------------------------------------------------------------------

/// One faulted transition from a settled history, then restart. `expect`: the reopen disposition, and whether
/// the faulted transition is present.
fn crash_case(action: u32, arg: u64, call: Result<(), Code>, expect: (Disposition, bool)) {
    let (t, mut a) = with_history(4);
    let (g, _) = cur(&a);
    let receipts_before = a.current().unwrap().state.receipt_count;
    a.testing_arm(1, action, arg).unwrap();
    let result = publish_next(&mut a, "faulted").map(|_| ());
    assert_eq!(result, call, "action {action} arg {arg}");
    if result == Err(Code::PublicationRefused) {
        // Nothing changed and the handle stays usable; restart now to observe the slot as left.
        assert_eq!(cur(&a).0, g);
    }
    a.close();
    let files_before = files(&t);
    let (opened, mut b) = reopen(&t);
    let report = opened.unwrap();
    let (disp, present) = expect;
    assert_eq!(disposition(&report), disp, "action {action} arg {arg}");
    let c = b.current().unwrap();
    assert_eq!(c.generation, if present { g + 1 } else { g }, "action {action} arg {arg}");
    assert_eq!(identity(&b) == Some(d("faulted")), present);
    assert_eq!(c.state.receipt_count, receipts_before + present as u64, "present once, never duplicated");
    assert_eq!(report.witness_completed, disp == Disposition::InterruptedCompleted);
    assert_eq!(files(&t).keys().collect::<Vec<_>>(), files_before.keys().collect::<Vec<_>>());
    // Work continues, and the next restart is clean.
    publish_next(&mut b, "after").unwrap();
    b.close();
    let (opened, _c) = reopen(&t);
    assert_eq!(disposition(&opened.unwrap()), Disposition::Clean);
}

#[test]
fn crash_matrix_resolves_every_fault_to_exactly_one_authority() {
    use Disposition::*;
    let r = record_bytes(4) as u64;
    let death = Err(Code::TestingProcessDeath);
    let refused = Err(Code::PublicationRefused);
    let uncertain = Err(Code::PublicationUncertain);
    // Process death at each named step.
    crash_case(1, Step::TransitionBegin as u64, death, (Clean, false));
    crash_case(1, Step::SlotWritten as u64, death, (InterruptedCompleted, true));
    crash_case(1, Step::SlotSynced as u64, death, (InterruptedCompleted, true));
    crash_case(1, Step::WitnessWritten as u64, death, (Clean, true));
    crash_case(1, Step::WitnessSynced as u64, death, (Clean, true));
    // The record slot.
    crash_case(2, 0, refused, (Clean, false));
    // The first 32 header bytes of every record of a deployment are identical: a write torn there changes nothing.
    for cut in [0, 1, 32] {
        crash_case(3, cut, refused, (Clean, false));
        crash_case(4, cut, death, (Clean, false));
    }
    // Torn anywhere else, including inside the MAC and the checksum: TORN, never mistaken for a substitution.
    for cut in [33, 255, 257, r / 2, r - 65, r - 64, r - 63, r - 33, r - 32, r - 31, r - 1] {
        crash_case(3, cut, refused, (TrailingIncomplete, false));
        crash_case(4, cut, death, (TrailingIncomplete, false));
    }
    crash_case(5, 0, uncertain, (Clean, false));
    crash_case(6, 0, uncertain, (InterruptedCompleted, true));
    // The witness (the record is already durable).
    crash_case(7, 0, uncertain, (InterruptedCompleted, true));
    for cut in [1, 100, 175, 177, 207, 209, 239] {
        crash_case(8, cut, uncertain, (InterruptedCompleted, true));
        crash_case(9, cut, death, (InterruptedCompleted, true));
    }
    crash_case(10, 0, uncertain, (InterruptedCompleted, true));
    crash_case(11, 0, uncertain, (Clean, true));
}

#[test]
fn an_uncertain_publication_poisons_the_handle_and_a_retry_is_a_duplicate() {
    let (t, mut a) = with_history(4);
    a.testing_arm(1, 6, 0).unwrap(); // sync reports failure, the record is durable
    let r = rt(&a);
    let s = seq(&a, &r);
    let before = identity(&a).unwrap();
    let e = Event::publish(&r, s, StateFormat::K1, &before, &d("x"), 0, &ZERO);
    assert_eq!(send!(a, &e).unwrap_err(), Code::PublicationUncertain);
    assert_eq!(a.current().unwrap_err(), Code::PublicationUncertain);
    let (opened, mut b) = reopen(&t);
    assert_eq!(disposition(&opened.unwrap()), Disposition::InterruptedCompleted);
    // The caller cannot know whether it landed; retrying the same request is refused, not applied twice.
    assert_eq!(send!(b, &e).unwrap_err(), Code::Duplicate);
}

#[test]
fn completing_the_witness_at_open_is_itself_crash_safe() {
    for (action, arg, open_code, after) in [
        (1, Step::OpenWitnessWritten as u64, Code::TestingProcessDeath, Disposition::Clean),
        (1, Step::OpenWitnessSynced as u64, Code::TestingProcessDeath, Disposition::Clean),
        (7, 0, Code::PublicationUncertain, Disposition::InterruptedCompleted),
        (10, 0, Code::PublicationUncertain, Disposition::InterruptedCompleted),
        (9, 50, Code::TestingProcessDeath, Disposition::InterruptedCompleted),
    ] {
        let (t, mut a) = with_history(4);
        a.testing_arm(1, 1, Step::SlotSynced as u64).unwrap();
        assert_eq!(publish_next(&mut a, "x").unwrap_err(), Code::TestingProcessDeath);
        let mut b = Authority::new(&t.state(), &t.authority()).unwrap();
        b.testing_arm(0, action, arg).unwrap();
        assert_eq!(b.open().unwrap_err().code, open_code, "action {action}");
        let (opened, c) = reopen(&t);
        assert_eq!(disposition(&opened.unwrap()), after, "action {action}");
        assert_eq!(identity(&c), Some(d("x")));
    }
}

#[test]
fn repeated_random_crashes_never_lose_an_acknowledged_or_duplicate_any_transition() {
    let (t, mut a) = with_history(4);
    let mut acknowledged = cur(&a).0;
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let r = record_bytes(4) as u64;
    for i in 0..300 {
        let (action, arg) = match next() % 13 {
            0 | 1 => (0, 0),
            2 => (1, next() % 5),
            3 => (2, 0),
            4 => (3, next() % r),
            5 => (4, next() % r),
            6 => (5, 0),
            7 => (6, 0),
            8 => (7, 0),
            9 => (8, next() % WITNESS_BYTES as u64),
            10 => (9, next() % WITNESS_BYTES as u64),
            11 => (10, 0),
            _ => (11, 0),
        };
        a.testing_arm(1, action, arg).unwrap();
        let before = cur(&a).0;
        let label = format!("k{i}");
        let result = publish_next(&mut a, &label);
        if result.is_ok() {
            acknowledged = before + 1;
        }
        if result.is_err() || next() % 4 == 0 {
            a.close();
            let (opened, b) = reopen(&t);
            let report = opened.expect("every crash recovers");
            a = b;
            let g = cur(&a).0;
            assert!(g == before || g == before + 1, "iteration {i}: {g} from {before}");
            assert!(g >= acknowledged, "iteration {i}: an acknowledged transition was lost");
            assert_eq!(a.current().unwrap().state.receipt_count, g, "iteration {i}: one receipt per generation");
            assert!(report.reads == 5 && report.replays <= 1);
            acknowledged = g;
        }
    }
}

// -- the T1 adversary: write access to the state directory only -----------------------------------------------

#[test]
fn a_forged_generation_is_unauthenticated_wherever_it_is_placed() {
    // In the non-current slot: a restricted open until the operator acknowledges it.
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let cs = slot_of(&t, &c.digest);
    let mut forged = raw(&t.slot(cs));
    forged[32..40].copy_from_slice(&(c.generation + 1).to_le_bytes());
    reseal_unkeyed(&mut forged);
    let observed = sha256::digest(&[&forged]);
    put_raw(&t.slot(1 - cs), &forged);
    let (opened, mut b) = reopen(&t);
    let report = opened.unwrap();
    assert_eq!(disposition(&report), Disposition::RestrictedUnauthenticated);
    assert_eq!(report.slot_class[1 - cs], SlotClass::Unauthenticated as u8);
    assert_eq!(*b.current().unwrap(), c, "the acknowledged authority is intact and readable");
    assert_eq!(publish_next(&mut b, "blocked").unwrap_err(), Code::ReconcileRequired);
    let o = op(&b);
    let e = Event::reconcile_unauthenticated(&o, seq(&b, &o), (1 - cs) as u8, 2, &d("not what was observed"));
    assert_eq!(send!(b, &e).unwrap_err(), Code::NotReconcilable);
    let e = Event::reconcile_unauthenticated(&o, seq(&b, &o), (1 - cs) as u8, 2, &observed);
    send!(b, &e).unwrap();
    assert_eq!(b.current().unwrap().state.reconcile_count, 1);
    publish_next(&mut b, "resumed").unwrap();
    b.close();
    assert_eq!(disposition(&reopen(&t).0.unwrap()), Disposition::Clean);

    // Over the acknowledged record: only an explicit acceptance of the older authority opens it.
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let cs = slot_of(&t, &c.digest);
    let mut forged = raw(&t.slot(cs));
    forged[32..40].copy_from_slice(&(c.generation + 5).to_le_bytes());
    reseal_unkeyed(&mut forged);
    put_raw(&t.slot(cs), &forged);
    let (opened, mut b) = reopen(&t);
    assert_eq!(disposition(&opened.unwrap()), Disposition::UnauthenticatedAcknowledged);
    assert_eq!(b.current().unwrap().generation, c.generation - 1, "the candidate is the older authority");
    accept_rollback(&mut b, c.generation, &c.digest, 2);
}

fn accept_rollback(b: &mut Authority, abandoned: u64, abandoned_digest: &Digest, class: u8) {
    assert_eq!(publish_next(b, "blocked").unwrap_err(), Code::ReconcileRequired);
    assert_eq!(b.check_evidence(&d("x"), 1, &d("y")), Err(Code::ReconcileRequired));
    let o = op(b);
    for (g, digest, cl) in [
        (abandoned + 1, *abandoned_digest, class),
        (abandoned, d("x"), class),
        (abandoned, *abandoned_digest, (class + 1) % 3),
    ] {
        let e = Event::reconcile_rollback(&o, seq(b, &o), g, &digest, cl);
        assert_eq!(tx(b, &e).unwrap_err(), Code::NotReconcilable);
    }
    let candidate = b.current().unwrap().clone();
    let e = Event::reconcile_rollback(&o, seq(b, &o), abandoned, abandoned_digest, class);
    let r = tx(b, &e).unwrap();
    assert_eq!(r.generation, abandoned + 1, "no generation is ever reused");
    let st = &b.current().unwrap().state;
    assert_eq!((st.cognition, st.reconcile_count), (candidate.state.cognition, candidate.state.reconcile_count + 1));
    publish_next(b, "resumed").unwrap();
}

#[test]
fn forged_evolution_revision_and_head_are_unauthenticated() {
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let cs = slot_of(&t, &c.digest);
    let mut forged = raw(&t.slot(cs));
    let state_at = HEADER_BYTES;
    forged[state_at + 56..state_at + 64].copy_from_slice(&7u64.to_le_bytes());
    forged[state_at + 64..state_at + 96].copy_from_slice(&d("forged head"));
    // The adversary recomputes every unkeyed field: state root, receipt and checksum.
    let state_end = state_at + state_bytes(4);
    let root = state_root(&c.lineage, c.generation, 4, &forged[state_at..state_end]);
    forged[176..208].copy_from_slice(&root);
    reseal_unkeyed(&mut forged);
    let keys = keys();
    assert_eq!(Record::classify(&forged, 4, &keys).0, SlotClass::Unauthenticated);
    put_raw(&t.slot(cs), &forged);
    let (opened, mut b) = reopen(&t);
    assert_eq!(disposition(&opened.unwrap()), Disposition::UnauthenticatedAcknowledged);
    assert_eq!(b.current().unwrap().state.evolution, EvolutionState::GENESIS);
    accept_rollback(&mut b, c.generation, &c.digest, 2);
}

#[test]
fn a_validly_checksummed_substitute_or_another_deployments_record_is_unauthenticated() {
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let cs = slot_of(&t, &c.digest);
    // A complete record sealed under another key: every checksum valid, the MAC not.
    let attacker = Keys::derive(&[0x77; 32]);
    let event = Event::anchor(&c.state.entities[1].id, 99, StateFormat::K1, &d("substitute"));
    let (_, substitute) = Record::seal(
        &attacker,
        4,
        c.generation + 1,
        c.generation,
        &c.lineage,
        &c.digest,
        &c.state_root,
        &d("r"),
        c.state.clone(),
        event,
    );
    assert_eq!(Record::classify(&substitute, 4, &keys()).0, SlotClass::Unauthenticated);
    put_raw(&t.slot(1 - cs), &substitute);
    assert_eq!(disposition(&reopen(&t).0.unwrap()), Disposition::RestrictedUnauthenticated);

    // The genuine genesis of another deployment.
    let other = TempDir::new();
    provision_fixed(&other, 4, &[0x42; 32], &NONCE).unwrap();
    put_raw(&t.slot(1 - cs), &raw(&other.slot(0)));
    assert_eq!(disposition(&reopen(&t).0.unwrap()), Disposition::RestrictedUnauthenticated);
    put_raw(&t.slot(cs), &raw(&other.slot(0)));
    assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(Disposition::Unrecoverable)));
}

#[test]
fn a_corrupt_newest_record_requires_an_explicit_rollback_acceptance() {
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let cs = slot_of(&t, &c.digest);
    let mut bytes = raw(&t.slot(cs));
    bytes[HEADER_BYTES + 100] ^= 0x01;
    put_raw(&t.slot(cs), &bytes);
    let (opened, mut b) = reopen(&t);
    let report = opened.unwrap();
    assert_eq!(disposition(&report), Disposition::CorruptAcknowledged);
    assert_eq!(report.slot_class[cs], SlotClass::Torn as u8);
    accept_rollback(&mut b, c.generation, &c.digest, 1);
    b.close();
    let (opened, _b) = reopen(&t);
    let report = opened.unwrap();
    assert_eq!(disposition(&report), Disposition::Clean);
    assert!(report.replay_verified, "the rollback acceptance itself replays");
}

#[test]
fn copied_back_older_records_and_a_lying_device_are_refused_rollbacks() {
    // Copy back older genuine records.
    let (t, mut a) = with_history(4);
    let saved = [raw(&t.slot(0)), raw(&t.slot(1))];
    let older = a.current().unwrap().clone();
    for i in 0..3 {
        publish_next(&mut a, &format!("later-{i}")).unwrap();
    }
    let acknowledged = a.current().unwrap().clone();
    drop(a);
    put_raw(&t.slot(0), &saved[0]);
    put_raw(&t.slot(1), &saved[1]);
    let (opened, mut b) = reopen(&t);
    let report = opened.unwrap();
    assert_eq!(disposition(&report), Disposition::Rollback);
    assert_eq!(report.witness_generation, acknowledged.generation);
    assert_eq!(*b.current().unwrap(), older);
    accept_rollback(&mut b, acknowledged.generation, &acknowledged.digest, 0);

    // A device that acknowledged the sync and lost the write.
    let (t, mut a) = with_history(4);
    let target = 1 - slot_of(&t, &a.current().unwrap().digest);
    let previous = raw(&t.slot(target));
    publish_next(&mut a, "acknowledged").unwrap();
    let acknowledged = a.current().unwrap().clone();
    drop(a);
    put_raw(&t.slot(target), &previous);
    let (opened, mut b) = reopen(&t);
    assert_eq!(disposition(&opened.unwrap()), Disposition::Rollback);
    accept_rollback(&mut b, acknowledged.generation, &acknowledged.digest, 0);
}

#[test]
fn swapped_or_duplicated_slots_recover_without_loss() {
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let (x, y) = (raw(&t.slot(0)), raw(&t.slot(1)));
    put_raw(&t.slot(0), &y);
    put_raw(&t.slot(1), &x);
    let (opened, b) = reopen(&t);
    assert_eq!(disposition(&opened.unwrap()), Disposition::Clean);
    assert_eq!(*b.current().unwrap(), c);
    drop(b);
    let cs = slot_of(&t, &c.digest);
    put_raw(&t.slot(1 - cs), &raw(&t.slot(cs)));
    let (opened, mut b) = reopen(&t);
    let report = opened.unwrap();
    assert_eq!((disposition(&report), report.replay_verified), (Disposition::PredecessorUnavailable, false));
    publish_next(&mut b, "continues").unwrap();
}

#[test]
fn authenticated_conflicts_replay_mismatches_and_witness_inconsistency_refuse_to_open() {
    // These need the deployment key: outside T1, but recovery still refuses rather than guessing.
    let cases: [(&str, Craft, Disposition); 5] = [
        (
            "a different authenticated record at the acknowledged generation",
            |c, older| {
                let r = older.state.entities[1].id;
                let w = older.state.entity(&r).unwrap().sender_watermark;
                let before = match older.state.cognition {
                    Cognition::Anchored { identity, .. } => identity,
                    _ => ZERO,
                };
                let e = Event::publish(&r, w + 1, StateFormat::K1, &before, &d("a different branch"), 0, &ZERO);
                let applied = apply(&older.prior(), 4, &e).unwrap();
                assert_eq!(applied.generation, c.generation);
                seal_after(older, applied.generation, applied.state, e)
            },
            Disposition::Conflict,
        ),
        (
            "a successor that does not follow the pin",
            |c, _| {
                let mut wrong = c.clone();
                wrong.digest = d("another predecessor");
                let e = Event::publish(&c.state.entities[1].id, 9, StateFormat::K1, &d("a"), &d("b"), 0, &ZERO);
                seal_after(&wrong, c.generation + 1, c.state.clone(), e)
            },
            Disposition::Conflict,
        ),
        (
            "a successor two generations ahead",
            |c, _| {
                let mut ahead = c.clone();
                ahead.generation += 1;
                let mut st = c.state.clone();
                st.receipt_count += 1;
                let e = Event::publish(&c.state.entities[1].id, 9, StateFormat::K1, &d("a"), &d("b"), 0, &ZERO);
                seal_after(&ahead, c.generation + 2, st, e)
            },
            Disposition::WitnessInconsistent,
        ),
        (
            "a successor whose state is not what its event produces",
            |c, _| {
                let r = c.state.entities[1].id;
                let w = c.state.entity(&r).unwrap().sender_watermark;
                let before = match c.state.cognition {
                    Cognition::Anchored { identity, .. } => identity,
                    _ => ZERO,
                };
                let e = Event::publish(&r, w + 1, StateFormat::K1, &before, &d("next"), 0, &ZERO);
                let mut applied = apply(&c.prior(), 4, &e).unwrap();
                applied.state.cognition = Cognition::Anchored { format: StateFormat::K1, identity: d("not next") };
                seal_after(c, applied.generation, applied.state, e)
            },
            Disposition::ReplayMismatch,
        ),
        (
            "an authenticated record that breaks an invariant",
            |c, _| {
                // The seal itself never validates: a key holder could write this, and recovery refuses it.
                let mut st = c.state.clone();
                st.entities[0].lifecycle = Lifecycle::Terminated;
                seal_after(c, c.generation + 1, st, Event::entity_reclaim(&d("x"), 1, &d("y")))
            },
            Disposition::ReplayMismatch,
        ),
    ];
    for (k, (name, make, expected)) in cases.into_iter().enumerate() {
        let (t, a) = with_history(4);
        let c = a.current().unwrap().clone();
        drop(a);
        let cs = slot_of(&t, &c.digest);
        let older = Record::classify(&raw(&t.slot(1 - cs)), 4, &keys()).1.unwrap();
        // The first case replaces the acknowledged record; the others fill the non-current slot.
        let at = if k == 0 { cs } else { 1 - cs };
        put_raw(&t.slot(at), &make(&c, &older));
        assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(expected)), "{name}");
    }
}

#[test]
fn an_authenticated_record_ahead_of_a_missing_pin_is_witness_inconsistent() {
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let cs = slot_of(&t, &c.digest);
    let mut ahead = c.clone();
    ahead.generation += 1;
    let mut st = c.state.clone();
    st.receipt_count += 1;
    let e = Event::publish(&c.state.entities[1].id, 9, StateFormat::K1, &d("a"), &d("b"), 0, &ZERO);
    put_raw(&t.slot(cs), &seal_after(&ahead, c.generation + 2, st, e));
    assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(Disposition::WitnessInconsistent)));
}

#[test]
fn authenticated_records_over_the_acknowledged_slot_never_open_as_a_rollback() {
    // A successor that does not follow the missing pin, with nothing beside it to arbitrate.
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let cs = slot_of(&t, &c.digest);
    let mut wrong = c.clone();
    wrong.digest = d("another predecessor");
    let e = Event::publish(&c.state.entities[1].id, 9, StateFormat::K1, &d("a"), &d("b"), 0, &ZERO);
    put_raw(&t.slot(cs), &seal_after(&wrong, c.generation + 1, c.state.clone(), e));
    let mut torn = raw(&t.slot(1 - cs));
    torn[300] ^= 1;
    put_raw(&t.slot(1 - cs), &torn);
    assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(Disposition::Conflict)));

    // An authenticated but invalid record in place of the acknowledged one.
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let cs = slot_of(&t, &c.digest);
    let mut st = c.state.clone();
    st.entities[0].lifecycle = Lifecycle::Terminated;
    put_raw(&t.slot(cs), &seal_after(&c, c.generation + 1, st, Event::entity_reclaim(&d("x"), 1, &d("y"))));
    assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(Disposition::ReplayMismatch)));
}

#[test]
fn two_witnesses_that_disagree_are_inconsistent() {
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let current = (0..2).find(|i| Witness::classify(&raw(&t.witness(*i)), &keys()).1 == Some(Witness::of(&c)));
    let other = 1 - current.expect("the current witness");
    let mut w = Witness::of(&c);
    w.record_digest = d("another record at the same generation");
    put_raw(&t.witness(other), &w.seal(&keys()));
    assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(Disposition::WitnessInconsistent)));
    let mut w = Witness::of(&c);
    w.lineage = d("another lineage");
    w.generation -= 1;
    put_raw(&t.witness(other), &w.seal(&keys()));
    assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(Disposition::WitnessInconsistent)));
}

#[test]
fn a_witness_of_another_capacity_is_inconsistent() {
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let mut w = Witness::of(&c);
    w.capacity = 5;
    put_raw(&t.witness(0), &w.seal(&keys()));
    put_raw(&t.witness(1), &[0u8; WITNESS_BYTES]);
    assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(Disposition::WitnessInconsistent)));
}

#[test]
fn a_candidate_must_be_one_history_with_its_predecessor() {
    // With no witness to arbitrate, two authenticated records that are not one history are a conflict.
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    let cs = slot_of(&t, &c.digest);
    let mut stranger = c.clone();
    stranger.digest = d("not this history");
    let e = Event::publish(&c.state.entities[1].id, 9, StateFormat::K1, &d("a"), &d("b"), 0, &ZERO);
    put_raw(&t.slot(1 - cs), &seal_after(&stranger, c.generation + 1, c.state.clone(), e));
    for i in 0..2 {
        put_raw(&t.witness(i), &[0u8; WITNESS_BYTES]);
    }
    assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(Disposition::Conflict)));
}

#[test]
fn state_invariants_refuse_every_violation() {
    let (_t, mut a) = fixture(4);
    active_source(&mut a, "s1");
    active_source(&mut a, "s2");
    let c = a.current().unwrap().clone();
    let (lineage, g) = (c.lineage, c.generation);
    assert_eq!(c.state.check(4, &lineage, g), Ok(()));
    type Mutation = (&'static str, fn(&mut State, u64));
    let mutations: [Mutation; 18] = [
        ("founders out of place", |s, _| s.entities.swap(0, 1)),
        ("rows out of founding order", |s, _| s.entities.swap(2, 3)),
        ("a terminated founder", |s, _| s.entities[1].lifecycle = Lifecycle::Terminated),
        ("a forged entity id", |s, _| s.entities[2].id = d("forged")),
        ("a row founded after the counter", |s, _| s.next_founding_index = 3),
        ("rows and reclaimed rows that do not add up", |s, _| {
            s.reclaimed_count = 1;
            s.reclaimed_head = d("h");
        }),
        ("no ACTIVE operator", |s, _| s.entities[0].lifecycle = Lifecycle::Dormant),
        ("a source without a scope", |s, _| s.entities[2].scope = ZERO),
        ("a source with no per-turn bound", |s, _| s.entities[2].per_turn_limit = 0),
        ("an operator with a version", |s, _| s.entities[0].source_version = 1),
        ("a row touched in the future", |s, g| s.entities[2].last_generation = g + 1),
        ("anchored without a publication", |s, _| {
            s.cognition = Cognition::Anchored { format: StateFormat::K1, identity: d("k") };
        }),
        ("a publication without its generation", |s, _| s.publications = 1),
        ("no receipt", |s, _| s.receipt_count = 0),
        ("more receipts than generations", |s, g| s.receipt_count = g + 1),
        ("an evidence count without a head", |s, _| s.evidence_count = 1),
        ("a legacy import without its digest", |s, _| {
            s.legacy = Some(Legacy { generation: 1, format: StateFormat::K1, anchored: false, record_digest: ZERO });
        }),
        ("a counter past its bound", |s, _| {
            s.publications = MAX_COUNTER + 1;
            s.last_publication_generation = 1;
            s.cognition = Cognition::Anchored { format: StateFormat::K1, identity: d("k") };
        }),
    ];
    for (name, mutate) in mutations {
        let mut st = c.state.clone();
        mutate(&mut st, g);
        assert_eq!(st.check(4, &lineage, g), Err(Code::Corrupt), "{name}");
        assert_eq!(State::decode(&st.to_bytes(4), 4, &lineage, g), Err(Code::Corrupt), "{name}");
    }
    let mut full = c.state.clone();
    full.entities.push(full.entities[3]);
    assert_eq!(full.check(4, &lineage, g), Err(Code::Corrupt), "more rows than the capacity");
    // The encoding is canonical: reserved bytes and unused rows are zero, and the round trip is exact.
    let bytes = c.state.to_bytes(4);
    assert_eq!(State::decode(&bytes, 4, &lineage, g), Ok(c.state.clone()));
    let mut short = c.state.clone();
    short.entities.pop();
    short.reclaimed_count = 1;
    short.reclaimed_head = d("h");
    let short_bytes = short.to_bytes(4);
    assert_eq!(State::decode(&short_bytes, 4, &lineage, g), Ok(short.clone()));
    let unused_row = STATE_FIXED + 3 * ENTITY_BYTES + 5;
    for (at, base) in [
        (2, &bytes),
        (97, &bytes),
        (308, &bytes),
        (315, &bytes),
        (400, &bytes),
        (STATE_FIXED + 34, &bytes),
        (STATE_FIXED + 150, &bytes),
        (unused_row, &short_bytes),
    ] {
        let mut b = base.clone();
        b[at] = 1;
        assert_eq!(State::decode(&b, 4, &lineage, g), Err(Code::Corrupt), "byte {at}");
    }
    assert_eq!(State::decode(&bytes[1..], 4, &lineage, g), Err(Code::Corrupt));
    assert_eq!(State::decode(&bytes, 4, &d("another lineage"), g), Err(Code::Corrupt), "ids bind the lineage");
}

#[test]
fn witness_loss_requires_an_explicit_repin() {
    let (t, a) = with_history(4);
    let c = a.current().unwrap().clone();
    drop(a);
    // One torn witness slot: the other still pins.
    let mut w = raw(&t.witness(0));
    w[50] ^= 1;
    put_raw(&t.witness(0), &w);
    let (opened, b) = reopen(&t);
    assert!(opened.is_ok());
    drop(b);
    for i in 0..2 {
        put_raw(&t.witness(i), &[0u8; WITNESS_BYTES]);
    }
    let (opened, mut b) = reopen(&t);
    let report = opened.unwrap();
    assert_eq!(disposition(&report), Disposition::WitnessMissing);
    assert!(report.replay_verified);
    assert_eq!(*b.current().unwrap(), c);
    assert_eq!(publish_next(&mut b, "blocked").unwrap_err(), Code::ReconcileRequired);
    let o = op(&b);
    let e = Event::reconcile_repin(&o, seq(&b, &o), [1, 0], &c.digest);
    assert_eq!(send!(b, &e).unwrap_err(), Code::NotReconcilable);
    let e = Event::reconcile_repin(&o, seq(&b, &o), [0, 0], &d("another record"));
    assert_eq!(send!(b, &e).unwrap_err(), Code::Stale);
    send!(b, &Event::reconcile_repin(&o, seq(&b, &o), [0, 0], &c.digest)).unwrap();
    b.close();
    let (opened, _b) = reopen(&t);
    let report = opened.unwrap();
    assert_eq!(disposition(&report), Disposition::Clean);
    assert_eq!(report.witness_class, [SlotClass::Valid as u8; 2], "a re-pin writes both witness slots");
}

#[test]
fn nothing_authenticated_is_unrecoverable() {
    let (t, a) = with_history(4);
    drop(a);
    for i in 0..2 {
        put_raw(&t.slot(i), &vec![0u8; record_bytes(4)]);
    }
    assert_eq!(refused(reopen(&t).0), (Code::RecoveryRefused, Some(Disposition::Unrecoverable)));
}

// -- boundedness --------------------------------------------------------------------------------------------

fn long_sequence_length() -> u64 {
    std::env::var("ELPIS_ECS_C_LONG_SEQUENCE").ok().and_then(|v| v.parse().ok()).unwrap_or(4000)
}

#[test]
fn long_sequences_keep_files_bytes_recovery_work_and_metadata_constant() {
    let capacity = 4;
    let (t, mut a) = fixture(capacity);
    anchor(&mut a, &d("k1-0")).unwrap();
    let layout = files(&t);
    let total: u64 = layout.values().map(|(len, _)| len).sum();
    let r = record_bytes(capacity) as u64;
    assert_eq!(total, 2 * r + 32 + 2 * WITNESS_BYTES as u64);
    let mut recovery: Option<(u64, u64, u64)> = None;
    let mut source: Option<Digest> = None;
    let n = long_sequence_length();
    for i in 0..n {
        a.testing_counters(true);
        // Cycle through every kind of transition, churning entities through the fixed capacity.
        match i % 8 {
            0 => source = Some(active_source(&mut a, &format!("s{i}"))),
            1 => {
                let (o, s) = (op(&a), source.unwrap());
                let v = a.current().unwrap().state.entity(&s).unwrap().source_version;
                let e = Event::source_version(&o, seq(&a, &o), &s, v, v + 1, 4, &d("scope"));
                send!(a, &e).unwrap();
            }
            2 | 3 => {
                let (r, before) = (rt(&a), identity(&a).unwrap());
                let e = Event::publish(&r, seq(&a, &r), StateFormat::K1, &before, &d(&format!("k{i}")), 2, &d("ev"));
                send!(a, &e).unwrap();
            }
            4 => {
                let o = op(&a);
                let ev = a.current().unwrap().state.evolution;
                send!(a, &Event::evolution_reserve(&o, seq(&a, &o), &ev, &d(&format!("a{i}")))).unwrap();
            }
            5 => {
                let o = op(&a);
                let ev = a.current().unwrap().state.evolution;
                send!(a, &Event::evolution_finalize(&o, seq(&a, &o), &ev, &d(&format!("r{i}")))).unwrap();
            }
            6 => {
                lifecycle(&mut a, &source.unwrap(), Lifecycle::Active, Lifecycle::Terminated).unwrap();
            }
            _ => {
                let o = op(&a);
                send!(a, &Event::entity_reclaim(&o, seq(&a, &o), &source.unwrap())).unwrap();
            }
        }
        if i % 8 != 0 {
            // Exactly two in-place writes of fixed size and two syncs; nothing is created, renamed or removed.
            let c = a.testing_counters(false);
            assert_eq!((c.pwrites, c.pwrite_bytes, c.data_syncs), (2, r + WITNESS_BYTES as u64, 2), "transition {i}");
            assert_eq!((c.opens, c.creates, c.renames, c.unlinks, c.dir_syncs, c.preads), (0, 0, 0, 0, 0, 0));
        }
        let st = &a.current().unwrap().state;
        assert!(st.entities.len() <= capacity as usize);
        if i % 500 == 0 || i + 1 == n {
            a.close();
            let (opened, b) = reopen(&t);
            a = b;
            let report = opened.unwrap();
            assert_eq!(disposition(&report), Disposition::Clean);
            assert!(report.replay_verified);
            let work = (report.reads, report.read_bytes, report.replays);
            assert_eq!(work, (5, 32 + 2 * WITNESS_BYTES as u64 + 2 * r, 1));
            assert_eq!(*recovery.get_or_insert(work), work, "recovery work does not depend on history length");
            assert_eq!(files(&t), layout, "the file set, sizes and modes never change");
        }
    }
    let st = &a.current().unwrap().state;
    assert!(st.receipt_count > n && st.next_founding_index >= n / 8);
    assert_eq!(st.entities.len() as u64 + st.reclaimed_count, st.next_founding_index);
    assert_eq!(st.to_bytes(capacity).len(), state_bytes(capacity), "retained metadata is fixed-size");
}
