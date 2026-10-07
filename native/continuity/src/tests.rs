//! Continuity register v2 qualification: frozen byte parity, the crash matrix and the evolution law.
//!
//! Deterministic and bounded: no sleeps, no threads racing, private temporary directories.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::Code;
use crate::record::{Cognition, Digest, EvolutionState, Snapshot, MAX_COUNTER, RECORD_SIZE, ZERO};
use crate::sha256;
use crate::store::probe::{Counters, Fault};
use crate::store::{Step, Store, SLOT_NAMES};

// -- helpers -------------------------------------------------------------------------------------------

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> TempDir {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("elpis-continuity-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }
    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
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

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn digest_of(s: &str) -> Digest {
    unhex(s).try_into().unwrap()
}

fn open(path: &Path) -> Store {
    let mut s = Store::new(path).unwrap();
    s.open().unwrap();
    s
}

fn files(path: &Path) -> Vec<(String, u64)> {
    let mut v: Vec<_> = fs::read_dir(path)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (e.file_name().to_string_lossy().into_owned(), e.metadata().unwrap().len())
        })
        .collect();
    v.sort();
    v
}

fn two_slots() -> Vec<(String, u64)> {
    vec![(SLOT_NAMES[0].into(), RECORD_SIZE as u64), (SLOT_NAMES[1].into(), RECORD_SIZE as u64)]
}

fn write_slots(path: &Path, a: &[u8], b: &[u8]) {
    fs::create_dir_all(path).unwrap();
    fs::write(path.join(SLOT_NAMES[0]), a).unwrap();
    fs::write(path.join(SLOT_NAMES[1]), b).unwrap();
}

fn snap(generation: u64, k1: Option<Digest>, evolution: EvolutionState) -> Snapshot {
    let cognition = k1.map(Cognition::Anchored).unwrap_or(Cognition::Unanchored);
    Snapshot::new(generation, cognition, evolution).unwrap()
}

fn fixture(name: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
}

/// The caller's side of the evolution law, as the runtime implements it: reserve durably, execute once,
/// finalize the exact pending authority. Returns the number of executions.
fn attempt(
    store: &mut Store,
    expected: EvolutionState,
    assertion: Digest,
    receipt: Digest,
) -> (u32, Result<Snapshot, Code>) {
    let pending = match store.reserve_evolution(Some(&expected), Some(&assertion)) {
        Ok(s) => s.evolution(),
        Err(code) => return (0, Err(code)),
    };
    (1, store.finalize_evolution(Some(&pending), Some(&receipt)))
}

// -- frozen v2 byte parity (vectors recorded from the qualified Python authority, PR #36) -------------

#[test]
fn frozen_v2_records_decode_encode_and_digest_exactly() {
    let text = fixture("continuity_v2_vectors.txt");
    let (mut ok, mut empty, mut corrupt) = (0, 0, 0);
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 12, "{line}");
        let raw = unhex(f[11]);
        match f[1] {
            "ok" => {
                let s = Snapshot::decode(&raw).unwrap().unwrap_or_else(|| panic!("{}", f[0]));
                assert_eq!(s.generation(), f[2].parse::<u64>().unwrap(), "{}", f[0]);
                let k1 = digest_of(f[4]);
                let expected_cognition = if f[3] == "1" { Cognition::Anchored(k1) } else { Cognition::Unanchored };
                assert_eq!(s.cognition(), expected_cognition, "{}", f[0]);
                let e = s.evolution();
                assert_eq!(e.revision(), f[5].parse::<u64>().unwrap());
                assert_eq!(hex(&e.head()), f[6]);
                assert_eq!(e.assertion().map(|a| hex(&a)), (f[7] == "1").then(|| f[8].to_string()));
                assert_eq!(hex(&e.digest()), f[9], "{} evolution digest", f[0]);
                assert_eq!(hex(&s.digest()), f[10], "{} record digest", f[0]);
                assert_eq!(s.encode().to_vec(), raw, "{} re-encodes to the frozen bytes", f[0]);
                ok += 1;
            }
            "empty" => {
                assert_eq!(Snapshot::decode(&raw), Ok(None), "{}", f[0]);
                empty += 1;
            }
            "corrupt" => {
                assert_eq!(Snapshot::decode(&raw), Err(Code::Corrupt), "{}", f[0]);
                corrupt += 1;
            }
            other => panic!("unknown expectation {other}"),
        }
    }
    assert_eq!((ok, empty, corrupt), (9, 1, 19));
}

#[test]
fn frozen_v2_store_session_reproduces_both_slots_byte_for_byte() {
    let dir = TempDir::new();
    let path = dir.join("c");
    let mut store = Store::new(&path).unwrap();
    let text = fixture("continuity_v2_sequence.txt");
    let mut steps = 0;
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let arg = |i: usize| digest_of(f[i]);
        match f[1] {
            "open" => {
                store.open().unwrap();
            }
            "reopen" => {
                store.close();
                store = open(&path);
            }
            "anchor" => {
                store.anchor_cognition(Some(&arg(2))).unwrap();
            }
            "commit" => {
                store.commit_cognition(Some(&arg(2)), Some(&arg(3))).unwrap();
            }
            "reserve" => {
                let current = store.snapshot().unwrap().evolution();
                store.reserve_evolution(Some(&current), Some(&arg(2))).unwrap();
            }
            "finalize" => {
                let current = store.snapshot().unwrap().evolution();
                store.finalize_evolution(Some(&current), Some(&arg(2))).unwrap();
            }
            other => panic!("unknown op {other}"),
        }
        for (i, name) in SLOT_NAMES.iter().enumerate() {
            assert_eq!(hex(&fs::read(path.join(name)).unwrap()), f[4 + i], "step {} {name}", f[0]);
        }
        steps += 1;
    }
    assert_eq!(steps, 10);
}

// -- basic law -----------------------------------------------------------------------------------------

#[test]
fn new_store_is_unanchored_generation_one_in_two_fixed_slots() {
    let dir = TempDir::new();
    let s = open(&dir.join("c"));
    assert_eq!(s.snapshot().unwrap(), Snapshot::GENESIS);
    assert_eq!(files(&dir.join("c")), two_slots());
}

#[test]
fn cognition_anchor_and_transitions_survive_restart() {
    let dir = TempDir::new();
    let path = dir.join("c");
    let mut s = open(&path);
    assert_eq!(s.commit_cognition(Some(&d("k1-0")), Some(&d("k1-1"))), Err(Code::Unanchored));
    s.anchor_cognition(Some(&d("k1-0"))).unwrap();
    assert_eq!(s.anchor_cognition(Some(&d("k1-9"))), Err(Code::AlreadyAnchored));
    s.commit_cognition(Some(&d("k1-0")), Some(&d("k1-1"))).unwrap();
    let before = s.snapshot().unwrap();
    // A transition that does not start from the expected identity writes nothing.
    assert_eq!(s.commit_cognition(Some(&d("k1-0")), Some(&d("k1-2"))), Err(Code::StateMismatch));
    assert_eq!(s.snapshot().unwrap(), before);
    assert_eq!(s.anchor_cognition(None), Err(Code::AlreadyAnchored).or(Err(Code::Invalid)));
    assert_eq!(s.commit_cognition(None, Some(&d("x"))), Err(Code::Invalid));
    drop(s);
    let s = open(&path);
    assert_eq!(s.snapshot().unwrap().cognition(), Cognition::Anchored(d("k1-1")));
    assert_eq!(s.snapshot().unwrap().generation(), 3);
}

#[test]
fn idle_to_pending_reservation_then_pending_to_next_idle_finalization() {
    let dir = TempDir::new();
    let mut s = open(&dir.join("c"));
    let a0 = s.snapshot().unwrap().evolution();
    assert_eq!(s.finalize_evolution(Some(&a0), Some(&d("r1"))), Err(Code::EvolutionNotPending));
    let pending = s.reserve_evolution(Some(&a0), Some(&d("assertion-100"))).unwrap().evolution();
    assert_eq!(pending, EvolutionState::pending(0, ZERO, d("assertion-100")).unwrap());
    assert_ne!(pending.digest(), a0.digest());
    let impostor = EvolutionState::pending(0, ZERO, d("assertion-999")).unwrap();
    assert_eq!(s.finalize_evolution(Some(&impostor), Some(&d("r1"))), Err(Code::AuthorityMismatch));
    assert_eq!(s.reserve_evolution(Some(&pending), Some(&d("assertion-101"))), Err(Code::EvolutionPending));
    assert_eq!(s.finalize_evolution(Some(&pending), Some(&ZERO)), Err(Code::Invalid));
    assert_eq!(s.finalize_evolution(Some(&pending), None), Err(Code::Invalid));
    assert_eq!(s.finalize_evolution(None, Some(&d("r1"))), Err(Code::AuthorityMismatch));
    let a1 = s.finalize_evolution(Some(&pending), Some(&d("r1"))).unwrap().evolution();
    assert_eq!(a1, EvolutionState::idle(1, d("r1")).unwrap());
    assert_ne!(a1.digest(), a0.digest());
    assert_eq!(s.finalize_evolution(Some(&pending), Some(&d("r1"))), Err(Code::AuthorityMismatch));
    assert_eq!(s.reserve_evolution(Some(&a0), Some(&d("assertion-2"))), Err(Code::AuthorityMismatch));
    assert_eq!(s.reserve_evolution(Some(&a1), None), Err(Code::Invalid));
    assert_eq!(s.snapshot().unwrap().evolution(), a1);
}

#[test]
fn idle_authority_cannot_carry_an_assertion_in_any_representation() {
    use crate::ffi::{elpis_continuity_evolution_digest, CEvolution};
    // The Rust type has no field for it; the C ABI refuses the encoding.
    let mut e = CEvolution { revision: 0, ..Default::default() };
    let mut out = [0u8; 32];
    assert_eq!(unsafe { elpis_continuity_evolution_digest(&e, out.as_mut_ptr()) }, 0);
    assert_eq!(out, EvolutionState::GENESIS.digest());
    e.assertion = d("arbitrary");
    assert_eq!(unsafe { elpis_continuity_evolution_digest(&e, out.as_mut_ptr()) }, Code::Corrupt as i32);
    e.pending = 2;
    assert_eq!(unsafe { elpis_continuity_evolution_digest(&e, out.as_mut_ptr()) }, Code::Corrupt as i32);
}

// -- generation selection --------------------------------------------------------------------------------

#[test]
fn the_higher_valid_generation_is_the_authority_in_either_slot() {
    let dir = TempDir::new();
    let older = snap(6, Some(d("k1-6")), EvolutionState::GENESIS);
    let newer = snap(7, Some(d("k1-7")), EvolutionState::GENESIS);
    for (i, (a, b)) in [(older, newer), (newer, older)].into_iter().enumerate() {
        let path = dir.join(&format!("c{i}"));
        write_slots(&path, &a.encode(), &b.encode());
        assert_eq!(open(&path).snapshot().unwrap(), newer);
    }
}

#[test]
fn equal_generations_are_ambiguous_and_both_invalid_is_corrupt() {
    let dir = TempDir::new();
    let one = snap(5, Some(d("k1-1")), EvolutionState::GENESIS);
    let two = snap(5, Some(d("k1-2")), EvolutionState::GENESIS);
    write_slots(&dir.join("eq"), &one.encode(), &two.encode());
    assert_eq!(Store::new(&dir.join("eq")).unwrap().open(), Err(Code::Corrupt));
    write_slots(&dir.join("bad"), &[1u8; RECORD_SIZE], &[0u8; RECORD_SIZE]);
    assert_eq!(Store::new(&dir.join("bad")).unwrap().open(), Err(Code::Corrupt));
}

#[test]
fn checksum_corruption_of_the_current_slot_falls_back_to_the_other() {
    let dir = TempDir::new();
    let path = dir.join("c");
    open(&path).anchor_cognition(Some(&d("k1-0"))).unwrap(); // generation 2 in slot b
    let mut raw = fs::read(path.join(SLOT_NAMES[1])).unwrap();
    raw[40] ^= 1;
    fs::write(path.join(SLOT_NAMES[1]), &raw).unwrap();
    // Integrity evidence only: the untouched older record is the authority left.
    assert_eq!(open(&path).snapshot().unwrap(), Snapshot::GENESIS);
}

// -- bounds -------------------------------------------------------------------------------------------------

#[test]
fn fixed_two_slot_footprint_and_no_history_growth() {
    let dir = TempDir::new();
    let path = dir.join("c");
    let mut s = open(&path);
    s.anchor_cognition(Some(&d("k1-0"))).unwrap();
    for n in 1..=2000u32 {
        s.probe.counters = Counters::default();
        s.commit_cognition(Some(&d(&format!("k1-{}", n - 1))), Some(&d(&format!("k1-{n}")))).unwrap();
        // One publication is exactly one complete fixed-size write and one data sync, nothing else.
        assert_eq!(s.probe.counters, Counters { pwrites: 1, pwrite_bytes: 176, data_syncs: 1, ..Default::default() });
        if n % 7 == 0 {
            let current = s.snapshot().unwrap().evolution();
            assert_eq!(attempt(&mut s, current, d(&format!("a{n}")), d(&format!("r{n}"))).0, 1);
        }
        assert_eq!(files(&path), two_slots());
    }
    assert!(s.snapshot().unwrap().generation() > 2000);
    drop(s);
    let total: u64 = files(&path).iter().map(|(_, n)| n).sum();
    assert_eq!(total, 352);
    // Restart work is two fixed reads, whatever the lifetime.
    let mut s = Store::new(&path).unwrap();
    s.open().unwrap();
    assert_eq!(s.probe.counters, Counters { opens: 2, preads: 2, pread_bytes: 352, ..Default::default() });
}

#[test]
fn restart_reads_two_slots_at_any_generation() {
    for generation in [10, 1_000_000, MAX_COUNTER] {
        let dir = TempDir::new();
        let current = snap(generation, Some(d("k1")), EvolutionState::idle(generation / 3, d("h")).unwrap());
        let older = snap(generation - 1, Some(d("k0")), current.evolution());
        write_slots(&dir.join("c"), &older.encode(), &current.encode());
        let mut s = Store::new(&dir.join("c")).unwrap();
        assert_eq!(s.open().unwrap(), current);
        assert_eq!(s.probe.counters.preads, 2);
        assert_eq!(s.probe.counters.pread_bytes, 352);
    }
}

#[test]
fn counter_exhaustion_is_refused_before_execution() {
    let dir = TempDir::new();
    let idle = snap(MAX_COUNTER - 1, None, EvolutionState::GENESIS);
    write_slots(&dir.join("g"), &idle.encode(), &[0u8; RECORD_SIZE]);
    let mut s = open(&dir.join("g"));
    assert_eq!(attempt(&mut s, idle.evolution(), d("a"), d("r")), (0, Err(Code::Exhausted)));
    assert_eq!(s.snapshot().unwrap(), idle);
    let at_max = snap(5, None, EvolutionState::idle(MAX_COUNTER, d("h")).unwrap());
    write_slots(&dir.join("r"), &at_max.encode(), &[0u8; RECORD_SIZE]);
    let mut s = open(&dir.join("r"));
    assert_eq!(attempt(&mut s, at_max.evolution(), d("a"), d("r")), (0, Err(Code::Exhausted)));
    // One generation below the reservation limit still reserves and finalizes.
    let ok = snap(MAX_COUNTER - 2, None, EvolutionState::idle(MAX_COUNTER - 1, d("h")).unwrap());
    write_slots(&dir.join("ok"), &ok.encode(), &[0u8; RECORD_SIZE]);
    let mut s = open(&dir.join("ok"));
    let (runs, result) = attempt(&mut s, ok.evolution(), d("a"), d("r"));
    assert_eq!(runs, 1);
    assert_eq!(result.unwrap().generation(), MAX_COUNTER);
}

// -- crash matrix ------------------------------------------------------------------------------------------

#[test]
fn death_at_each_publication_step_yields_previous_or_next() {
    for step in [Step::PublishBegin, Step::PublishWritten, Step::PublishSynced] {
        let dir = TempDir::new();
        let path = dir.join("c");
        let mut s = open(&path);
        s.anchor_cognition(Some(&d("k1-0"))).unwrap();
        let previous = s.snapshot().unwrap();
        s.probe.publications = 0;
        s.probe.plan = Some((1, Fault::Die(step)));
        assert_eq!(s.commit_cognition(Some(&d("k1-0")), Some(&d("k1-1"))), Err(Code::TestingProcessDeath));
        drop(s);
        let next = snap(previous.generation() + 1, Some(d("k1-1")), previous.evolution());
        // Death after the write leaves the complete record in the page cache; before it, nothing changed.
        let expected = if step == Step::PublishBegin { previous } else { next };
        assert_eq!(open(&path).snapshot().unwrap(), expected, "{step:?}");
    }
}

#[test]
fn torn_write_never_destroys_the_previous_authority() {
    for cut in [1usize, 8, 16, 63, 64, 104, RECORD_SIZE - 1] {
        let dir = TempDir::new();
        let path = dir.join("c");
        let mut s = open(&path);
        s.anchor_cognition(Some(&d("k1-0"))).unwrap();
        s.commit_cognition(Some(&d("k1-0")), Some(&d("k1-1"))).unwrap();
        let previous = s.snapshot().unwrap();
        s.probe.publications = 0;
        s.probe.plan = Some((1, Fault::TornDie(cut)));
        assert_eq!(s.commit_cognition(Some(&d("k1-1")), Some(&d("k1-2"))), Err(Code::TestingProcessDeath));
        drop(s);
        let mut s = open(&path);
        assert_eq!(s.snapshot().unwrap(), previous, "cut {cut}");
        // And the store keeps publishing into the torn slot afterwards.
        assert_eq!(
            s.commit_cognition(Some(&d("k1-1")), Some(&d("k1-2"))).unwrap().cognition(),
            Cognition::Anchored(d("k1-2"))
        );
    }
}

#[test]
fn write_failure_is_refused_and_the_store_stays_usable() {
    for fault in [Fault::WriteFail, Fault::TornFail(80)] {
        let dir = TempDir::new();
        let mut s = open(&dir.join("c"));
        s.anchor_cognition(Some(&d("k1-0"))).unwrap();
        let previous = s.snapshot().unwrap();
        s.probe.publications = 0;
        s.probe.plan = Some((1, fault));
        assert_eq!(s.commit_cognition(Some(&d("k1-0")), Some(&d("k1-1"))), Err(Code::PublicationRefused));
        s.probe.plan = None;
        assert_eq!(s.snapshot().unwrap(), previous);
        assert_eq!(
            s.commit_cognition(Some(&d("k1-0")), Some(&d("k1-1"))).unwrap().generation(),
            previous.generation() + 1
        );
    }
}

#[test]
fn sync_failure_is_uncertain_poisons_until_reopen_and_resolves_to_one_record() {
    for (fault, durable) in [(Fault::SyncFailLost, false), (Fault::SyncFailDurable, true)] {
        let dir = TempDir::new();
        let path = dir.join("c");
        let mut s = open(&path);
        s.anchor_cognition(Some(&d("k1-0"))).unwrap();
        let previous = s.snapshot().unwrap();
        s.probe.publications = 0;
        s.probe.plan = Some((1, fault));
        assert_eq!(s.commit_cognition(Some(&d("k1-0")), Some(&d("k1-1"))), Err(Code::PublicationUncertain));
        assert_eq!(s.snapshot(), Err(Code::PublicationUncertain));
        assert_eq!(s.anchor_cognition(Some(&d("x"))), Err(Code::PublicationUncertain));
        let next = snap(previous.generation() + 1, Some(d("k1-1")), previous.evolution());
        assert_eq!(s.open().unwrap(), if durable { next } else { previous });
    }
}

#[test]
fn death_during_initialization_restarts_or_completes_it() {
    for step in [Step::InitBWritten, Step::InitAWritten, Step::InitBRenamed, Step::InitARenamed, Step::InitDirSync] {
        let dir = TempDir::new();
        let path = dir.join("c");
        let mut s = Store::new(&path).unwrap();
        s.probe.plan = Some((0, Fault::Die(step)));
        assert_eq!(s.open(), Err(Code::TestingProcessDeath), "{step:?}");
        drop(s);
        assert_eq!(open(&path).snapshot().unwrap(), Snapshot::GENESIS, "{step:?}");
        assert_eq!(files(&path), two_slots());
    }
}

// -- the evolution law -----------------------------------------------------------------------------------

#[test]
fn reservation_refusal_executes_nothing() {
    let dir = TempDir::new();
    let path = dir.join("c");
    let mut s = open(&path);
    let before = s.snapshot().unwrap();
    s.probe.publications = 0;
    s.probe.plan = Some((1, Fault::WriteFail));
    assert_eq!(attempt(&mut s, before.evolution(), d("a"), d("r")), (0, Err(Code::PublicationRefused)));
    assert_eq!(s.snapshot().unwrap(), before);
    drop(s);
    assert_eq!(open(&path).snapshot().unwrap(), before);
}

#[test]
fn uncertain_reservation_resolves_idle_or_pending_before_any_execution() {
    for (fault, durable) in [(Fault::SyncFailLost, false), (Fault::SyncFailDurable, true)] {
        let dir = TempDir::new();
        let path = dir.join("c");
        let mut s = open(&path);
        let idle = s.snapshot().unwrap().evolution();
        s.probe.publications = 0;
        s.probe.plan = Some((1, fault));
        assert_eq!(attempt(&mut s, idle, d("a"), d("r")), (0, Err(Code::PublicationUncertain)));
        drop(s);
        for _ in 0..3 {
            let e = open(&path).snapshot().unwrap().evolution();
            let expected = if durable { EvolutionState::pending(0, ZERO, d("a")).unwrap() } else { idle };
            assert_eq!(e, expected, "{fault:?}");
        }
        let mut s = open(&path);
        if durable {
            // The reserved assertion cannot run again: reservation is refused while pending.
            assert_eq!(attempt(&mut s, idle, d("a"), d("r")).0, 0);
            let pending = s.snapshot().unwrap().evolution();
            assert_eq!(attempt(&mut s, pending, d("a"), d("r")), (0, Err(Code::EvolutionPending)));
        }
    }
}

#[test]
fn uncertain_finalization_resolves_pending_or_next_idle_and_the_old_assertion_never_runs_again() {
    for (fault, durable) in [(Fault::SyncFailLost, false), (Fault::SyncFailDurable, true)] {
        let dir = TempDir::new();
        let path = dir.join("c");
        let mut s = open(&path);
        let idle = s.snapshot().unwrap().evolution();
        s.probe.publications = 0;
        s.probe.plan = Some((2, fault)); // publication 1 reserves, publication 2 finalizes
        assert_eq!(attempt(&mut s, idle, d("a"), d("r")), (1, Err(Code::PublicationUncertain)));
        drop(s);
        let pending = EvolutionState::pending(0, ZERO, d("a")).unwrap();
        let next = EvolutionState::idle(1, d("r")).unwrap();
        for _ in 0..3 {
            assert_eq!(open(&path).snapshot().unwrap().evolution(), if durable { next } else { pending });
        }
        let mut s = open(&path);
        // The old assertion was bound to `idle`: in either outcome it cannot be reserved again.
        let (runs, result) = attempt(&mut s, idle, d("a"), d("r"));
        assert_eq!(runs, 0);
        assert_eq!(result, Err(Code::AuthorityMismatch));
        if !durable {
            assert_eq!(attempt(&mut s, pending, d("a"), d("r")), (0, Err(Code::EvolutionPending)));
        }
    }
}

#[test]
fn repeated_restart_leaves_pending_intact_and_cognition_preserves_it() {
    let dir = TempDir::new();
    let path = dir.join("c");
    let mut s = open(&path);
    let pending = s.reserve_evolution(Some(&EvolutionState::GENESIS), Some(&d("a1"))).unwrap().evolution();
    s.anchor_cognition(Some(&d("k1-0"))).unwrap();
    for n in 1..5 {
        let snap = s.commit_cognition(Some(&d(&format!("k1-{}", n - 1))), Some(&d(&format!("k1-{n}")))).unwrap();
        assert_eq!(snap.evolution(), pending);
    }
    drop(s);
    for _ in 0..5 {
        let s = open(&path);
        assert_eq!(s.snapshot().unwrap().evolution(), pending);
        assert_eq!(s.snapshot().unwrap().cognition(), Cognition::Anchored(d("k1-4")));
    }
}

#[test]
fn explicit_reconciliation_advances_exactly_once() {
    let dir = TempDir::new();
    let path = dir.join("c");
    let mut s = open(&path);
    let pending = s.reserve_evolution(Some(&EvolutionState::GENESIS), Some(&d("a1"))).unwrap().evolution();
    drop(s); // the attempt's outcome was not finalized (restart)
    let mut s = open(&path);
    assert_eq!(s.snapshot().unwrap().evolution(), pending);
    let finalized = s.finalize_evolution(Some(&pending), Some(&d("completed"))).unwrap().evolution();
    assert_eq!(finalized, EvolutionState::idle(1, d("completed")).unwrap());
    assert_eq!(s.finalize_evolution(Some(&pending), Some(&d("completed"))), Err(Code::AuthorityMismatch));
    drop(s);
    assert_eq!(open(&path).snapshot().unwrap().evolution(), finalized);
}

// -- fail closed -------------------------------------------------------------------------------------------

#[test]
fn old_v1_store_is_refused_without_reinterpretation_or_mutation() {
    let dir = TempDir::new();
    let path = dir.join("old");
    let mut body = Vec::new();
    body.extend_from_slice(b"ELPCONT\x01");
    body.extend_from_slice(&1u16.to_be_bytes());
    body.extend_from_slice(&[0; 6]);
    body.extend_from_slice(&1u64.to_be_bytes());
    body.extend_from_slice(&[0; 8 + 32 + 8 + 32]);
    let old: Vec<u8> = [body.clone(), sha256::digest(&[b"elpis.continuity.register.v1\x00", &body]).to_vec()].concat();
    assert_eq!(old.len(), 136);
    write_slots(&path, &old, &[0u8; 136]);
    assert_eq!(Store::new(&path).unwrap().open(), Err(Code::Corrupt));
    assert_eq!(fs::read(path.join(SLOT_NAMES[0])).unwrap(), old);
    assert_eq!(fs::read(path.join(SLOT_NAMES[1])).unwrap(), vec![0u8; 136]);
}

#[test]
fn damaged_layouts_fail_closed() {
    type Mutate = fn(&Path);
    let cases: [Mutate; 4] = [
        |p| fs::remove_file(p.join(SLOT_NAMES[1])).unwrap(),
        |p| fs::write(p.join(SLOT_NAMES[0]), vec![b'x'; RECORD_SIZE + 1]).unwrap(),
        |p| fs::write(p.join("stray"), b"").unwrap(),
        |p| {
            // Deleting the first slot of a published store.
            fs::remove_file(p.join(SLOT_NAMES[0])).unwrap();
        },
    ];
    for (i, mutate) in cases.iter().enumerate() {
        let dir = TempDir::new();
        let path = dir.join("c");
        let mut s = open(&path);
        if i == 3 {
            s.anchor_cognition(Some(&d("k1-0"))).unwrap(); // published into continuity.b
        }
        drop(s);
        mutate(&path);
        let before = files(&path);
        assert_eq!(Store::new(&path).unwrap().open(), Err(Code::Corrupt), "case {i}");
        assert_eq!(files(&path), before, "case {i} touched nothing");
    }
}

#[test]
fn retired_receipt_history_layout_is_refused_not_read() {
    for name in ["MANIFEST", "events.log", "g0000000000000001.seg", "checkpoint.bin", "LOCK"] {
        let dir = TempDir::new();
        let path = dir.join("c");
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join(name), b"legacy").unwrap();
        assert_eq!(Store::new(&path).unwrap().open(), Err(Code::LegacyStorage), "{name}");
        assert_eq!(files(&path), vec![(name.to_string(), 6)]);
    }
}

#[test]
fn one_owner_per_directory() {
    let dir = TempDir::new();
    let path = dir.join("c");
    let mut first = open(&path);
    assert_eq!(Store::new(&path).unwrap().open(), Err(Code::Locked));
    assert_eq!(first.open(), Err(Code::Open));
    first.close();
    let second = open(&path);
    drop(second);
    drop(first);
    open(&path);
}

#[test]
fn paths_closed_stores_and_codes_are_stable() {
    assert_eq!(Store::new(Path::new("relative")).err(), Some(Code::Path));
    let dir = TempDir::new();
    assert_eq!(Store::new(&dir.join("c")).unwrap().snapshot(), Err(Code::Uninitialized));
    let expected = [
        (1, "CONTINUITY_UNINITIALIZED"),
        (2, "CONTINUITY_UNANCHORED"),
        (3, "CONTINUITY_ALREADY_ANCHORED"),
        (4, "CONTINUITY_STATE_MISMATCH"),
        (5, "CONTINUITY_CORRUPT"),
        (6, "CONTINUITY_PUBLICATION_REFUSED"),
        (7, "CONTINUITY_PUBLICATION_UNCERTAIN"),
        (8, "CONTINUITY_LOCKED"),
        (9, "CONTINUITY_AUTHORITY_MISMATCH"),
        (10, "CONTINUITY_EVOLUTION_PENDING"),
        (11, "CONTINUITY_EVOLUTION_NOT_PENDING"),
        (12, "CONTINUITY_EXHAUSTED"),
        (13, "CONTINUITY_INVALID"),
        (14, "CONTINUITY_PATH"),
        (15, "CONTINUITY_OPEN"),
        (16, "CONTINUITY_LEGACY_STORAGE"),
        (17, "CONTINUITY_IO"),
        (255, "CONTINUITY_TESTING_PROCESS_DEATH"),
    ];
    for (value, name) in expected {
        let code = Code::from_i32(value).unwrap();
        assert_eq!((code as i32, code.name()), (value, name));
        let c = unsafe { std::ffi::CStr::from_ptr(crate::ffi::elpis_continuity_code_name(value)) };
        assert_eq!(c.to_str().unwrap(), name);
    }
    assert_eq!(Code::ALL.len(), expected.len());
    assert!(crate::ffi::elpis_continuity_code_name(99).is_null());
}
