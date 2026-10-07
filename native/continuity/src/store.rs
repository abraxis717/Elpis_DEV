//! The two-slot crash-safe continuity register (docs/CONTINUITY.md).
//!
//! A directory holds exactly `continuity.a` and `continuity.b`, each one 176-byte record. The current
//! authority is the valid record with the higher generation. Publication writes the complete next record
//! into the slot that does not hold the current authority, then `fdatasync`s it: one write, one sync, no
//! rename, no log. Restart reads both slots. Nothing grows, nothing is replayed, nothing is compacted.
//!
//! Evolution law: an assertion is durably reserved (idle -> pending) before the caller may execute it, and
//! the exact pending authority is finalized (pending -> next idle) with the established result. After an
//! uncertain publication, reopening resolves to exactly one complete record: for a reservation, idle or
//! pending; for a finalization, pending or the next idle. Nothing retries, reconciles or clears a
//! reservation automatically.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, ErrorKind};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use crate::error::Code;
use crate::record::{Cognition, Digest, EvolutionState, Snapshot, MAX_COUNTER, RECORD_SIZE, ZERO};

pub const SLOT_NAMES: [&str; 2] = ["continuity.a", "continuity.b"];
const TMP_SUFFIX: &str = ".tmp";
/// Names of the retired receipt-history layout: refused, never read.
const LEGACY_NAMES: [&str; 6] =
    ["MANIFEST", "MANIFEST.tmp", "LOCK", "events.log", "checkpoint.bin", "checkpoint.bin.tmp"];

/// Named steps of the crash matrix (fault injection is compiled only into test and testing builds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Step {
    PublishBegin = 0,
    PublishWritten = 1,
    PublishSynced = 2,
    InitBWritten = 10,
    InitAWritten = 11,
    InitBRenamed = 12,
    InitARenamed = 13,
    InitDirSync = 14,
}

impl Step {
    pub fn from_u32(v: u32) -> Option<Step> {
        [
            Step::PublishBegin,
            Step::PublishWritten,
            Step::PublishSynced,
            Step::InitBWritten,
            Step::InitAWritten,
            Step::InitBRenamed,
            Step::InitARenamed,
            Step::InitDirSync,
        ]
        .into_iter()
        .find(|s| *s as u32 == v)
    }
}

#[cfg(any(test, feature = "testing"))]
pub mod probe {
    //! Test-only fault plan and I/O counters for one store.

    use super::Step;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Fault {
        /// Simulated process death at a step: nothing more is written, descriptors close.
        Die(Step),
        /// The record write fails before any byte is written.
        WriteFail,
        /// The first `n` bytes are written, then the write fails (the caller sees a refusal).
        TornFail(usize),
        /// The first `n` bytes are written, then the process dies.
        TornDie(usize),
        /// The write succeeds, the unsynced bytes are lost (the slot's previous bytes return), the sync
        /// "fails": the publication is uncertain and was in fact not durable.
        SyncFailLost,
        /// The write and the sync succeed but the sync reports failure: uncertain, in fact durable.
        SyncFailDurable,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    pub struct Counters {
        pub opens: u64,
        pub preads: u64,
        pub pread_bytes: u64,
        pub pwrites: u64,
        pub pwrite_bytes: u64,
        pub data_syncs: u64,
        pub dir_syncs: u64,
        pub renames: u64,
        pub unlinks: u64,
    }

    #[derive(Default)]
    pub struct Probe {
        /// `(publication, fault)`: publication 0 is the next open's initialization, n >= 1 the n-th
        /// publication after the plan was set.
        pub plan: Option<(u64, Fault)>,
        pub publications: u64,
        pub counters: Counters,
        /// The target slot's previous bytes while a `SyncFailLost` publication is in flight.
        pub lost: Option<[u8; crate::record::RECORD_SIZE]>,
    }
}

#[cfg(any(test, feature = "testing"))]
use probe::{Fault, Probe};

struct Opened {
    _lock: File,
    slots: [File; 2],
    current: Snapshot,
    slot: usize,
}

/// Exclusive owner of one continuity directory. Single writer; callers serialize access.
pub struct Store {
    directory: PathBuf,
    opened: Option<Opened>,
    poisoned: Option<Code>,
    #[cfg(any(test, feature = "testing"))]
    pub(crate) probe: Probe,
}

fn io_code(_: io::Error) -> Code {
    Code::Io
}

impl Store {
    pub fn new(directory: &Path) -> Result<Store, Code> {
        if !directory.is_absolute() {
            return Err(Code::Path);
        }
        Ok(Store {
            directory: directory.to_path_buf(),
            opened: None,
            poisoned: None,
            #[cfg(any(test, feature = "testing"))]
            probe: Probe::default(),
        })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    // -- test seams (no-ops in the production library) ---------------------------------------------

    #[cfg(any(test, feature = "testing"))]
    fn fault(&self, publication: u64) -> Option<Fault> {
        match self.probe.plan {
            Some((p, f)) if p == publication => Some(f),
            _ => None,
        }
    }

    #[inline]
    fn step(&mut self, _step: Step, _publication: u64) -> Result<(), Code> {
        #[cfg(any(test, feature = "testing"))]
        if self.fault(_publication) == Some(Fault::Die(_step)) {
            return Err(self.die());
        }
        Ok(())
    }

    #[cfg(any(test, feature = "testing"))]
    fn die(&mut self) -> Code {
        self.probe.plan = None;
        self.opened = None; // the process is gone: descriptors (and the lock) close
        self.poisoned = Some(Code::TestingProcessDeath);
        Code::TestingProcessDeath
    }

    #[inline]
    fn count(&mut self, _op: fn(&mut probe_counters::C, u64), _bytes: u64) {
        #[cfg(any(test, feature = "testing"))]
        _op(&mut self.probe.counters, _bytes);
    }

    // -- lifecycle -----------------------------------------------------------------------------------

    pub fn open(&mut self) -> Result<Snapshot, Code> {
        if self.opened.is_some() {
            return Err(Code::Open);
        }
        self.poisoned = None;
        let created = !self.directory.exists();
        fs::create_dir_all(&self.directory).map_err(io_code)?;
        if created {
            if let Some(parent) = self.directory.parent().map(Path::to_path_buf) {
                self.sync_dir(&parent).map_err(io_code)?;
            }
        }
        // The directory itself is the lock: no lock file, and nothing is inspected or initialized before
        // exclusive ownership.
        let lock = File::open(&self.directory).map_err(io_code)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(Code::Locked),
            Err(TryLockError::Error(e)) => return Err(io_code(e)),
        }
        self.admit_directory()?;
        let mut files = Vec::with_capacity(2);
        let mut records = [[0u8; RECORD_SIZE]; 2];
        for (i, name) in SLOT_NAMES.iter().enumerate() {
            let file = OpenOptions::new().read(true).write(true).open(self.directory.join(name)).map_err(io_code)?;
            self.count(probe_counters::open, 0);
            if file.metadata().map_err(io_code)?.len() != RECORD_SIZE as u64 {
                return Err(Code::Corrupt);
            }
            file.read_exact_at(&mut records[i], 0).map_err(io_code)?;
            self.count(probe_counters::pread, RECORD_SIZE as u64);
            files.push(file);
        }
        let (current, slot) = resolve(&records)?;
        let b = files.pop().expect("two slots");
        let a = files.pop().expect("two slots");
        self.opened = Some(Opened { _lock: lock, slots: [a, b], current, slot });
        Ok(current)
    }

    pub fn close(&mut self) {
        self.opened = None;
    }

    fn admit_directory(&mut self) -> Result<(), Code> {
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.directory).map_err(io_code)? {
            names.push(entry.map_err(io_code)?.file_name().to_string_lossy().into_owned());
        }
        let legacy = names.iter().any(|n| {
            LEGACY_NAMES.contains(&n.as_str()) || (n.starts_with('g') && (n.ends_with(".ckpt") || n.ends_with(".seg")))
        });
        if legacy {
            return Err(Code::LegacyStorage);
        }
        let allowed = |n: &str| SLOT_NAMES.iter().any(|s| n == *s || n.strip_suffix(TMP_SUFFIX) == Some(*s));
        if names.iter().any(|n| !allowed(n)) {
            return Err(Code::Corrupt);
        }
        let has = |n: &str| names.iter().any(|x| x == n);
        if has(SLOT_NAMES[0]) {
            if !has(SLOT_NAMES[1]) {
                return Err(Code::Corrupt);
            }
            for name in SLOT_NAMES {
                // Debris of an initialization that already completed.
                let tmp = format!("{name}{TMP_SUFFIX}");
                if has(&tmp) {
                    fs::remove_file(self.directory.join(&tmp)).map_err(io_code)?;
                    self.count(probe_counters::unlink, 0);
                }
            }
            return Ok(());
        }
        // No continuity.a: initialization never reached its publication point. Its debris is a renamed but
        // still empty continuity.b at most. A continuity.b that carries a record was published to, so its
        // partner was deleted: fail closed rather than re-initialize over authority.
        if has(SLOT_NAMES[1]) {
            let b = fs::read(self.directory.join(SLOT_NAMES[1])).map_err(io_code)?;
            if b != [0u8; RECORD_SIZE] {
                return Err(Code::Corrupt);
            }
        }
        self.initialize(&names)
    }

    /// First open of an empty directory: publish generation 1 (unanchored, evolution genesis).
    fn initialize(&mut self, names: &[String]) -> Result<(), Code> {
        for name in names {
            // An initialization that never reached its publication point.
            fs::remove_file(self.directory.join(name)).map_err(io_code)?;
            self.count(probe_counters::unlink, 0);
        }
        let payloads: [(&str, [u8; RECORD_SIZE], Step, Step); 2] = [
            (SLOT_NAMES[1], [0u8; RECORD_SIZE], Step::InitBWritten, Step::InitBRenamed),
            (SLOT_NAMES[0], Snapshot::GENESIS.encode(), Step::InitAWritten, Step::InitARenamed),
        ];
        for (name, payload, written, _) in &payloads {
            let path = self.directory.join(format!("{name}{TMP_SUFFIX}"));
            let file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path).map_err(io_code)?;
            self.count(probe_counters::open, 0);
            file.write_all_at(payload, 0).map_err(io_code)?;
            self.count(probe_counters::pwrite, RECORD_SIZE as u64);
            sync_data(&file).map_err(io_code)?;
            self.count(probe_counters::data_sync, 0);
            drop(file);
            self.step(*written, 0)?;
        }
        // b first: the existence of continuity.a is the publication point.
        for (name, _, _, renamed) in &payloads {
            fs::rename(self.directory.join(format!("{name}{TMP_SUFFIX}")), self.directory.join(name))
                .map_err(io_code)?;
            self.count(probe_counters::rename, 0);
            self.step(*renamed, 0)?;
        }
        let dir = self.directory.clone();
        self.sync_dir(&dir).map_err(io_code)?;
        self.step(Step::InitDirSync, 0)
    }

    fn sync_dir(&mut self, path: &Path) -> io::Result<()> {
        let dir = File::open(path)?;
        sync_data(&dir)?;
        self.count(probe_counters::dir_sync, 0);
        Ok(())
    }

    // -- reads -----------------------------------------------------------------------------------------

    fn require(&self) -> Result<&Opened, Code> {
        if let Some(code) = self.poisoned {
            return Err(code);
        }
        self.opened.as_ref().ok_or(Code::Uninitialized)
    }

    pub fn snapshot(&self) -> Result<Snapshot, Code> {
        Ok(self.require()?.current)
    }

    // -- the only transitions --------------------------------------------------------------------------

    fn publish(&mut self, next: Snapshot) -> Result<Snapshot, Code> {
        let record = next.encode();
        #[cfg(any(test, feature = "testing"))]
        {
            self.probe.publications += 1;
        }
        #[cfg(any(test, feature = "testing"))]
        let n = self.probe.publications;
        #[cfg(not(any(test, feature = "testing")))]
        let n = 1;
        self.step(Step::PublishBegin, n)?;
        let target = 1 - self.require()?.slot;
        self.write_slot(target, &record, n)?;
        self.step(Step::PublishWritten, n)?;
        if self.sync_slot(target, n).is_err() {
            // Durable outcome unknown: reopen resolves to exactly one complete record.
            self.poisoned = Some(Code::PublicationUncertain);
            self.close();
            return Err(Code::PublicationUncertain);
        }
        self.step(Step::PublishSynced, n)?;
        let opened = self.opened.as_mut().expect("open");
        opened.current = next;
        opened.slot = target;
        Ok(next)
    }

    fn write_slot(&mut self, target: usize, record: &[u8; RECORD_SIZE], _n: u64) -> Result<(), Code> {
        #[cfg(any(test, feature = "testing"))]
        match self.fault(_n) {
            Some(Fault::WriteFail) => return Err(Code::PublicationRefused),
            Some(Fault::TornFail(cut)) | Some(Fault::TornDie(cut)) => {
                let cut = cut.min(RECORD_SIZE);
                let file = &self.opened.as_ref().expect("open").slots[target];
                file.write_all_at(&record[..cut], 0).map_err(|_| Code::PublicationRefused)?;
                self.count(probe_counters::pwrite, cut as u64);
                if matches!(self.fault(_n), Some(Fault::TornDie(_))) {
                    return Err(self.die());
                }
                self.probe.plan = None;
                return Err(Code::PublicationRefused);
            }
            Some(Fault::SyncFailLost) => {
                // Remember the slot's previous bytes; sync_slot restores them before "failing".
                let file = &self.opened.as_ref().expect("open").slots[target];
                let mut previous = [0u8; RECORD_SIZE];
                file.read_exact_at(&mut previous, 0).map_err(|_| Code::PublicationRefused)?;
                self.probe.lost = Some(previous);
            }
            _ => {}
        }
        let file = &self.opened.as_ref().expect("open").slots[target];
        // The current authority's slot is untouched; a failed write leaves it in force.
        file.write_all_at(record, 0).map_err(|_| Code::PublicationRefused)?;
        self.count(probe_counters::pwrite, RECORD_SIZE as u64);
        Ok(())
    }

    fn sync_slot(&mut self, target: usize, _n: u64) -> io::Result<()> {
        #[cfg(any(test, feature = "testing"))]
        match self.fault(_n) {
            Some(Fault::SyncFailLost) => {
                let previous = self.probe.lost.take().expect("previous bytes");
                let file = &self.opened.as_ref().expect("open").slots[target];
                file.write_all_at(&previous, 0)?;
                sync_data(file)?;
                self.probe.plan = None;
                return Err(io::Error::other("simulated: sync outcome unknown (bytes lost)"));
            }
            Some(Fault::SyncFailDurable) => {
                sync_data(&self.opened.as_ref().expect("open").slots[target])?;
                self.probe.plan = None;
                return Err(io::Error::other("simulated: sync outcome unknown (durable)"));
            }
            _ => {}
        }
        sync_data(&self.opened.as_ref().expect("open").slots[target])?;
        self.count(probe_counters::data_sync, 0);
        Ok(())
    }

    /// Explicitly anchor the first K1 lineage at a retained-state identity.
    pub fn anchor_cognition(&mut self, k1: Option<&Digest>) -> Result<Snapshot, Code> {
        let current = self.require()?.current;
        let k1 = *k1.ok_or(Code::Invalid)?;
        if current.cognition() != Cognition::Unanchored {
            return Err(Code::AlreadyAnchored);
        }
        let next = Snapshot::new(current.generation() + 1, Cognition::Anchored(k1), current.evolution())?;
        self.publish(next)
    }

    /// Publish `after` as the expected K1 identity of a committed turn from `before`.
    pub fn commit_cognition(&mut self, before: Option<&Digest>, after: Option<&Digest>) -> Result<Snapshot, Code> {
        let current = self.require()?.current;
        let (before, after) = (*before.ok_or(Code::Invalid)?, *after.ok_or(Code::Invalid)?);
        match current.cognition() {
            Cognition::Unanchored => return Err(Code::Unanchored),
            Cognition::Anchored(expected) if expected != before => return Err(Code::StateMismatch),
            Cognition::Anchored(_) => {}
        }
        let next = Snapshot::new(current.generation() + 1, Cognition::Anchored(after), current.evolution())?;
        self.publish(next)
    }

    /// Durably reserve one exact assertion (idle -> pending) before the caller may execute it.
    pub fn reserve_evolution(
        &mut self,
        expected: Option<&EvolutionState>,
        assertion: Option<&Digest>,
    ) -> Result<Snapshot, Code> {
        let current = self.require()?.current;
        let expected = expected.ok_or(Code::AuthorityMismatch)?;
        if *expected != current.evolution() {
            return Err(Code::AuthorityMismatch);
        }
        let (revision, head) = match *expected {
            EvolutionState::Pending { .. } => return Err(Code::EvolutionPending),
            EvolutionState::Idle { revision, head } => (revision, head),
        };
        let assertion = *assertion.ok_or(Code::Invalid)?;
        // Refuse exhaustion before an irreversible attempt, not at finalization: the reservation and its
        // finalization each need one generation, the finalization one revision.
        if current.generation() >= MAX_COUNTER - 1 || revision == MAX_COUNTER {
            return Err(Code::Exhausted);
        }
        let pending = EvolutionState::pending(revision, head, assertion)?;
        self.publish(Snapshot::new(current.generation() + 1, current.cognition(), pending)?)
    }

    /// Finalize the exact pending authority (pending -> next idle) with an explicitly established result.
    ///
    /// Also the explicit reconciliation primitive after restart: the caller must establish the completed
    /// result externally. Never retries an attempt or clears a reservation back to its previous idle state.
    pub fn finalize_evolution(
        &mut self,
        expected: Option<&EvolutionState>,
        receipt: Option<&Digest>,
    ) -> Result<Snapshot, Code> {
        let current = self.require()?.current;
        let expected = expected.ok_or(Code::AuthorityMismatch)?;
        if *expected != current.evolution() {
            return Err(Code::AuthorityMismatch);
        }
        let revision = match *expected {
            EvolutionState::Idle { .. } => return Err(Code::EvolutionNotPending),
            EvolutionState::Pending { revision, .. } => revision,
        };
        let receipt = *receipt.ok_or(Code::Invalid)?;
        if receipt == ZERO {
            return Err(Code::Invalid);
        }
        let next = EvolutionState::idle(revision + 1, receipt).map_err(|_| Code::Invalid)?;
        self.publish(Snapshot::new(current.generation() + 1, current.cognition(), next)?)
    }
}

fn sync_data(file: &File) -> io::Result<()> {
    loop {
        match file.sync_data() {
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            other => return other,
        }
    }
}

/// The current authority: the valid record with the higher generation. Torn or stale slots are skipped;
/// no valid record or two records of equal generation fail closed.
pub fn resolve(records: &[[u8; RECORD_SIZE]; 2]) -> Result<(Snapshot, usize), Code> {
    let decoded: Vec<Option<Snapshot>> = records.iter().map(|r| Snapshot::decode(r).ok().flatten()).collect();
    match (decoded[0], decoded[1]) {
        (None, None) => Err(Code::Corrupt),
        (Some(a), Some(b)) if a.generation() == b.generation() => Err(Code::Corrupt),
        (Some(a), Some(b)) => Ok(if b.generation() > a.generation() { (b, 1) } else { (a, 0) }),
        (Some(a), None) => Ok((a, 0)),
        (None, Some(b)) => Ok((b, 1)),
    }
}

/// Counter updates for the test probe (inert in the production library).
pub(crate) mod probe_counters {
    #[cfg(any(test, feature = "testing"))]
    pub type C = super::probe::Counters;
    #[cfg(not(any(test, feature = "testing")))]
    pub struct C;

    #[cfg(any(test, feature = "testing"))]
    mod imp {
        use super::C;
        pub fn open(c: &mut C, _: u64) {
            c.opens += 1;
        }
        pub fn pread(c: &mut C, b: u64) {
            c.preads += 1;
            c.pread_bytes += b;
        }
        pub fn pwrite(c: &mut C, b: u64) {
            c.pwrites += 1;
            c.pwrite_bytes += b;
        }
        pub fn data_sync(c: &mut C, _: u64) {
            c.data_syncs += 1;
        }
        pub fn dir_sync(c: &mut C, _: u64) {
            c.dir_syncs += 1;
        }
        pub fn rename(c: &mut C, _: u64) {
            c.renames += 1;
        }
        pub fn unlink(c: &mut C, _: u64) {
            c.unlinks += 1;
        }
    }
    #[cfg(not(any(test, feature = "testing")))]
    mod imp {
        use super::C;
        pub fn open(_: &mut C, _: u64) {}
        pub fn pread(_: &mut C, _: u64) {}
        pub fn pwrite(_: &mut C, _: u64) {}
        pub fn data_sync(_: &mut C, _: u64) {}
        pub fn dir_sync(_: &mut C, _: u64) {}
        pub fn rename(_: &mut C, _: u64) {}
        pub fn unlink(_: &mut C, _: u64) {}
    }
    pub use imp::*;
}
