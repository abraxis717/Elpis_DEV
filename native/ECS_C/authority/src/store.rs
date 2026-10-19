//! The ECS_C authority store: provisioning, open and bounded recovery, transitions, operator reconciliation.
//!
//! **Files (fixed for the life of a deployment; nothing is ever created, renamed or removed after provisioning):**
//!
//! | directory | file | size |
//! |---|---|---|
//! | state | `authority.a`, `authority.b` | `R(capacity)` each |
//! | authority (protected: owner-only, never inside the state directory) | `deployment.key` (mode 0400) | 32 |
//! | authority | `witness.a`, `witness.b` | 240 each |
//!
//! **Transition** (WARM / control lane, never HOT): validate the event against the current authority with the
//! transition law, write the complete next record into the slot that does not hold the current authority and
//! `fdatasync` it (the commit point), then write the witness into the witness slot that does not hold the current
//! pin and `fdatasync` it, then acknowledge. Two writes, two syncs, `R + 240` bytes, whatever the history length.
//!
//! **Recovery** reads two records, two witnesses and the key (constant work), classifies each slot, and re-replays
//! the newest transition against the predecessor the other slot still holds. It never depends on how many
//! transitions happened since provisioning.
//!
//! **Threat model.** T0: process death at any point, torn writes, a failed or lying `fdatasync`. T1: an adversary
//! who can read and write the state directory (substitute, forge, copy back older records, delete) but can neither
//! read the deployment key nor write the authority directory. The MAC defeats forgery and substitution; the witness
//! in the directory T1 cannot write defeats rollback and replay of older genuine records. Not defended: a holder of
//! the deployment key, code running inside the process, an adversary who can write the authority directory, and a
//! device that loses acknowledged writes of *both* directories coherently. The witness is a second record of the
//! same digest only in appearance: its independence is the access-control separation of its directory, which
//! `open` verifies (owner, mode, no symlinks, no nesting), not a second copy beside the first.

use std::collections::BTreeSet;
use std::fs::{self, DirBuilder, File, OpenOptions, TryLockError};
use std::io::{self, ErrorKind, Read};
use std::os::unix::fs::{DirBuilderExt, FileExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use elpis_continuity::sha256;

use crate::error::Code;
use crate::hmac::wipe;
use crate::model::*;
use crate::record::*;

pub const STATE_FILES: [&str; 2] = ["authority.a", "authority.b"];
pub const KEY_FILE: &str = "deployment.key";
pub const WITNESS_FILES: [&str; 2] = ["witness.a", "witness.b"];
const TMP: &str = ".tmp";

extern "C" {
    fn geteuid() -> u32;
}

fn euid() -> u32 {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { geteuid() }
}

/// Named steps of the crash matrix (process death is injectable only in test and testing builds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Step {
    TransitionBegin = 0,
    SlotWritten = 1,
    SlotSynced = 2,
    WitnessWritten = 3,
    WitnessSynced = 4,
    ProvisionFilesWritten = 10,
    ProvisionStateRenamed = 11,
    ProvisionStateSynced = 12,
    ProvisionKeyRenamed = 13,
    ProvisionWitnessBRenamed = 14,
    ProvisionAuthoritySynced = 15,
    ProvisionPublished = 16,
    ProvisionDone = 17,
    OpenWitnessWritten = 20,
    OpenWitnessSynced = 21,
}

impl Step {
    pub const ALL: [Step; 15] = [
        Step::TransitionBegin,
        Step::SlotWritten,
        Step::SlotSynced,
        Step::WitnessWritten,
        Step::WitnessSynced,
        Step::ProvisionFilesWritten,
        Step::ProvisionStateRenamed,
        Step::ProvisionStateSynced,
        Step::ProvisionKeyRenamed,
        Step::ProvisionWitnessBRenamed,
        Step::ProvisionAuthoritySynced,
        Step::ProvisionPublished,
        Step::ProvisionDone,
        Step::OpenWitnessWritten,
        Step::OpenWitnessSynced,
    ];

    pub fn from_u32(v: u32) -> Option<Step> {
        Step::ALL.into_iter().find(|s| *s as u32 == v)
    }
}

/// Which in-place write a fault applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Slot,
    Witness,
}

/// I/O counters. Recovery reports its own reads and replays from them in every build (the bounded-recovery claim
/// is observable in production); the testing library also exports them whole.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Counters {
    pub opens: u64,
    pub creates: u64,
    pub preads: u64,
    pub pread_bytes: u64,
    pub pwrites: u64,
    pub pwrite_bytes: u64,
    pub data_syncs: u64,
    pub dir_syncs: u64,
    pub renames: u64,
    pub unlinks: u64,
    pub replays: u64,
    pub key_reads: u64,
}

#[cfg(any(test, feature = "testing"))]
pub mod probe {
    //! Test-only fault plan for one store.

    use super::{Step, Target};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum WriteFault {
        /// The write fails before any byte is written.
        Fail,
        /// The first `n` bytes are written, then the write fails.
        TornFail(usize),
        /// The first `n` bytes are written, then the process dies.
        TornDie(usize),
        /// The write succeeds, the unsynced bytes are lost (the previous bytes return), the sync "fails".
        SyncLost,
        /// The write and the sync succeed, but the sync reports failure.
        SyncDurable,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Fault {
        Die(Step),
        Write(Target, WriteFault),
    }

    #[derive(Default)]
    pub struct Probe {
        /// `(publication, fault)`: publication 0 is the next open (witness completion) or a provisioning, n >= 1
        /// the n-th transition after the plan was set.
        pub plan: Option<(u64, Fault)>,
        pub publications: u64,
        pub lost: Option<Vec<u8>>,
    }
}

#[derive(Default)]
pub(crate) struct Io {
    pub(crate) counters: Counters,
    #[cfg(any(test, feature = "testing"))]
    pub(crate) probe: probe::Probe,
}

fn sync_data(file: &File) -> io::Result<()> {
    loop {
        match file.sync_data() {
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            other => return other,
        }
    }
}

impl Io {
    #[cfg(any(test, feature = "testing"))]
    fn planned(&self, n: u64) -> Option<probe::Fault> {
        match self.probe.plan {
            Some((p, f)) if p == n => Some(f),
            _ => None,
        }
    }

    #[cfg(any(test, feature = "testing"))]
    fn next_publication(&mut self) -> u64 {
        self.probe.publications += 1;
        self.probe.publications
    }

    #[cfg(not(any(test, feature = "testing")))]
    fn next_publication(&mut self) -> u64 {
        1
    }

    fn step(&mut self, _step: Step, _n: u64) -> Result<(), Code> {
        #[cfg(any(test, feature = "testing"))]
        if self.planned(_n) == Some(probe::Fault::Die(_step)) {
            self.probe.plan = None;
            return Err(Code::TestingProcessDeath);
        }
        Ok(())
    }

    fn read_at(&mut self, file: &File, buf: &mut [u8]) -> io::Result<()> {
        file.read_exact_at(buf, 0)?;
        self.counters.preads += 1;
        self.counters.pread_bytes += buf.len() as u64;
        Ok(())
    }

    fn write_at(&mut self, file: &File, buf: &[u8]) -> io::Result<()> {
        file.write_all_at(buf, 0)?;
        self.counters.pwrites += 1;
        self.counters.pwrite_bytes += buf.len() as u64;
        Ok(())
    }

    fn sync_dir(&mut self, path: &Path) -> io::Result<()> {
        sync_data(&File::open(path)?)?;
        self.counters.dir_syncs += 1;
        Ok(())
    }

    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()> {
        fs::rename(from, to)?;
        self.counters.renames += 1;
        Ok(())
    }

    fn unlink(&mut self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)?;
        self.counters.unlinks += 1;
        Ok(())
    }

    /// Provisioning only: a new file of exactly `bytes` with exactly `mode`, synced.
    fn create(&mut self, path: &Path, bytes: &[u8], mode: u32) -> io::Result<()> {
        let file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?;
        self.counters.creates += 1;
        self.write_at(&file, bytes)?;
        file.set_permissions(fs::Permissions::from_mode(mode))?;
        sync_data(&file)?;
        self.counters.data_syncs += 1;
        Ok(())
    }

    /// Write a slot or a witness in place, under the fault plan of publication `n`.
    fn put(&mut self, file: &File, bytes: &[u8], _target: Target, _n: u64) -> Result<(), Code> {
        #[cfg(any(test, feature = "testing"))]
        if let Some(probe::Fault::Write(t, f)) = self.planned(_n) {
            use probe::WriteFault;
            if t == _target {
                match f {
                    WriteFault::Fail => {
                        self.probe.plan = None;
                        return Err(Code::PublicationRefused);
                    }
                    WriteFault::TornFail(cut) | WriteFault::TornDie(cut) => {
                        let cut = cut.min(bytes.len());
                        self.write_at(file, &bytes[..cut]).map_err(|_| Code::PublicationRefused)?;
                        self.probe.plan = None;
                        return Err(if matches!(f, WriteFault::TornDie(_)) {
                            Code::TestingProcessDeath
                        } else {
                            Code::PublicationRefused
                        });
                    }
                    WriteFault::SyncLost => {
                        let mut previous = vec![0u8; bytes.len()];
                        file.read_exact_at(&mut previous, 0).map_err(|_| Code::PublicationRefused)?;
                        self.probe.lost = Some(previous);
                    }
                    WriteFault::SyncDurable => {}
                }
            }
        }
        self.write_at(file, bytes).map_err(|_| Code::PublicationRefused)
    }

    fn settle(&mut self, file: &File, _target: Target, _n: u64) -> io::Result<()> {
        #[cfg(any(test, feature = "testing"))]
        if let Some(probe::Fault::Write(t, f)) = self.planned(_n) {
            use probe::WriteFault;
            if t == _target {
                match f {
                    WriteFault::SyncLost => {
                        let previous = self.probe.lost.take().expect("previous bytes");
                        file.write_all_at(&previous, 0)?;
                        sync_data(file)?;
                        self.probe.plan = None;
                        return Err(io::Error::other("simulated: sync outcome unknown (bytes lost)"));
                    }
                    WriteFault::SyncDurable => {
                        sync_data(file)?;
                        self.probe.plan = None;
                        return Err(io::Error::other("simulated: sync outcome unknown (durable)"));
                    }
                    _ => {}
                }
            }
        }
        sync_data(file)?;
        self.counters.data_syncs += 1;
        Ok(())
    }
}

// -- recovery dispositions and modes ---------------------------------------------------------------------

/// What recovery concluded. The numeric values are C ABI. 1..=5 open normally (4 restricted), 10..=13 open in a
/// reconcile-only mode, 20..=23 refuse to open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Disposition {
    /// The acknowledged record, verified against its predecessor by replay (or genesis recomputed).
    Clean = 1,
    /// The non-current slot holds an incomplete write: an interrupted transition that was never acknowledged.
    TrailingIncomplete = 2,
    /// A durable but unwitnessed transition (the process stopped between the record sync and the witness): it is
    /// verified, adopted, and the witness completed. Present once, never duplicated.
    InterruptedCompleted = 3,
    /// The acknowledged record is intact; the other slot holds a checksummed record without a valid MAC (a
    /// substitution). Reads only, until the operator acknowledges it.
    RestrictedUnauthenticated = 4,
    /// The acknowledged record is intact and authenticated, but its predecessor is not available to replay.
    PredecessorUnavailable = 5,
    /// Every authenticated record is older than the acknowledged one (copied-back files, a lying device).
    Rollback = 10,
    /// The acknowledged record is torn or corrupt; an older authenticated record remains.
    CorruptAcknowledged = 11,
    /// The acknowledged record was replaced by an unauthenticated one; an older authenticated record remains.
    UnauthenticatedAcknowledged = 12,
    /// Neither witness slot holds a valid witness.
    WitnessMissing = 13,
    /// Two authenticated records that do not form one history.
    Conflict = 20,
    /// An authenticated record newer than the witness allows, or witnesses that disagree.
    WitnessInconsistent = 21,
    /// An authenticated record whose transition does not replay from its predecessor (or is invalid).
    ReplayMismatch = 22,
    /// No authenticated record at all.
    Unrecoverable = 23,
}

impl Disposition {
    pub const ALL: [Disposition; 13] = [
        Disposition::Clean,
        Disposition::TrailingIncomplete,
        Disposition::InterruptedCompleted,
        Disposition::RestrictedUnauthenticated,
        Disposition::PredecessorUnavailable,
        Disposition::Rollback,
        Disposition::CorruptAcknowledged,
        Disposition::UnauthenticatedAcknowledged,
        Disposition::WitnessMissing,
        Disposition::Conflict,
        Disposition::WitnessInconsistent,
        Disposition::ReplayMismatch,
        Disposition::Unrecoverable,
    ];

    pub fn from_u32(v: u32) -> Option<Disposition> {
        Disposition::ALL.into_iter().find(|d| *d as u32 == v)
    }
}

/// What the open authority accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Normal,
    /// Reads; only RECONCILE_UNAUTHENTICATED for exactly this slot and these observed bytes.
    Restricted {
        slot: u8,
        observed: Digest,
    },
    /// The current authority is only a candidate; only RECONCILE_ROLLBACK naming exactly this abandonment.
    Rollback {
        abandoned_generation: u64,
        abandoned_digest: Digest,
        class: u8,
    },
    /// The current authority is only a candidate; only RECONCILE_REPIN naming exactly these witness classes.
    WitnessMissing {
        classes: [u8; 2],
    },
}

impl Mode {
    pub fn code(&self) -> u32 {
        match self {
            Mode::Normal => 0,
            Mode::Restricted { .. } => 1,
            Mode::Rollback { .. } => 2,
            Mode::WitnessMissing { .. } => 3,
        }
    }
}

/// The recovery report of one open.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// `Disposition` as u32 (0 before recovery ran).
    pub disposition: u32,
    pub mode: u32,
    pub slot_class: [u8; 2],
    pub witness_class: [u8; 2],
    pub replay_verified: bool,
    pub witness_completed: bool,
    /// The authority holds an evolution reservation whose result must be established and finalized: an
    /// outstanding cross-domain operation. Never cleared automatically.
    pub evolution_pending: bool,
    /// Generations the slot headers claim (untrusted; 0 when unreadable).
    pub slot_generation: [u64; 2],
    /// The effective witness's pinned generation (0 when none).
    pub witness_generation: u64,
    /// The generation of the authority (or candidate) the handle holds.
    pub current_generation: u64,
    /// SHA-256 of each slot's raw bytes as read.
    pub observed: [Digest; 2],
    /// Recovery work: reads, bytes read, transitions replayed.
    pub reads: u64,
    pub read_bytes: u64,
    pub replays: u64,
}

/// A refused open: the code, and the recovery report when recovery ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub code: Code,
    pub report: Option<Box<Report>>,
}

impl From<Code> for Refusal {
    fn from(code: Code) -> Refusal {
        Refusal { code, report: None }
    }
}

/// The acknowledgment of one transition, returned to its caller. ECS_C keeps only the chain head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Receipt {
    pub generation: u64,
    pub predecessor_generation: u64,
    pub receipt: Digest,
    pub record_digest: Digest,
    pub state_root: Digest,
    pub receipt_head: Digest,
    pub receipt_count: u64,
    pub founded: Option<Digest>,
}

/// What provisioning published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provisioned {
    pub capacity: u32,
    pub record_bytes: u64,
    pub lineage: Digest,
    pub deployment: Digest,
    pub record_digest: Digest,
    pub state_root: Digest,
}

struct Opened {
    _locks: [File; 2],
    slots: [File; 2],
    witnesses: [File; 2],
    keys: Keys,
    capacity: u32,
    current: Record,
    slot: usize,
    witness_slot: usize,
    mode: Mode,
    report: Report,
}

/// Exclusive owner of one ECS_C authority (a state directory and its authority directory). Single writer;
/// callers serialize access.
pub struct Authority {
    state_dir: PathBuf,
    authority_dir: PathBuf,
    opened: Option<Opened>,
    poisoned: Option<Code>,
    pub(crate) io: Io,
}

// -- filesystem admission ----------------------------------------------------------------------------------

fn io_code(_: io::Error) -> Code {
    Code::Io
}

fn check_paths(state_dir: &Path, authority_dir: &Path) -> Result<(), Code> {
    if !state_dir.is_absolute() || !authority_dir.is_absolute() || state_dir == authority_dir {
        return Err(Code::Path);
    }
    Ok(())
}

/// Neither directory may contain the other: a T1 adversary who can write the state directory must not be able
/// to reach the authority directory.
fn distinct(state_dir: &Path, authority_dir: &Path) -> Result<(), Code> {
    let s = fs::canonicalize(state_dir).map_err(io_code)?;
    let a = fs::canonicalize(authority_dir).map_err(io_code)?;
    if s == a || s.starts_with(&a) || a.starts_with(&s) {
        return Err(Code::Path);
    }
    Ok(())
}

fn protected_dir(m: &fs::Metadata, euid: u32) -> bool {
    m.is_dir() && !m.file_type().is_symlink() && m.uid() == euid && m.mode() & 0o077 == 0
}

fn lock(dir: &Path) -> Result<File, Code> {
    let file = File::open(dir).map_err(io_code)?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(Code::Locked),
        Err(TryLockError::Error(e)) => Err(io_code(e)),
    }
}

fn names(dir: &Path) -> Result<BTreeSet<String>, Code> {
    let mut out = BTreeSet::new();
    for entry in fs::read_dir(dir).map_err(io_code)? {
        out.insert(entry.map_err(io_code)?.file_name().to_string_lossy().into_owned());
    }
    Ok(out)
}

fn set_of(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| n.to_string()).collect()
}

/// Open an existing slot or witness file read-write, refusing symlinks and swapped files.
fn open_checked(io: &mut Io, path: &Path, euid: u32, protected: bool) -> Result<File, Code> {
    let m = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == ErrorKind::NotFound => return Err(Code::Corrupt),
        Err(e) => return Err(io_code(e)),
    };
    if !m.file_type().is_file() {
        return Err(if protected { Code::Unprotected } else { Code::Corrupt });
    }
    if protected && (m.uid() != euid || m.mode() & 0o077 != 0) {
        return Err(Code::Unprotected);
    }
    let file = OpenOptions::new().read(true).write(true).open(path).map_err(io_code)?;
    io.counters.opens += 1;
    let fm = file.metadata().map_err(io_code)?;
    if fm.dev() != m.dev() || fm.ino() != m.ino() {
        return Err(Code::Corrupt);
    }
    Ok(file)
}

fn read_key(io: &mut Io, path: &Path, euid: u32) -> Result<[u8; KEY_BYTES], Code> {
    let m = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == ErrorKind::NotFound => return Err(Code::Corrupt),
        Err(e) => return Err(io_code(e)),
    };
    if !m.file_type().is_file() || m.uid() != euid || m.mode() & 0o777 != 0o400 || m.len() != KEY_BYTES as u64 {
        return Err(Code::Unprotected);
    }
    let file = File::open(path).map_err(io_code)?;
    io.counters.opens += 1;
    let fm = file.metadata().map_err(io_code)?;
    if fm.dev() != m.dev() || fm.ino() != m.ino() {
        return Err(Code::Unprotected);
    }
    let mut key = [0u8; KEY_BYTES];
    io.read_at(&file, &mut key).map_err(io_code)?;
    io.counters.key_reads += 1;
    if zero(&key) {
        return Err(Code::Corrupt);
    }
    Ok(key)
}

fn random32() -> Result<[u8; 32], Code> {
    let mut source = File::open("/dev/urandom").map_err(io_code)?;
    loop {
        let mut b = [0u8; 32];
        source.read_exact(&mut b).map_err(io_code)?;
        if !zero(&b) {
            return Ok(b);
        }
    }
}

/// Create `dir` owner-only, or admit an existing one (the authority directory must already be protected).
fn prepare_dir(io: &mut Io, dir: &Path, protected: bool, euid: u32) -> Result<(), Code> {
    match fs::symlink_metadata(dir) {
        Ok(m) => {
            if !m.is_dir() || m.file_type().is_symlink() {
                return Err(if protected { Code::Unprotected } else { Code::Path });
            }
            if protected && !protected_dir(&m, euid) {
                return Err(Code::Unprotected);
            }
            Ok(())
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let parent = dir.parent().ok_or(Code::Path)?;
            fs::create_dir_all(parent).map_err(io_code)?;
            DirBuilder::new().mode(0o700).create(dir).map_err(io_code)?;
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).map_err(io_code)?;
            io.counters.creates += 1;
            io.sync_dir(parent).map_err(io_code)
        }
        Err(e) => Err(io_code(e)),
    }
}

// -- recovery ------------------------------------------------------------------------------------------------

struct Recovered {
    slot: usize,
    witness_slot: usize,
    mode: Mode,
    disposition: Disposition,
    replay_verified: bool,
    complete_witness: bool,
}

/// `newer` is exactly `older` plus `newer.event` under the transition law.
fn replays(io: &mut Io, capacity: u32, older: &Record, newer: &Record) -> bool {
    io.counters.replays += 1;
    match apply(&older.prior(), capacity, &newer.event) {
        Ok(a) => a.generation == newer.generation && a.receipt == newer.receipt && a.state == newer.state,
        Err(_) => false,
    }
}

/// A record without its predecessor: a genesis is recomputed from its event; any other is unverified.
fn verify_alone(io: &mut Io, capacity: u32, deployment: &Digest, r: &Record) -> Result<bool, Disposition> {
    if r.generation != 1 {
        return Ok(false);
    }
    io.counters.replays += 1;
    match genesis(deployment, capacity, &r.event) {
        Ok((lineage, state, receipt)) if lineage == r.lineage && state == r.state && receipt == r.receipt => Ok(true),
        _ => Err(Disposition::ReplayMismatch),
    }
}

fn newest_consistent(
    io: &mut Io,
    capacity: u32,
    deployment: &Digest,
    valid: &[(usize, &Record)],
) -> Result<(usize, bool), Disposition> {
    match valid {
        [] => Err(Disposition::Unrecoverable),
        [(i, r)] => Ok((*i, verify_alone(io, capacity, deployment, r)?)),
        [(i, a), (j, b)] => {
            let ((ni, n), o) = if a.generation >= b.generation { ((*i, *a), *b) } else { ((*j, *b), *a) };
            if n.generation == o.generation {
                if n.digest != o.digest {
                    return Err(Disposition::Conflict);
                }
                return Ok((ni, verify_alone(io, capacity, deployment, n)?));
            }
            if !n.follows(o) {
                return Err(Disposition::Conflict);
            }
            if !replays(io, capacity, o, n) {
                return Err(Disposition::ReplayMismatch);
            }
            Ok((ni, true))
        }
        _ => unreachable!("two slots"),
    }
}

fn recover(
    io: &mut Io,
    capacity: u32,
    deployment: &Digest,
    slots: &[(SlotClass, Option<Record>); 2],
    observed: &[Digest; 2],
    witnesses: &[(SlotClass, Option<Witness>); 2],
) -> Result<Recovered, Disposition> {
    use Disposition as D;
    if slots.iter().any(|(c, _)| *c == SlotClass::Invalid) {
        return Err(D::ReplayMismatch);
    }
    let valid: Vec<(usize, &Record)> = (0..2).filter_map(|i| slots[i].1.as_ref().map(|r| (i, r))).collect();
    if valid.len() == 2 && valid[0].1.lineage != valid[1].1.lineage {
        return Err(D::Conflict);
    }
    let pins: Vec<(usize, Witness)> = (0..2).filter_map(|i| witnesses[i].1.map(|w| (i, w))).collect();
    let pin = match pins.as_slice() {
        [] => None,
        [one] => Some(*one),
        [a, b] => {
            if a.1.lineage != b.1.lineage || (a.1.generation == b.1.generation && a.1 != b.1) {
                return Err(D::WitnessInconsistent);
            }
            Some(if a.1.generation >= b.1.generation { *a } else { *b })
        }
        _ => unreachable!("two witness slots"),
    };
    let Some((wi, w)) = pin else {
        let (slot, replay_verified) = newest_consistent(io, capacity, deployment, &valid)?;
        let classes = [witnesses[0].0 as u8, witnesses[1].0 as u8];
        return Ok(Recovered {
            slot,
            witness_slot: 1,
            mode: Mode::WitnessMissing { classes },
            disposition: D::WitnessMissing,
            replay_verified,
            complete_witness: false,
        });
    };
    if w.capacity != capacity {
        return Err(D::WitnessInconsistent);
    }
    if valid.iter().any(|(_, r)| r.lineage != w.lineage) {
        return Err(D::Conflict);
    }
    let normal = |slot, disposition, replay_verified, complete_witness, mode| Recovered {
        slot,
        witness_slot: wi,
        mode,
        disposition,
        replay_verified,
        complete_witness,
    };

    // The acknowledged record is present.
    if let Some(&(si, s)) = valid.iter().find(|(_, r)| r.generation == w.generation) {
        if s.digest != w.record_digest || s.state_root != w.state_root {
            return Err(D::Conflict);
        }
        let oi = 1 - si;
        return Ok(match &slots[oi] {
            (SlotClass::Valid, Some(o)) => {
                if o.generation == s.generation {
                    if o.digest != s.digest {
                        return Err(D::Conflict);
                    }
                    let verified = verify_alone(io, capacity, deployment, s)?;
                    normal(si, D::PredecessorUnavailable, verified, false, Mode::Normal)
                } else if o.follows(s) {
                    if !replays(io, capacity, s, o) {
                        return Err(D::ReplayMismatch);
                    }
                    normal(oi, D::InterruptedCompleted, true, true, Mode::Normal)
                } else if s.follows(o) {
                    if !replays(io, capacity, o, s) {
                        return Err(D::ReplayMismatch);
                    }
                    normal(si, D::Clean, true, false, Mode::Normal)
                } else if o.generation > s.generation + 1 {
                    return Err(D::WitnessInconsistent);
                } else {
                    // An authenticated record at the next generation (or older) that is not one history with the
                    // acknowledged record: a conflicting candidate.
                    return Err(D::Conflict);
                }
            }
            (SlotClass::Empty, _) => {
                let verified = verify_alone(io, capacity, deployment, s)?;
                let d = if s.generation == 1 { D::Clean } else { D::PredecessorUnavailable };
                normal(si, d, verified, false, Mode::Normal)
            }
            (SlotClass::Torn, _) => {
                let verified = verify_alone(io, capacity, deployment, s)?;
                normal(si, D::TrailingIncomplete, verified, false, Mode::Normal)
            }
            (SlotClass::Unauthenticated, _) => {
                let verified = verify_alone(io, capacity, deployment, s)?;
                let mode = Mode::Restricted { slot: oi as u8, observed: observed[oi] };
                normal(si, D::RestrictedUnauthenticated, verified, false, mode)
            }
            _ => return Err(D::ReplayMismatch),
        });
    }

    // The acknowledged record is absent. A durable, unwitnessed successor bound to the pin completes it: the
    // ordinary next transition, or an operator rollback acceptance that abandoned exactly the pinned record.
    if let Some(&(ni, n)) = valid.iter().find(|(_, r)| r.generation == w.generation + 1) {
        let follows_pin = n.predecessor_generation == w.generation
            && n.predecessor_digest == w.record_digest
            && n.predecessor_state_root == w.state_root;
        let abandons_pin = n.event.kind == EventKind::ReconcileRollback
            && u64_at(&n.event.payload, 0) == w.generation
            && digest_at(&n.event.payload, 16) == w.record_digest;
        if !(follows_pin || abandons_pin) {
            return Err(D::Conflict);
        }
        let verified = match &slots[1 - ni] {
            (SlotClass::Valid, Some(o)) if n.follows(o) => {
                if !replays(io, capacity, o, n) {
                    return Err(D::ReplayMismatch);
                }
                true
            }
            (SlotClass::Valid, Some(_)) => return Err(D::Conflict),
            _ => false,
        };
        return Ok(normal(ni, D::InterruptedCompleted, verified, true, Mode::Normal));
    }
    if valid.iter().any(|(_, r)| r.generation > w.generation) {
        return Err(D::WitnessInconsistent);
    }
    if valid.is_empty() {
        return Err(D::Unrecoverable);
    }
    // Everything authenticated is older than the acknowledged record.
    let class = if slots.iter().any(|(c, _)| *c == SlotClass::Torn) {
        1
    } else if slots.iter().any(|(c, _)| *c == SlotClass::Unauthenticated) {
        2
    } else {
        0
    };
    let (slot, replay_verified) = newest_consistent(io, capacity, deployment, &valid)?;
    let disposition = [D::Rollback, D::CorruptAcknowledged, D::UnauthenticatedAcknowledged][class as usize];
    let mode = Mode::Rollback { abandoned_generation: w.generation, abandoned_digest: w.record_digest, class };
    Ok(normal(slot, disposition, replay_verified, false, mode))
}

/// Only the reconciliation that matches what recovery observed is accepted outside the normal mode.
fn gate(mode: &Mode, event: &Event) -> Result<(), Code> {
    let p = &event.payload;
    match (mode, event.kind) {
        (Mode::Normal, kind) if kind.reconciliation() => Err(Code::NotReconcilable),
        (Mode::Normal, _) => Ok(()),
        (Mode::Restricted { slot, observed }, EventKind::ReconcileUnauthenticated) => {
            if p[0] == *slot && digest_at(p, 8) == *observed {
                Ok(())
            } else {
                Err(Code::NotReconcilable)
            }
        }
        (Mode::Rollback { abandoned_generation, abandoned_digest, class }, EventKind::ReconcileRollback) => {
            if u64_at(p, 0) == *abandoned_generation && p[8] == *class && digest_at(p, 16) == *abandoned_digest {
                Ok(())
            } else {
                Err(Code::NotReconcilable)
            }
        }
        (Mode::WitnessMissing { classes }, EventKind::ReconcileRepin) => {
            if p[0] == classes[0] && p[1] == classes[1] {
                Ok(())
            } else {
                Err(Code::NotReconcilable)
            }
        }
        _ => Err(Code::ReconcileRequired),
    }
}

// -- the authority -------------------------------------------------------------------------------------------

impl Authority {
    pub fn new(state_dir: &Path, authority_dir: &Path) -> Result<Authority, Code> {
        check_paths(state_dir, authority_dir)?;
        Ok(Authority {
            state_dir: state_dir.to_path_buf(),
            authority_dir: authority_dir.to_path_buf(),
            opened: None,
            poisoned: None,
            io: Io::default(),
        })
    }

    /// Provision a new deployment: a fresh deployment key and lineage, the genesis record, the first witness.
    /// The publication point is `witness.a`; before it, provisioning may simply be repeated.
    pub fn provision(
        state_dir: &Path,
        authority_dir: &Path,
        capacity: u32,
        operator_label: &Digest,
        runtime_label: &Digest,
    ) -> Result<Provisioned, Code> {
        let mut master = random32()?;
        let nonce = random32()?;
        let result = provision_with(
            &mut Io::default(),
            state_dir,
            authority_dir,
            capacity,
            operator_label,
            runtime_label,
            &master,
            &nonce,
        );
        wipe(&mut master);
        result
    }

    // -- test seams (inert in the production library) -------------------------------------------------------

    /// Testing builds only: arm one fault for the `publication`-th transition from now (0: the next open's
    /// witness completion). Actions: 0 clear; 1 die at step `arg`; for the record slot 2 write fails, 3 torn write
    /// of `arg` bytes then refusal, 4 torn write of `arg` bytes then death, 5 sync fails with the bytes lost,
    /// 6 sync fails but durable; 7..=11 the same five for the witness.
    #[cfg(any(test, feature = "testing"))]
    pub fn testing_arm(&mut self, publication: u64, action: u32, arg: u64) -> Result<(), Code> {
        use probe::{Fault, WriteFault};
        let write = |a: u32| match a {
            0 => WriteFault::Fail,
            1 => WriteFault::TornFail(arg as usize),
            2 => WriteFault::TornDie(arg as usize),
            3 => WriteFault::SyncLost,
            _ => WriteFault::SyncDurable,
        };
        let fault = match action {
            0 => None,
            1 => Some(Fault::Die(Step::from_u32(arg as u32).ok_or(Code::Invalid)?)),
            2..=6 => Some(Fault::Write(Target::Slot, write(action - 2))),
            7..=11 => Some(Fault::Write(Target::Witness, write(action - 7))),
            _ => return Err(Code::Invalid),
        };
        self.io.probe.publications = 0;
        self.io.probe.plan = fault.map(|f| (publication, f));
        Ok(())
    }

    /// Testing builds only: the I/O counters, reset when `reset`.
    #[cfg(any(test, feature = "testing"))]
    pub fn testing_counters(&mut self, reset: bool) -> Counters {
        let counters = self.io.counters;
        if reset {
            self.io.counters = Counters::default();
        }
        counters
    }

    // -- open and recovery ----------------------------------------------------------------------------------

    pub fn open(&mut self) -> Result<Report, Refusal> {
        if self.opened.is_some() {
            return Err(Code::Open.into());
        }
        self.poisoned = None;
        let before = self.io.counters;
        let mut report = Report::default();
        let result = self.open_inner(&mut report);
        report.reads = self.io.counters.preads - before.preads;
        report.read_bytes = self.io.counters.pread_bytes - before.pread_bytes;
        report.replays = self.io.counters.replays - before.replays;
        match result {
            Ok(()) => {
                let opened = self.opened.as_mut().expect("opened");
                opened.report = report;
                Ok(report)
            }
            Err(code) => {
                let ran = report.disposition != 0;
                Err(Refusal { code, report: ran.then(|| Box::new(report)) })
            }
        }
    }

    fn open_inner(&mut self, report: &mut Report) -> Result<(), Code> {
        let (sd, ad) = (self.state_dir.clone(), self.authority_dir.clone());
        let euid = euid();
        let am = match fs::symlink_metadata(&ad) {
            Ok(m) => m,
            Err(e) if e.kind() == ErrorKind::NotFound => return Err(Code::NotProvisioned),
            Err(e) => return Err(io_code(e)),
        };
        if !protected_dir(&am, euid) {
            return Err(Code::Unprotected);
        }
        let sm = fs::symlink_metadata(&sd).map_err(|e| {
            if e.kind() == ErrorKind::NotFound {
                Code::Corrupt
            } else {
                io_code(e)
            }
        })?;
        if !sm.is_dir() || sm.file_type().is_symlink() {
            return Err(Code::Corrupt);
        }
        distinct(&sd, &ad)?;
        let locks = [lock(&ad)?, lock(&sd)?];
        let anames = names(&ad)?;
        let snames = names(&sd)?;
        if !anames.contains(WITNESS_FILES[0]) {
            return Err(if anames.is_empty() && snames.is_empty() {
                Code::NotProvisioned
            } else {
                Code::ProvisionIncomplete
            });
        }
        if anames != set_of(&[KEY_FILE, WITNESS_FILES[0], WITNESS_FILES[1]]) || snames != set_of(&STATE_FILES) {
            return Err(Code::Corrupt);
        }
        let mut master = read_key(&mut self.io, &ad.join(KEY_FILE), euid)?;
        let keys = Keys::derive(&master);
        wipe(&mut master);

        let mut wraw = [[0u8; WITNESS_BYTES]; 2];
        let mut witness_files = Vec::with_capacity(2);
        for (i, name) in WITNESS_FILES.iter().enumerate() {
            let file = open_checked(&mut self.io, &ad.join(name), euid, true)?;
            if file.metadata().map_err(io_code)?.len() != WITNESS_BYTES as u64 {
                return Err(Code::Corrupt);
            }
            self.io.read_at(&file, &mut wraw[i]).map_err(io_code)?;
            witness_files.push(file);
        }
        let mut slot_files = Vec::with_capacity(2);
        let mut lens = [0u64; 2];
        for (i, name) in STATE_FILES.iter().enumerate() {
            let file = open_checked(&mut self.io, &sd.join(name), euid, false)?;
            lens[i] = file.metadata().map_err(io_code)?.len();
            slot_files.push(file);
        }
        if lens[0] != lens[1] {
            return Err(Code::Corrupt);
        }
        let capacity = capacity_of(lens[0]).ok_or(Code::Corrupt)?;
        let mut raw = [vec![0u8; lens[0] as usize], vec![0u8; lens[0] as usize]];
        for i in 0..2 {
            self.io.read_at(&slot_files[i], &mut raw[i]).map_err(io_code)?;
        }

        let slots = [Record::classify(&raw[0], capacity, &keys), Record::classify(&raw[1], capacity, &keys)];
        let observed = [sha256::digest(&[&raw[0]]), sha256::digest(&[&raw[1]])];
        let witnesses = [Witness::classify(&wraw[0], &keys), Witness::classify(&wraw[1], &keys)];
        report.slot_class = [slots[0].0 as u8, slots[1].0 as u8];
        report.witness_class = [witnesses[0].0 as u8, witnesses[1].0 as u8];
        report.slot_generation = [Record::claimed_generation(&raw[0]), Record::claimed_generation(&raw[1])];
        report.observed = observed;
        report.witness_generation = witnesses.iter().filter_map(|(_, w)| w.map(|w| w.generation)).max().unwrap_or(0);

        let recovered = match recover(&mut self.io, capacity, &keys.deployment, &slots, &observed, &witnesses) {
            Ok(r) => r,
            Err(disposition) => {
                report.disposition = disposition as u32;
                return Err(Code::RecoveryRefused);
            }
        };
        let current = slots[recovered.slot].1.clone().expect("recovered a valid record");
        report.disposition = recovered.disposition as u32;
        report.mode = recovered.mode.code();
        report.replay_verified = recovered.replay_verified;
        report.current_generation = current.generation;
        report.evolution_pending = current.state.evolution.assertion().is_some();

        let mut witness_slot = recovered.witness_slot;
        if recovered.complete_witness {
            // The durable, unwitnessed transition is the authority: pin it before anything is built on it.
            let target = 1 - witness_slot;
            let bytes = Witness::of(&current).seal(&keys);
            match self.io.put(&witness_files[target], &bytes, Target::Witness, 0) {
                Ok(()) => {}
                Err(Code::TestingProcessDeath) => return Err(Code::TestingProcessDeath),
                Err(_) => return Err(Code::PublicationUncertain),
            }
            self.io.step(Step::OpenWitnessWritten, 0)?;
            if self.io.settle(&witness_files[target], Target::Witness, 0).is_err() {
                return Err(Code::PublicationUncertain);
            }
            self.io.step(Step::OpenWitnessSynced, 0)?;
            witness_slot = target;
            report.witness_completed = true;
            report.witness_generation = current.generation;
        }
        let b = slot_files.pop().expect("two slots");
        let a = slot_files.pop().expect("two slots");
        let wb = witness_files.pop().expect("two witnesses");
        let wa = witness_files.pop().expect("two witnesses");
        self.opened = Some(Opened {
            _locks: locks,
            slots: [a, b],
            witnesses: [wa, wb],
            keys,
            capacity,
            current,
            slot: recovered.slot,
            witness_slot,
            mode: recovered.mode,
            report: *report,
        });
        Ok(())
    }

    pub fn close(&mut self) {
        self.opened = None;
    }

    // -- reads ----------------------------------------------------------------------------------------------

    fn require(&self) -> Result<&Opened, Code> {
        if let Some(code) = self.poisoned {
            return Err(code);
        }
        self.opened.as_ref().ok_or(Code::Closed)
    }

    /// The current authority (in a reconcile-only mode: the candidate recovery found).
    pub fn current(&self) -> Result<&Record, Code> {
        Ok(&self.require()?.current)
    }

    pub fn mode(&self) -> Result<Mode, Code> {
        Ok(self.require()?.mode)
    }

    pub fn report(&self) -> Result<Report, Code> {
        Ok(self.require()?.report)
    }

    pub fn capacity(&self) -> Result<u32, Code> {
        Ok(self.require()?.capacity)
    }

    pub fn deployment(&self) -> Result<Digest, Code> {
        Ok(self.require()?.keys.deployment)
    }

    pub fn next_entity_id(&self, kind: EntityKind, label: &Digest) -> Result<Digest, Code> {
        let o = self.require()?;
        Ok(o.current.state.next_entity_id(&o.current.lineage, kind, label))
    }

    /// Admission of typed evidence (pure: nothing is recorded). Refused while the authority is only a candidate.
    pub fn check_evidence(&self, source: &Digest, version: u64, scope: &Digest) -> Result<u32, Code> {
        let o = self.require()?;
        if matches!(o.mode, Mode::Rollback { .. } | Mode::WitnessMissing { .. }) {
            return Err(Code::ReconcileRequired);
        }
        o.current.state.check_evidence(source, version, scope)
    }

    // -- the only mutation ----------------------------------------------------------------------------------

    /// Compare-and-publish one event: refused `Stale` unless the caller names the current generation and state
    /// root; validated by the transition law; durable and witnessed before it is acknowledged.
    pub fn transition(
        &mut self,
        expected_generation: u64,
        expected_state_root: &Digest,
        event: &Event,
    ) -> Result<Receipt, Code> {
        let (record, bytes, witness, founded) = {
            let o = self.require()?;
            gate(&o.mode, event)?;
            if expected_generation != o.current.generation || *expected_state_root != o.current.state_root {
                return Err(Code::Stale);
            }
            let applied = apply(&o.current.prior(), o.capacity, event)?;
            let founded = applied.founded;
            let c = &o.current;
            let (record, bytes) = Record::seal(
                &o.keys,
                o.capacity,
                applied.generation,
                c.generation,
                &c.lineage,
                &c.digest,
                &c.state_root,
                &applied.receipt,
                applied.state,
                *event,
            );
            let witness = Witness::of(&record).seal(&o.keys);
            (record, bytes, witness, founded)
        };
        let (slot, witness_slot) = self.publish(&bytes, &witness, event.kind == EventKind::ReconcileRepin)?;
        let o = self.opened.as_mut().expect("open");
        o.current = record;
        o.slot = slot;
        o.witness_slot = witness_slot;
        o.mode = Mode::Normal;
        let c = &o.current;
        Ok(Receipt {
            generation: c.generation,
            predecessor_generation: c.predecessor_generation,
            receipt: c.receipt,
            record_digest: c.digest,
            state_root: c.state_root,
            receipt_head: c.state.receipt_head,
            receipt_count: c.state.receipt_count,
            founded,
        })
    }

    fn die(&mut self) -> Code {
        self.opened = None; // the process is gone: descriptors and locks close
        self.poisoned = Some(Code::TestingProcessDeath);
        Code::TestingProcessDeath
    }

    fn poison(&mut self) -> Code {
        // Durable outcome unknown, or durable but unwitnessed: reopening resolves to exactly one authority.
        self.opened = None;
        self.poisoned = Some(Code::PublicationUncertain);
        Code::PublicationUncertain
    }

    fn checkpoint(&mut self, step: Step, n: u64) -> Result<(), Code> {
        match self.io.step(step, n) {
            Ok(()) => Ok(()),
            Err(_) => Err(self.die()),
        }
    }

    fn publish(&mut self, bytes: &[u8], witness: &[u8; WITNESS_BYTES], repin: bool) -> Result<(usize, usize), Code> {
        let n = self.io.next_publication();
        self.checkpoint(Step::TransitionBegin, n)?;
        let (target, wtarget) = {
            let o = self.opened.as_ref().expect("open");
            (1 - o.slot, 1 - o.witness_slot)
        };
        // 1. The complete next record, into the slot that does not hold the current authority (left untouched).
        let o = self.opened.as_ref().expect("open");
        match self.io.put(&o.slots[target], bytes, Target::Slot, n) {
            Ok(()) => {}
            Err(Code::TestingProcessDeath) => return Err(self.die()),
            Err(code) => return Err(code),
        }
        self.checkpoint(Step::SlotWritten, n)?;
        let o = self.opened.as_ref().expect("open");
        if self.io.settle(&o.slots[target], Target::Slot, n).is_err() {
            return Err(self.poison());
        }
        self.checkpoint(Step::SlotSynced, n)?;
        // 2. Durable from here. A failure leaves it unwitnessed, which reopening verifies and completes.
        let mut written = vec![wtarget];
        if repin {
            written.push(1 - wtarget);
        }
        for (k, w) in written.iter().enumerate() {
            let o = self.opened.as_ref().expect("open");
            match self.io.put(&o.witnesses[*w], witness, Target::Witness, n) {
                Ok(()) => {}
                Err(Code::TestingProcessDeath) => return Err(self.die()),
                Err(_) => return Err(self.poison()),
            }
            if k == 0 {
                self.checkpoint(Step::WitnessWritten, n)?;
            }
            let o = self.opened.as_ref().expect("open");
            if self.io.settle(&o.witnesses[*w], Target::Witness, n).is_err() {
                return Err(self.poison());
            }
            if k == 0 {
                self.checkpoint(Step::WitnessSynced, n)?;
            }
        }
        Ok((target, *written.last().expect("one witness written")))
    }
}

/// Testing builds only: the authority recovery would resolve to, read without the directory locks and without
/// writing anything (a witness completion is not performed). For tests that inspect what a restart would find
/// while a live owner holds the directories.
#[cfg(any(test, feature = "testing"))]
pub fn testing_peek(state_dir: &Path, authority_dir: &Path) -> Result<Record, Code> {
    let mut io = Io::default();
    let euid = euid();
    let mut master = read_key(&mut io, &authority_dir.join(KEY_FILE), euid)?;
    let keys = Keys::derive(&master);
    wipe(&mut master);
    let read = |path: PathBuf| fs::read(path).map_err(io_code);
    let raw = [read(state_dir.join(STATE_FILES[0]))?, read(state_dir.join(STATE_FILES[1]))?];
    let wraw = [read(authority_dir.join(WITNESS_FILES[0]))?, read(authority_dir.join(WITNESS_FILES[1]))?];
    if raw[0].len() != raw[1].len() {
        return Err(Code::Corrupt);
    }
    let capacity = capacity_of(raw[0].len() as u64).ok_or(Code::Corrupt)?;
    let slots = [Record::classify(&raw[0], capacity, &keys), Record::classify(&raw[1], capacity, &keys)];
    let observed = [sha256::digest(&[&raw[0]]), sha256::digest(&[&raw[1]])];
    let witnesses = [Witness::classify(&wraw[0], &keys), Witness::classify(&wraw[1], &keys)];
    let recovered = recover(&mut io, capacity, &keys.deployment, &slots, &observed, &witnesses)
        .map_err(|_| Code::RecoveryRefused)?;
    Ok(slots[recovered.slot].1.clone().expect("recovered a valid record"))
}

/// Provisioning with explicit key material (tests use fixed material for reproducible vectors).
#[allow(clippy::too_many_arguments)]
pub(crate) fn provision_with(
    io: &mut Io,
    state_dir: &Path,
    authority_dir: &Path,
    capacity: u32,
    operator_label: &Digest,
    runtime_label: &Digest,
    master: &[u8; KEY_BYTES],
    nonce: &Digest,
) -> Result<Provisioned, Code> {
    if !(MIN_CAPACITY..=MAX_CAPACITY).contains(&capacity) || zero(master) || *nonce == ZERO {
        return Err(Code::Invalid);
    }
    check_paths(state_dir, authority_dir)?;
    let euid = euid();
    prepare_dir(io, authority_dir, true, euid)?;
    prepare_dir(io, state_dir, false, euid)?;
    distinct(state_dir, authority_dir)?;
    let _locks = [lock(authority_dir)?, lock(state_dir)?];
    let anames = names(authority_dir)?;
    let snames = names(state_dir)?;
    if anames.contains(WITNESS_FILES[0]) {
        return Err(Code::AlreadyProvisioned);
    }
    // Only the debris of a provisioning that never reached its publication point may be present.
    let with_tmp = |list: &[&str]| -> BTreeSet<String> {
        list.iter().flat_map(|n| [n.to_string(), format!("{n}{TMP}")]).collect()
    };
    let adebris = with_tmp(&[KEY_FILE, WITNESS_FILES[0], WITNESS_FILES[1]]);
    let sdebris = with_tmp(&STATE_FILES);
    if !anames.is_subset(&adebris) || !snames.is_subset(&sdebris) {
        return Err(Code::Corrupt);
    }
    for (dir, list) in [(authority_dir, &anames), (state_dir, &snames)] {
        for name in list.iter() {
            io.unlink(&dir.join(name)).map_err(io_code)?;
        }
        if !list.is_empty() {
            io.sync_dir(dir).map_err(io_code)?;
        }
    }
    let keys = Keys::derive(master);
    let event = Event::genesis(nonce, operator_label, runtime_label);
    let (lineage, state, receipt) = genesis(&keys.deployment, capacity, &event)?;
    let (record, bytes) = Record::seal(&keys, capacity, 1, 0, &lineage, &ZERO, &ZERO, &receipt, state, event);
    let witness = Witness::of(&record).seal(&keys);
    let tmp = |dir: &Path, name: &str| dir.join(format!("{name}{TMP}"));
    let (sd, ad) = (state_dir, authority_dir);
    io.create(&tmp(ad, KEY_FILE), master, 0o400).map_err(io_code)?;
    io.create(&tmp(sd, STATE_FILES[1]), &vec![0u8; bytes.len()], 0o600).map_err(io_code)?;
    io.create(&tmp(sd, STATE_FILES[0]), &bytes, 0o600).map_err(io_code)?;
    io.create(&tmp(ad, WITNESS_FILES[1]), &[0u8; WITNESS_BYTES], 0o600).map_err(io_code)?;
    io.create(&tmp(ad, WITNESS_FILES[0]), &witness, 0o600).map_err(io_code)?;
    io.step(Step::ProvisionFilesWritten, 0)?;
    for name in [STATE_FILES[1], STATE_FILES[0]] {
        io.rename(&tmp(sd, name), &sd.join(name)).map_err(io_code)?;
    }
    io.step(Step::ProvisionStateRenamed, 0)?;
    io.sync_dir(sd).map_err(io_code)?;
    io.step(Step::ProvisionStateSynced, 0)?;
    io.rename(&tmp(ad, KEY_FILE), &ad.join(KEY_FILE)).map_err(io_code)?;
    io.step(Step::ProvisionKeyRenamed, 0)?;
    io.rename(&tmp(ad, WITNESS_FILES[1]), &ad.join(WITNESS_FILES[1])).map_err(io_code)?;
    io.step(Step::ProvisionWitnessBRenamed, 0)?;
    io.sync_dir(ad).map_err(io_code)?;
    io.step(Step::ProvisionAuthoritySynced, 0)?;
    // The publication point: the existence of witness.a.
    io.rename(&tmp(ad, WITNESS_FILES[0]), &ad.join(WITNESS_FILES[0])).map_err(io_code)?;
    io.step(Step::ProvisionPublished, 0)?;
    io.sync_dir(ad).map_err(io_code)?;
    io.step(Step::ProvisionDone, 0)?;
    Ok(Provisioned {
        capacity,
        record_bytes: bytes.len() as u64,
        lineage,
        deployment: keys.deployment,
        record_digest: record.digest,
        state_root: record.state_root,
    })
}
