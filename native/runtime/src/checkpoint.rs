//! K1 Recovery R0: a bounded, crash-recoverable store of the complete K1 retained state (docs/K1_RECOVERY_R0.md).
//!
//! Continuity tells a restarting runtime *which* K1 retained-state identity is authoritative; it holds no state. FMS
//! COLD is scratch residency that reclaims orphans. Neither may hold the state that recovery needs, so this module
//! is a distinct owner with one job: keep a complete `(W, epoch, H, a)` envelope (`ELPISGK1`) of the authorized
//! state, and of at most one newer candidate, in exactly two fixed-capacity slot files.
//!
//! **Checkpoint bytes never authorize themselves.** A slot is resumable only when its verified retained-state
//! identity is the one continuity holds. A checksum-valid slot of any other identity is evidence, never authority.
//!
//! Physical bound: an operator provisions a directory with exactly `k1-checkpoint.a` and `k1-checkpoint.b`, each
//! `HEADER + capacity` bytes, where `capacity` is the envelope size of the admitted K1 shape. Slots are rewritten in
//! place; files never grow, shrink, rename or multiply; there is no history and nothing to reclaim. A state of
//! another shape is refused, never resized for.
//!
//! Slot layout (little-endian):
//!
//! ```text
//!   0   magic "ELPK1CK\x01"
//!   8   format version (u16) = 1, then 6 reserved zero bytes
//!   16  slot generation (u64, >= 1; the newest slot has the highest)
//!   24  envelope length (u64) = capacity
//!   32  retained-state identity (32): the envelope's own SHA-256 trailer
//!   64  reserved, zero (32)
//!   96  SHA-256 over "elpis.k1-checkpoint.slot.v1\0" || bytes 0..96 || envelope
//!   128 envelope (capacity bytes)
//! ```
//!
//! A slot is valid only if every field checks, the slot checksum matches, and the envelope's trailer is the
//! SHA-256 of the envelope that precedes it and equals the header identity. An all-zero header is an empty slot.
//! Anything else (a torn or corrupted write) is invalid and simply not a checkpoint.

use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use elpis_continuity::sha256::digest as sha256;
use elpis_continuity::Digest;

use crate::code::Rt;

pub const SLOT_NAMES: [&str; 2] = ["k1-checkpoint.a", "k1-checkpoint.b"];
pub const HEADER: usize = 128;
const MAGIC: [u8; 8] = *b"ELPK1CK\x01";
const VERSION: u16 = 1;
const DOMAIN: &[u8] = b"elpis.k1-checkpoint.slot.v1\0";
/// The smallest envelope any K1 shape has (header, one W value, F = 3 at dim 1, the trailer).
const MIN_ENVELOPE: usize = 64 + 8 * (1 + 6 + 3) + 32;
/// The largest envelope (the K1 image budget, including its header, plus the trailer).
const MAX_ENVELOPE: usize = 64 * 1024 * 1024 + 32;

/// A valid slot's header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub generation: u64,
    pub identity: Digest,
}

fn sync_data(file: &File) -> io::Result<()> {
    loop {
        match file.sync_data() {
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            other => return other,
        }
    }
}

fn io(_: io::Error) -> Rt {
    Rt::CheckpointIo
}

fn header(generation: u64, envelope: &[u8], identity: &Digest) -> [u8; HEADER] {
    let mut h = [0u8; HEADER];
    h[0..8].copy_from_slice(&MAGIC);
    h[8..10].copy_from_slice(&VERSION.to_le_bytes());
    h[16..24].copy_from_slice(&generation.to_le_bytes());
    h[24..32].copy_from_slice(&(envelope.len() as u64).to_le_bytes());
    h[32..64].copy_from_slice(identity);
    let sum = sha256(&[DOMAIN, &h[..96], envelope]);
    h[96..128].copy_from_slice(&sum);
    h
}

/// The verified identity of an envelope: its SHA-256 trailer, if it is the SHA-256 of everything before it.
pub fn envelope_identity(envelope: &[u8]) -> Option<Digest> {
    if envelope.len() < MIN_ENVELOPE || &envelope[..8] != b"ELPISGK1" {
        return None;
    }
    let (body, trailer) = envelope.split_at(envelope.len() - 32);
    let mut identity = [0u8; 32];
    identity.copy_from_slice(trailer);
    (sha256(&[body]) == identity).then_some(identity)
}

/// Parse and verify one slot (header and envelope). `None` for an empty or invalid slot.
fn verify(raw: &[u8], capacity: usize) -> Option<Slot> {
    let (h, envelope) = raw.split_at(HEADER);
    if h[0..8] != MAGIC
        || h[8..10] != VERSION.to_le_bytes()
        || h[10..16] != [0u8; 6]
        || h[64..96] != [0u8; 32]
        || u64::from_le_bytes(h[24..32].try_into().ok()?) != capacity as u64
    {
        return None;
    }
    let generation = u64::from_le_bytes(h[16..24].try_into().ok()?);
    let mut identity = [0u8; 32];
    identity.copy_from_slice(&h[32..64]);
    if generation == 0 || sha256(&[DOMAIN, &h[..96], envelope]) != h[96..128] {
        return None;
    }
    (envelope_identity(envelope) == Some(identity)).then_some(Slot { generation, identity })
}

/// The open checkpoint store: the two slot files, exclusively owned, and their verified headers.
pub struct Store {
    directory: PathBuf,
    _lock: File,
    files: [File; 2],
    capacity: usize,
    slots: [Option<Slot>; 2],
    /// One slot's bytes, preallocated: a LEARN's checkpoint allocates nothing.
    buffer: Vec<u8>,
}

impl Store {
    /// Operator provisioning: create `directory` (it must not exist) with exactly the two empty slot files of
    /// `HEADER + capacity` bytes each, fully written and synced. Refuses an existing directory: provisioning never
    /// overwrites, resizes or adopts anything.
    pub fn provision(directory: &Path, capacity: usize) -> Result<(), Rt> {
        if !directory.is_absolute() || !(MIN_ENVELOPE..=MAX_ENVELOPE).contains(&capacity) {
            return Err(Rt::CheckpointInvalid);
        }
        fs::create_dir(directory).map_err(|e| {
            if e.kind() == ErrorKind::AlreadyExists {
                Rt::CheckpointInvalid
            } else {
                Rt::CheckpointIo
            }
        })?;
        let zeros = vec![0u8; HEADER + capacity];
        for name in SLOT_NAMES {
            let file =
                OpenOptions::new().write(true).create_new(true).mode(0o600).open(directory.join(name)).map_err(io)?;
            file.write_all_at(&zeros, 0).map_err(io)?; // physically allocated now, never later
            sync_data(&file).map_err(io)?;
        }
        sync_data(&File::open(directory).map_err(io)?).map_err(io)?;
        if let Some(parent) = directory.parent() {
            sync_data(&File::open(parent).map_err(io)?).map_err(io)?;
        }
        Ok(())
    }

    /// Open a provisioned store: exactly the two slot files of one equal size, exclusively locked; both headers are
    /// read and verified (an invalid slot is merely empty).
    pub fn open(directory: &Path) -> Result<Store, Rt> {
        if !directory.is_absolute() {
            return Err(Rt::CheckpointInvalid);
        }
        let lock = File::open(directory).map_err(io)?;
        lock.try_lock().map_err(|_| Rt::CheckpointInvalid)?;
        let mut names = Vec::new();
        for entry in fs::read_dir(directory).map_err(io)? {
            names.push(entry.map_err(io)?.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        if names != SLOT_NAMES {
            return Err(Rt::CheckpointInvalid);
        }
        let open = |name: &str| OpenOptions::new().read(true).write(true).open(directory.join(name)).map_err(io);
        let files = [open(SLOT_NAMES[0])?, open(SLOT_NAMES[1])?];
        let sizes = [files[0].metadata().map_err(io)?.len(), files[1].metadata().map_err(io)?.len()];
        let capacity = usize::try_from(sizes[0]).map_err(|_| Rt::CheckpointInvalid)?.saturating_sub(HEADER);
        if sizes[0] != sizes[1] || !(MIN_ENVELOPE..=MAX_ENVELOPE).contains(&capacity) {
            return Err(Rt::CheckpointInvalid);
        }
        let mut store = Store {
            directory: directory.to_path_buf(),
            _lock: lock,
            files,
            capacity,
            slots: [None, None],
            buffer: vec![0u8; HEADER + capacity],
        };
        for i in 0..2 {
            store.slots[i] = store.read(i)?;
        }
        Ok(store)
    }

    fn read(&mut self, i: usize) -> Result<Option<Slot>, Rt> {
        self.files[i].read_exact_at(&mut self.buffer, 0).map_err(io)?;
        Ok(verify(&self.buffer, self.capacity))
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// The envelope size every slot holds (the admitted K1 shape's).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn slots(&self) -> [Option<Slot>; 2] {
        self.slots
    }

    /// The buffer one envelope is written into before [`Store::write`] (capacity bytes).
    pub fn envelope_buffer(&mut self) -> &mut [u8] {
        &mut self.buffer[HEADER..]
    }

    /// The slot holding `identity`, if a verified one does.
    pub fn holding(&self, identity: &Digest) -> Option<usize> {
        (0..2).find(|&i| matches!(self.slots[i], Some(s) if s.identity == *identity))
    }

    /// Write the envelope in the buffer as the newest slot, into the slot that does not hold `protect` (the
    /// authorized identity), and sync it. Returns the slot written. The envelope must verify (its own trailer) and
    /// its identity is returned through the slot. Nothing else is touched: the protected slot stays intact whatever
    /// happens to this write.
    pub fn write(&mut self, protect: Option<&Digest>) -> Result<(usize, Slot), Rt> {
        let identity = envelope_identity(&self.buffer[HEADER..]).ok_or(Rt::CheckpointInvalid)?;
        let target = match protect.and_then(|p| self.holding(p)) {
            Some(i) => 1 - i,
            // Nothing to protect: overwrite the older slot (an empty one first).
            None => match self.slots {
                [None, _] => 0,
                [_, None] => 1,
                [Some(a), Some(b)] => usize::from(a.generation > b.generation),
            },
        };
        let generation = self.slots.iter().flatten().map(|s| s.generation).max().unwrap_or(0) + 1;
        let h = header(generation, &self.buffer[HEADER..], &identity);
        self.buffer[..HEADER].copy_from_slice(&h);
        self.slots[target] = None; // whatever the write leaves behind is not trusted until it verifies
        self.files[target].write_all_at(&self.buffer, 0).map_err(io)?;
        sync_data(&self.files[target]).map_err(io)?;
        let slot = Slot { generation, identity };
        self.slots[target] = Some(slot);
        Ok((target, slot))
    }

    /// Withdraw a slot (an unauthorized candidate): zero its header and sync. The slot is empty from then on.
    pub fn retract(&mut self, i: usize) -> Result<(), Rt> {
        self.slots[i] = None;
        self.files[i].write_all_at(&[0u8; HEADER], 0).map_err(io)?;
        sync_data(&self.files[i]).map_err(io)
    }

    /// The verified envelope of slot `i` (re-read and re-verified from the file).
    pub fn envelope(&mut self, i: usize) -> Result<Option<(Slot, &[u8])>, Rt> {
        let slot = self.read(i)?;
        self.slots[i] = slot;
        Ok(slot.map(|s| (s, &self.buffer[HEADER..])))
    }
}

/// What restart may do with the K1 lineage (continuity decides; the slots are only evidence).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Disposition {
    /// Continuity is unanchored: there is no lineage, and a slot never creates one.
    NothingToRecover = 1,
    /// Continuity holds D and a verified complete slot holds D; no newer candidate exists. Resume exactly D.
    Resumable = 2,
    /// Continuity holds D0, but a newer verified slot holds another identity D1: a LEARN checkpointed D1 and the
    /// process ended before continuity published it (the commit may or may not have happened in memory). Not
    /// resumable until an operator discards the candidate (resume D0) or adopts it (continuity D0 -> D1).
    CandidateUnresolved = 3,
    /// Continuity holds D but no verified slot holds D: the authorized state cannot be resumed from here.
    CheckpointMissing = 4,
}
