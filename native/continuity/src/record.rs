//! The continuity register record (docs/CONTINUITY.md): one fixed 176-byte current authority.
//!
//! Two record versions share one layout. **v2** (`ELPCONT\x02`) is unchanged: its anchored digest is a K1 retained-state
//! identity (`ELPISGK1`). **v3** (`ELPCONT\x03`, register domain v3) is written only for a lineage anchored at an ECS
//! composite retained-state identity (`ELPISEC1`, docs/ECS_COMPOSITE_STATE.md): it is the v2 layout with byte 25 = 1
//! (the state format) and requires the anchored flag. Every K1 or unanchored authority is still encoded as v2, byte
//! for byte, so v2 parity holds; a v2-only reader refuses a v3 record (fail closed) instead of misreading it.
//!
//! Layout (big-endian integers):
//!
//! | offset | size | field                                                   |
//! |--------|------|---------------------------------------------------------|
//! | 0      | 8    | magic `ELPCONT\x02`                                     |
//! | 8      | 2    | format version 2                                        |
//! | 10     | 6    | reserved, zero                                          |
//! | 16     | 8    | generation, 1..=2^63-1                                  |
//! | 24     | 1    | cognition: 0 unanchored, 1 anchored                     |
//! | 25     | 1    | v2: reserved, zero; v3: state format 1 (ELPISEC1)       |
//! | 26     | 6    | reserved, zero                                          |
//! | 32     | 32   | retained-state digest (zero when unanchored)            |
//! | 64     | 8    | evolution revision, 0..=2^63-1                          |
//! | 72     | 32   | evolution head (zero exactly at revision 0)             |
//! | 104    | 1    | evolution: 0 idle, 1 pending                            |
//! | 105    | 7    | reserved, zero                                          |
//! | 112    | 32   | pending assertion identity (zero when idle)             |
//! | 144    | 32   | SHA-256(`elpis.continuity.register.v{2,3}\0` ‖ bytes 0..144) |

use crate::error::Code;
use crate::sha256;

pub type Digest = [u8; 32];

pub const RECORD_SIZE: usize = 176;
pub const BODY_SIZE: usize = 144;
pub const MAGIC: &[u8; 8] = b"ELPCONT\x02";
pub const FORMAT_VERSION: u16 = 2;
/// The v3 record: an authority anchored at an ECS composite retained-state identity.
pub const MAGIC_V3: &[u8; 8] = b"ELPCONT\x03";
pub const FORMAT_VERSION_V3: u16 = 3;
/// The largest generation or revision a record may carry.
pub const MAX_COUNTER: u64 = (1 << 63) - 1;
pub const ZERO: Digest = [0; 32];

const REGISTER_DOMAIN: &[u8] = b"elpis.continuity.register.v2\x00";
const REGISTER_DOMAIN_V3: &[u8] = b"elpis.continuity.register.v3\x00";
const EVOLUTION_DOMAIN: &[u8] = b"elpis.continuity.evolution-authority.v2\x00";

/// The evolution authority: idle at a head, or reserved for exactly one assertion.
///
/// An idle authority has no assertion field at all, so "idle with a pending digest" cannot be represented.
/// Construct through [`EvolutionState::idle`] / [`EvolutionState::pending`], which enforce the counter bound
/// and the genesis rule (revision 0 if and only if the head is zero).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvolutionState {
    Idle { revision: u64, head: Digest },
    Pending { revision: u64, head: Digest, assertion: Digest },
}

impl EvolutionState {
    pub const GENESIS: EvolutionState = EvolutionState::Idle { revision: 0, head: ZERO };

    fn check(revision: u64, head: &Digest) -> Result<(), Code> {
        if revision > MAX_COUNTER || (revision == 0) != (*head == ZERO) {
            return Err(Code::Corrupt);
        }
        Ok(())
    }

    pub fn idle(revision: u64, head: Digest) -> Result<Self, Code> {
        Self::check(revision, &head)?;
        Ok(EvolutionState::Idle { revision, head })
    }

    pub fn pending(revision: u64, head: Digest, assertion: Digest) -> Result<Self, Code> {
        Self::check(revision, &head)?;
        Ok(EvolutionState::Pending { revision, head, assertion })
    }

    pub fn revision(&self) -> u64 {
        match *self {
            EvolutionState::Idle { revision, .. } | EvolutionState::Pending { revision, .. } => revision,
        }
    }

    pub fn head(&self) -> Digest {
        match *self {
            EvolutionState::Idle { head, .. } | EvolutionState::Pending { head, .. } => head,
        }
    }

    pub fn assertion(&self) -> Option<Digest> {
        match *self {
            EvolutionState::Idle { .. } => None,
            EvolutionState::Pending { assertion, .. } => Some(assertion),
        }
    }

    /// `elpis.continuity.evolution-authority.v2`: the identity an evolution assertion binds.
    pub fn digest(&self) -> Digest {
        let pending = self.assertion();
        sha256::digest(&[
            EVOLUTION_DOMAIN,
            &self.revision().to_be_bytes(),
            &self.head(),
            &[pending.is_some() as u8],
            &pending.unwrap_or(ZERO),
        ])
    }
}

/// The K1 lineage: no lineage yet, or the expected retained-state digest of the committed state (of the format
/// [`Snapshot::format`] names).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cognition {
    Unanchored,
    Anchored(Digest),
}

/// The retained-state format an anchored digest identifies. A lineage keeps its format: a commit never changes
/// it, and the only transition between formats is the explicit legacy import (`Store::import_cognition`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateFormat {
    /// `ELPISGK1` v1, the K1 retained state (every v2 record; the format of an unanchored authority).
    K1 = 0,
    /// `ELPISEC1` v1, the ECS composite retained state (K1 and the ECS memory; v3 records only).
    Composite = 1,
}

/// One complete, verified continuity authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    generation: u64,
    cognition: Cognition,
    format: StateFormat,
    evolution: EvolutionState,
}

impl Snapshot {
    pub const GENESIS: Snapshot = Snapshot {
        generation: 1,
        cognition: Cognition::Unanchored,
        format: StateFormat::K1,
        evolution: EvolutionState::GENESIS,
    };

    /// A K1-format authority (every v2 record).
    pub fn new(generation: u64, cognition: Cognition, evolution: EvolutionState) -> Result<Self, Code> {
        Self::with_format(generation, cognition, StateFormat::K1, evolution)
    }

    /// An authority of an explicit format. A composite format names an anchored identity: an unanchored
    /// composite authority does not exist.
    pub fn with_format(
        generation: u64,
        cognition: Cognition,
        format: StateFormat,
        evolution: EvolutionState,
    ) -> Result<Self, Code> {
        if generation == 0 || generation > MAX_COUNTER {
            return Err(Code::Corrupt);
        }
        if format == StateFormat::Composite && cognition == Cognition::Unanchored {
            return Err(Code::Corrupt);
        }
        Ok(Snapshot { generation, cognition, format, evolution })
    }

    pub fn format(&self) -> StateFormat {
        self.format
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn cognition(&self) -> Cognition {
        self.cognition
    }

    pub fn evolution(&self) -> EvolutionState {
        self.evolution
    }

    fn body(&self) -> [u8; BODY_SIZE] {
        let mut b = [0u8; BODY_SIZE];
        if self.format == StateFormat::Composite {
            b[0..8].copy_from_slice(MAGIC_V3);
            b[8..10].copy_from_slice(&FORMAT_VERSION_V3.to_be_bytes());
            b[25] = StateFormat::Composite as u8;
        } else {
            b[0..8].copy_from_slice(MAGIC);
            b[8..10].copy_from_slice(&FORMAT_VERSION.to_be_bytes());
        }
        b[16..24].copy_from_slice(&self.generation.to_be_bytes());
        if let Cognition::Anchored(k1) = self.cognition {
            b[24] = 1;
            b[32..64].copy_from_slice(&k1);
        }
        b[64..72].copy_from_slice(&self.evolution.revision().to_be_bytes());
        b[72..104].copy_from_slice(&self.evolution.head());
        if let Some(assertion) = self.evolution.assertion() {
            b[104] = 1;
            b[112..144].copy_from_slice(&assertion);
        }
        b
    }

    pub fn encode(&self) -> [u8; RECORD_SIZE] {
        let body = self.body();
        let mut out = [0u8; RECORD_SIZE];
        out[..BODY_SIZE].copy_from_slice(&body);
        out[BODY_SIZE..].copy_from_slice(&checksum(&body, self.format));
        out
    }

    /// The record checksum: the content identity of this whole authority (integrity, not authentication).
    pub fn digest(&self) -> Digest {
        checksum(&self.body(), self.format)
    }

    /// A verified snapshot, `None` for an empty (all-zero) slot, `Corrupt` for anything else.
    pub fn decode(raw: &[u8]) -> Result<Option<Snapshot>, Code> {
        if raw.len() != RECORD_SIZE {
            return Err(Code::Corrupt);
        }
        if raw.iter().all(|&b| b == 0) {
            return Ok(None);
        }
        let (body, sum) = raw.split_at(BODY_SIZE);
        let format = if &body[0..8] == MAGIC && body[8..10] == FORMAT_VERSION.to_be_bytes() && body[25] == 0 {
            StateFormat::K1
        } else if &body[0..8] == MAGIC_V3
            && body[8..10] == FORMAT_VERSION_V3.to_be_bytes()
            && body[25] == StateFormat::Composite as u8
        {
            StateFormat::Composite
        } else {
            return Err(Code::Corrupt);
        };
        if checksum(body, format) != sum {
            return Err(Code::Corrupt);
        }
        let zero = |r: std::ops::Range<usize>| body[r].iter().all(|&b| b == 0);
        if !zero(10..16) || !zero(26..32) {
            return Err(Code::Corrupt);
        }
        let k1 = digest_at(body, 32);
        let cognition = match body[24] {
            0 if k1 == ZERO => Cognition::Unanchored,
            1 => Cognition::Anchored(k1),
            _ => return Err(Code::Corrupt),
        };
        let revision = u64_at(body, 64);
        let head = digest_at(body, 72);
        let assertion = digest_at(body, 112);
        if !zero(105..112) {
            return Err(Code::Corrupt);
        }
        let evolution = match body[104] {
            0 if assertion == ZERO => EvolutionState::idle(revision, head)?,
            1 => EvolutionState::pending(revision, head, assertion)?,
            _ => return Err(Code::Corrupt),
        };
        Snapshot::with_format(u64_at(body, 16), cognition, format, evolution).map(Some)
    }
}

fn checksum(body: &[u8], format: StateFormat) -> Digest {
    let domain = match format {
        StateFormat::K1 => REGISTER_DOMAIN,
        StateFormat::Composite => REGISTER_DOMAIN_V3,
    };
    sha256::digest(&[domain, body])
}

fn digest_at(b: &[u8], at: usize) -> Digest {
    let mut d = ZERO;
    d.copy_from_slice(&b[at..at + 32]);
    d
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[at..at + 8]);
    u64::from_be_bytes(v)
}
