//! The authority record and the rollback witness: fixed-size, sealed by an unkeyed checksum and a keyed MAC.
//!
//! The checksum (SHA-256 under a domain) is the record's **integrity identity**: anyone can recompute it, so it
//! detects accidents (a torn or rotted write) and names the record, but it authenticates nothing. The MAC
//! (HMAC-SHA256 under a key derived from the deployment key, which lives in the protected authority directory)
//! is the **authentication**: a record whose checksum is right and whose MAC is wrong was produced without the
//! key. Recovery uses the difference: TORN (accident) versus UNAUTHENTICATED (substitution). The checksum is the
//! last field and covers the MAC, so a write torn anywhere (including inside the MAC) reads as TORN, never as a
//! substitution; only bytes deliberately checksummed without the key read as UNAUTHENTICATED.
//!
//! Record (little-endian; `R = 1344 + 160 * capacity` bytes):
//!
//! | offset | size | field |
//! |---|---|---|
//! | 0 | 8 | magic `ELPECA01` |
//! | 8 | 4 | format version 1 |
//! | 12 | 4 | flags, zero |
//! | 16 | 4 | entity capacity |
//! | 20 | 4 | event bytes (512) |
//! | 24 | 8 | record bytes `R` |
//! | 32 | 8 | generation (the authority's logical clock), >= 1 |
//! | 40 | 8 | predecessor generation (0 at genesis) |
//! | 48 | 32 | lineage |
//! | 80 | 32 | deployment id |
//! | 112 | 32 | predecessor record digest (zero at genesis) |
//! | 144 | 32 | predecessor state root (zero at genesis) |
//! | 176 | 32 | state root |
//! | 208 | 32 | receipt of the transition that produced this record |
//! | 240 | 16 | reserved, zero |
//! | 256 | 512 + 160 * capacity | the materialized state ([`crate::model::State`]) |
//! | R - 576 | 512 | the event that produced this record ([`crate::model::Event`]) |
//! | R - 64 | 32 | MAC: HMAC-SHA256(record key, `elpis.ecs_c.authority.record-mac.v1\0` ‖ bytes 0..R-64) |
//! | R - 32 | 32 | checksum: SHA-256(`elpis.ecs_c.authority.record.v1\0` ‖ bytes 0..R-32) — the record digest |
//!
//! Witness (240 bytes):
//!
//! | offset | size | field |
//! |---|---|---|
//! | 0 | 8 | magic `ELPECW01` |
//! | 8 | 4 | format version 1 |
//! | 12 | 4 | flags, zero |
//! | 16 | 4 | entity capacity |
//! | 20 | 4 | reserved, zero |
//! | 24 | 8 | record bytes `R` |
//! | 32 | 8 | pinned generation |
//! | 40 | 8 | reserved, zero |
//! | 48 | 32 | lineage |
//! | 80 | 32 | deployment id |
//! | 112 | 32 | pinned record digest |
//! | 144 | 32 | pinned state root |
//! | 176 | 32 | MAC: HMAC-SHA256(witness key, `elpis.ecs_c.authority.witness-mac.v1\0` ‖ bytes 0..176) |
//! | 208 | 32 | checksum: SHA-256(`elpis.ecs_c.authority.witness.v1\0` ‖ bytes 0..208) |

use crate::error::Code;
use crate::hmac::{equal, hmac, wipe};
use crate::model::*;
use crate::register::MAX_COUNTER;
use crate::sha256;

pub const RECORD_MAGIC: &[u8; 8] = b"ELPECA01";
pub const WITNESS_MAGIC: &[u8; 8] = b"ELPECW01";
pub const FORMAT_VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 256;
pub const TRAILER_BYTES: usize = 64;
pub const WITNESS_BYTES: usize = 240;
pub const KEY_BYTES: usize = 32;

pub(crate) const RECORD_CHECKSUM: &[u8] = b"elpis.ecs_c.authority.record.v1\x00";
const RECORD_MAC: &[u8] = b"elpis.ecs_c.authority.record-mac.v1\x00";
pub(crate) const WITNESS_CHECKSUM: &[u8] = b"elpis.ecs_c.authority.witness.v1\x00";
const WITNESS_MAC: &[u8] = b"elpis.ecs_c.authority.witness-mac.v1\x00";
const KEY_RECORD: &[u8] = b"elpis.ecs_c.authority.key.record.v1";
const KEY_WITNESS: &[u8] = b"elpis.ecs_c.authority.key.witness.v1";
const KEY_DEPLOYMENT: &[u8] = b"elpis.ecs_c.authority.key.deployment.v1";
const DEPLOYMENT_ID: &[u8] = b"elpis.ecs_c.authority.deployment.v1\x00";

pub const fn record_bytes(capacity: u32) -> usize {
    HEADER_BYTES + state_bytes(capacity) + EVENT_BYTES + TRAILER_BYTES
}

/// The capacity a record file of `len` bytes holds, if `len` is a valid record size.
pub fn capacity_of(len: u64) -> Option<u32> {
    let fixed = record_bytes(0) as u64;
    if len < fixed || !(len - fixed).is_multiple_of(ENTITY_BYTES as u64) {
        return None;
    }
    let capacity = u32::try_from((len - fixed) / ENTITY_BYTES as u64).ok()?;
    (MIN_CAPACITY..=MAX_CAPACITY).contains(&capacity).then_some(capacity)
}

/// The keys of one deployment, derived from its 32-byte deployment key. Wiped on drop.
pub struct Keys {
    record: [u8; 32],
    witness: [u8; 32],
    pub deployment: Digest,
}

impl Keys {
    pub fn derive(master: &[u8; KEY_BYTES]) -> Keys {
        let mut label = hmac(master, &[KEY_DEPLOYMENT]);
        let deployment = sha256::digest(&[DEPLOYMENT_ID, &label]);
        wipe(&mut label);
        Keys { record: hmac(master, &[KEY_RECORD]), witness: hmac(master, &[KEY_WITNESS]), deployment }
    }
}

impl Drop for Keys {
    fn drop(&mut self) {
        wipe(&mut self.record);
        wipe(&mut self.witness);
    }
}

/// What recovery found in one slot (record or witness). The numeric values are C ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SlotClass {
    /// All zero: never written.
    Empty = 0,
    /// The checksum or the fixed structure fails: an incomplete or damaged write (an accident).
    Torn = 1,
    /// The checksum holds and the MAC does not: bytes produced without the deployment key.
    Unauthenticated = 2,
    /// Authenticated, yet structurally or semantically invalid (another lineage, a broken invariant).
    Invalid = 3,
    Valid = 4,
}

/// One complete, authenticated, validated authority record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub capacity: u32,
    pub generation: u64,
    pub predecessor_generation: u64,
    pub lineage: Digest,
    pub deployment: Digest,
    pub predecessor_digest: Digest,
    pub predecessor_state_root: Digest,
    pub state_root: Digest,
    pub receipt: Digest,
    pub state: State,
    pub event: Event,
    /// The checksum: the record's integrity identity.
    pub digest: Digest,
}

pub fn state_root(lineage: &Digest, generation: u64, capacity: u32, state: &[u8]) -> Digest {
    sha256::digest(&[domain::STATE_ROOT, lineage, &generation.to_le_bytes(), &capacity.to_le_bytes(), state])
}

impl Record {
    /// Seal a record: the state root, checksum and MAC are computed here. Returns the record and its bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn seal(
        keys: &Keys,
        capacity: u32,
        generation: u64,
        predecessor_generation: u64,
        lineage: &Digest,
        predecessor_digest: &Digest,
        predecessor_state_root: &Digest,
        receipt: &Digest,
        state: State,
        event: Event,
    ) -> (Record, Vec<u8>) {
        let total = record_bytes(capacity);
        let mut b = vec![0u8; total];
        let state_at = HEADER_BYTES;
        let event_at = HEADER_BYTES + state_bytes(capacity);
        state.encode(capacity, &mut b[state_at..event_at]);
        let root = state_root(lineage, generation, capacity, &b[state_at..event_at]);
        b[0..8].copy_from_slice(RECORD_MAGIC);
        put_u32(&mut b, 8, FORMAT_VERSION);
        put_u32(&mut b, 16, capacity);
        put_u32(&mut b, 20, EVENT_BYTES as u32);
        put_u64(&mut b, 24, total as u64);
        put_u64(&mut b, 32, generation);
        put_u64(&mut b, 40, predecessor_generation);
        put_digest(&mut b, 48, lineage);
        put_digest(&mut b, 80, &keys.deployment);
        put_digest(&mut b, 112, predecessor_digest);
        put_digest(&mut b, 144, predecessor_state_root);
        put_digest(&mut b, 176, &root);
        put_digest(&mut b, 208, receipt);
        b[event_at..event_at + EVENT_BYTES].copy_from_slice(&event.encode());
        let body = total - TRAILER_BYTES;
        let mac = hmac(&keys.record, &[RECORD_MAC, &b[..body]]);
        b[body..body + 32].copy_from_slice(&mac);
        let digest = sha256::digest(&[RECORD_CHECKSUM, &b[..body + 32]]);
        b[body + 32..].copy_from_slice(&digest);
        let record = Record {
            capacity,
            generation,
            predecessor_generation,
            lineage: *lineage,
            deployment: keys.deployment,
            predecessor_digest: *predecessor_digest,
            predecessor_state_root: *predecessor_state_root,
            state_root: root,
            receipt: *receipt,
            state,
            event,
            digest,
        };
        (record, b)
    }

    /// The generation a slot's header claims (0 when unreadable). Untrusted: for reports only.
    pub fn claimed_generation(raw: &[u8]) -> u64 {
        if raw.len() >= 40 && &raw[0..8] == RECORD_MAGIC {
            u64_at(raw, 32)
        } else {
            0
        }
    }

    /// Classify one slot's bytes for a deployment of `capacity` and `keys`.
    pub fn classify(raw: &[u8], capacity: u32, keys: &Keys) -> (SlotClass, Option<Record>) {
        let total = record_bytes(capacity);
        if raw.len() != total {
            return (SlotClass::Torn, None);
        }
        if zero(raw) {
            return (SlotClass::Empty, None);
        }
        let body = total - TRAILER_BYTES;
        let structure_ok = &raw[0..8] == RECORD_MAGIC
            && u32_at(raw, 8) == FORMAT_VERSION
            && u32_at(raw, 12) == 0
            && u32_at(raw, 16) == capacity
            && u32_at(raw, 20) == EVENT_BYTES as u32
            && u64_at(raw, 24) == total as u64;
        let digest = sha256::digest(&[RECORD_CHECKSUM, &raw[..body + 32]]);
        if !structure_ok || !equal(&digest, &raw[body + 32..]) {
            return (SlotClass::Torn, None);
        }
        let mac = hmac(&keys.record, &[RECORD_MAC, &raw[..body]]);
        if !equal(&mac, &raw[body..body + 32]) {
            return (SlotClass::Unauthenticated, None);
        }
        match Record::decode_authenticated(raw, capacity, keys, digest) {
            Ok(record) => (SlotClass::Valid, Some(record)),
            Err(_) => (SlotClass::Invalid, None),
        }
    }

    fn decode_authenticated(raw: &[u8], capacity: u32, keys: &Keys, digest: Digest) -> Result<Record, Code> {
        let state_at = HEADER_BYTES;
        let event_at = HEADER_BYTES + state_bytes(capacity);
        let (generation, predecessor_generation) = (u64_at(raw, 32), u64_at(raw, 40));
        let lineage = digest_at(raw, 48);
        let deployment = digest_at(raw, 80);
        let (predecessor_digest, predecessor_state_root) = (digest_at(raw, 112), digest_at(raw, 144));
        let (root, receipt_digest) = (digest_at(raw, 176), digest_at(raw, 208));
        if deployment != keys.deployment || !zero(&raw[240..HEADER_BYTES]) {
            return Err(Code::Corrupt);
        }
        if generation == 0 || generation > MAX_COUNTER || predecessor_generation >= generation {
            return Err(Code::Corrupt);
        }
        let genesis = predecessor_generation == 0;
        if genesis != (generation == 1)
            || genesis != (predecessor_digest == ZERO)
            || genesis != (predecessor_state_root == ZERO)
        {
            return Err(Code::Corrupt);
        }
        let state_raw = &raw[state_at..event_at];
        if root != state_root(&lineage, generation, capacity, state_raw) {
            return Err(Code::Corrupt);
        }
        let event_raw: &[u8; EVENT_BYTES] = raw[event_at..event_at + EVENT_BYTES].try_into().expect("event");
        let event = Event::decode(event_raw).map_err(|_| Code::Corrupt)?;
        if genesis != (event.kind == EventKind::Genesis) {
            return Err(Code::Corrupt);
        }
        let expected_receipt = receipt(
            &lineage,
            generation,
            predecessor_generation,
            &predecessor_digest,
            &predecessor_state_root,
            event_raw,
        );
        if expected_receipt != receipt_digest {
            return Err(Code::Corrupt);
        }
        let state = State::decode(state_raw, capacity, &lineage, generation)?;
        Ok(Record {
            capacity,
            generation,
            predecessor_generation,
            lineage,
            deployment,
            predecessor_digest,
            predecessor_state_root,
            state_root: root,
            receipt: receipt_digest,
            state,
            event,
            digest,
        })
    }

    /// `self` was produced from `older` (generation, digest and state root bindings).
    pub fn follows(&self, older: &Record) -> bool {
        self.predecessor_generation == older.generation
            && self.predecessor_digest == older.digest
            && self.predecessor_state_root == older.state_root
    }

    pub fn prior(&self) -> Prior<'_> {
        Prior {
            lineage: self.lineage,
            generation: self.generation,
            record_digest: self.digest,
            state_root: self.state_root,
            state: &self.state,
        }
    }
}

/// The rollback witness: the newest acknowledged generation and its digests, authenticated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Witness {
    pub capacity: u32,
    pub generation: u64,
    pub lineage: Digest,
    pub deployment: Digest,
    pub record_digest: Digest,
    pub state_root: Digest,
}

impl Witness {
    pub fn of(record: &Record) -> Witness {
        Witness {
            capacity: record.capacity,
            generation: record.generation,
            lineage: record.lineage,
            deployment: record.deployment,
            record_digest: record.digest,
            state_root: record.state_root,
        }
    }

    pub fn seal(&self, keys: &Keys) -> [u8; WITNESS_BYTES] {
        let mut b = [0u8; WITNESS_BYTES];
        b[0..8].copy_from_slice(WITNESS_MAGIC);
        put_u32(&mut b, 8, FORMAT_VERSION);
        put_u32(&mut b, 16, self.capacity);
        put_u64(&mut b, 24, record_bytes(self.capacity) as u64);
        put_u64(&mut b, 32, self.generation);
        put_digest(&mut b, 48, &self.lineage);
        put_digest(&mut b, 80, &self.deployment);
        put_digest(&mut b, 112, &self.record_digest);
        put_digest(&mut b, 144, &self.state_root);
        let mac = hmac(&keys.witness, &[WITNESS_MAC, &b[..176]]);
        b[176..208].copy_from_slice(&mac);
        let checksum = sha256::digest(&[WITNESS_CHECKSUM, &b[..208]]);
        b[208..].copy_from_slice(&checksum);
        b
    }

    pub fn classify(raw: &[u8], keys: &Keys) -> (SlotClass, Option<Witness>) {
        if raw.len() != WITNESS_BYTES {
            return (SlotClass::Torn, None);
        }
        if zero(raw) {
            return (SlotClass::Empty, None);
        }
        let structure_ok = &raw[0..8] == WITNESS_MAGIC && u32_at(raw, 8) == FORMAT_VERSION && u32_at(raw, 12) == 0;
        let checksum = sha256::digest(&[WITNESS_CHECKSUM, &raw[..208]]);
        if !structure_ok || !equal(&checksum, &raw[208..]) {
            return (SlotClass::Torn, None);
        }
        let mac = hmac(&keys.witness, &[WITNESS_MAC, &raw[..176]]);
        if !equal(&mac, &raw[176..208]) {
            return (SlotClass::Unauthenticated, None);
        }
        let capacity = u32_at(raw, 16);
        let w = Witness {
            capacity,
            generation: u64_at(raw, 32),
            lineage: digest_at(raw, 48),
            deployment: digest_at(raw, 80),
            record_digest: digest_at(raw, 112),
            state_root: digest_at(raw, 144),
        };
        let valid = (MIN_CAPACITY..=MAX_CAPACITY).contains(&capacity)
            && u32_at(raw, 20) == 0
            && u64_at(raw, 24) == record_bytes(capacity) as u64
            && u64_at(raw, 40) == 0
            && (1..=MAX_COUNTER).contains(&w.generation)
            && w.deployment == keys.deployment
            && w.record_digest != ZERO;
        if valid {
            (SlotClass::Valid, Some(w))
        } else {
            (SlotClass::Invalid, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes that carry a valid MAC and checksum under `keys`, after `edit` (a key holder's forgery).
    fn resealed(keys: &Keys, mut b: Vec<u8>, edit: impl Fn(&mut [u8])) -> Vec<u8> {
        edit(&mut b);
        let body = b.len() - TRAILER_BYTES;
        let mac = hmac(&keys.record, &[RECORD_MAC, &b[..body]]);
        b[body..body + 32].copy_from_slice(&mac);
        let sum = sha256::digest(&[RECORD_CHECKSUM, &b[..body + 32]]);
        b[body + 32..].copy_from_slice(&sum);
        b
    }

    #[test]
    fn authenticated_records_are_still_validated_completely() {
        let keys = Keys::derive(&[9; 32]);
        let event = Event::genesis(&[1; 32], &[2; 32], &[3; 32]);
        let (lineage, state, receipt) = genesis(&keys.deployment, 4, &event).unwrap();
        let (record, bytes) = Record::seal(&keys, 4, 1, 0, &lineage, &ZERO, &ZERO, &receipt, state, event);
        assert_eq!(Record::classify(&bytes, 4, &keys), (SlotClass::Valid, Some(record.clone())));
        assert_eq!(Record::classify(&resealed(&keys, bytes.clone(), |_| {}), 4, &keys).0, SlotClass::Valid);
        type Edit = (&'static str, fn(&mut [u8]));
        let edits: [Edit; 9] = [
            ("state root", |b| b[176] ^= 1),
            ("receipt", |b| b[208] ^= 1),
            ("deployment id", |b| b[80] ^= 1),
            ("reserved header", |b| b[250] = 1),
            ("generation zero", |b| b[32..40].fill(0)),
            ("a genesis with a predecessor", |b| b[112] = 1),
            ("a predecessor not older", |b| b[40] = 1),
            ("the event", |b| b[record_bytes(4) - TRAILER_BYTES - EVENT_BYTES + 100] ^= 1),
            ("an invalid state", |b| b[HEADER_BYTES + 96] = 7),
        ];
        for (name, edit) in edits {
            let forged = resealed(&keys, bytes.clone(), edit);
            assert_eq!(Record::classify(&forged, 4, &keys).0, SlotClass::Invalid, "{name}");
        }
        // The structural header is checked before authentication: a wrong size or magic is TORN.
        let mut torn = bytes.clone();
        torn[0] = b'X';
        assert_eq!(Record::classify(&torn, 4, &keys).0, SlotClass::Torn);
        assert_eq!(Record::classify(&bytes, 5, &keys).0, SlotClass::Torn);
        assert_eq!(Record::classify(&bytes[1..], 4, &keys).0, SlotClass::Torn);
        assert_eq!(Record::classify(&vec![0; bytes.len()], 4, &keys).0, SlotClass::Empty);
        // The record's identity covers the MAC: equal content under another key is another identity.
        let other = Keys::derive(&[8; 32]);
        let (o, _) = Record::seal(&other, 4, 1, 0, &lineage, &ZERO, &ZERO, &receipt, record.state.clone(), event);
        assert_ne!(o.digest, record.digest);
    }
}
