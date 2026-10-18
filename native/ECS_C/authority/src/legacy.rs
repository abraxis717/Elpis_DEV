//! Explicit import of a standalone continuity register (v2 or v3): read, never reinterpreted in place.
//!
//! The current record is read under the continuity directory's own lock and resolved exactly as continuity
//! resolves it. It is then carried verbatim inside an IMPORT_LEGACY event, so the import replays at recovery like
//! every other transition. The continuity directory is never written.

use std::fs::{self, File, TryLockError};
use std::os::unix::fs::FileExt;
use std::path::Path;

use elpis_continuity::store::resolve;
use elpis_continuity::{RECORD_SIZE, SLOT_NAMES};

use crate::error::Code;

pub fn read_continuity(directory: &Path) -> Result<[u8; RECORD_SIZE], Code> {
    if !directory.is_absolute() {
        return Err(Code::Path);
    }
    let lock = File::open(directory).map_err(|_| Code::LegacyRefused)?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return Err(Code::Locked),
        Err(TryLockError::Error(_)) => return Err(Code::Io),
    }
    let mut records = [[0u8; RECORD_SIZE]; 2];
    for (i, name) in SLOT_NAMES.iter().enumerate() {
        let path = directory.join(name);
        let m = fs::symlink_metadata(&path).map_err(|_| Code::LegacyRefused)?;
        if !m.file_type().is_file() || m.len() != RECORD_SIZE as u64 {
            return Err(Code::LegacyRefused);
        }
        let file = File::open(&path).map_err(|_| Code::LegacyRefused)?;
        file.read_exact_at(&mut records[i], 0).map_err(|_| Code::LegacyRefused)?;
    }
    let (_, slot) = resolve(&records).map_err(|_| Code::LegacyRefused)?;
    Ok(records[slot])
}
