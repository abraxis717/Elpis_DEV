//! Elpis continuity: the minimal durable runtime authority (docs/CONTINUITY.md).
//!
//! Continuity holds the current durable lineage/authority the runtime needs for safe restart: the expected
//! K1 retained-state identity of the ECS lineage, and the evolution authority (idle at a head, or reserved
//! for one exact assertion). It is one fixed-size record in a two-slot crash-safe register v2. It is not an
//! ECS, not a history, not an event log, not a message bus and not an audit ledger.
//!
//! The production surface is the C ABI in [`ffi`] (include/elpis/continuity.h). The `testing` feature adds
//! fault injection and I/O counters and is built only into `libelpis_continuity_testing.so`.

pub mod error;
pub mod ffi;
pub mod record;
mod sha256;
pub mod store;

pub use error::Code;
pub use record::{Cognition, Digest, EvolutionState, Snapshot, MAX_COUNTER, RECORD_SIZE};
pub use store::{Store, SLOT_NAMES};

#[cfg(test)]
mod tests;
