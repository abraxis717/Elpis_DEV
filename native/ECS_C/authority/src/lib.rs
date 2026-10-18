//! Elpis ECS_C authority: the bounded, authenticated continuity and entity authority (docs/ECS_C_AUTHORITY.md).
//!
//! ECS_C owns entities and their lifecycles, sender attribution and logical clocks, source versions and scopes,
//! the cognition lineage (which retained-state identity is authoritative), the evolution authority, receipts and
//! reconciliation. It holds the **complete materialized authority** in one fixed-size record, written alternately
//! into two preallocated slots, with a rollback witness in a separately protected directory. Every transition is
//! validated by one transition law before it is written, and recovery replays the newest transition against its
//! predecessor. Nothing grows: no log, no segments, no generations, no compaction (census, section 5a).
//!
//! ECS_C performs no geometric learning and never touches ECS_G state; it names retained-state identities only.
//! The production surface is the C ABI in `ffi` (include/elpis/ecs_c_authority.h, the default `c-abi` feature);
//! RuntimeCore embeds the crate without it. The `testing` feature adds fault injection and is built only into
//! `libelpis_ecs_c_authority_testing.so`.

pub mod error;
#[cfg(feature = "c-abi")]
pub mod ffi;
pub mod hmac;
pub mod legacy;
pub mod model;
pub mod record;
pub mod store;

pub use error::Code;
pub use model::{
    apply, entity_id, genesis, receipt_chain, Applied, Cognition, Digest, Entity, EntityKind, Event, EventKind, Legacy,
    Lifecycle, Prior, State, MAX_CAPACITY, MAX_PER_TURN, MIN_CAPACITY, ZERO,
};
pub use record::{capacity_of, record_bytes, Record, SlotClass, Witness, WITNESS_BYTES};
pub use store::{
    Authority, Counters, Disposition, Mode, Provisioned, Receipt, Refusal, Report, Step, KEY_FILE, STATE_FILES,
    WITNESS_FILES,
};

#[cfg(test)]
mod tests;
