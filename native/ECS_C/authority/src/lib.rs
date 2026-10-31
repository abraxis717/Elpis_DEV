//! Elpis ECS_C authority: the bounded, authenticated continuity and entity authority (docs/ECS_C_AUTHORITY.md).
//!
//! RETIRING (docs/ELPIS_MISSION.md, Amendment 3): this durable implementation writes persistent records and witnesses
//! and is non-conforming under the volatile-runtime directive. Its logic now lives in the volatile session crate
//! (`elpis_ecs_c_session`), which it re-exports; it remains only until RuntimeCore stops composing it.
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

// The transition law and its primitives belong to the volatile session crate (native/ECS_C/session,
// docs/ECS_C_SESSION.md), which outlives this durable authority; it is re-exported here unchanged.
pub use elpis_ecs_c_session::{error, model, register, sha256};

#[cfg(feature = "c-abi")]
pub mod ffi;
pub mod hmac;
pub mod legacy;
pub mod record;
pub mod store;

pub use error::Code;
pub use model::{
    apply, entity_id, genesis, receipt_chain, Applied, Cognition, Digest, Entity, EntityKind, Event, EventKind, Legacy,
    Lifecycle, Prior, State, MAX_CAPACITY, MAX_PER_TURN, MIN_CAPACITY, ZERO,
};
pub use record::{capacity_of, record_bytes, Record, SlotClass, Witness, WITNESS_BYTES};
pub use register::{EvolutionState, StateFormat, MAX_COUNTER};
pub use store::{
    Authority, Counters, Disposition, Mode, Provisioned, Receipt, Refusal, Report, Step, KEY_FILE, STATE_FILES,
    WITNESS_FILES,
};

#[cfg(any(test, feature = "testing"))]
pub use store::testing_peek;

#[cfg(test)]
mod tests;
