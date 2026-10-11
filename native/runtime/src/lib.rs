//! Elpis RuntimeCore: the runtime's mutable systems authority (docs/RUNTIME_CORE.md).
//!
//! One open runtime owns:
//! * its lifecycle (open / close) and its fail-stop disposition;
//! * the continuity authority (the `elpis_continuity` store, embedded: one implementation);
//! * the binding of its K1 lineage to one native K1 retained state, verified against continuity;
//! * the managed canonical turn's native K1 transaction: begin, the experience schedule, commit or abort, then
//!   the continuity publication of the committed retained-state identity, fail-stopping when that publication
//!   is not certain;
//! * the evolution attempt's durable reservation and finalization (one bounded attempt in flight).
//!
//! It owns no semantics: the token codec, the (unqualified) ECS codec map and the evolution gate's validation
//! stay at the language boundary, which hands RuntimeCore a validated native-ready stimulus or an assertion
//! digest. It holds no history, no events, no receipts and no trajectory: everything durable is the one
//! continuity record. The native K1 state is reached through caller-supplied function tables over the K1 C ABI
//! (`ecsg_k1.h`, `ecsg_k1_fms.h`), so RuntimeCore links no ECS code and computes no ECS mathematics.
//!
//! The production surface is the C ABI in [`ffi`] (include/elpis/runtime.h). The library also exports the
//! continuity C ABI of the embedded store's crate, so one library serves both.

pub mod checkpoint;
pub mod code;
pub mod core;
pub mod ffi;
pub mod fuel;
pub mod substrate;

pub use crate::core::{Core, Counters, Stimulus};
pub use code::{Error, Rt};
pub use fuel::{Budget, CEILING};
pub use substrate::{CommitIdentity, Experience, K1Ops, Key, ScheduleOutcome, Transition};

// The continuity C ABI is part of this library's exported surface (one authority, one implementation).
pub use elpis_continuity;

#[cfg(test)]
mod tests;
