//! Elpis RuntimeCore: the runtime's mutable systems authority (docs/RUNTIME_CORE.md), ABI v8: volatile only.
//!
//! One open runtime owns, in memory and for one session (docs/ELPIS_MISSION.md, Amendment 3):
//! * its lifecycle (open = a new session, close = the session ends) and its fail-stop disposition;
//! * the authority (`authority`): one volatile ECS_C session (`elpis_ecs_c_session`, embedded without its C ABI):
//!   the lineage's cognition identity, the evolution authority and the sources typed evidence is admitted against;
//! * the binding of its K1 lineage to one native K1 retained state, verified against the session;
//! * the managed canonical turn's native K1 transaction: begin, the experience schedule, commit or abort. A managed
//!   commit advances the live coordinate of the bound state and writes nothing;
//! * the evolution attempt's reservation and finalization in the session (one bounded attempt in flight).
//!
//! It owns no semantics: the token codec, the (unqualified) ECS codec map and the evolution gate's validation
//! stay at the language boundary, which hands RuntimeCore a validated native-ready stimulus or an assertion
//! digest. It holds no history, no events, no receipts and no trajectory, and it has no persistent-state write path
//! at all: no durability lane, checkpoint store, publication, recovery or file (ABI v7's, retired by v8). The
//! native K1 state is reached through caller-supplied function tables over the K1 C ABI (`ecsg_k1.h`,
//! `ecsg_k1_fms.h`), so RuntimeCore links no ECS code and computes no ECS mathematics.
//!
//! The production surface is the C ABI in [`ffi`] (include/elpis/runtime.h), and nothing else: the library exports
//! no other crate's C ABI.

pub mod authority;
pub mod code;
pub mod core;
pub mod ffi;
pub mod fuel;
pub mod substrate;

pub use crate::core::{Core, Counters, Stimulus, VolatileCommit};
pub use authority::SessionTemplate;
pub use code::{Error, Rt};
pub use fuel::{Budget, CEILING};
pub use substrate::{Experience, K1Ops, Key, ScheduleOutcome, Transition};

#[cfg(test)]
mod tests;
