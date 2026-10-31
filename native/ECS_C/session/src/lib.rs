//! Elpis ECS_C volatile session authority (ECSC_VOLATILE_SESSION_R0; docs/ECS_C_SESSION.md).
//!
//! The ECS_C authority's logic — entities and lifecycles, sender attribution and ordering, source versions, scopes
//! and per-turn bounds, typed-evidence admission, the cognition lineage, the evolution authority, receipts — as a
//! bounded state machine that lives only in volatile memory. A [`session::Session`] starts at a fresh genesis, evolves
//! by staged, all-or-nothing transitions of the one transition law ([`model::apply_into`]), and is gone when it is
//! destroyed or the process ends. Nothing is persisted, recovered, reconciled or resumed: every operation that existed
//! only for persistence (legacy import, the three reconciliations) is refused `PersistenceRetired`.
//!
//! `no_std`: the crate has no file, clock, thread or environment access. Its one system call is getrandom(2), for the
//! session nonce when the caller supplies none (a zero nonce). The C ABI (`ffi`, the `c-abi` feature, off by default)
//! adds a malloc-backed allocator and an abort-on-panic handler; an embedding Rust crate leaves it off.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod error;
#[cfg(feature = "c-abi")]
pub mod ffi;
pub mod model;
pub mod register;
pub mod session;
pub mod sha256;

#[cfg(all(feature = "c-abi", not(test)))]
mod runtime_support;

pub use error::Code;
pub use session::{Coordinate, Session, SessionConfig, SessionReceipt};

#[cfg(test)]
mod tests;
