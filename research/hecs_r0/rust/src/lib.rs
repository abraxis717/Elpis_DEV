//! H-ECS R0: a research laboratory testing whether a hierarchy of canonical ECS states at distinct temporal
//! scales (and optionally distinct learned representation spaces) differs in behaviour from width scaling
//! of one ECS. RESEARCH_ONLY: no runtime authority, never on the canonical inference path.
//!
//! Every level owns exactly one native K1 state (W, epoch, H, a) reached through the existing C ABI
//! (`k1`); the ECS equations are not touched. Hierarchy ownership, scheduling, encoders, planning and the
//! measurements are this crate's.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

pub mod analysis;
pub mod encoder;
pub mod experiment;
pub mod hierarchy;
pub mod json;
pub mod k1;
pub mod linalg;
pub mod model;
pub mod planner;
pub mod rng;
pub mod spec;
pub mod world;

#[cfg(test)]
mod tests;
