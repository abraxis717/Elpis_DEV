//! H-ECS R2: does the qualified ECS/K1 world-model primitive, trained one step ahead, give a multi-step
//! predictive object stable enough for planning at the horizons H-ECS consumes? Hierarchy (width scaling vs a
//! temporal-shared vs a distinct-representation hierarchy of ECS states) is adjudicated only if it does.
//! RESEARCH_ONLY: no runtime authority, never on the canonical inference path. A fresh experiment after the
//! closed H-ECS R0 (TASK_INVALID_ON_DEV) and R1 (TASK_INVALID); R1's mechanics are reused byte for byte where
//! unchanged (research/hecs_r2/SOURCE_EQUIVALENCE.json) and `multistep` is new.
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
pub mod multistep;
pub mod planner;
pub mod rng;
pub mod spec;
pub mod validity;
pub mod world;

#[cfg(test)]
mod tests;
