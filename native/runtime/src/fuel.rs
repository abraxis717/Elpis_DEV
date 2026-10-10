//! Total cognitive fuel: a deterministic integer bound on the native ECS work one operation may cause, admitted
//! before any reserve, transaction or candidate mutation.
//!
//! Per-field bounds (64 experiences, 256 rows and 2^20 K1 steps each) still let one schedule ask for tens of
//! millions of learning steps. Admission therefore bounds the totals: experiences, rows, rows of one experience
//! (which is also the reservation ceiling, so the memory a schedule can make RuntimeCore reserve), learning steps,
//! query rows, and **ECS work units**.
//!
//! An ECS work unit is one multiply-accumulate-class operation of the native K1 kernels as counted by this formula
//! (not measured, not wall-clock, not machine-dependent). For a state of input dimension `d`, width `w` and
//! `F = features(d)` (the S3 length):
//!
//! ```text
//!   one K1 learning step on r rows   2*r*d*w            G1: forward and gradient over the rows
//!                                  + 2*d*w*F            the K1 correction: S3(W) and its Jacobian contraction
//!                                  + F*F                the H mat-vec
//!   one consolidation of r rows      r*F*F              the input statistics phi(x) phi(x)^T
//!                                  + d*w*F              the anchor a = S3(W)
//!   experience (r rows, k steps)     k * step(r) + consolidation(r)
//!   schedule                         sum over its experiences
//!   QUERY of r rows                  r*d*w              f_W(x)
//! ```
//!
//! Every sum and product is checked; a total that does not fit `u64` is refused, never wrapped. The formula is a
//! conservative proxy of the dominant loops, not a cost model: two states with the same units need not take the
//! same time. No wall-clock deadline is claimed: the native K1 ABI is synchronous and has no cooperative
//! cancellation point, so a running schedule cannot be preempted; fuel admission bounds it before it starts.

use crate::substrate::{features, Experience, MAX_EXPERIENCES};

/// `elpis_runtime_budget`: the totals one cognitive operation may use. A caller budget can only narrow
/// [`CEILING`]; every field must be at least 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub max_experiences: u64,
    /// Total rows of one schedule.
    pub max_rows: u64,
    /// Rows of one experience; also the ceiling of the row capacity RuntimeCore may reserve.
    pub max_experience_rows: u64,
    /// Total K1 learning steps of one schedule.
    pub max_steps: u64,
    /// ECS work units of one operation.
    pub max_work_units: u64,
    pub max_query_rows: u64,
}

/// The canonical ceiling, compiled in. No budget may exceed it.
pub const CEILING: Budget = Budget {
    max_experiences: MAX_EXPERIENCES as u64,
    max_rows: 16_384,
    max_experience_rows: 256,
    max_steps: 1 << 20,
    max_work_units: 1 << 30,
    max_query_rows: 4096,
};

/// Why a request is outside its budget (all refused as `COGNITION_FUEL_EXCEEDED`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Over {
    Budget,
    Experiences,
    Rows,
    ExperienceRows,
    Steps,
    WorkUnits,
    QueryRows,
}

impl Budget {
    /// Whether this budget is a valid narrowing of [`CEILING`].
    pub fn within_ceiling(&self) -> bool {
        let pairs = [
            (self.max_experiences, CEILING.max_experiences),
            (self.max_rows, CEILING.max_rows),
            (self.max_experience_rows, CEILING.max_experience_rows),
            (self.max_steps, CEILING.max_steps),
            (self.max_work_units, CEILING.max_work_units),
            (self.max_query_rows, CEILING.max_query_rows),
        ];
        pairs.iter().all(|&(v, c)| (1..=c).contains(&v))
    }
}

fn mul(a: u64, b: u64) -> Option<u64> {
    a.checked_mul(b)
}

/// ECS work units of one K1 learning step on `rows` rows, and of one consolidation of them.
fn step_and_consolidation(dim: u64, width: u64, f: u64, rows: u64) -> Option<(u64, u64)> {
    let dw = mul(dim, width)?;
    let dwf = mul(dw, f)?;
    let ff = mul(f, f)?;
    let step = mul(mul(2, rows)?, dw)?.checked_add(mul(2, dwf)?)?.checked_add(ff)?;
    let consolidation = mul(rows, ff)?.checked_add(dwf)?;
    Some((step, consolidation))
}

/// ECS work units of an experience schedule on a `dim x width` state; `None` for an invalid shape or schedule, or a
/// total beyond `u64`.
pub fn schedule_units(dim: usize, width: usize, schedule: &[Experience]) -> Option<u64> {
    let f = features(dim) as u64;
    if f == 0 || width == 0 || schedule.is_empty() {
        return None;
    }
    let (dim, width) = (dim as u64, width as u64);
    schedule.iter().try_fold(0u64, |total, e| {
        if e.rows == 0 || e.steps == 0 {
            return None;
        }
        let (step, consolidation) = step_and_consolidation(dim, width, f, e.rows)?;
        total.checked_add(mul(e.steps, step)?.checked_add(consolidation)?)
    })
}

/// ECS work units of a QUERY of `rows` rows on a `dim x width` state.
pub fn query_units(dim: usize, width: usize, rows: usize) -> Option<u64> {
    if features(dim) == 0 || width == 0 || rows == 0 {
        return None;
    }
    mul(mul(rows as u64, dim as u64)?, width as u64)
}

/// Admit an experience schedule under `budget` (the budget itself must narrow [`CEILING`]).
pub fn admit_schedule(budget: &Budget, dim: usize, width: usize, schedule: &[Experience]) -> Result<u64, Over> {
    if !budget.within_ceiling() {
        return Err(Over::Budget);
    }
    if schedule.len() as u64 > budget.max_experiences {
        return Err(Over::Experiences);
    }
    let mut rows = 0u64;
    let mut steps = 0u64;
    for e in schedule {
        if e.rows > budget.max_experience_rows {
            return Err(Over::ExperienceRows);
        }
        rows = rows.checked_add(e.rows).ok_or(Over::Rows)?;
        steps = steps.checked_add(e.steps).ok_or(Over::Steps)?;
    }
    if rows > budget.max_rows {
        return Err(Over::Rows);
    }
    if steps > budget.max_steps {
        return Err(Over::Steps);
    }
    match schedule_units(dim, width, schedule) {
        Some(units) if units <= budget.max_work_units => Ok(units),
        _ => Err(Over::WorkUnits),
    }
}

/// Admit a QUERY of `rows` rows under `budget`.
pub fn admit_query(budget: &Budget, dim: usize, width: usize, rows: usize) -> Result<u64, Over> {
    if !budget.within_ceiling() {
        return Err(Over::Budget);
    }
    if rows as u64 > budget.max_query_rows {
        return Err(Over::QueryRows);
    }
    match query_units(dim, width, rows) {
        Some(units) if units <= budget.max_work_units => Ok(units),
        _ => Err(Over::WorkUnits),
    }
}
