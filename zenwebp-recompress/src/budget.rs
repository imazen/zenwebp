//! Budget plumbing.
//!
//! Translates [`crate::Budget`] into a per-call stop condition usable by the
//! iterative refinement loop. The default [`crate::Budget::OneShot`] becomes
//! `max_passes = 1, max_time = None`, meaning the strategy runs once and
//! ships.

use crate::api::Budget;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // deadline/max_passes consumed once MaxTime iteration lands in router.
pub(crate) struct Stopwatch {
    pub deadline: Option<Instant>,
    pub max_passes: u32,
}

#[allow(dead_code)]
impl Stopwatch {
    pub fn from_budget(budget: Budget) -> Self {
        match budget {
            Budget::OneShot => Self {
                deadline: None,
                max_passes: 1,
            },
            Budget::MaxIterations(n) => Self {
                deadline: None,
                max_passes: n.max(1),
            },
            Budget::MaxTime(d) => Self {
                deadline: Some(Instant::now() + d),
                max_passes: u32::MAX,
            },
        }
    }

    pub fn time_left(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    pub fn expired(&self) -> bool {
        match self.deadline {
            Some(d) => Instant::now() >= d,
            None => false,
        }
    }
}
