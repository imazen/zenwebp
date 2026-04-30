//! zenpredict-driven encoder-knob picker for zenwebp (v0.1).
//!
//! Replaces the 3-row [`crate::encoder::analysis::content_type_to_tuning`]
//! lookup table with a baked hybrid-heads MLP (ZNPR v2) that predicts
//! `log_bytes + (sns, filter_strength, filter_sharpness)` per cell
//! over a 6-cell `(method × segments)` Cartesian grid. Argmin over
//! the bytes_log block under a constraint mask gives the cell; the
//! scalar heads at that cell index give the per-cell scalar tuning.
//!
//! Status: experimental. Gated on the `picker` cargo feature so
//! production builds are unaffected. The `analyzer` feature is also
//! required (the picker consumes 32 zenanalyze features).
//!
//! ## Cell grid (6 cells)
//!
//! | axis       | values    |
//! |------------|-----------|
//! | `method`   | {4, 5, 6} |
//! | `segments` | {1, 4}    |
//!
//! Scalar heads predicted per-cell:
//!
//! | head              | range  |
//! |-------------------|--------|
//! | `sns_strength`    | 0..100 |
//! | `filter_strength` | 0..100 |
//! | `filter_sharpness`| 0..7   |

pub mod spec;

#[cfg(feature = "picker")]
pub mod runtime;

#[cfg(feature = "picker")]
pub use runtime::{PickError, TuningPick, pick_tuning, pick_tuning_from_features};
