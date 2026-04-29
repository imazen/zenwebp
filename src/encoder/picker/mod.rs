//! zenpicker-driven encoder-knob picker for zenwebp (research spike).
//!
//! Replaces the 3-row [`crate::encoder::analysis::content_type_to_tuning`]
//! lookup table with a baked MLP that predicts log-bytes per
//! `(sns_strength, filter_strength, segments)` cell from zenanalyze
//! features + `(target_zensim, log_pixels)`. Argmin of the predicted
//! bytes under a constraint mask gives the encoder tuple to use.
//!
//! Status: **spike**. Gated on the `picker` cargo feature so the
//! production build is unaffected. v0.1 pure-categorical model — no
//! continuous heads. `filter_sharpness` is fixed at 0 and `method`
//! at 4 for this spike (continuous axes belong in v0.2 hybrid heads;
//! out of scope for this measurement).
//!
//! ## Cell grid (16 cells)
//!
//! | axis             | values                |
//! |------------------|------------------------|
//! | `sns_strength`   | {0, 25, 50, 80}       |
//! | `filter_strength`| {30, 60}              |
//! | `segments`       | {1, 4}                |
//!
//! `filter_sharpness = 0` and the encoder method are fixed by the
//! caller — the picker only chooses the four-tuple
//! `(sns, filter_strength, filter_sharpness, segments)` returned to
//! the existing `content_type_to_tuning` callsite.

pub mod spec;

#[cfg(feature = "picker")]
pub mod runtime;

#[cfg(feature = "picker")]
pub use runtime::pick_tuning;
