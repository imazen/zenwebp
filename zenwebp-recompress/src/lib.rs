#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

//! See [`recompress`] for the single entry point and [`RecompressOptions`]
//! / [`RecompressResult`] for the contract.

extern crate alloc;

mod api;
mod budget;
mod calibration;
mod classify;
mod error;
mod estimate;
mod measure;
mod router;
mod source;
mod strategies;
mod target;
mod vp8x;

pub use api::{
    Budget, LosslessReason, NoOpReason, Plan, RecompressOptions, RecompressResult, StrategyKind,
    plan, recompress,
};
pub use error::Error;

/// Expert-only internals.
///
/// **Stability:** Anything under this module may change between any two
/// releases — it is NOT covered by semver. The frozen public surface lives
/// at the crate root.
#[cfg(feature = "expert")]
pub mod expert {
    pub use crate::calibration::{CalibrationLookup, CellEstimate};
    pub use crate::classify::classify;
    pub use crate::estimate::estimate_quality_by_recompression;
    pub use crate::measure::{score_against_reference, score_recompression, score_rgba};
    pub use crate::router::{RouterDecision, decide_strategy, dispatch};
    pub use crate::source::{ContentClass, EncoderFamily, SourceAnalysis, analyze_source};
    pub use crate::strategies::{
        coeff_edit::{run_coeff_edit, run_coeff_edit_keep, run_coeff_edit_requant},
        deblock::deblock_rgba,
        deblock_reencode::run_deblock_reencode,
        lossless_reencode::run_lossless_reencode,
        lossless_remux::run_lossless_remux,
        reencode::run_reencode,
    };
    pub use crate::target::target_zensim_a_to_libwebp_q;
}

#[cfg(test)]
mod public_api_lock {
    use super::*;

    /// Locks the frozen public API surface — fails fast if names shift.
    /// Bumping the major version is the only way to break this test.
    #[test]
    fn public_api_is_frozen() {
        let _opts = RecompressOptions {
            target_zensim_a: 80.0,
            budget: Budget::OneShot,
            ..Default::default()
        };
        let _r1: RecompressResult = RecompressResult::NoOp {
            reason: NoOpReason::SourceAlreadyMeetsTarget,
        };
        let _r2: RecompressResult = RecompressResult::LosslessOnly {
            bytes: Vec::new(),
            reason: LosslessReason::NoStrategyShrinksFile,
            better_handled_by_jxl: false,
        };
        let _r3: RecompressResult = RecompressResult::Recompressed {
            bytes: Vec::new(),
            strategy: StrategyKind::Reencode,
            projected_zensim_a: 80.0,
            measured_zensim_a: None,
            source_to_output_ratio: 1.0,
            better_handled_by_jxl: false,
        };
        // budget variants exist
        let _b1 = Budget::OneShot;
        let _b2 = Budget::MaxIterations(3);
        let _b3 = Budget::MaxTime(std::time::Duration::from_millis(500));
        // strategy variants exist
        let _s = [
            StrategyKind::CoeffEdit,
            StrategyKind::DeblockReencode,
            StrategyKind::Reencode,
            StrategyKind::LosslessReencode,
            StrategyKind::LosslessRemux,
        ];
    }
}
