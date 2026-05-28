//! Lock the frozen public API surface — fails fast if anything shifts.
//!
//! Bumping a major version (0.x → 0.(x+1) for 0.x, 1.0+) is the only way
//! to break this test.

use zenwebp_recompress::{
    Budget, LosslessReason, NoOpReason, RecompressOptions, RecompressResult, StrategyKind,
    recompress,
};

#[test]
fn frozen_api_compiles() {
    let _opts = RecompressOptions {
        target_zensim_a: 80.0,
        budget: Budget::OneShot,
        ..Default::default()
    };
    // All Budget variants.
    let _ = [
        Budget::OneShot,
        Budget::MaxIterations(3),
        Budget::MaxTime(std::time::Duration::from_millis(500)),
    ];
    // All StrategyKind variants.
    let _ = [
        StrategyKind::CoeffEdit,
        StrategyKind::DeblockReencode,
        StrategyKind::Reencode,
        StrategyKind::LosslessReencode,
        StrategyKind::LosslessRemux,
    ];
    // All LosslessReason variants.
    let _ = [
        LosslessReason::NoStrategyShrinksFile,
        LosslessReason::NoStrategyMeetsTarget,
        LosslessReason::SourceWasLossless,
        LosslessReason::SourceIsAnimated,
        LosslessReason::SourceQualityTooLow,
    ];
    // NoOpReason.
    let _ = [NoOpReason::SourceAlreadyMeetsTarget];
    // RecompressResult constructors.
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
    // Entry point signature.
    let _: fn(
        &[u8],
        &RecompressOptions,
    ) -> Result<RecompressResult, zenwebp_recompress::Error> = recompress;
}

#[test]
fn rejects_target_out_of_range() {
    let bytes = vec![0u8; 100];
    let opts = RecompressOptions {
        target_zensim_a: 150.0,
        budget: Budget::OneShot,
        ..Default::default()
    };
    let res = recompress(&bytes, &opts);
    assert!(matches!(
        res,
        Err(zenwebp_recompress::Error::TargetOutOfRange(_))
    ));

    let opts2 = RecompressOptions {
        target_zensim_a: -1.0,
        budget: Budget::OneShot,
        ..Default::default()
    };
    let res2 = recompress(&bytes, &opts2);
    assert!(matches!(
        res2,
        Err(zenwebp_recompress::Error::TargetOutOfRange(_))
    ));
}

#[test]
fn rejects_non_webp_input() {
    let bytes = vec![0u8; 100];
    let opts = RecompressOptions::default();
    let res = recompress(&bytes, &opts);
    assert!(matches!(
        res,
        Err(zenwebp_recompress::Error::InvalidInput(_))
    ));
}

#[test]
fn result_helpers_work() {
    let r = RecompressResult::NoOp {
        reason: NoOpReason::SourceAlreadyMeetsTarget,
    };
    assert!(r.strategy().is_none());
    assert!(!r.better_handled_by_jxl());

    let r = RecompressResult::LosslessOnly {
        bytes: Vec::new(),
        reason: LosslessReason::SourceWasLossless,
        better_handled_by_jxl: true,
    };
    assert_eq!(r.strategy(), Some(StrategyKind::LosslessRemux));
    assert!(r.better_handled_by_jxl());
}
