//! Tests for the opt-in `Config::validate()` API.
//!
//! Each test constructs a `Config` directly (bypassing the clamping
//! `with_*` setters) so the offending value reaches `validate()`
//! unmodified. The happy-path test exercises the public builder
//! constructor.

use zenwebp::{EncoderConfig, LosslessConfig, LossyConfig, Preset, ValidationError, ZensimTarget};

// ---------------------------------------------------------------------------
// Happy path — defaults validate.
// ---------------------------------------------------------------------------

#[test]
fn default_lossy_validates() {
    LossyConfig::new().validate().expect("default lossy ok");
}

#[test]
fn default_lossless_validates() {
    LosslessConfig::new()
        .validate()
        .expect("default lossless ok");
}

#[test]
fn preset_lossy_validates() {
    LossyConfig::with_preset(Preset::Photo, 80.0)
        .validate()
        .expect("photo preset ok");
}

#[test]
fn encoder_config_lossy_validates() {
    EncoderConfig::new_lossy().validate().expect("ok");
}

#[test]
fn encoder_config_lossless_validates() {
    EncoderConfig::new_lossless().validate().expect("ok");
}

// ---------------------------------------------------------------------------
// Per-variant range failures (LossyConfig).
// ---------------------------------------------------------------------------

#[test]
fn quality_out_of_range() {
    let mut cfg = LossyConfig::new();
    cfg.quality = 150.0;
    match cfg.validate().unwrap_err() {
        ValidationError::QualityOutOfRange { value, .. } => assert_eq!(value, 150.0),
        e => panic!("unexpected variant: {e:?}"),
    }
}

#[test]
fn quality_negative_rejected() {
    let mut cfg = LossyConfig::new();
    cfg.quality = -1.0;
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::QualityOutOfRange { .. })
    ));
}

#[test]
fn quality_nan_rejected() {
    let mut cfg = LossyConfig::new();
    cfg.quality = f32::NAN;
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::QualityNotFinite { .. })
    ));
}

#[test]
fn method_out_of_range() {
    let mut cfg = LossyConfig::new();
    cfg.method = 7;
    match cfg.validate().unwrap_err() {
        ValidationError::MethodOutOfRange { value, .. } => assert_eq!(value, 7),
        e => panic!("unexpected variant: {e:?}"),
    }
}

#[test]
fn alpha_quality_out_of_range() {
    let mut cfg = LossyConfig::new();
    cfg.alpha_quality = 200;
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::AlphaQualityOutOfRange { .. })
    ));
}

#[test]
fn sns_strength_out_of_range() {
    let mut cfg = LossyConfig::new();
    cfg.sns_strength = Some(200);
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::SnsStrengthOutOfRange { .. })
    ));
}

#[test]
fn filter_strength_out_of_range() {
    let mut cfg = LossyConfig::new();
    cfg.filter_strength = Some(200);
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::FilterStrengthOutOfRange { .. })
    ));
}

#[test]
fn filter_sharpness_out_of_range() {
    let mut cfg = LossyConfig::new();
    cfg.filter_sharpness = Some(8);
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::FilterSharpnessOutOfRange { .. })
    ));
}

#[test]
fn segments_zero_rejected() {
    let mut cfg = LossyConfig::new();
    cfg.segments = Some(0);
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::SegmentsOutOfRange { .. })
    ));
}

#[test]
fn segments_too_many_rejected() {
    let mut cfg = LossyConfig::new();
    cfg.segments = Some(5);
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::SegmentsOutOfRange { .. })
    ));
}

#[test]
fn partition_limit_out_of_range() {
    let mut cfg = LossyConfig::new();
    cfg.partition_limit = Some(200);
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::PartitionLimitOutOfRange { .. })
    ));
}

#[test]
fn target_psnr_out_of_range() {
    let mut cfg = LossyConfig::new();
    cfg.target_psnr = 200.0;
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::TargetPsnrOutOfRange { .. })
    ));
}

#[test]
fn target_psnr_negative_rejected() {
    let mut cfg = LossyConfig::new();
    cfg.target_psnr = -10.0;
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::TargetPsnrOutOfRange { .. })
    ));
}

#[test]
fn target_psnr_zero_is_disabled_ok() {
    let mut cfg = LossyConfig::new();
    cfg.target_psnr = 0.0;
    cfg.validate().expect("0.0 = disabled, must pass");
}

#[test]
fn target_zensim_out_of_range() {
    let mut cfg = LossyConfig::new();
    cfg.target_zensim = Some(ZensimTarget::new(150.0));
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::TargetZensimOutOfRange { .. })
    ));
}

#[test]
fn target_zensim_max_passes_zero_rejected() {
    let mut cfg = LossyConfig::new();
    cfg.target_zensim = Some(ZensimTarget::new(80.0).with_max_passes(0));
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::TargetZensimMaxPassesZero { .. })
    ));
}

#[test]
fn target_zensim_negative_overshoot_rejected() {
    let mut cfg = LossyConfig::new();
    cfg.target_zensim = Some(ZensimTarget::new(80.0).with_max_overshoot(Some(-1.0)));
    match cfg.validate().unwrap_err() {
        ValidationError::TargetZensimToleranceInvalid { field, value } => {
            assert_eq!(field, "max_overshoot");
            assert_eq!(value, -1.0);
        }
        e => panic!("unexpected variant: {e:?}"),
    }
}

#[test]
fn sharp_yuv_negative_threshold_rejected() {
    let mut cfg = LossyConfig::new();
    cfg.sharp_yuv = Some(zenwebp::SharpYuvConfig {
        max_iterations: 2,
        convergence_threshold: -0.1,
        refine_y: true,
    });
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::SharpYuvConvergenceThresholdInvalid { .. })
    ));
}

// ---------------------------------------------------------------------------
// Cross-param: target_size / target_psnr / target_zensim are mutually
// exclusive. All three pairs must be rejected.
// ---------------------------------------------------------------------------

#[test]
fn target_size_and_psnr_mutually_exclusive() {
    let mut cfg = LossyConfig::new();
    cfg.target_size = 50_000;
    cfg.target_psnr = 40.0;
    match cfg.validate().unwrap_err() {
        ValidationError::TargetMutuallyExclusive { first, second } => {
            assert_eq!(first, "target_size");
            assert_eq!(second, "target_psnr");
        }
        e => panic!("unexpected variant: {e:?}"),
    }
}

#[test]
fn target_size_and_zensim_mutually_exclusive() {
    let mut cfg = LossyConfig::new();
    cfg.target_size = 50_000;
    cfg.target_zensim = Some(ZensimTarget::new(80.0));
    match cfg.validate().unwrap_err() {
        ValidationError::TargetMutuallyExclusive { first, second } => {
            assert_eq!(first, "target_size");
            assert_eq!(second, "target_zensim");
        }
        e => panic!("unexpected variant: {e:?}"),
    }
}

#[test]
fn target_psnr_and_zensim_mutually_exclusive() {
    let mut cfg = LossyConfig::new();
    cfg.target_psnr = 40.0;
    cfg.target_zensim = Some(ZensimTarget::new(80.0));
    match cfg.validate().unwrap_err() {
        ValidationError::TargetMutuallyExclusive { first, second } => {
            assert_eq!(first, "target_psnr");
            assert_eq!(second, "target_zensim");
        }
        e => panic!("unexpected variant: {e:?}"),
    }
}

#[test]
fn all_three_targets_set_rejected() {
    let mut cfg = LossyConfig::new();
    cfg.target_size = 50_000;
    cfg.target_psnr = 40.0;
    cfg.target_zensim = Some(ZensimTarget::new(80.0));
    // First pair detected wins (size+psnr).
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::TargetMutuallyExclusive { .. })
    ));
}

#[test]
fn single_target_ok() {
    let mut cfg = LossyConfig::new();
    cfg.target_size = 50_000;
    cfg.validate().expect("single target ok");

    let mut cfg = LossyConfig::new();
    cfg.target_psnr = 40.0;
    cfg.validate().expect("single target ok");

    let mut cfg = LossyConfig::new();
    cfg.target_zensim = Some(ZensimTarget::new(80.0));
    cfg.validate().expect("single target ok");
}

// ---------------------------------------------------------------------------
// LosslessConfig.
// ---------------------------------------------------------------------------

#[test]
fn lossless_quality_out_of_range() {
    let mut cfg = LosslessConfig::new();
    cfg.quality = 150.0;
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::QualityOutOfRange { .. })
    ));
}

#[test]
fn lossless_method_out_of_range() {
    let mut cfg = LosslessConfig::new();
    cfg.method = 9;
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::MethodOutOfRange { .. })
    ));
}

#[test]
fn lossless_alpha_quality_out_of_range() {
    let mut cfg = LosslessConfig::new();
    cfg.alpha_quality = 200;
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::AlphaQualityOutOfRange { .. })
    ));
}

#[test]
fn lossless_near_lossless_out_of_range() {
    let mut cfg = LosslessConfig::new();
    cfg.near_lossless = 200;
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::NearLosslessOutOfRange { .. })
    ));
}

// ---------------------------------------------------------------------------
// EncoderConfig dispatch.
// ---------------------------------------------------------------------------

#[test]
fn encoder_config_dispatches_to_inner() {
    let mut inner = LossyConfig::new();
    inner.quality = 200.0;
    let cfg = EncoderConfig::Lossy(inner);
    assert!(matches!(
        cfg.validate(),
        Err(ValidationError::QualityOutOfRange { .. })
    ));
}

// ---------------------------------------------------------------------------
// __expert: InternalParams.validate()
// ---------------------------------------------------------------------------

#[cfg(feature = "__expert")]
mod expert {
    use zenwebp::{InternalParams, SharpYuvSetting, ValidationError};

    #[test]
    fn internal_params_default_validates() {
        InternalParams::default().validate().expect("ok");
    }

    #[test]
    fn internal_params_partition_limit_out_of_range() {
        let mut p = InternalParams::default();
        p.partition_limit = Some(200);
        assert!(matches!(
            p.validate(),
            Err(ValidationError::PartitionLimitOutOfRange { .. })
        ));
    }

    #[test]
    fn internal_params_sharp_yuv_custom_invalid() {
        let mut p = InternalParams::default();
        p.sharp_yuv = Some(SharpYuvSetting::Custom(zenwebp::SharpYuvConfig {
            max_iterations: 2,
            convergence_threshold: f32::NAN,
            refine_y: true,
        }));
        assert!(matches!(
            p.validate(),
            Err(ValidationError::SharpYuvConvergenceThresholdInvalid { .. })
        ));
    }
}
