//! Runtime that loads the baked picker model and runs argmin over
//! cells + reads scalar heads to assemble an encoder tuning tuple.
//!
//! Loads `zenwebp_picker_v0.1.bin` (ZNPR v2 format) at first call,
//! verifies `schema_hash` against the compile-time const, and reuses
//! a `Predictor` scratch state across invocations.
//!
//! API contract:
//!   `pick_tuning(raw_feats, w, h, target_zensim, &constraints)` →
//!   `Result<TuningPick, PickError>`
//! where `TuningPick = (sns, filter_strength, filter_sharpness,
//! method, segments)`. The codec composes these into a `LossyConfig`.
//!
//! See `zenanalyze--zenpredict/MIGRATION.md` for the
//! `zenpicker → zenpredict` API rename + ZNPR v1 → v2 format change.

use alloc::vec::Vec;
use core::f32;

use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpredict::{
    AllowedMask, Model, Predictor, ScoreTransform,
    argmin::argmin_masked_in_range as argmin_in_range,
};

use super::spec::{
    CELLS, FEAT_COLS, N_CELLS, PickerConstraints, RANGE_BYTES_LOG, RANGE_FILTER_SHARPNESS,
    RANGE_FILTER_STRENGTH, RANGE_SNS, SCHEMA_HASH,
};

/// Compile-time map from FEAT_COLS string names → zenanalyze
/// AnalysisFeature enum variants. Order MUST match `FEAT_COLS`. Used
/// by `extract_raw_features_rgb8` to walk the analysis result in the
/// exact order the picker's input vector expects.
const ANALYSIS_FEATURES: &[AnalysisFeature] = &[
    AnalysisFeature::LaplacianVarianceP50,
    AnalysisFeature::LaplacianVarianceP75,
    AnalysisFeature::LaplacianVariance,
    AnalysisFeature::QuantSurvivalY,
    AnalysisFeature::CbSharpness,
    AnalysisFeature::PixelCount,
    AnalysisFeature::Uniformity,
    AnalysisFeature::DistinctColorBins,
    AnalysisFeature::CrSharpness,
    AnalysisFeature::EdgeDensity,
    AnalysisFeature::NoiseFloorYP50,
    AnalysisFeature::LumaHistogramEntropy,
    AnalysisFeature::NaturalLikelihood,
    AnalysisFeature::QuantSurvivalYP50,
    AnalysisFeature::NoiseFloorUvP50,
    AnalysisFeature::AqMapMean,
    AnalysisFeature::CrHorizSharpness,
    AnalysisFeature::MinDim,
    AnalysisFeature::EdgeSlopeStdev,
    AnalysisFeature::LaplacianVarianceP90,
    AnalysisFeature::PatchFraction,
    AnalysisFeature::MaxDim,
    AnalysisFeature::AspectMinOverMax,
    AnalysisFeature::AqMapP75,
    AnalysisFeature::CbHorizSharpness,
    AnalysisFeature::NoiseFloorYP25,
    AnalysisFeature::NoiseFloorUV,
    AnalysisFeature::ChromaComplexity,
    AnalysisFeature::QuantSurvivalYP75,
    AnalysisFeature::AqMapStd,
    AnalysisFeature::GradientFraction,
    AnalysisFeature::NoiseFloorYP75,
    AnalysisFeature::ScreenContentLikelihood,
    AnalysisFeature::HighFreqEnergyRatio,
    AnalysisFeature::Colourfulness,
    AnalysisFeature::QuantSurvivalUv,
];

const _: () = {
    // Compile-time invariant: ANALYSIS_FEATURES must align with FEAT_COLS
    // length so the runtime vector and the schema_hash stay in lockstep.
    if ANALYSIS_FEATURES.len() != FEAT_COLS.len() {
        panic!("ANALYSIS_FEATURES.len() != FEAT_COLS.len()");
    }
};

/// Wrap the include_bytes blob in a u64-aligned struct. The ZNPR v2
/// loader requires ≥4-byte alignment for f32 weight slices and fails
/// loudly with `PredictError::SectionMisaligned` otherwise (the v1
/// loader silently allocated a copy on misalignment — v2 trades that
/// for typed failure).
#[repr(C, align(16))]
struct AlignedModel<const N: usize>([u8; N]);

const MODEL_BYTES_RAW: &[u8] =
    &AlignedModel(*include_bytes!("zenwebp_picker_v0.1.bin")).0;

/// What the codec gets back when the picker succeeds. The four-tuple
/// matches the existing `content_type_to_tuning` shape, plus the
/// categorical (method, segments) the picker chose.
#[derive(Clone, Copy, Debug)]
pub struct TuningPick {
    pub sns_strength: u8,
    pub filter_strength: u8,
    pub filter_sharpness: u8,
    pub method: u8,
    pub segments: u8,
    /// Cell index in [0, N_CELLS). For diagnostics + the codec's
    /// optional fallback path.
    pub cell_idx: usize,
}

/// Reasons the picker can fail. Codec consumers can fall through to
/// the bucket-table path for any of these.
#[derive(Clone, Copy, Debug)]
pub enum PickError {
    /// `MODEL_BYTES_RAW` is empty (no .bin shipped at this build).
    NoBakedModel,
    /// Header parse / alignment / version fail. See zenpredict for
    /// the typed underlying error.
    Parse,
    /// Loaded model's schema_hash differs from compile-time
    /// `SCHEMA_HASH` — codec + bake drifted out of sync.
    SchemaMismatch { expected: u64, got: u64 },
    /// Forward-pass failure (feature length mismatch etc).
    Forward,
    /// All cells masked out by `PickerConstraints` — caller must relax.
    NoAllowedCell,
}

/// Engineered feature vector layout:
///   `raw_feats[..N_FEAT_COLS] || size_oh[4] || poly[5] || cross[N_FEAT_COLS] || icc_bytes[1]`
///   = `2 * N_FEAT_COLS + 10` floats.
/// Order MUST match the Python `train_hybrid.py` builder. Drift is
/// caught at load time via `from_bytes_with_schema`.
fn engineered_features(
    raw_feats: &[f32],
    width: u32,
    height: u32,
    target_zensim: f32,
) -> Vec<f32> {
    debug_assert_eq!(
        raw_feats.len(),
        FEAT_COLS.len(),
        "raw feature count mismatch"
    );
    let pixels = (width as f32) * (height as f32);
    let log_px = libm::logf(pixels.max(1.0));
    let target_norm = target_zensim / 100.0;

    // size_class one-hot — must match Python `SIZE_INDEX` ordering.
    let size_oh = match (width as u64) * (height as u64) {
        n if n < 64 * 64 => [1.0_f32, 0.0, 0.0, 0.0],     // tiny
        n if n < 256 * 256 => [0.0, 1.0, 0.0, 0.0],       // small
        n if n < 1024 * 1024 => [0.0, 0.0, 1.0, 0.0],     // medium
        _ => [0.0, 0.0, 0.0, 1.0],                         // large
    };

    let n_feat = raw_feats.len();
    let mut out = Vec::with_capacity(n_feat + 4 + 5 + n_feat + 1);
    out.extend_from_slice(raw_feats);
    out.extend_from_slice(&size_oh);
    out.extend_from_slice(&[
        log_px,
        log_px * log_px,
        target_norm,
        target_norm * target_norm,
        target_norm * log_px,
    ]);
    for f in raw_feats {
        out.push(target_norm * f);
    }
    out.push(0.0); // icc_bytes — runtime icc-size plumbing is a follow-up.
    out
}

/// Run zenanalyze on the RGB8 source and pull our 32 features in
/// `FEAT_COLS` order. Returns `None` if any feature fails to extract
/// (the trainer's load_features uses 0.0 as the default, so we
/// mirror that behavior — out-of-range or unsupported features
/// surface as 0.0 rather than failing the whole pick).
fn extract_raw_features_rgb8(rgb: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut feats = FeatureSet::new();
    for f in ANALYSIS_FEATURES {
        feats = feats.with(*f);
    }
    let query = AnalysisQuery::new(feats);
    let analysis = zenanalyze::analyze_features_rgb8(rgb, width, height, &query);
    ANALYSIS_FEATURES
        .iter()
        .map(|f| analysis.get_f32(*f).unwrap_or(0.0))
        .collect()
}

/// Pick the encoder tuning tuple via the baked picker.
///
/// Internally extracts the 32-feature zenanalyze vector from `rgb`
/// in `FEAT_COLS` order (so the codec doesn't have to track the
/// schema), then runs the picker forward pass + masked argmin.
///
/// Returns `Err(PickError)` for any failure; codec callers can fall
/// through to the `content_type_to_tuning` bucket-table path on
/// `Err(_)`.
pub fn pick_tuning(
    rgb: &[u8],
    width: u32,
    height: u32,
    target_zensim: f32,
    constraints: &PickerConstraints,
) -> Result<TuningPick, PickError> {
    let raw_feats = extract_raw_features_rgb8(rgb, width, height);
    pick_tuning_from_features(&raw_feats, width, height, target_zensim, constraints)
}

/// Lower-level entry: pick from a pre-extracted feature vector.
/// Useful for the dev/A-B harness which already has features in hand
/// and doesn't want to pay zenanalyze cost again.
pub fn pick_tuning_from_features(
    raw_feats: &[f32],
    width: u32,
    height: u32,
    target_zensim: f32,
    constraints: &PickerConstraints,
) -> Result<TuningPick, PickError> {
    if MODEL_BYTES_RAW.is_empty() {
        return Err(PickError::NoBakedModel);
    }

    let model =
        Model::from_bytes_with_schema(MODEL_BYTES_RAW, SCHEMA_HASH).map_err(|e| {
            // Map the zenpredict typed error back to ours. Two cases
            // we care about: schema mismatch (drift) vs anything else.
            match e {
                zenpredict::PredictError::SchemaHashMismatch { expected, got } => {
                    PickError::SchemaMismatch { expected, got }
                }
                _ => PickError::Parse,
            }
        })?;

    let mut predictor = Predictor::new(model);
    let feats = engineered_features(raw_feats, width, height, target_zensim);
    let mask_arr = constraints.allowed_mask();
    let mask = AllowedMask::new(&mask_arr);

    // Run forward pass once; argmin + scalar reads share the output.
    let output = predictor.predict(&feats).map_err(|_| PickError::Forward)?;

    // Argmin over the bytes_log sub-range in linear-bytes space
    // (ScoreTransform::Exp converts log-bytes to bytes monotonically).
    let cell_idx = argmin_in_range(
        output,
        (RANGE_BYTES_LOG.start, RANGE_BYTES_LOG.end),
        &mask,
        ScoreTransform::Exp,
        None,
    )
    .ok_or(PickError::NoAllowedCell)?;

    if cell_idx >= N_CELLS {
        return Err(PickError::Forward);
    }

    // Read the scalar heads at the chosen cell index. The picker was
    // trained against `(0..100, 0..100, 0..7)` ranges; clamp on read
    // so an out-of-distribution input vector can't hand the codec a
    // negative sns or a sharpness of 99.
    let sns = clamp_to_u8(output[RANGE_SNS.start + cell_idx], 0.0, 100.0);
    let filter_strength = clamp_to_u8(
        output[RANGE_FILTER_STRENGTH.start + cell_idx],
        0.0,
        100.0,
    );
    let filter_sharpness = clamp_to_u8(
        output[RANGE_FILTER_SHARPNESS.start + cell_idx],
        0.0,
        7.0,
    );

    let cell = CELLS[cell_idx];
    Ok(TuningPick {
        sns_strength: sns,
        filter_strength,
        filter_sharpness,
        method: cell.method,
        segments: cell.segments,
        cell_idx,
    })
}

fn clamp_to_u8(v: f32, lo: f32, hi: f32) -> u8 {
    let clamped = if v.is_nan() {
        lo
    } else {
        v.max(lo).min(hi)
    };
    libm::roundf(clamped) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engineered_features_layout_correct() {
        const N: usize = FEAT_COLS.len();
        let raw = [0.1_f32; N];
        let v = engineered_features(&raw, 512, 512, 80.0);
        // N raw + 4 size_oh + 5 poly + N cross + 1 icc
        assert_eq!(v.len(), N + 4 + 5 + N + 1);
        // size_class for 512×512 = 262144 → "medium". size_oh sits
        // right after the raw block.
        assert_eq!(v[N..N + 4], [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn engineered_features_size_class_buckets() {
        const N: usize = FEAT_COLS.len();
        let raw = [0.0_f32; N];
        // tiny: 32×32 = 1024 px
        let v = engineered_features(&raw, 32, 32, 80.0);
        assert_eq!(v[N..N + 4], [1.0, 0.0, 0.0, 0.0]);
        // small: 200×200 = 40000 px
        let v = engineered_features(&raw, 200, 200, 80.0);
        assert_eq!(v[N..N + 4], [0.0, 1.0, 0.0, 0.0]);
        // large: 2048×2048
        let v = engineered_features(&raw, 2048, 2048, 80.0);
        assert_eq!(v[N..N + 4], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn picker_loads_and_picks_a_cell() {
        // Smoke: feed a synthetic raw vector through the lower-level
        // entry point; default mask allows everything.
        let raw = [0.5_f32; FEAT_COLS.len()];
        let pick = pick_tuning_from_features(
            &raw,
            512,
            512,
            80.0,
            &PickerConstraints::default(),
        );
        assert!(pick.is_ok(), "pick_tuning failed: {:?}", pick);
        let p = pick.unwrap();
        assert!(p.cell_idx < N_CELLS);
        assert!(p.sns_strength <= 100);
        assert!(p.filter_strength <= 100);
        assert!(p.filter_sharpness <= 7);
        assert!([4, 5, 6].contains(&p.method));
        assert!([1, 4].contains(&p.segments));
    }

    #[test]
    fn picker_respects_method_constraint() {
        let raw = [0.5_f32; FEAT_COLS.len()];
        let constraints = PickerConstraints {
            allowed_methods: Some(&[4]),
            ..Default::default()
        };
        let pick = pick_tuning_from_features(&raw, 512, 512, 80.0, &constraints);
        assert!(pick.is_ok());
        assert_eq!(pick.unwrap().method, 4);
    }

    #[test]
    fn picker_returns_no_allowed_cell_when_all_masked() {
        let raw = [0.5_f32; FEAT_COLS.len()];
        let constraints = PickerConstraints {
            allowed_methods: Some(&[3]), // not in {4,5,6} — masks every cell
            ..Default::default()
        };
        let pick = pick_tuning_from_features(&raw, 512, 512, 80.0, &constraints);
        assert!(matches!(pick, Err(PickError::NoAllowedCell)));
    }
}
