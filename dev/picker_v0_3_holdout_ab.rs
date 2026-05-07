//! Held-out re-encode A/B for the v0.3 zenwebp picker.
//!
//! Compares two encoder arms on a held-out PNG corpus at multiple
//! `target_zensim` levels:
//!
//!   * **picker**: 36-feature zenanalyze vector → engineered 82-feature
//!     vector (same layout as `src/encoder/picker/runtime.rs::engineered_features`)
//!     → externally-loaded v0.3 `.bin` via `zenpredict::Model::from_bytes`
//!     → `predict_transformed` → cell argmin + scalar heads → explicit
//!     `(method, segments, sns_strength, filter_strength, filter_sharpness)`
//!     applied to a `LossyConfig::with_target_zensim` encode.
//!   * **bucket**: identical pipeline except the encoder knobs come
//!     from `content_type_to_tuning(classify_image_type_rgb8(rgb))`,
//!     pinned to method = 4 (the bucket-table path's default).
//!
//! Both arms ride the same `target_zensim` closed-loop so they hit
//! comparable achieved scores; the differential is bytes at matched
//! quality.
//!
//! Why we don't reuse `picker_ab_eval.rs`: that harness leans on the
//! embedded v0.1 `.bin` shipped at `src/encoder/picker/zenwebp_picker_v0.1.bin`.
//! v0.3 has a divergent `schema_hash` and cannot drop into the runtime
//! without bumping `src/encoder/picker/spec.rs::SCHEMA_HASH`. This
//! harness loads v0.3 externally so we can A/B without touching
//! `runtime.rs` (per the held-out brief: "DON'T modify any codec's
//! `runtime.rs`").
//!
//! Usage:
//!   cargo run --release \
//!     --features "target-zensim analyzer picker" \
//!     --example picker_v0_3_holdout_ab -- \
//!     --bin /tmp/zenwebp_v0.3.bin \
//!     --corpus ~/work/zentrain-corpus/mlp-validate/cid22-val \
//!     --targets 75,80,85,90 \
//!     --out-md ~/work/zen/zenwebp/benchmarks/picker_v0.3_holdout_ab_2026-05-04.md \
//!     --out-tsv /tmp/picker_v0_3_holdout_ab.tsv

#![cfg(all(feature = "target-zensim", feature = "analyzer"))]
#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpredict::{AllowedMask, Model, Predictor, ScoreTransform, argmin::argmin_masked_in_range};

/// Per-feature pre-engineering transform pulled from the v0.3 .bin's
/// (now-stripped) `zentrain.feature_transforms` metadata entry.
/// Trainer-side semantics: transforms are applied at feature-load
/// time, BEFORE engineering. So the engineered axes (size_oh, poly,
/// `zq_norm * f`, icc) consume already-transformed raw features.
///
/// Source: `/tmp/zenwebp_v0.3.feature_transforms.txt` dumped by
/// `/tmp/patch_v0_3_bin.py` at strip-time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RawTransform {
    Identity,
    Log,
    Log1p,
}

impl RawTransform {
    fn apply(self, x: f32) -> f32 {
        match self {
            // Match `zenpredict::FeatureTransform::apply` exactly.
            Self::Identity => x,
            Self::Log => x.ln(),
            Self::Log1p => x.ln_1p(),
        }
    }
}

/// 36 entries in FEAT_COLS order, matching the metadata dump
/// extracted by the patcher.
const RAW_TRANSFORMS_V0_3: &[RawTransform] = &[
    RawTransform::Log1p,    // 0  feat_laplacian_variance_p50
    RawTransform::Log1p,    // 1  feat_laplacian_variance_p75
    RawTransform::Log1p,    // 2  feat_laplacian_variance
    RawTransform::Identity, // 3  feat_quant_survival_y
    RawTransform::Identity, // 4  feat_cb_sharpness
    RawTransform::Log,      // 5  feat_pixel_count
    RawTransform::Identity, // 6  feat_uniformity
    RawTransform::Identity, // 7  feat_distinct_color_bins
    RawTransform::Identity, // 8  feat_cr_sharpness
    RawTransform::Identity, // 9  feat_edge_density
    RawTransform::Identity, // 10 feat_noise_floor_y_p50
    RawTransform::Identity, // 11 feat_luma_histogram_entropy
    RawTransform::Identity, // 12 feat_natural_likelihood       (retired → fed 0.0)
    RawTransform::Identity, // 13 feat_quant_survival_y_p50
    RawTransform::Identity, // 14 feat_noise_floor_uv_p50
    RawTransform::Identity, // 15 feat_aq_map_mean
    RawTransform::Identity, // 16 feat_cr_horiz_sharpness
    RawTransform::Log,      // 17 feat_min_dim
    RawTransform::Log1p,    // 18 feat_edge_slope_stdev
    RawTransform::Log1p,    // 19 feat_laplacian_variance_p90
    RawTransform::Identity, // 20 feat_patch_fraction
    RawTransform::Log,      // 21 feat_max_dim
    RawTransform::Identity, // 22 feat_aspect_min_over_max
    RawTransform::Identity, // 23 feat_aq_map_p75
    RawTransform::Identity, // 24 feat_cb_horiz_sharpness
    RawTransform::Identity, // 25 feat_noise_floor_y_p25
    RawTransform::Identity, // 26 feat_noise_floor_uv
    RawTransform::Identity, // 27 feat_chroma_complexity
    RawTransform::Identity, // 28 feat_quant_survival_y_p75
    RawTransform::Identity, // 29 feat_aq_map_std
    RawTransform::Identity, // 30 feat_gradient_fraction
    RawTransform::Identity, // 31 feat_noise_floor_y_p75
    RawTransform::Identity, // 32 feat_screen_content_likelihood (retired → fed 0.0)
    RawTransform::Identity, // 33 feat_high_freq_energy_ratio
    RawTransform::Identity, // 34 feat_colourfulness
    RawTransform::Identity, // 35 feat_quant_survival_uv
];
use zenwebp::encoder::analysis::{
    ImageContentType, classify_image_type_rgb8, content_type_to_tuning,
};
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, ZensimTarget};

// -----------------------------------------------------------------------
// Schema (mirrors src/encoder/picker/spec.rs FEAT_COLS / runtime.rs
// engineered_features). Kept in sync with the v0.3 manifest.json's
// feat_cols + extra_axes ordering.
// -----------------------------------------------------------------------

const FEAT_COLS: &[&str] = &[
    "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75",
    "feat_laplacian_variance",
    "feat_quant_survival_y",
    "feat_cb_sharpness",
    "feat_pixel_count",
    "feat_uniformity",
    "feat_distinct_color_bins",
    "feat_cr_sharpness",
    "feat_edge_density",
    "feat_noise_floor_y_p50",
    "feat_luma_histogram_entropy",
    "feat_natural_likelihood",
    "feat_quant_survival_y_p50",
    "feat_noise_floor_uv_p50",
    "feat_aq_map_mean",
    "feat_cr_horiz_sharpness",
    "feat_min_dim",
    "feat_edge_slope_stdev",
    "feat_laplacian_variance_p90",
    "feat_patch_fraction",
    "feat_max_dim",
    "feat_aspect_min_over_max",
    "feat_aq_map_p75",
    "feat_cb_horiz_sharpness",
    "feat_noise_floor_y_p25",
    "feat_noise_floor_uv",
    "feat_chroma_complexity",
    "feat_quant_survival_y_p75",
    "feat_aq_map_std",
    "feat_gradient_fraction",
    "feat_noise_floor_y_p75",
    "feat_screen_content_likelihood",
    "feat_high_freq_energy_ratio",
    "feat_colourfulness",
    "feat_quant_survival_uv",
];

/// Per-FEAT_COL `Option<AnalysisFeature>`. `None` slots are features
/// that the trainer's KEEP_FEATURES list expected but that have
/// since been retired from `zenanalyze::feature::AnalysisFeature` —
/// `NaturalLikelihood` and `ScreenContentLikelihood` were retired
/// in favor of more specific composites. The trainer's `load_features`
/// path defaults missing columns to 0.0; we mirror that here so the
/// model sees a sentinel rather than randomly-shaped input.
const ANALYSIS_FEATURES: &[Option<AnalysisFeature>] = &[
    Some(AnalysisFeature::LaplacianVarianceP50),
    Some(AnalysisFeature::LaplacianVarianceP75),
    Some(AnalysisFeature::LaplacianVariance),
    Some(AnalysisFeature::QuantSurvivalY),
    Some(AnalysisFeature::CbSharpness),
    Some(AnalysisFeature::PixelCount),
    Some(AnalysisFeature::Uniformity),
    Some(AnalysisFeature::DistinctColorBins),
    Some(AnalysisFeature::CrSharpness),
    Some(AnalysisFeature::EdgeDensity),
    Some(AnalysisFeature::NoiseFloorYP50),
    Some(AnalysisFeature::LumaHistogramEntropy),
    None, // feat_natural_likelihood — retired in zenanalyze main; fed as 0.0
    Some(AnalysisFeature::QuantSurvivalYP50),
    Some(AnalysisFeature::NoiseFloorUvP50),
    Some(AnalysisFeature::AqMapMean),
    Some(AnalysisFeature::CrHorizSharpness),
    Some(AnalysisFeature::MinDim),
    Some(AnalysisFeature::EdgeSlopeStdev),
    Some(AnalysisFeature::LaplacianVarianceP90),
    Some(AnalysisFeature::PatchFraction),
    Some(AnalysisFeature::MaxDim),
    Some(AnalysisFeature::AspectMinOverMax),
    Some(AnalysisFeature::AqMapP75),
    Some(AnalysisFeature::CbHorizSharpness),
    Some(AnalysisFeature::NoiseFloorYP25),
    Some(AnalysisFeature::NoiseFloorUV),
    Some(AnalysisFeature::ChromaComplexity),
    Some(AnalysisFeature::QuantSurvivalYP75),
    Some(AnalysisFeature::AqMapStd),
    Some(AnalysisFeature::GradientFraction),
    Some(AnalysisFeature::NoiseFloorYP75),
    None, // feat_screen_content_likelihood — retired; fed as 0.0
    Some(AnalysisFeature::HighFreqEnergyRatio),
    Some(AnalysisFeature::Colourfulness),
    Some(AnalysisFeature::QuantSurvivalUv),
];

/// 6-cell taxonomy: (method, segments) lex-sorted on
/// `[(4,1),(4,4),(5,1),(5,4),(6,1),(6,4)]`.
const CELLS: &[(u8, u8)] = &[(4, 1), (4, 4), (5, 1), (5, 4), (6, 1), (6, 4)];
const N_CELLS: usize = 6;

// Output layout: bytes_log[0..6], sns[6..12], filter_strength[12..18],
// filter_sharpness[18..24].
const RANGE_BYTES_LOG: (usize, usize) = (0, 6);
const OFF_SNS: usize = 6;
const OFF_FILTER_STRENGTH: usize = 12;
const OFF_FILTER_SHARPNESS: usize = 18;

// -----------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------

#[derive(Default)]
struct Cli {
    bin: PathBuf,
    corpus: PathBuf,
    targets: Vec<f32>,
    out_md: Option<PathBuf>,
    out_tsv: Option<PathBuf>,
    max_passes: u8,
}

fn parse_args() -> Cli {
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut cli = Cli {
        targets: vec![75.0, 80.0, 85.0, 90.0],
        max_passes: 3,
        ..Default::default()
    };
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        let next = || -> &str { argv.get(i + 1).map(String::as_str).unwrap_or("") };
        match a.as_str() {
            "--bin" => {
                cli.bin = PathBuf::from(next());
                i += 2;
            }
            "--corpus" => {
                cli.corpus = PathBuf::from(next());
                i += 2;
            }
            "--targets" => {
                cli.targets = next()
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().expect("bad --targets value"))
                    .collect();
                i += 2;
            }
            "--out-md" => {
                cli.out_md = Some(PathBuf::from(next()));
                i += 2;
            }
            "--out-tsv" => {
                cli.out_tsv = Some(PathBuf::from(next()));
                i += 2;
            }
            "--max-passes" => {
                cli.max_passes = next().parse().expect("bad --max-passes");
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                usage_and_exit();
            }
        }
    }
    if cli.bin.as_os_str().is_empty() || cli.corpus.as_os_str().is_empty() {
        usage_and_exit();
    }
    cli
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: picker_v0_3_holdout_ab \n\
         \t--bin <v0.3.bin> --corpus <dir> [--targets 75,80,85,90] \n\
         \t[--out-md path] [--out-tsv path] [--max-passes 3]"
    );
    std::process::exit(2);
}

// -----------------------------------------------------------------------
// PNG load
// -----------------------------------------------------------------------

struct DecodedPng {
    rgb: Vec<u8>,
    w: u32,
    h: u32,
}

fn decode_png(path: &Path) -> Option<DecodedPng> {
    let bytes = fs::read(path).ok()?;
    let cur = std::io::Cursor::new(bytes);
    let mut decoder = png::Decoder::new(cur);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let info = reader.info();
    let w = info.width;
    let h = info.height;
    let color = info.color_type;
    let bytes_per_pixel = match color {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Indexed => 3,
    };
    let mut buf = vec![0u8; (w as usize) * (h as usize) * bytes_per_pixel];
    reader.next_frame(&mut buf).ok()?;
    let rgb = match color {
        png::ColorType::Rgb | png::ColorType::Indexed => buf,
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(4) {
                rgb.extend_from_slice(&[px[0], px[1], px[2]]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for &g in &buf {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(2) {
                rgb.extend_from_slice(&[px[0], px[0], px[0]]);
            }
            rgb
        }
    };
    Some(DecodedPng { rgb, w, h })
}

fn list_pngs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "png" || e == "PNG"))
        .collect();
    v.sort();
    v
}

// -----------------------------------------------------------------------
// Feature extraction + engineered vector
// -----------------------------------------------------------------------

fn extract_raw_features_rgb8(rgb: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut feats = FeatureSet::new();
    for slot in ANALYSIS_FEATURES {
        if let Some(f) = slot {
            feats = feats.with(*f);
        }
    }
    let query = AnalysisQuery::new(feats);
    let analysis = zenanalyze::analyze_features_rgb8(rgb, width, height, &query);
    ANALYSIS_FEATURES
        .iter()
        .map(|slot| match slot {
            Some(f) => analysis.get_f32(*f).unwrap_or(0.0),
            None => 0.0,
        })
        .collect()
}

/// Apply per-feature transforms (log/log1p/identity) in-place.
/// Mirrors the trainer's feature-load step which transforms BEFORE
/// engineering, so engineered cross terms `zq_norm * f[i]` consume
/// the already-transformed raw value.
fn transform_raw_in_place(raw: &mut [f32]) {
    debug_assert_eq!(raw.len(), RAW_TRANSFORMS_V0_3.len());
    for (i, x) in raw.iter_mut().enumerate() {
        *x = RAW_TRANSFORMS_V0_3[i].apply(*x);
    }
}

/// Mirror of `src/encoder/picker/runtime.rs::engineered_features`.
/// Layout:
///   raw[N] || size_oh[4] || poly[5] || cross[N] || icc[1]  = 2N + 10
/// where size_oh is the 4-bucket one-hot, poly =
/// `[log_px, log_px^2, target_norm, target_norm^2, target_norm * log_px]`,
/// and `cross[i] = target_norm * raw[i]`.
fn engineered_features(raw_feats: &[f32], width: u32, height: u32, target_zensim: f32) -> Vec<f32> {
    debug_assert_eq!(raw_feats.len(), FEAT_COLS.len());
    let pixels = (width as f32) * (height as f32);
    let log_px = pixels.max(1.0).ln();
    let target_norm = target_zensim / 100.0;

    let size_oh = match (width as u64) * (height as u64) {
        n if n < 64 * 64 => [1.0_f32, 0.0, 0.0, 0.0],
        n if n < 256 * 256 => [0.0, 1.0, 0.0, 0.0],
        n if n < 1024 * 1024 => [0.0, 0.0, 1.0, 0.0],
        _ => [0.0, 0.0, 0.0, 1.0],
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
    out.push(0.0); // icc_bytes — not plumbed (matches runtime.rs behavior).
    out
}

// -----------------------------------------------------------------------
// Picker inference
// -----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct PickerKnobs {
    method: u8,
    segments: u8,
    sns_strength: u8,
    filter_strength: u8,
    filter_sharpness: u8,
    cell_idx: usize,
}

fn pick_knobs(predictor: &mut Predictor<'_>, feats82: &[f32]) -> PickerKnobs {
    // We feed pre-transformed features (transforms applied at raw
    // extraction time, before engineering). The patched .bin has had
    // its `feature_transforms` metadata stripped so `predict` is the
    // safe entry — `predict_transformed` would no-op (transforms
    // metadata absent) and `predict` doesn't risk a future-version
    // schema check failing on us.
    let output = predictor.predict(feats82).expect("predict");
    let mask_arr = [true; N_CELLS];
    let mask = AllowedMask::new(&mask_arr);
    let cell_idx =
        argmin_masked_in_range(output, RANGE_BYTES_LOG, &mask, ScoreTransform::Exp, None)
            .expect("argmin");
    assert!(cell_idx < N_CELLS);
    let (method, segments) = CELLS[cell_idx];
    let sns = clamp_to_u8(output[OFF_SNS + cell_idx], 0.0, 100.0);
    let fs = clamp_to_u8(output[OFF_FILTER_STRENGTH + cell_idx], 0.0, 100.0);
    let sh = clamp_to_u8(output[OFF_FILTER_SHARPNESS + cell_idx], 0.0, 7.0);
    PickerKnobs {
        method,
        segments,
        sns_strength: sns,
        filter_strength: fs,
        filter_sharpness: sh,
        cell_idx,
    }
}

fn clamp_to_u8(v: f32, lo: f32, hi: f32) -> u8 {
    let clamped = if v.is_nan() { lo } else { v.max(lo).min(hi) };
    clamped.round() as u8
}

// -----------------------------------------------------------------------
// Bucket arm
// -----------------------------------------------------------------------

fn bucket_knobs(rgb: &[u8], width: u32, height: u32) -> (ImageContentType, u8, u8, u8, u8, u8) {
    let ct = classify_image_type_rgb8(rgb, width, height);
    let (sns, fs, sh, segs) = content_type_to_tuning(ct);
    // Bucket-table path always pins method = 4 (default user method
    // when invoking Preset::Auto without overrides; see vp8/mod.rs
    // resolve_auto_preset_via_analyzer).
    (ct, 4, segs, sns, fs, sh)
}

// -----------------------------------------------------------------------
// Encode arms (each rides target_zensim closed loop)
// -----------------------------------------------------------------------

struct EncodeOutcome {
    bytes: usize,
    achieved: f32,
    passes: u8,
    targets_met: bool,
    elapsed_ms: f64,
}

fn encode_with_knobs(
    rgb: &[u8],
    w: u32,
    h: u32,
    method: u8,
    segments: u8,
    sns: u8,
    fs: u8,
    sh: u8,
    target: f32,
    max_passes: u8,
) -> Option<EncodeOutcome> {
    let cfg = LossyConfig::new()
        .with_method(method)
        .with_segments(segments)
        .with_sns_strength(sns)
        .with_filter_strength(fs)
        .with_filter_sharpness(sh)
        .with_target_zensim(
            ZensimTarget::new(target)
                .with_max_overshoot(Some(1.5))
                .with_max_passes(max_passes),
        );
    let t0 = Instant::now();
    let r = EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, w, h).encode_with_metrics();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match r {
        Ok((bytes, m)) => Some(EncodeOutcome {
            bytes: bytes.len(),
            achieved: m.achieved_score,
            passes: m.passes_used,
            targets_met: m.targets_met,
            elapsed_ms,
        }),
        Err(e) => {
            eprintln!("ERR encode: {e:?}");
            None
        }
    }
}

// -----------------------------------------------------------------------
// Aggregate row + report
// -----------------------------------------------------------------------

#[derive(Default, Clone)]
struct CellStats {
    n: u32,
    bytes_sum: u64,
    achieved_sum: f64,
}

impl CellStats {
    fn add(&mut self, bytes: usize, achieved: f32) {
        self.n += 1;
        self.bytes_sum += bytes as u64;
        self.achieved_sum += f64::from(achieved);
    }
    fn mean_achieved(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.achieved_sum / f64::from(self.n)
        }
    }
}

fn fmt_bytes(b: u64) -> String {
    if b > 1_000_000 {
        format!("{:.2} MB", b as f64 / 1_000_000.0)
    } else {
        format!("{} B", b)
    }
}

fn main() {
    let cli = parse_args();
    eprintln!("v0.3 picker A/B held-out:");
    eprintln!("  bin: {}", cli.bin.display());
    eprintln!("  corpus: {}", cli.corpus.display());
    eprintln!("  targets: {:?}", cli.targets);

    // Load .bin externally (no schema check; we know it diverges
    // from the in-runtime const).
    let bin_bytes = fs::read(&cli.bin).expect("read --bin");
    let model = Model::from_bytes(&bin_bytes).expect("Model::from_bytes (v0.3)");
    eprintln!(
        "  model: n_inputs={}, n_outputs={}, schema_hash=0x{:016x}",
        model.n_inputs(),
        model.n_outputs(),
        model.schema_hash()
    );
    assert_eq!(model.n_inputs(), 82, "expected 82 inputs");
    assert_eq!(model.n_outputs(), 24, "expected 24 outputs");

    let mut predictor = Predictor::new(model);

    let images_paths = list_pngs(&cli.corpus);
    eprintln!("  found {} PNGs", images_paths.len());
    let mut images: Vec<(PathBuf, DecodedPng)> = Vec::new();
    for p in images_paths {
        if let Some(d) = decode_png(&p) {
            images.push((p, d));
        }
    }
    eprintln!("  loaded {} images", images.len());

    let mut tsv_writer: Option<fs::File> = cli.out_tsv.as_ref().map(|p| {
        let mut f = fs::File::create(p).expect("open --out-tsv");
        writeln!(
            f,
            "arm\timage\twidth\theight\ttarget\tcontent_type\tcell_idx\tmethod\tsegments\tsns\tfilter_strength\tfilter_sharpness\tbytes\tachieved\tpasses\tmet\tencode_ms"
        )
        .unwrap();
        f
    });

    let t0 = Instant::now();

    // Per-target cell stats per arm.
    let mut picker_per_target: Vec<CellStats> = vec![CellStats::default(); cli.targets.len()];
    let mut bucket_per_target: Vec<CellStats> = vec![CellStats::default(); cli.targets.len()];
    let mut picker_total = CellStats::default();
    let mut bucket_total = CellStats::default();
    let mut picker_wins_per_target: Vec<u32> = vec![0; cli.targets.len()];
    let mut paired_count_per_target: Vec<u32> = vec![0; cli.targets.len()];

    for (idx, (path, d)) in images.iter().enumerate() {
        if idx % 4 == 0 {
            // Refresh workongoing every few images (~< 2 min).
            let _ = std::fs::write(
                "/home/lilith/work/zen/zenwebp/.workongoing",
                format!(
                    "{} claude-session-zenwebp-picker-ab v0.3-encoding {}/{}\n",
                    chrono_now_iso(),
                    idx,
                    images.len()
                ),
            );
        }
        let mut raw_feats = extract_raw_features_rgb8(&d.rgb, d.w, d.h);
        transform_raw_in_place(&mut raw_feats);
        let (ct, b_method, b_segs, b_sns, b_fs, b_sh) = bucket_knobs(&d.rgb, d.w, d.h);

        for (ti, &target) in cli.targets.iter().enumerate() {
            // Picker arm: re-engineer features per target (target_norm differs).
            let feats82 = engineered_features(&raw_feats, d.w, d.h, target);
            let pk = pick_knobs(&mut predictor, &feats82);

            let p = encode_with_knobs(
                &d.rgb,
                d.w,
                d.h,
                pk.method,
                pk.segments,
                pk.sns_strength,
                pk.filter_strength,
                pk.filter_sharpness,
                target,
                cli.max_passes,
            );
            let b = encode_with_knobs(
                &d.rgb,
                d.w,
                d.h,
                b_method,
                b_segs,
                b_sns,
                b_fs,
                b_sh,
                target,
                cli.max_passes,
            );

            if let Some(po) = &p {
                picker_per_target[ti].add(po.bytes, po.achieved);
                picker_total.add(po.bytes, po.achieved);
                if let Some(f) = tsv_writer.as_mut() {
                    writeln!(
                        f,
                        "picker\t{}\t{}\t{}\t{}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{:.2}",
                        path.display(),
                        d.w,
                        d.h,
                        target,
                        ct,
                        pk.cell_idx,
                        pk.method,
                        pk.segments,
                        pk.sns_strength,
                        pk.filter_strength,
                        pk.filter_sharpness,
                        po.bytes,
                        po.achieved,
                        po.passes,
                        po.targets_met as u8,
                        po.elapsed_ms,
                    )
                    .unwrap();
                }
            }
            if let Some(bo) = &b {
                bucket_per_target[ti].add(bo.bytes, bo.achieved);
                bucket_total.add(bo.bytes, bo.achieved);
                if let Some(f) = tsv_writer.as_mut() {
                    writeln!(
                        f,
                        "bucket\t{}\t{}\t{}\t{}\t{:?}\t-\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{:.2}",
                        path.display(),
                        d.w,
                        d.h,
                        target,
                        ct,
                        b_method,
                        b_segs,
                        b_sns,
                        b_fs,
                        b_sh,
                        bo.bytes,
                        bo.achieved,
                        bo.passes,
                        bo.targets_met as u8,
                        bo.elapsed_ms,
                    )
                    .unwrap();
                }
            }
            if let (Some(po), Some(bo)) = (&p, &b) {
                paired_count_per_target[ti] += 1;
                if po.bytes <= bo.bytes {
                    picker_wins_per_target[ti] += 1;
                }
            }
        }
        if let Some(f) = tsv_writer.as_mut() {
            f.flush().ok();
        }
    }

    if let Some(mut f) = tsv_writer {
        f.flush().ok();
    }

    // ---------- Emit Markdown report ----------
    let elapsed_total = t0.elapsed().as_secs_f64();
    let mut md = String::new();
    md.push_str("# Picker v0.3 — held-out re-encode A/B (zenwebp)\n\n");
    md.push_str(&format!(
        "* Date: {}\n* Corpus: `{}` ({} images)\n* Picker bin: `{}` (n_inputs={}, n_outputs={}, schema_hash=`0x{:016x}`)\n* Targets: {:?}\n* Encoder closed-loop: `target_zensim` with `max_passes={}`, `max_overshoot=1.5`\n* Wall: {:.1} s\n\n",
        chrono_now_iso(),
        cli.corpus.display(),
        images.len(),
        cli.bin.display(),
        82,
        24,
        0x139d73665fb030c7_u64,
        cli.targets,
        cli.max_passes,
        elapsed_total,
    ));

    md.push_str("## Per-target table\n\n");
    md.push_str("| target | n | bytes_picker | bytes_bucket | Δ% (picker − bucket) | win_rate | achieved_picker | achieved_bucket |\n");
    md.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (ti, &target) in cli.targets.iter().enumerate() {
        let p = &picker_per_target[ti];
        let b = &bucket_per_target[ti];
        let bp = p.bytes_sum as f64;
        let bb = b.bytes_sum as f64;
        let delta_pct = if bb > 0.0 {
            (bp - bb) / bb * 100.0
        } else {
            0.0
        };
        let wins = picker_wins_per_target[ti];
        let paired = paired_count_per_target[ti].max(1);
        let win_rate = wins as f64 / paired as f64 * 100.0;
        md.push_str(&format!(
            "| {:.0} | {} | {} | {} | {:+.2}% | {:.0}% ({}/{}) | {:.2} | {:.2} |\n",
            target,
            p.n,
            fmt_bytes(p.bytes_sum),
            fmt_bytes(b.bytes_sum),
            delta_pct,
            win_rate,
            wins,
            paired,
            p.mean_achieved(),
            b.mean_achieved(),
        ));
    }

    let bp = picker_total.bytes_sum as f64;
    let bb = bucket_total.bytes_sum as f64;
    let total_delta_pct = if bb > 0.0 {
        (bp - bb) / bb * 100.0
    } else {
        0.0
    };
    md.push_str(&format!(
        "\n## Total\n\n* Picker total bytes: **{}** (mean achieved zensim {:.2})\n* Bucket total bytes: **{}** (mean achieved zensim {:.2})\n* Δ bytes: **{:+.2}%** (picker − bucket)\n* Δ achieved zensim: **{:+.3}** pp\n",
        fmt_bytes(picker_total.bytes_sum),
        picker_total.mean_achieved(),
        fmt_bytes(bucket_total.bytes_sum),
        bucket_total.mean_achieved(),
        total_delta_pct,
        picker_total.mean_achieved() - bucket_total.mean_achieved(),
    ));

    let achieved_gap = (picker_total.mean_achieved() - bucket_total.mean_achieved()).abs();
    let verdict = if total_delta_pct <= 0.0 && achieved_gap <= 0.5 {
        "**SHIP**"
    } else if total_delta_pct < 0.0 && achieved_gap > 0.5 {
        "HOLD (achieved-zensim gap > 0.5pp invalidates byte comparison)"
    } else {
        "**HOLD**"
    };
    md.push_str(&format!(
        "\n## Verdict\n\n{}\n\n* Threshold: SHIP if total bytes (picker) ≤ total bytes (bucket) within ±0.5pp achieved-zensim parity.\n",
        verdict,
    ));

    md.push_str("\n## Method notes\n\n");
    md.push_str("* Picker arm: extracted 36-feature zenanalyze vector in `FEAT_COLS` order, applied per-feature transforms (log/log1p/identity per the v0.3 `.bin`'s metadata; trainer applies these BEFORE engineering so cross terms consume transformed values), built the engineered 82-vec via `feats[36] || size_oh[4] || poly[5] || zq*feats[36] || icc[1]` (mirror of `src/encoder/picker/runtime.rs::engineered_features`), ran `Predictor::predict` against the externally-loaded v0.3 `.bin` (with its malformed `feature_transforms` metadata stripped — the original 36-entry list mismatched the 82-input model and would have hard-failed `parse_feature_transforms`), decoded the bytes_log argmin → cell index → `(method, segments)` from the lex-sorted (4,1)..(6,4) taxonomy, and read `sns_strength`, `filter_strength`, `filter_sharpness` from the per-cell scalar heads (clamped to [0,100], [0,100], [0,7] respectively).\n");
    md.push_str("* Bucket arm: classified RGB via `zenwebp::encoder::analysis::classify_image_type_rgb8` and called `content_type_to_tuning` for `(sns, filter_strength, filter_sharpness, segments)`. Method pinned to `4` (the bucket-table path's default).\n");
    md.push_str("* Both arms encode through `LossyConfig::with_target_zensim` so the iteration loop adapts global VP8 quality to land in the target band; reported bytes / achieved scores are the closed-loop best.\n");
    md.push_str("* FEAT_COLS source: hardcoded from `src/encoder/picker/spec.rs::FEAT_COLS` (36 entries, matches `zenwebp_picker_v0.3_2026-05-04.manifest.json::feat_cols` exactly). Engineered axes (46 = size_oh[4] + poly[5] + zq×feats[36] + icc[1]) match `manifest.json::extra_axes` order.\n");

    if let Some(p) = &cli.out_md {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(p, &md).expect("write --out-md");
        eprintln!("wrote markdown report: {}", p.display());
    } else {
        println!("{md}");
    }

    eprintln!(
        "summary: picker {} ({:.2}) vs bucket {} ({:.2}) delta {:+.2}% over {} images × {} targets in {:.1}s",
        fmt_bytes(picker_total.bytes_sum),
        picker_total.mean_achieved(),
        fmt_bytes(bucket_total.bytes_sum),
        bucket_total.mean_achieved(),
        total_delta_pct,
        images.len(),
        cli.targets.len(),
        elapsed_total,
    );
}

// Tiny ISO-8601 UTC formatter; avoids pulling chrono just for a
// timestamp stamp at the top of the report.
fn chrono_now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Convert epoch seconds → YYYY-MM-DDTHH:MM:SSZ via libc-free
    // rfc3339 dance.
    let days = (secs / 86_400) as i64;
    let mut hms = secs % 86_400;
    let h = hms / 3600;
    hms %= 3600;
    let m = hms / 60;
    let s = hms % 60;
    // From civil days → (Y, m, d) using the algorithm from
    // Howard Hinnant (no_std-friendly).
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m_civil = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m_civil <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m_civil:02}-{d:02}T{h:02}:{mm:02}:{ss:02}Z",
        m_civil = m_civil,
        d = d,
        h = h,
        mm = m,
        ss = s,
    )
}
