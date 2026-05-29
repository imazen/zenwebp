//! `zwr-calibrate` — corpus sweep harness for zenwebp-recompress.
//!
//! Two modes:
//!
//! 1. **`--sources DIR`** — walks DIR for `*.webp`. For each WebP and each
//!    `(target_zensim_a × strategy)` cell, dispatches the strategy via the
//!    expert API, measures `(size_ratio, zensim_a_vs_source)`. The
//!    cumulative-vs-reference column is empty (we don't have the original).
//!
//! 2. **`--refs DIR --q-grid 20:95:5`** — walks DIR for `*.webp` (must be
//!    lossless, used as the reference) or `*.png`. For each reference and
//!    each source quality in the grid, in-memory re-encodes the RGBA at
//!    that quality to produce a synthetic source. For each synthetic
//!    source and each `(target_zensim_a × strategy)` cell, dispatches the
//!    strategy, measures `(size_ratio, zensim_a_vs_source,
//!    zensim_a_vs_reference)`. This is the calibration-quality data.

use clap::Parser;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use walkdir::WalkDir;
use zenwebp::PixelLayout;
use zenwebp::encoder::{EncodeRequest, LossyConfig};
use zenwebp::oneshot::decode_rgba;
use zenwebp_recompress::expert::{
    SourceAnalysis, analyze_source, run_deblock_reencode, run_lossless_reencode,
    run_lossless_remux, run_reencode, run_reencode_at_q, score_recompression, score_rgba,
};
use zenwebp_recompress::{Budget, RecompressOptions};

#[derive(Debug, Parser)]
#[command(
    name = "zwr-calibrate",
    about = "Per-strategy corpus sweep for zenwebp-recompress",
    version
)]
struct Cli {
    /// Path to a directory of WebP sources (recurses). Use this for a
    /// generation-loss-only sweep on existing WebPs.
    #[arg(long, conflicts_with = "refs")]
    sources: Option<PathBuf>,

    /// Path to a directory of lossless WebP (or PNG) reference files.
    /// Each reference is decoded to RGBA, encoded at each source-q in
    /// `--q-grid`, then swept. Provides cumulative zensim-A measurements.
    #[arg(long)]
    refs: Option<PathBuf>,

    /// Source quality grid for `--refs` mode, e.g. `20:95:5`.
    #[arg(long, default_value = "20:95:5", requires = "refs")]
    q_grid: String,

    /// Encoder method for synthetic source encodes (`--refs` mode). 0=fast,
    /// 6=slow. Default 4 mirrors libwebp's default.
    #[arg(long, default_value_t = 4)]
    method: u8,

    /// Loop-filter strength for synthetic source encodes (`--refs` mode),
    /// 0..100. `None` = encoder default. Set `0` to produce weakly-filtered
    /// sources for testing the deblock strategy (see
    /// `benchmarks/deblock_experiment_2026-05-28.md`).
    #[arg(long)]
    source_filter: Option<u8>,

    /// Target zensim-A grid, e.g. `0:100:5` (start:end:step). Used by the
    /// calibration-mediated strategy sweep (`--strategies`).
    #[arg(long, default_value = "50:95:5")]
    targets: String,

    /// RAW reencode-q grid, e.g. `20:100:2`. When set (with `--refs`), the
    /// harness sweeps `run_reencode_at_q` at each explicit q instead of the
    /// calibration-mediated `--targets`/`--strategies` Reencode — this is the
    /// non-circular data used to BUILD the calibration tables. Each row
    /// records `reencode_q`, `size_ratio`, gen-loss and cumulative zensim-A.
    /// `LosslessRemux` and `vp8l` are still measured once per source.
    #[arg(long, requires = "refs")]
    reencode_qs: Option<String>,

    /// Strategies to sweep. Comma-separated.
    /// Supported: `remux,reencode,deblock,vp8l`.
    #[arg(long, default_value = "remux,reencode,deblock,vp8l")]
    strategies: String,

    /// Output CSV path.
    #[arg(long)]
    output: PathBuf,

    /// Cap on number of input files (sorted walk order).
    #[arg(long)]
    max_files: Option<usize>,
}

#[derive(Debug, serde::Serialize, Clone)]
struct Row {
    input_path: String,
    input_bytes: u64,
    width: u32,
    height: u32,
    /// Synthetic source quality. For `--sources` mode this is the
    /// detect-derived estimate; for `--refs` mode it's the exact q used
    /// to make the synthetic source.
    source_q: f32,
    source_quantizer_index: u32,
    encoder_family: String,
    source_kind: String,
    has_alpha: bool,
    target_zensim_a: f32,
    /// Explicit reencode quality for raw-sweep rows (`--reencode-qs`).
    /// `NaN` for calibration-mediated / remux / vp8l rows.
    reencode_q: f32,
    strategy: String,
    output_bytes: u64,
    size_ratio: f32,
    /// Generation-loss zensim-A: recompressed vs source.
    measured_zensim_a_vs_source: f32,
    /// Cumulative zensim-A: recompressed vs original reference. Empty in
    /// `--sources` mode (no reference available).
    measured_zensim_a_vs_reference: f32,
    result_kind: String,
    error: String,
}

fn parse_range(s: &str) -> anyhow::Result<Vec<f32>> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        anyhow::bail!("range must be start:end:step (got {s:?})");
    }
    let start: f32 = parts[0].parse()?;
    let end: f32 = parts[1].parse()?;
    let step: f32 = parts[2].parse()?;
    if step <= 0.0 {
        anyhow::bail!("step must be > 0");
    }
    let mut out = Vec::new();
    let mut x = start;
    while x <= end + 1e-6 {
        out.push(x);
        x += step;
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
enum StrategyName {
    Remux,
    Reencode,
    Deblock,
    Vp8l,
}

impl StrategyName {
    fn parse(s: &str) -> Option<Self> {
        Some(match s.trim() {
            "remux" => Self::Remux,
            "reencode" => Self::Reencode,
            "deblock" => Self::Deblock,
            "vp8l" => Self::Vp8l,
            _ => return None,
        })
    }
    fn label(self) -> &'static str {
        match self {
            Self::Remux => "remux",
            Self::Reencode => "reencode",
            Self::Deblock => "deblock",
            Self::Vp8l => "vp8l",
        }
    }
}

fn dispatch_strategy(
    s: StrategyName,
    bytes: &[u8],
    analysis: &SourceAnalysis,
    target: f32,
) -> Result<Vec<u8>, String> {
    let opts = RecompressOptions {
        target_zensim_a: target,
        budget: Budget::OneShot,
        ..Default::default()
    };
    let res = match s {
        StrategyName::Remux => run_lossless_remux(bytes, analysis),
        StrategyName::Reencode => run_reencode(bytes, analysis, &opts),
        StrategyName::Deblock => run_deblock_reencode(bytes, analysis, &opts),
        StrategyName::Vp8l => run_lossless_reencode(bytes, analysis),
    };
    res.map_err(|e| format!("{e:?}"))
}

fn sweep_one_source(
    label: &str,
    bytes: &[u8],
    analysis: &SourceAnalysis,
    reference_rgba: Option<(&[u8], u32, u32)>,
    targets: &[f32],
    strategies: &[StrategyName],
) -> Vec<Row> {
    let mut rows = Vec::new();
    for &target in targets {
        for &s in strategies {
            let row = match dispatch_strategy(s, bytes, analysis, target) {
                Ok(out) => {
                    let ratio = out.len() as f32 / bytes.len().max(1) as f32;
                    let gen_loss = score_recompression(bytes, &out).unwrap_or(f32::NAN);
                    let cumulative = if let Some((ref_rgba, w, h)) = reference_rgba {
                        // Decode the recompressed output to RGBA, score
                        // vs the canonical reference RGBA.
                        match decode_rgba(&out) {
                            Ok((out_rgba, ow, oh)) if ow == w && oh == h => {
                                score_rgba(ref_rgba, &out_rgba, w, h).unwrap_or(f32::NAN)
                            }
                            Ok(_) => f32::NAN, // dimension mismatch
                            Err(_) => f32::NAN,
                        }
                    } else {
                        f32::NAN
                    };
                    Row {
                        input_path: label.to_string(),
                        input_bytes: bytes.len() as u64,
                        width: analysis.width,
                        height: analysis.height,
                        source_q: analysis.source_q,
                        source_quantizer_index: analysis.vp8_quantizer_index as u32,
                        encoder_family: format!("{:?}", analysis.encoder_family),
                        source_kind: format!("{:?}", analysis.kind),
                        has_alpha: analysis.has_alpha,
                        target_zensim_a: target,
                        reencode_q: f32::NAN,
                        strategy: s.label().to_string(),
                        output_bytes: out.len() as u64,
                        size_ratio: ratio,
                        measured_zensim_a_vs_source: gen_loss,
                        measured_zensim_a_vs_reference: cumulative,
                        result_kind: "ok".to_string(),
                        error: String::new(),
                    }
                }
                Err(e) => Row {
                    input_path: label.to_string(),
                    input_bytes: bytes.len() as u64,
                    width: analysis.width,
                    height: analysis.height,
                    source_q: analysis.source_q,
                    source_quantizer_index: analysis.vp8_quantizer_index as u32,
                    encoder_family: format!("{:?}", analysis.encoder_family),
                    source_kind: format!("{:?}", analysis.kind),
                    has_alpha: analysis.has_alpha,
                    target_zensim_a: target,
                    reencode_q: f32::NAN,
                    strategy: s.label().to_string(),
                    output_bytes: 0,
                    size_ratio: f32::NAN,
                    measured_zensim_a_vs_source: f32::NAN,
                    measured_zensim_a_vs_reference: f32::NAN,
                    result_kind: "Error".to_string(),
                    error: e,
                },
            };
            rows.push(row);
        }
    }
    rows
}

/// Raw, non-circular calibration sweep for one synthetic source: sweep
/// `run_reencode_at_q` at each explicit `reencode_q`, plus measure
/// `LosslessRemux` and `vp8l` once. Records cumulative zensim-A vs the clean
/// reference (the quantity the calibration must predict) and the size ratio.
fn sweep_one_source_raw(
    label: &str,
    bytes: &[u8],
    analysis: &SourceAnalysis,
    reference_rgba: (&[u8], u32, u32),
    reencode_qs: &[f32],
) -> Vec<Row> {
    let (ref_rgba, w, h) = reference_rgba;
    let cumulative = |out: &[u8]| -> f32 {
        match decode_rgba(out) {
            Ok((out_rgba, ow, oh)) if ow == w && oh == h => {
                score_rgba(ref_rgba, &out_rgba, w, h).unwrap_or(f32::NAN)
            }
            _ => f32::NAN,
        }
    };
    let base = |strategy: &str, reencode_q: f32, out: Option<&[u8]>, err: String| -> Row {
        let (output_bytes, size_ratio, gen_loss, cum, kind) = match out {
            Some(o) => (
                o.len() as u64,
                o.len() as f32 / bytes.len().max(1) as f32,
                score_recompression(bytes, o).unwrap_or(f32::NAN),
                cumulative(o),
                "ok",
            ),
            None => (0, f32::NAN, f32::NAN, f32::NAN, "Error"),
        };
        Row {
            input_path: label.to_string(),
            input_bytes: bytes.len() as u64,
            width: analysis.width,
            height: analysis.height,
            source_q: analysis.source_q,
            source_quantizer_index: analysis.vp8_quantizer_index as u32,
            encoder_family: format!("{:?}", analysis.encoder_family),
            source_kind: format!("{:?}", analysis.kind),
            has_alpha: analysis.has_alpha,
            target_zensim_a: f32::NAN,
            reencode_q,
            strategy: strategy.to_string(),
            output_bytes,
            size_ratio,
            measured_zensim_a_vs_source: gen_loss,
            measured_zensim_a_vs_reference: cum,
            result_kind: kind.to_string(),
            error: err,
        }
    };

    let mut rows = Vec::with_capacity(reencode_qs.len() + 2);
    for &q in reencode_qs {
        let qi = q.clamp(1.0, 100.0).round() as u8;
        match run_reencode_at_q(bytes, analysis, qi) {
            Ok(out) => rows.push(base("reencode", q, Some(&out), String::new())),
            Err(e) => rows.push(base("reencode", q, None, format!("{e:?}"))),
        }
    }
    // Lossless paths are q-independent: measure once.
    match run_lossless_remux(bytes, analysis) {
        Ok(out) => rows.push(base("remux", f32::NAN, Some(&out), String::new())),
        Err(e) => rows.push(base("remux", f32::NAN, None, format!("{e:?}"))),
    }
    match run_lossless_reencode(bytes, analysis) {
        Ok(out) => rows.push(base("vp8l", f32::NAN, Some(&out), String::new())),
        Err(e) => rows.push(base("vp8l", f32::NAN, None, format!("{e:?}"))),
    }
    rows
}

fn sweep_existing_file(path: &Path, targets: &[f32], strategies: &[StrategyName]) -> Vec<Row> {
    let mut rows = Vec::new();
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            rows.push(error_row(
                &path.display().to_string(),
                0,
                0,
                0,
                "read",
                &e.to_string(),
            ));
            return rows;
        }
    };
    let analysis = match analyze_source(&bytes) {
        Ok(a) => a,
        Err(e) => {
            rows.push(error_row(
                &path.display().to_string(),
                bytes.len() as u64,
                0,
                0,
                "analyze",
                &format!("{e:?}"),
            ));
            return rows;
        }
    };
    let label = path.display().to_string();
    rows.extend(sweep_one_source(
        &label, &bytes, &analysis, None, targets, strategies,
    ));
    rows
}

#[allow(clippy::too_many_arguments)]
fn sweep_one_reference(
    path: &Path,
    q_grid: &[f32],
    method: u8,
    source_filter: Option<u8>,
    targets: &[f32],
    strategies: &[StrategyName],
    reencode_qs: Option<&[f32]>,
) -> Vec<Row> {
    let mut rows = Vec::new();
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            rows.push(error_row(
                &path.display().to_string(),
                0,
                0,
                0,
                "read",
                &e.to_string(),
            ));
            return rows;
        }
    };
    let (ref_rgba, w, h) = match decode_rgba(&bytes) {
        Ok(t) => t,
        Err(e) => {
            rows.push(error_row(
                &path.display().to_string(),
                bytes.len() as u64,
                0,
                0,
                "decode_ref",
                &format!("{e:?}"),
            ));
            return rows;
        }
    };

    for &q in q_grid {
        let mut cfg = LossyConfig::new().with_quality(q).with_method(method);
        if let Some(f) = source_filter {
            cfg = cfg.with_filter_strength(f);
        }
        let synthetic =
            match EncodeRequest::lossy(&cfg, &ref_rgba, PixelLayout::Rgba8, w, h).encode() {
                Ok(b) => b,
                Err(e) => {
                    rows.push(error_row(
                        &path.display().to_string(),
                        bytes.len() as u64,
                        w,
                        h,
                        "encode_synth",
                        &format!("{e:?}"),
                    ));
                    continue;
                }
            };
        let analysis = match analyze_source(&synthetic) {
            Ok(a) => a,
            Err(e) => {
                rows.push(error_row(
                    &path.display().to_string(),
                    synthetic.len() as u64,
                    w,
                    h,
                    "analyze_synth",
                    &format!("{e:?}"),
                ));
                continue;
            }
        };
        let label = format!("{}_synth_q{}", path.display(), q.round() as u32);
        if let Some(rqs) = reencode_qs {
            rows.extend(sweep_one_source_raw(
                &label,
                &synthetic,
                &analysis,
                (&ref_rgba, w, h),
                rqs,
            ));
        } else {
            rows.extend(sweep_one_source(
                &label,
                &synthetic,
                &analysis,
                Some((&ref_rgba, w, h)),
                targets,
                strategies,
            ));
        }
    }
    rows
}

fn error_row(
    label: &str,
    input_bytes: u64,
    width: u32,
    height: u32,
    stage: &str,
    err: &str,
) -> Row {
    Row {
        input_path: label.to_string(),
        input_bytes,
        width,
        height,
        source_q: f32::NAN,
        source_quantizer_index: 0,
        encoder_family: String::new(),
        source_kind: String::new(),
        has_alpha: false,
        target_zensim_a: f32::NAN,
        reencode_q: f32::NAN,
        strategy: stage.to_string(),
        output_bytes: 0,
        size_ratio: f32::NAN,
        measured_zensim_a_vs_source: f32::NAN,
        measured_zensim_a_vs_reference: f32::NAN,
        result_kind: "Error".to_string(),
        error: err.to_string(),
    }
}

fn collect_webps(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("webp") || s.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .map(|e| e.into_path())
        .collect();
    paths.sort();
    paths
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let targets = parse_range(&cli.targets)?;
    let strategies: Vec<StrategyName> = cli
        .strategies
        .split(',')
        .map(|s| StrategyName::parse(s).ok_or_else(|| anyhow::anyhow!("unknown strategy {s:?}")))
        .collect::<anyhow::Result<_>>()?;

    let writer = Mutex::new(csv::Writer::from_path(&cli.output)?);

    let reencode_qs = cli.reencode_qs.as_deref().map(parse_range).transpose()?;

    if let Some(refs_dir) = &cli.refs {
        let q_grid = parse_range(&cli.q_grid)?;
        let mut paths = collect_webps(refs_dir);
        if let Some(cap) = cli.max_files {
            paths.truncate(cap);
        }
        let cells_per = match &reencode_qs {
            Some(rqs) => rqs.len() + 2, // reencode-q grid + remux + vp8l
            None => targets.len() * strategies.len(),
        };
        eprintln!(
            "zwr-calibrate: {} refs, {} q-source levels, {} = {} cells{}",
            paths.len(),
            q_grid.len(),
            match &reencode_qs {
                Some(rqs) => format!("{} raw reencode-qs (+remux+vp8l)", rqs.len()),
                None => format!(
                    "{} targets × {} strategies",
                    targets.len(),
                    strategies.len()
                ),
            },
            paths.len() * q_grid.len() * cells_per,
            if reencode_qs.is_some() {
                " [RAW calibration-building mode]"
            } else {
                ""
            },
        );
        paths.par_iter().for_each(|p| {
            let rows = sweep_one_reference(
                p,
                &q_grid,
                cli.method,
                cli.source_filter,
                &targets,
                &strategies,
                reencode_qs.as_deref(),
            );
            let mut w = writer.lock().unwrap();
            for r in rows {
                let _ = w.serialize(&r);
            }
            let _ = w.flush();
        });
    } else if let Some(sources_dir) = &cli.sources {
        let mut paths = collect_webps(sources_dir);
        if let Some(cap) = cli.max_files {
            paths.truncate(cap);
        }
        eprintln!(
            "zwr-calibrate: {} files, {} targets, {} strategies = {} cells",
            paths.len(),
            targets.len(),
            strategies.len(),
            paths.len() * targets.len() * strategies.len()
        );
        paths.par_iter().for_each(|p| {
            let rows = sweep_existing_file(p, &targets, &strategies);
            let mut w = writer.lock().unwrap();
            for r in rows {
                let _ = w.serialize(&r);
            }
            let _ = w.flush();
        });
    } else {
        anyhow::bail!("must provide --sources OR --refs");
    }

    eprintln!("done -> {}", cli.output.display());
    Ok(())
}
