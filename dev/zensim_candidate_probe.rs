//! Artifact output for the existing WebP phase-3 trace binary.
//! Opaque packed sRGB8 only; PNG color metadata is not converted.
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Read},
    path::{Path, PathBuf},
    time::Instant,
};
use zensim_candidate::{BakeScorer, RgbSlice};
use zenwebp::{
    AblationToggles, EncodeRequest, LossyConfig, PixelLayout, ZensimTarget, set_ablation_toggles,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn file_sha(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = f.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn load_png(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let mut reader = png::Decoder::new(BufReader::new(File::open(path)?)).read_info()?;
    if reader.info().bit_depth != png::BitDepth::Eight
        || !matches!(
            reader.info().color_type,
            png::ColorType::Rgb | png::ColorType::Grayscale
        )
    {
        return Err("source must be opaque RGB/grayscale 8-bit PNG".into());
    }
    let mut pixels = vec![0; reader.output_buffer_size().ok_or("PNG buffer overflow")?];
    let info = reader.next_frame(&mut pixels)?;
    pixels.truncate(info.buffer_size());
    if info.color_type == png::ColorType::Grayscale {
        pixels = pixels.into_iter().flat_map(|v| [v; 3]).collect();
    }
    Ok((pixels, info.width, info.height))
}
fn write_png(path: &Path, pixels: &[u8], width: u32, height: u32) -> Result<()> {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    input: &Path,
    out: &Path,
    target: f32,
    overshoot: f32,
    passes: u8,
    method: u8,
    fine_gap: Option<f32>,
) -> Result<()> {
    if out.exists() {
        return Err("output must be fresh".into());
    }
    if fine_gap.is_some_and(|v| !v.is_finite() || v < 0.) {
        return Err("fine gap must be finite and nonnegative".into());
    }
    let input = input.canonicalize()?;
    let bake =
        PathBuf::from(std::env::var_os("ZENWEBP_ZQ_BAKE").ok_or("ZENWEBP_ZQ_BAKE required")?);
    let bake_bytes = std::fs::read(&bake)?;
    let model = zenpredict_serving::Model::from_bytes(&bake_bytes)?;
    let mut scorer = BakeScorer::new(&model)?;
    let (pixels, width, height) = load_png(&input)?;
    let source = RgbSlice::new(pixels.as_chunks::<3>().0, width as usize, height as usize);
    let config = LossyConfig::new()
        .with_method(method)
        .with_segments(4)
        .with_target_zensim(
            ZensimTarget::new(target)
                .with_max_overshoot(Some(overshoot))
                .with_max_undershoot_ship(Some(0.))
                .with_max_passes(passes),
        );
    config.validate()?;
    set_ablation_toggles(AblationToggles {
        phase3_fine_gap: fine_gap,
        trace_phase3: true,
        no_multi_pass_stats: true,
        ..Default::default()
    });
    let started = Instant::now();
    let (bytes, metrics) = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, width, height)
        .encode_with_metrics()?;
    let loop_seconds = started.elapsed().as_secs_f64();
    let terminal_start = Instant::now();
    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgb(&bytes)?;
    if (dw, dh) != (width, height) || decoded.len() != pixels.len() {
        return Err("decoded packed RGB shape mismatch".into());
    }
    let actual = scorer
        .compute(
            &source,
            &RgbSlice::new(decoded.as_chunks::<3>().0, width as usize, height as usize),
            Some("webp"),
        )?
        .score();
    if !actual.is_finite() || (actual - f64::from(metrics.achieved_score)).abs() > 1e-5 {
        return Err(format!(
            "terminal mismatch {actual} versus {}",
            metrics.achieved_score
        )
        .into());
    }
    let terminal_seconds = terminal_start.elapsed().as_secs_f64();
    if file_sha(&bake)? != sha(&bake_bytes) {
        return Err("model changed during encode".into());
    }
    std::fs::create_dir(out)?;
    std::fs::write(out.join("selected.webp"), &bytes)?;
    write_png(&out.join("selected.png"), &decoded, width, height)?;
    let report = serde_json::json!({
        "source":input,"source_sha256":file_sha(&input)?,"width":width,"height":height,
        "model_sha256":sha(&bake_bytes),"driver_sha256":file_sha(&std::env::current_exe()?)?,
        "target":target,"max_overshoot":overshoot,"max_undershoot_ship":0.,
        "full_encode_budget":passes,"full_encodes":metrics.passes_used,
        "ordinary_decodes":metrics.passes_used,"scalar_comparisons":metrics.passes_used,
        "terminal_decodes":1,"terminal_scalar_comparisons":1,
        "achieved":metrics.achieved_score,"terminal_score":actual,"target_met":metrics.targets_met,
        "bytes":bytes.len(),"encoded_sha256":sha(&bytes),"decoded_sha256":sha(&decoded),
        "loop_seconds":loop_seconds,"terminal_seconds":terminal_seconds,
        "total_seconds":loop_seconds+terminal_seconds,"method":method,"fine_gap":fine_gap,
        "spatial_mode":std::env::var("ZENWEBP_ZQ_SPATIAL")?,"seed_q":std::env::var("ZENWEBP_ZQ_START_Q")?,
        "formula_revision":std::env::var("ZENSIM_FORMULA_REV")?,
        "scope":"fixed-seed native binding screen; no targeting or spatial qualification"
    });
    let text = serde_json::to_string_pretty(&report)?;
    std::fs::write(out.join("result.json"), format!("{text}\n"))?;
    println!("{text}");
    Ok(())
}
