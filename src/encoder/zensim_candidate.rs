//! Private complete-model measurement for the existing target-zensim loop.
use super::iteration::SpatialMap;
use crate::PixelLayout;
use crate::encoder::api::{EncodeDiagnostics, EncodeError, EncodeStats};
use std::io::Write;
use std::path::PathBuf;
use whereat::{At, at};
use zenpredict_serving::Model;
use zensim_candidate::{BakeScorer, Fused944Session, PrecomputedReference, RgbSlice};

type Result<T> = core::result::Result<T, At<EncodeError>>;
fn failure(e: impl core::fmt::Display) -> At<EncodeError> {
    at!(EncodeError::InvalidBufferSize(format!(
        "WebP candidate: {e}"
    )))
}

pub(super) struct Config {
    model: Model,
    pub(super) seed_q: f32,
    pub(super) mode: String,
    trace: Option<PathBuf>,
}
impl Config {
    pub(super) fn from_env(layout: PixelLayout, width: u32, height: u32) -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("ZENWEBP_ZQ_BAKE") else {
            if ["ZENWEBP_ZQ_SPATIAL", "ZENWEBP_ZQ_TRACE_DIR"]
                .iter()
                .any(|key| std::env::var_os(key).is_some())
            {
                return Err(failure("research options require ZENWEBP_ZQ_BAKE"));
            }
            // START_Q predates this binding and remains a legacy census override.
            return Ok(None);
        };
        if layout != PixelLayout::Rgb8 || width < 8 || height < 8 {
            return Err(failure("requires packed opaque sRGB8 with dimensions >=8"));
        }
        if std::env::var("ZENSIM_FORMULA_REV").as_deref() != Ok("1") {
            return Err(failure("requires explicit ZENSIM_FORMULA_REV=1"));
        }
        if super::ablation_runtime::USE_QUADRANT_PROXY.with(|c| c.get()) {
            return Err(failure(
                "candidate integrals require the actual segment map",
            ));
        }
        let seed_q: f32 = std::env::var("ZENWEBP_ZQ_START_Q")
            .map_err(failure)?
            .parse()
            .map_err(failure)?;
        if !seed_q.is_finite() || !(0.0..=100.0).contains(&seed_q) {
            return Err(failure("seed q must be finite in [0,100]"));
        }
        let mode = std::env::var("ZENWEBP_ZQ_SPATIAL").map_err(failure)?;
        if !matches!(mode.as_str(), "scalar" | "neutral" | "active") {
            return Err(failure("spatial mode must be scalar, neutral or active"));
        }
        let model = Model::from_bytes(&std::fs::read(path).map_err(failure)?).map_err(failure)?;
        Ok(Some(Self {
            model,
            seed_q,
            mode,
            trace: std::env::var_os("ZENWEBP_ZQ_TRACE_DIR").map(PathBuf::from),
        }))
    }
}

pub(super) struct Measurement<'a> {
    config: &'a Config,
    scorer: BakeScorer<'a>,
    source: &'a [[u8; 3]],
    width: usize,
    height: usize,
    pre: Option<PrecomputedReference>,
    session: Fused944Session,
    pass: usize,
}
impl<'a> Measurement<'a> {
    pub(super) fn new(
        config: &'a Config,
        pixels: &'a [u8],
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let (width, height) = (width as usize, height as usize);
        let (source, rest) = pixels.as_chunks::<3>();
        if !rest.is_empty() || Some(source.len()) != width.checked_mul(height) {
            return Err(failure("packed source shape mismatch"));
        }
        let scorer = BakeScorer::new(&config.model).map_err(failure)?;
        let pre = if config.mode == "scalar" {
            None
        } else {
            Some(
                scorer
                    .precompute_reference(&RgbSlice::new(source, width, height))
                    .map_err(failure)?,
            )
        };
        if let Some(path) = &config.trace {
            std::fs::create_dir(path).map_err(failure)?;
        }
        Ok(Self {
            config,
            scorer,
            source,
            width,
            height,
            pre,
            session: Fused944Session::new(),
            pass: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn measure(
        &mut self,
        bytes: &[u8],
        q: f32,
        stats: &EncodeStats,
        diag: &EncodeDiagnostics,
        overrides: Option<[i8; 4]>,
        encode_seconds: f64,
    ) -> Result<(f32, SpatialMap)> {
        let start = std::time::Instant::now();
        let (pixels, width, height) = crate::oneshot::decode_rgb(bytes).map_err(failure)?;
        let decode_seconds = start.elapsed().as_secs_f64();
        if (width as usize, height as usize) != (self.width, self.height) {
            return Err(failure("decoded dimensions changed"));
        }
        let (decoded, rest) = pixels.as_chunks::<3>();
        if !rest.is_empty() || decoded.len() != self.source.len() {
            return Err(failure("decoded packed RGB8 shape mismatch"));
        }
        let start = std::time::Instant::now();
        let source = RgbSlice::new(self.source, self.width, self.height);
        let distorted = RgbSlice::new(decoded, self.width, self.height);
        let (cols, rows) = (self.width.div_ceil(16), self.height.div_ceil(16));
        if (diag.mb_width as usize, diag.mb_height as usize) != (cols, rows)
            || (!diag.segment_map.is_empty() && diag.segment_map.len() != cols * rows)
        {
            return Err(failure("encoder segment-map geometry mismatch"));
        }
        let mut map = Vec::new();
        let score = if let Some(pre) = &self.pre {
            let spatial = self
                .scorer
                .compute_with_ref_and_attribution(
                    &source,
                    pre,
                    &distorted,
                    Some("webp"),
                    &mut self.session,
                    8,
                )
                .map_err(failure)?;
            if !spatial.unsupported_feature_ids().is_empty() || spatial.has_corruption_gate() {
                return Err(failure(
                    "unsupported spatial terms or discontinuous corruption gate",
                ));
            }
            map.try_reserve_exact(cols * rows).map_err(failure)?;
            for y in 0..rows {
                for x in 0..cols {
                    map.push(
                        spatial
                            .attribution()
                            .query_rect(
                                x * 16,
                                y * 16,
                                ((x + 1) * 16).min(self.width),
                                ((y + 1) * 16).min(self.height),
                            )
                            .abs() as f32,
                    );
                }
            }
            spatial.result().score() as f32
        } else {
            self.scorer
                .compute(&source, &distorted, Some("webp"))
                .map_err(failure)?
                .score() as f32
        };
        if !score.is_finite() || map.iter().any(|v| !v.is_finite()) {
            return Err(failure("nonfinite score or block integral"));
        }
        let measure_seconds = start.elapsed().as_secs_f64();
        if let Some(path) = &self.config.trace {
            std::fs::write(path.join(format!("pass-{}.webp", self.pass)), bytes)
                .map_err(failure)?;
            std::fs::write(path.join(format!("pass-{}.rgb8", self.pass)), &pixels)
                .map_err(failure)?;
            std::fs::write(
                path.join(format!("pass-{}.segments.u8", self.pass)),
                &diag.segment_map,
            )
            .map_err(failure)?;
            if self.pre.is_some() {
                let values: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(path.join(format!("pass-{}.map.f32", self.pass)), values)
                    .map_err(failure)?;
            }
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path.join("measurements.tsv"))
                .map_err(failure)?;
            if self.pass == 0 {
                writeln!(f, "pass\tq\tscore\tbytes\tmap_evaluations\tmap_control_attempted\tnon_neutral_override_segments\tsegment_quants\toverrides\tencode_seconds\tdecode_seconds\tmeasure_seconds")
                    .map_err(failure)?;
            }
            let non_neutral = overrides.map_or(0, |v| v.iter().filter(|&&x| x != 0).count());
            writeln!(f, "{}\t{q:.9}\t{score:.9}\t{}\t{}\t{}\t{non_neutral}\t{:?}\t{:?}\t{encode_seconds:.9}\t{decode_seconds:.9}\t{measure_seconds:.9}",
                self.pass, bytes.len(), usize::from(self.pre.is_some()), usize::from(overrides.is_some()),
                stats.segment_quant, overrides.unwrap_or([0; 4])).map_err(failure)?;
        }
        self.pass += 1;
        Ok((score, SpatialMap::Macroblocks(map)))
    }
}
