//! Budgeted sweep-plan construction over the WebP encoder knob space.
//!
//! Port of the variant-generation playbook
//! (`zenjpeg/docs/VARIANT_GENERATION.md`) to zenwebp. Mode is the
//! discrimination axis (pattern 1): lossy (VP8) and lossless (VP8L)
//! share no knob space, so [`SweepVariant`] is an enum and a lossless
//! cell cannot spell an SNS strength structurally. The quality grid
//! multiplies lossy cells only; lossless strata emit one cell each
//! (VP8L "quality" is an effort dial — **byte-only, trial-class** —
//! and rides the axes as an ordinary value, rendered as a `-ql` flag).
//!
//! Deliberately excluded from the curated axes, with reasons:
//!
//! - `target_size` / `target_psnr` / `target_zensim` — closed-loop
//!   search modes, not open-loop cells; a sweep cell must be a pure
//!   function of its config.
//! - `preset` — a macro-knob that resolves into sns/filter values; the
//!   underlying knobs are swept directly so cells stay self-describing.
//! - `alpha_quality` / `exact` — class-conditional (alpha-bearing
//!   corpus + alpha-aware metric needed; playbook pattern 10).
//! - `near_lossless` — changes pixels (metric-class); the curated
//!   lossless space stays 100 % trial-class (decoded pixels identical
//!   across every lossless cell, so `min(bytes)` comparisons are exact).
//! - `multi_pass_stats` outside m4 — gate-shadowed (the encoder ignores
//!   it at other method tiers); the `mpass` probe is curated at m4 via
//!   the main axes only.
//!
//! Scalar/step provenance (ship-derived rule — values that already ship
//! as defaults or documented bounds, not invented grids):
//!
//! | knob | bound | curated steps | provenance |
//! |---|---|---|---|
//! | method (lossy) | 0–6 | 4, 6, 2 | 4 = zenwebp/libwebp default; 6 = max effort; 2 = fast tier |
//! | sns_strength (SCALAR) | 0–100 | None, 0, 100, 25, 80 | None resolves to 50 (no-preset default, libwebp `config_enc.c` parity — `Some(50)` would byte-alias it, so 50 is reachable only as the default); 0/100 = off/max bounds; 25 = Drawing preset, 80 = Photo/Picture preset (ship-derived mid-ladder, 2026-06-12 dense-sweep program). Effective ladder {0, 25, 50, 80, 100} |
//! | filter_strength (SCALAR) | 0–100 | None, 0, 30, 10, 100 | None resolves to 60 (no-preset default — `Some(60)` would byte-alias it); 0 = off bound; 30 = Photo preset, 10 = Drawing preset, 100 = max bound (ship-derived mid-ladder, 2026-06-12). Effective ladder {0, 10, 30, 60, 100} |
//! | filter_sharpness (SCALAR) | 0–7 | None, 3, 6, 7 | new axis 2026-06-12 (dense-sweep program — was missing entirely). None resolves to 0 (no-preset default — `Some(0)` would byte-alias it); 3 = Photo preset, 6 = Drawing preset, 7 = bound endpoint (`with_filter_sharpness` clamps to ≤7). Live: 3 header bits (`header.rs` `write_literal(3, sharpness_level)`) + in-loop filter thresholds feed reconstruction. Effective ladder {0, 3, 6, 7} |
//! | segments | 1–4 | None, 1 | default auto; 1 = segmentation off |
//! | sharp_yuv | off/on | off, on | `cwebp -sharp_yuv` |
//! | method (lossless) | 0–6 | 4, 6, 0 | 4 = default; 6/0 effort bounds |
//! | quality (lossless) | 0–100 | 75, 100, 25 | 75 = default; bounds-ish effort steps |
//! | internal probes | — | def, parity, mpass, smooth, plim50 | label registry below |
//!
//! Everything here is `__expert`: it drives the [`InternalParams`]
//! escape hatch and inherits its no-semver-guarantees contract.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::encoder::CostModel;
use crate::encoder::config::{
    EncoderConfig, InternalParams, LosslessConfig, LossyConfig, SharpYuvSetting,
};

// ============================================================================
// Named internal-params probes (the label registry)
// ============================================================================

/// A labelled lossy internal-params bundle. Labels are id tokens: no
/// `-` (the flag separator) or `_` (the token separator) inside a
/// label, or downstream id parsing breaks.
#[derive(Clone, Debug)]
pub struct NamedLossyProbe {
    /// Registry label (an id token).
    pub label: String,
    /// The bundle this label denotes.
    pub params: InternalParams,
}

impl NamedLossyProbe {
    /// The all-defaults probe (`"def"`).
    #[must_use]
    pub fn default_probe() -> Self {
        Self {
            label: "def".to_string(),
            params: InternalParams::default(),
        }
    }
}

/// The curated lossy internal-params probes. Each deviates in exactly
/// one field so dev-1 strata answer "does this knob matter".
#[must_use]
pub fn lossy_internal_probes() -> Vec<NamedLossyProbe> {
    vec![
        // Strict libwebp parity: disables zenwebp's perceptual extensions.
        NamedLossyProbe {
            label: "parity".to_string(),
            params: InternalParams {
                cost_model: Some(CostModel::StrictLibwebpParity),
                ..InternalParams::default()
            },
        },
        // Two-pass statistics (live at m4 only — gate-shadowed elsewhere,
        // documented at the field; the method axis includes 4).
        NamedLossyProbe {
            label: "mpass".to_string(),
            params: InternalParams {
                multi_pass_stats: Some(true),
                ..InternalParams::default()
            },
        },
        // 3×3 majority-filter segment-map smoothing (`cwebp -pre 1`).
        NamedLossyProbe {
            label: "smooth".to_string(),
            params: InternalParams {
                smooth_segment_map: Some(true),
                ..InternalParams::default()
            },
        },
        // Pinned partition-0 budget mid-point (disables the retry ladder).
        NamedLossyProbe {
            label: "plim50".to_string(),
            params: InternalParams {
                partition_limit: Some(50),
                ..InternalParams::default()
            },
        },
    ]
}

/// Registry lookup: `"def"` + the curated probe labels.
#[must_use]
pub fn lossy_probe_by_label(label: &str) -> Option<NamedLossyProbe> {
    if label == "def" {
        return Some(NamedLossyProbe::default_probe());
    }
    lossy_internal_probes()
        .into_iter()
        .find(|p| p.label == label)
}

// ============================================================================
// Variants
// ============================================================================

/// One lossy (VP8) encode variant with its resolved quality.
#[derive(Clone, Debug)]
pub struct LossyVariant {
    /// Grid quality (fingerprint-hashed; pixels change with it).
    pub quality: f32,
    /// Effort/method tier 0–6.
    pub method: u8,
    /// Labelled internal-params probe.
    pub internal: NamedLossyProbe,
    /// Sharp-YUV chroma downsampling.
    pub sharp_yuv: bool,
    /// SNS strength override (`None` = encoder-derived default).
    pub sns_strength: Option<u8>,
    /// Filter strength override (`None` = encoder-derived default).
    pub filter_strength: Option<u8>,
    /// Filter sharpness override 0–7 (`None` = encoder-derived default).
    pub filter_sharpness: Option<u8>,
    /// Segment-count override (`None` = auto).
    pub segments: Option<u8>,
}

/// One lossless (VP8L) encode variant. Every field here is byte-only
/// (trial-class): decoded pixels are identical across all lossless
/// cells.
#[derive(Clone, Debug)]
pub struct LosslessVariant {
    /// Effort/method tier 0–6.
    pub method: u8,
    /// VP8L effort dial 0–100 (NOT pixel quality).
    pub quality: f32,
}

/// A sweep variant: lossy or lossless, structurally discriminated.
#[derive(Clone, Debug)]
pub enum SweepVariant {
    /// Lossy (VP8) cell.
    Lossy(LossyVariant),
    /// Lossless (VP8L) cell.
    Lossless(LosslessVariant),
}

impl SweepVariant {
    /// Build the actual encoder config for this variant.
    #[must_use]
    pub fn build(&self) -> EncoderConfig {
        match self {
            Self::Lossy(v) => {
                let mut cfg = LossyConfig::new()
                    .with_quality(v.quality)
                    .with_method(v.method)
                    .with_internal_params(v.internal.params.clone());
                if v.sharp_yuv {
                    cfg = cfg.with_sharp_yuv(true);
                }
                if let Some(s) = v.sns_strength {
                    cfg = cfg.with_sns_strength(s);
                }
                if let Some(f) = v.filter_strength {
                    cfg = cfg.with_filter_strength(f);
                }
                if let Some(h) = v.filter_sharpness {
                    cfg = cfg.with_filter_sharpness(h);
                }
                if let Some(n) = v.segments {
                    cfg = cfg.with_segments(n);
                }
                EncoderConfig::Lossy(cfg)
            }
            Self::Lossless(v) => EncoderConfig::Lossless(
                LosslessConfig::new()
                    .with_quality(v.quality)
                    .with_method(v.method),
            ),
        }
    }
}

// ============================================================================
// Axes
// ============================================================================

/// Lossy axes, most-important value first (index 0 = production default).
#[derive(Clone, Debug)]
pub struct LossyAxes {
    /// Effort/method tiers.
    pub methods: Vec<u8>,
    /// Labelled internal-params probes.
    pub internal: Vec<NamedLossyProbe>,
    /// Sharp-YUV on/off.
    pub sharp_yuv: Vec<bool>,
    /// SNS overrides (`None` = encoder default).
    pub sns_strength: Vec<Option<u8>>,
    /// Filter-strength overrides.
    pub filter_strength: Vec<Option<u8>>,
    /// Filter-sharpness overrides (0–7).
    pub filter_sharpness: Vec<Option<u8>>,
    /// Segment-count overrides.
    pub segments: Vec<Option<u8>>,
}

/// Lossless axes (all trial-class), most-important first.
#[derive(Clone, Debug)]
pub struct LosslessAxes {
    /// Effort/method tiers.
    pub methods: Vec<u8>,
    /// VP8L effort dials (byte-only).
    pub qualities: Vec<f32>,
}

/// The full axis bundle: either or both modes. `None` = that mode is
/// not swept at all.
#[derive(Clone, Debug)]
pub struct SweepAxes {
    /// Lossy axes; cells multiply by the quality grid.
    pub lossy: Option<LossyAxes>,
    /// Lossless axes; one cell per stratum.
    pub lossless: Option<LosslessAxes>,
}

impl SweepAxes {
    /// RD-front axes: lossy m4/m6 at defaults, lossless m4 default.
    #[must_use]
    pub fn rd_core() -> Self {
        Self {
            lossy: Some(LossyAxes {
                methods: vec![4, 6],
                internal: vec![NamedLossyProbe::default_probe()],
                sharp_yuv: vec![false],
                sns_strength: vec![None],
                filter_strength: vec![None],
                filter_sharpness: vec![None],
                segments: vec![None],
            }),
            lossless: Some(LosslessAxes {
                methods: vec![4],
                qualities: vec![75.0],
            }),
        }
    }

    /// Every curated mode axis on top of [`rd_core`](Self::rd_core).
    #[must_use]
    pub fn modes_full() -> Self {
        let mut axes = Self::rd_core();
        let lossy = axes.lossy.as_mut().expect("rd_core has lossy");
        lossy.methods.push(2);
        lossy.internal.extend(lossy_internal_probes());
        lossy.sharp_yuv.push(true);
        lossy.sns_strength.push(Some(0));
        lossy.sns_strength.push(Some(100));
        lossy.filter_strength.push(Some(0));
        // SCALAR mid-ladders (2026-06-12, dense-sweep program) — appended
        // after the established endpoint steps so the budget ladder
        // (tail-shed) drops the newest values first. Values are
        // ship-derived preset constants; the no-preset defaults
        // (sns 50, filter 60, sharpness 0) are deliberately NOT spelled
        // — Some(default) would byte-alias the None stratum.
        lossy.sns_strength.push(Some(25)); // Drawing preset
        lossy.sns_strength.push(Some(80)); // Photo/Picture preset
        lossy.filter_strength.push(Some(30)); // Photo preset
        lossy.filter_strength.push(Some(10)); // Drawing preset
        lossy.filter_strength.push(Some(100)); // max bound
        lossy.filter_sharpness.push(Some(3)); // Photo preset
        lossy.filter_sharpness.push(Some(6)); // Drawing preset
        lossy.filter_sharpness.push(Some(7)); // bound endpoint
        lossy.segments.push(Some(1));
        let lossless = axes.lossless.as_mut().expect("rd_core has lossless");
        lossless.methods.push(6);
        lossless.methods.push(0);
        lossless.qualities.push(100.0);
        lossless.qualities.push(25.0);
        axes
    }

    /// Dense isolated scalar ladders for both modes
    /// (VARIANT_GENERATION pattern 17): every categorical axis pinned to
    /// its production default, each CONTINUOUS knob given a dense single-
    /// axis ladder — the data a trained **scalar head** trains on. Pair
    /// with [`SweepBuilder::with_max_deviations`]`(1)` for one clean
    /// per-knob response curve across the quality grid, free of the
    /// cartesian interaction blow-up.
    #[must_use]
    pub fn scalar_dense() -> Self {
        Self {
            lossy: Some(LossyAxes::scalar_dense()),
            lossless: Some(LosslessAxes::scalar_dense()),
        }
    }
}

impl LossyAxes {
    /// Dense, isolated ladders over the lossy CONTINUOUS knobs — what a
    /// trained **scalar head** needs (VARIANT_GENERATION pattern 17).
    /// Every categorical axis is pinned to its default (internal `def`,
    /// sharp-yuv off, segments auto); the continuous knobs get dense
    /// ladders:
    ///
    /// - the **full method ladder m0–m6** — `method` is the compute dial
    ///   (and the [`compute_tier`]), but `modes_full` only carries
    ///   `{4, 6, 2}`; a per-effort cost/quality head needs every step, so
    ///   the gap (m0/m1/m3/m5) is filled here. The default 4 leads so the
    ///   m4 cell is deviation-0.
    /// - `sns_strength` — the dense 0,10,…,100 ladder. The no-preset
    ///   default 50 is reachable only as the `None` stratum (leads); the
    ///   `Some` ladder deliberately omits 50 because `Some(50)` would
    ///   byte-alias the `None` cell (config.rs `to_params`).
    /// - `filter_strength` — dense 0,10,…,100, with 60 (the no-preset
    ///   default) omitted for the same alias reason; `None` is its cell.
    /// - `filter_sharpness` — the full 0–7 ladder, with 0 (the no-preset
    ///   default) omitted; `None` is its cell.
    ///
    /// Pair with [`SweepBuilder::with_max_deviations`]`(1)` for one
    /// isolated ladder per knob across the quality grid.
    #[must_use]
    pub fn scalar_dense() -> Self {
        Self {
            // Default 4 first, then the dense ladder (m0–m6 minus 4); the
            // dense COMPUTE axis (method == compute_tier).
            methods: vec![4, 0, 1, 2, 3, 5, 6],
            internal: vec![NamedLossyProbe::default_probe()],
            sharp_yuv: vec![false],
            // None (= default 50) leads; dense Some ladder omits 50 (alias).
            sns_strength: vec![
                None,
                Some(0),
                Some(10),
                Some(20),
                Some(30),
                Some(40),
                Some(60),
                Some(70),
                Some(80),
                Some(90),
                Some(100),
            ],
            // None (= default 60) leads; dense Some ladder omits 60 (alias).
            filter_strength: vec![
                None,
                Some(0),
                Some(10),
                Some(20),
                Some(30),
                Some(40),
                Some(50),
                Some(70),
                Some(80),
                Some(90),
                Some(100),
            ],
            // None (= default 0) leads; dense Some ladder omits 0 (alias).
            filter_sharpness: vec![
                None,
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
            ],
            segments: vec![None],
        }
    }
}

impl LosslessAxes {
    /// Dense method ladder (m0–m6) for the lossless compute/bytes scalar
    /// head, the VP8L effort dial pinned to its 75 default
    /// (VARIANT_GENERATION pattern 17). Lossless pixels are exact, so the
    /// scalar of interest is purely bytes-vs-CPU per method; `method` is
    /// also the [`compute_tier`]. The default 4 leads (its cell is
    /// deviation-0).
    #[must_use]
    pub fn scalar_dense() -> Self {
        Self {
            methods: vec![4, 0, 1, 2, 3, 5, 6],
            qualities: vec![75.0],
        }
    }
}

// ============================================================================
// Quality grid (lossy cells only)
// ============================================================================

/// Quality grids per the sweep discipline (low-q never thinned
/// preferentially).
#[derive(Clone, Debug)]
pub enum QualityGrid {
    /// q ∈ {1, 5, 10, …, 100} — the 21-point floor.
    Step5,
    /// Step 5 through q65, step 2 from q70 (31 points).
    TrainingDense,
    /// Caller-provided points (kept in order, deduplicated).
    Explicit(Vec<f32>),
}

impl QualityGrid {
    /// Materialize the grid points.
    #[must_use]
    pub fn points(&self) -> Vec<f32> {
        match self {
            Self::Step5 => {
                let mut v = vec![1.0];
                v.extend((1..=20).map(|i| (i * 5) as f32));
                v
            }
            Self::TrainingDense => {
                let mut v = vec![1.0];
                v.extend((1..=13).map(|i| (i * 5) as f32));
                v.extend((35..=50).map(|i| (i * 2) as f32));
                v
            }
            Self::Explicit(pts) => {
                let mut v = Vec::new();
                for &p in pts {
                    if !v.contains(&p) {
                        v.push(p);
                    }
                }
                v
            }
        }
    }
}

// ============================================================================
// Ids (the durable identity contract — playbook pattern 7)
// ============================================================================

impl LossyVariant {
    /// Base id (no quality token):
    /// `vp8-m<m>_<label>[-syuv][-sns<v>][-flt<v>][-shp<v>][-seg<n>]`.
    fn base_id(&self) -> String {
        let mut s = format!("vp8-m{}_{}", self.method, self.internal.label);
        if self.sharp_yuv {
            s.push_str("-syuv");
        }
        if let Some(v) = self.sns_strength {
            s.push_str(&format!("-sns{v}"));
        }
        if let Some(v) = self.filter_strength {
            s.push_str(&format!("-flt{v}"));
        }
        if let Some(v) = self.filter_sharpness {
            s.push_str(&format!("-shp{v}"));
        }
        if let Some(n) = self.segments {
            s.push_str(&format!("-seg{n}"));
        }
        s
    }
}

impl LosslessVariant {
    /// Id: `vp8l-m<m>[-ql<q>]` (no quality-grid token; `ql` renders the
    /// effort dial when it deviates from the 75 default; `Display` is
    /// shortest-roundtrip so the value is lossless).
    fn base_id(&self) -> String {
        let mut s = format!("vp8l-m{}", self.method);
        if self.quality != 75.0 {
            s.push_str(&format!("-ql{}", self.quality));
        }
        s
    }
}

/// Reconstruct the [`SweepVariant`] a cell id denotes (full id,
/// including the lossy `_q<q>` token — the variant carries its resolved
/// quality, which is fingerprint-hashed). Grammar at the `base_id`
/// renderers; renderer and parser move in lockstep, enforced by
/// `cell_ids_roundtrip_to_their_variants`. Internal-params labels
/// resolve through [`lossy_probe_by_label`]; unknown labels error.
/// Evolution is additive-only — never rename a token or change numeric
/// formatting (stored ledger identity).
pub fn variant_from_cell_id(id: &str) -> Result<SweepVariant, String> {
    if let Some(rest) = id.strip_prefix("vp8-m") {
        let mut toks = rest.splitn(2, '_');
        let (Some(m_s), Some(tail)) = (toks.next(), toks.next()) else {
            return Err(format!("lossy id {id:?} missing tokens"));
        };
        let method: u8 = m_s
            .parse()
            .map_err(|e| format!("bad method in {id:?}: {e}"))?;
        let (flags_part, q_part) = match tail.rsplit_once('_') {
            Some((f, q)) if q.starts_with('q') => (f, Some(q)),
            _ => (tail, None),
        };
        let Some(q_tok) = q_part else {
            return Err(format!("lossy id {id:?} missing _q quality token"));
        };
        let quality: f32 = q_tok[1..]
            .parse()
            .map_err(|e| format!("bad q in {id:?}: {e}"))?;
        let mut parts = flags_part.split('-');
        let label = parts.next().unwrap_or_default();
        let internal = lossy_probe_by_label(label)
            .ok_or_else(|| format!("internal-params label {label:?} not in the registry"))?;
        let mut v = LossyVariant {
            quality,
            method,
            internal,
            sharp_yuv: false,
            sns_strength: None,
            filter_strength: None,
            filter_sharpness: None,
            segments: None,
        };
        for f in parts {
            match f {
                "syuv" => v.sharp_yuv = true,
                f if f.starts_with("sns") => {
                    v.sns_strength = Some(
                        f[3..]
                            .parse()
                            .map_err(|e| format!("bad sns in {id:?}: {e}"))?,
                    );
                }
                f if f.starts_with("flt") => {
                    v.filter_strength = Some(
                        f[3..]
                            .parse()
                            .map_err(|e| format!("bad flt in {id:?}: {e}"))?,
                    );
                }
                f if f.starts_with("shp") => {
                    v.filter_sharpness = Some(
                        f[3..]
                            .parse()
                            .map_err(|e| format!("bad shp in {id:?}: {e}"))?,
                    );
                }
                f if f.starts_with("seg") => {
                    v.segments = Some(
                        f[3..]
                            .parse()
                            .map_err(|e| format!("bad seg in {id:?}: {e}"))?,
                    );
                }
                other => return Err(format!("unknown lossy flag {other:?} in {id:?}")),
            }
        }
        Ok(SweepVariant::Lossy(v))
    } else if let Some(rest) = id.strip_prefix("vp8l-m") {
        let mut parts = rest.split('-');
        let m_s = parts.next().unwrap_or_default();
        let method: u8 = m_s
            .parse()
            .map_err(|e| format!("bad method in {id:?}: {e}"))?;
        let mut v = LosslessVariant {
            method,
            quality: 75.0,
        };
        for f in parts {
            if let Some(q) = f.strip_prefix("ql") {
                v.quality = q.parse().map_err(|e| format!("bad ql in {id:?}: {e}"))?;
            } else {
                return Err(format!("unknown lossless flag {f:?} in {id:?}"));
            }
        }
        Ok(SweepVariant::Lossless(v))
    } else {
        Err(format!(
            "cell id {id:?} is neither a vp8- (lossy) nor vp8l- (lossless) id"
        ))
    }
}

// ============================================================================
// Byte-identity fingerprint (over resolved state)
// ============================================================================

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u8(&mut self, v: u8) {
        self.write(&[v]);
    }
    fn f32(&mut self, v: f32) {
        self.write(&v.to_bits().to_le_bytes());
    }
    fn opt_u8(&mut self, v: Option<u8>) {
        match v {
            None => self.u8(0xFF),
            Some(x) => {
                self.u8(1);
                self.u8(x);
            }
        }
    }
}

/// Byte-identity fingerprint of a variant's resolved state. Equal
/// fingerprints produce identical bytes for the same input. Every
/// field is hashed (no exclusions to prove wrong — search-effort knobs
/// like `method` and the VP8L effort dial change bytes by design).
#[must_use]
pub fn fingerprint(variant: &SweepVariant) -> u64 {
    let mut h = Fnv::new();
    match variant {
        SweepVariant::Lossy(v) => {
            h.u8(1);
            h.f32(v.quality);
            h.u8(v.method);
            // Resolved internal fields, not the label.
            h.opt_u8(v.internal.params.partition_limit);
            h.u8(match v.internal.params.multi_pass_stats {
                None => 0xFF,
                Some(false) => 0,
                Some(true) => 1,
            });
            h.u8(match v.internal.params.smooth_segment_map {
                None => 0xFF,
                Some(false) => 0,
                Some(true) => 1,
            });
            h.u8(match v.internal.params.cost_model {
                None => 0xFF,
                Some(CostModel::ZenwebpDefault) => 0,
                Some(CostModel::StrictLibwebpParity) => 1,
            });
            h.u8(match &v.internal.params.sharp_yuv {
                None => 0xFF,
                Some(SharpYuvSetting::Off) => 0,
                Some(SharpYuvSetting::On) => 1,
                // Not reachable from the curated axes; hashed distinctly.
                Some(SharpYuvSetting::Custom(_)) => 2,
            });
            h.u8(u8::from(v.sharp_yuv));
            h.opt_u8(v.sns_strength);
            h.opt_u8(v.filter_strength);
            h.opt_u8(v.filter_sharpness);
            h.opt_u8(v.segments);
        }
        SweepVariant::Lossless(v) => {
            h.u8(2);
            h.u8(v.method);
            h.f32(v.quality);
        }
    }
    h.0
}

// ============================================================================
// Compute-resource tier
// ============================================================================

/// Compute-cost tier of a variant (higher = more CPU per encode). Used
/// by [`SweepBuilder::with_compute_limit`] to bound a sweep by compute
/// budget, and public so the fleet and pickers can constrain candidates
/// the same way (cf. zenavif `auto_tune`'s speed range). For WebP the
/// tier **is the `method` level** — libwebp's RD-optimization dial
/// (m0–m2 no RD opt, m3–m4 RD scoring, m5 trellis, m6 trellis-all), the
/// single knob that drives per-method encoder effort — for both VP8
/// (lossy) and VP8L (lossless), so `with_compute_limit(4)` is literally
/// "sweep `m ≤ 4`". Quality and the VP8L effort dial change *bytes*, not
/// the CPU schedule, so they are not part of the tier.
#[must_use]
pub fn compute_tier(variant: &SweepVariant) -> u8 {
    match variant {
        SweepVariant::Lossy(v) => v.method,
        SweepVariant::Lossless(v) => v.method,
    }
}

// ============================================================================
// Plan output + builder
// ============================================================================

/// One encode cell.
#[derive(Clone, Debug)]
pub struct SweepCell {
    /// Stable id (base + `_q<q>` for lossy cells).
    pub id: String,
    /// The variant to encode with.
    pub variant: SweepVariant,
    /// Grid quality for lossy cells; `None` for lossless.
    pub quality: Option<f32>,
    /// Byte-identity fingerprint of the resolved state.
    pub fingerprint: u64,
    /// Ids merged into this cell (identical fingerprints).
    pub aliases: Vec<String>,
    /// Axes deviating from the default stratum (0 = production default).
    pub deviations: u8,
}

/// A collapsed axis record (the no-silent-caps report).
#[derive(Clone, Debug)]
pub struct DroppedAxis {
    /// Axis name.
    pub axis: &'static str,
    /// Values kept (Debug-rendered).
    pub kept: String,
    /// Values dropped.
    pub dropped: Vec<String>,
}

/// The finite, auditable plan.
#[derive(Clone, Debug)]
pub struct SweepPlan {
    /// Deduplicated encode cells, main-effects-first.
    pub cells: Vec<SweepCell>,
    /// Strata rejected by validity checks (none in the curated axes).
    pub invalid_skipped: Vec<String>,
    /// Cell ids dropped because their [`compute_tier`] (the `method`
    /// level) exceeded the [`SweepBuilder::with_compute_limit`] budget —
    /// the explicit no-silent-caps report for the compute constraint
    /// (empty when no limit was set).
    pub compute_tier_skipped: Vec<String>,
    /// Budget-ladder drop report (no silent caps).
    pub dropped: Vec<DroppedAxis>,
    /// Candidate cells merged by fingerprint identity.
    pub duplicates_merged: usize,
    /// Uniform quality-grid coarsenings applied.
    pub q_coarsenings: u32,
    /// Budget unreachable even after the full ladder.
    pub over_budget: bool,
}

/// Plan builder: axes × grid under an optional cell budget.
#[derive(Clone, Debug)]
pub struct SweepBuilder {
    axes: SweepAxes,
    grid: QualityGrid,
    budget: Option<usize>,
    compute_limit: Option<u8>,
    max_deviations: Option<u8>,
}

impl SweepBuilder {
    /// New builder over the given axes and quality grid.
    #[must_use]
    pub fn new(axes: SweepAxes, grid: QualityGrid) -> Self {
        Self {
            axes,
            grid,
            budget: None,
            compute_limit: None,
            max_deviations: None,
        }
    }

    /// Cap the deduplicated cell count (ladder reductions reported).
    #[must_use]
    pub fn with_budget(mut self, max_cells: usize) -> Self {
        self.budget = Some(max_cells);
        self
    }

    /// Constrain the plan to cells whose [`compute_tier`] (the `method`
    /// level) is `<= max_tier`, dropping the more expensive cells
    /// (recorded in [`SweepPlan::compute_tier_skipped`], never silently).
    /// For WebP this is exactly "sweep `m <= max_tier`" — the dense
    /// method ladder in [`SweepAxes::scalar_dense`] stays, capped at the
    /// budget. Composes with [`with_budget`](Self::with_budget) (compute
    /// filter first, then the budget ladder reduces the survivors).
    #[must_use]
    pub fn with_compute_limit(mut self, max_tier: u8) -> Self {
        self.compute_limit = Some(max_tier);
        self
    }

    /// Keep only cells within `max_deviations` axes of their mode's
    /// default stratum. `1` = main-effects only (the default cell plus
    /// every single-axis probe, no interaction combos) — the
    /// isolated-axis regime a trained **scalar head** trains on. Pair
    /// with [`SweepAxes::scalar_dense`] for dense per-knob response
    /// curves without the cartesian blow-up of the full cross.
    #[must_use]
    pub fn with_max_deviations(mut self, max_deviations: u8) -> Self {
        self.max_deviations = Some(max_deviations);
        self
    }

    /// Cross + apply the user constraints (compute-tier limit, then
    /// deviation scope). Returns `(cells, duplicates_merged,
    /// compute_tier_skipped)`.
    fn build_cells(
        &self,
        axes: &SweepAxes,
        q_points: &[f32],
    ) -> (Vec<SweepCell>, usize, Vec<String>) {
        let (mut cells, duplicates_merged) = cross(axes, q_points);
        let mut compute_tier_skipped = Vec::new();
        if let Some(max_tier) = self.compute_limit {
            cells.retain(|c| {
                if compute_tier(&c.variant) <= max_tier {
                    true
                } else {
                    compute_tier_skipped.push(c.id.clone());
                    false
                }
            });
        }
        if let Some(max_dev) = self.max_deviations {
            cells.retain(|c| c.deviations <= max_dev);
        }
        (cells, duplicates_merged, compute_tier_skipped)
    }

    /// Build the plan.
    #[must_use]
    pub fn plan(&self) -> SweepPlan {
        let mut axes = self.axes.clone();
        let mut q_points = self.grid.points();
        let mut dropped: Vec<DroppedAxis> = Vec::new();
        let mut q_coarsenings = 0u32;
        let mut over_budget = false;

        loop {
            let (cells, merged, compute_tier_skipped) = self.build_cells(&axes, &q_points);
            let within = match self.budget {
                None => true,
                Some(b) => cells.len() <= b,
            };
            if within {
                return SweepPlan {
                    cells,
                    invalid_skipped: Vec::new(),
                    compute_tier_skipped,
                    dropped,
                    duplicates_merged: merged,
                    q_coarsenings,
                    over_budget,
                };
            }
            if let Some(d) = collapse_one_axis(&mut axes) {
                if let Some(last) = dropped.last_mut()
                    && last.axis == d.axis
                {
                    last.dropped.extend(d.dropped);
                    last.kept = d.kept;
                    continue;
                }
                dropped.push(d);
                continue;
            }
            if q_points.len() > 11 {
                let last = q_points.len() - 1;
                q_points = q_points
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i == 0 || *i == last || i % 2 == 0)
                    .map(|(_, &p)| p)
                    .collect();
                q_coarsenings += 1;
                continue;
            }
            over_budget = true;
            let (cells, merged, compute_tier_skipped) = self.build_cells(&axes, &q_points);
            return SweepPlan {
                cells,
                invalid_skipped: Vec::new(),
                compute_tier_skipped,
                dropped,
                duplicates_merged: merged,
                q_coarsenings,
                over_budget,
            };
        }
    }
}

fn collapse<T: core::fmt::Debug>(
    name: &'static str,
    v: &mut Vec<T>,
    floor: usize,
) -> Option<DroppedAxis> {
    if v.len() <= floor {
        return None;
    }
    let dropped = vec![format!("{:?}", v[v.len() - 1])];
    v.truncate(v.len() - 1);
    let kept = v
        .iter()
        .map(|x| format!("{x:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    Some(DroppedAxis {
        axis: name,
        kept,
        dropped,
    })
}

/// One-value-at-a-time ladder: lossy mode axes first (they multiply by
/// the grid), then lossless, never below the rd_core floors.
fn collapse_one_axis(axes: &mut SweepAxes) -> Option<DroppedAxis> {
    if let Some(l) = axes.lossy.as_mut() {
        let probes = l
            .internal
            .iter()
            .map(|p| p.label.clone())
            .collect::<Vec<_>>();
        let _ = probes;
        if let Some(d) = collapse("lossy.filter_sharpness", &mut l.filter_sharpness, 1)
            .or_else(|| collapse("lossy.segments", &mut l.segments, 1))
            .or_else(|| collapse("lossy.filter_strength", &mut l.filter_strength, 1))
            .or_else(|| collapse("lossy.sns_strength", &mut l.sns_strength, 1))
            .or_else(|| collapse("lossy.sharp_yuv", &mut l.sharp_yuv, 1))
            .or_else(|| {
                if l.internal.len() > 1 {
                    let last = l.internal.pop().expect("len > 1");
                    Some(DroppedAxis {
                        axis: "lossy.internal",
                        kept: l
                            .internal
                            .iter()
                            .map(|p| p.label.clone())
                            .collect::<Vec<_>>()
                            .join(", "),
                        dropped: vec![last.label],
                    })
                } else {
                    None
                }
            })
            .or_else(|| collapse("lossy.methods", &mut l.methods, 2))
        {
            return Some(d);
        }
    }
    if let Some(l) = axes.lossless.as_mut()
        && let Some(d) = collapse("lossless.qualities", &mut l.qualities, 1)
            .or_else(|| collapse("lossless.methods", &mut l.methods, 1))
    {
        return Some(d);
    }
    None
}

/// Cross axes × grid into deduplicated, main-effects-first cells.
/// Lossy strata expand quality-ascending within each stratum; lossless
/// strata follow all lossy strata of the same deviation class.
fn cross(axes: &SweepAxes, q_points: &[f32]) -> (Vec<SweepCell>, usize) {
    struct Entry {
        variant: SweepVariant,
        deviations: u8,
        idx_sum: usize,
        lossless: bool,
        seq: usize,
    }
    let mut entries: Vec<Entry> = Vec::new();
    let mut seq = 0usize;

    if let Some(l) = &axes.lossy {
        for (mi, &method) in l.methods.iter().enumerate() {
            for (ii, internal) in l.internal.iter().enumerate() {
                for (yi, &sharp) in l.sharp_yuv.iter().enumerate() {
                    for (si, &sns) in l.sns_strength.iter().enumerate() {
                        for (fi, &flt) in l.filter_strength.iter().enumerate() {
                            for (hi, &shp) in l.filter_sharpness.iter().enumerate() {
                                for (gi, &seg) in l.segments.iter().enumerate() {
                                    let idxs = [mi, ii, yi, si, fi, hi, gi];
                                    entries.push(Entry {
                                        variant: SweepVariant::Lossy(LossyVariant {
                                            quality: 0.0, // grid applies in pass 2
                                            method,
                                            internal: internal.clone(),
                                            sharp_yuv: sharp,
                                            sns_strength: sns,
                                            filter_strength: flt,
                                            filter_sharpness: shp,
                                            segments: seg,
                                        }),
                                        deviations: idxs.iter().filter(|&&x| x != 0).count() as u8,
                                        idx_sum: idxs.iter().sum(),
                                        lossless: false,
                                        seq,
                                    });
                                    seq += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(l) = &axes.lossless {
        for (mi, &method) in l.methods.iter().enumerate() {
            for (qi, &quality) in l.qualities.iter().enumerate() {
                let idxs = [mi, qi];
                entries.push(Entry {
                    variant: SweepVariant::Lossless(LosslessVariant { method, quality }),
                    deviations: idxs.iter().filter(|&&x| x != 0).count() as u8,
                    idx_sum: idxs.iter().sum(),
                    lossless: true,
                    seq,
                });
                seq += 1;
            }
        }
    }

    // Main effects first; lossy before lossless inside each class.
    entries.sort_by_key(|e| (e.deviations, e.lossless, e.idx_sum, e.seq));

    let mut cells: Vec<SweepCell> = Vec::new();
    let mut by_fp: alloc::collections::BTreeMap<u64, usize> = alloc::collections::BTreeMap::new();
    let mut merged = 0usize;
    for e in &entries {
        match &e.variant {
            SweepVariant::Lossy(base) => {
                for &q in q_points {
                    let mut v = base.clone();
                    v.quality = q;
                    let variant = SweepVariant::Lossy(v.clone());
                    let fp = fingerprint(&variant);
                    let id = format!("{}_q{q}", v.base_id());
                    if let Some(&i) = by_fp.get(&fp) {
                        cells[i].aliases.push(id);
                        merged += 1;
                    } else {
                        by_fp.insert(fp, cells.len());
                        cells.push(SweepCell {
                            id,
                            variant,
                            quality: Some(q),
                            fingerprint: fp,
                            aliases: Vec::new(),
                            deviations: e.deviations,
                        });
                    }
                }
            }
            SweepVariant::Lossless(v) => {
                let variant = SweepVariant::Lossless(v.clone());
                let fp = fingerprint(&variant);
                let id = v.base_id();
                if let Some(&i) = by_fp.get(&fp) {
                    cells[i].aliases.push(id);
                    merged += 1;
                } else {
                    by_fp.insert(fp, cells.len());
                    cells.push(SweepCell {
                        id,
                        variant,
                        quality: None,
                        fingerprint: fp,
                        aliases: Vec::new(),
                        deviations: e.deviations,
                    });
                }
            }
        }
    }
    (cells, merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_ids_roundtrip_to_their_variants() {
        // Grammar-totality gate: every id the planner emits — canonical
        // and alias spellings, both modes — parses back to a variant
        // whose fingerprint is identical.
        let mut checked = 0usize;
        for (axes, grid) in [
            (SweepAxes::rd_core(), QualityGrid::Step5),
            (
                SweepAxes::modes_full(),
                QualityGrid::Explicit(vec![10.0, 85.0]),
            ),
        ] {
            let plan = SweepBuilder::new(axes, grid).plan();
            for cell in &plan.cells {
                for id in core::iter::once(&cell.id).chain(cell.aliases.iter()) {
                    let v = variant_from_cell_id(id).unwrap_or_else(|e| panic!("{id}: {e}"));
                    assert_eq!(
                        fingerprint(&v),
                        cell.fingerprint,
                        "fingerprint drift for {id}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 60,
            "grammar coverage suspiciously thin: {checked}"
        );
    }

    #[test]
    fn malformed_ids_error() {
        for bad in [
            "vp8-m4_nolabel_q75",  // unknown registry label
            "vp8-m4_def",          // missing quality token
            "vp8-m4_def-warp_q75", // unknown flag
            "vp8-m4_def-shpx_q75", // non-numeric sharpness
            "vp8l-m4-warp",        // unknown lossless flag
            "gif-m4_def_q75",      // unknown mode prefix
        ] {
            assert!(
                variant_from_cell_id(bad).is_err(),
                "{bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn queue_is_main_effects_first_and_ids_unique() {
        let plan = SweepBuilder::new(
            SweepAxes::modes_full(),
            QualityGrid::Explicit(vec![50.0, 85.0]),
        )
        .plan();
        assert_eq!(plan.cells[0].deviations, 0);
        assert!(
            plan.cells[0].id.starts_with("vp8-m4_def"),
            "{}",
            plan.cells[0].id
        );
        for w in plan.cells.windows(2) {
            assert!(w[1].deviations >= w[0].deviations);
        }
        let mut seen = alloc::collections::BTreeSet::new();
        for c in &plan.cells {
            for id in core::iter::once(&c.id).chain(c.aliases.iter()) {
                assert!(seen.insert(id.clone()), "duplicate id {id}");
            }
        }
    }

    #[test]
    fn scalar_ladders_pinned_distinct_and_roundtrip() {
        // Dense-sweep program contract: the SCALAR ladders stay pinned
        // (sns mid-ladder, filter mid-ladder, the new filter_sharpness
        // axis), no curated value spells the no-preset default (it
        // would byte-alias the None stratum), every value fingerprints
        // distinctly at a fixed stratum, and ids round-trip.
        let axes = SweepAxes::modes_full();
        let lossy = axes.lossy.as_ref().unwrap();
        assert_eq!(
            lossy.sns_strength,
            vec![None, Some(0), Some(100), Some(25), Some(80)],
            "sns ladder drifted"
        );
        assert_eq!(
            lossy.filter_strength,
            vec![None, Some(0), Some(30), Some(10), Some(100)],
            "filter ladder drifted"
        );
        assert_eq!(
            lossy.filter_sharpness,
            vec![None, Some(3), Some(6), Some(7)],
            "sharpness ladder drifted"
        );
        // No-preset defaults must not be spelled (alias trap):
        // sns 50, filter 60, sharpness 0 (config.rs to_params).
        assert!(!lossy.sns_strength.contains(&Some(50)));
        assert!(!lossy.filter_strength.contains(&Some(60)));
        assert!(!lossy.filter_sharpness.contains(&Some(0)));

        let base = LossyVariant {
            quality: 75.0,
            method: 4,
            internal: NamedLossyProbe::default_probe(),
            sharp_yuv: false,
            sns_strength: None,
            filter_strength: None,
            filter_sharpness: None,
            segments: None,
        };
        let mut fps = alloc::collections::BTreeMap::new();
        fps.insert(
            fingerprint(&SweepVariant::Lossy(base.clone())),
            base.base_id(),
        );
        let probe = |v: LossyVariant, fps: &mut alloc::collections::BTreeMap<u64, String>| {
            let id = format!("{}_q75", v.base_id());
            let fp = fingerprint(&SweepVariant::Lossy(v));
            if let Some(prev) = fps.insert(fp, id.clone()) {
                panic!("fingerprint collision between {prev} and {id}");
            }
            let parsed = variant_from_cell_id(&id).unwrap();
            assert_eq!(fingerprint(&parsed), fp, "parser drift for {id}");
        };
        for &sns in lossy.sns_strength.iter().flatten() {
            let mut v = base.clone();
            v.sns_strength = Some(sns);
            probe(v, &mut fps);
        }
        for &flt in lossy.filter_strength.iter().flatten() {
            let mut v = base.clone();
            v.filter_strength = Some(flt);
            probe(v, &mut fps);
        }
        for &shp in lossy.filter_sharpness.iter().flatten() {
            let mut v = base.clone();
            v.filter_sharpness = Some(shp);
            probe(v, &mut fps);
        }
        // default + 4 sns + 4 filter + 3 sharpness.
        assert_eq!(fps.len(), 12, "scalar ladder size drifted");
    }

    #[test]
    fn budget_ladder_reports_and_never_drops_silently() {
        let unbudgeted = SweepBuilder::new(SweepAxes::modes_full(), QualityGrid::Step5).plan();
        let budget = unbudgeted.cells.len() / 3;
        let plan = SweepBuilder::new(SweepAxes::modes_full(), QualityGrid::Step5)
            .with_budget(budget)
            .plan();
        assert!(plan.cells.len() <= budget);
        assert!(!plan.dropped.is_empty());
        for d in &plan.dropped {
            assert!(!d.dropped.is_empty());
        }
        assert!(!plan.over_budget);
    }

    #[test]
    fn plan_is_deterministic() {
        let a = SweepBuilder::new(SweepAxes::rd_core(), QualityGrid::Step5).plan();
        let b = SweepBuilder::new(SweepAxes::rd_core(), QualityGrid::Step5).plan();
        assert_eq!(a.cells.len(), b.cells.len());
        for (x, y) in a.cells.iter().zip(&b.cells) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.fingerprint, y.fingerprint);
        }
    }

    #[test]
    fn lossless_space_is_grid_free() {
        let mut axes = SweepAxes::rd_core();
        axes.lossy = None;
        let plan = SweepBuilder::new(axes, QualityGrid::Step5).plan();
        assert_eq!(plan.cells.len(), 1, "one cell per lossless stratum");
        assert!(plan.cells[0].quality.is_none());
        assert_eq!(plan.cells[0].id, "vp8l-m4");
    }

    #[test]
    fn scalar_dense_is_isolated_and_dense() {
        // VARIANT_GENERATION pattern 17/18: scalar_dense + max_deviations(1)
        // is a clean per-knob ladder regime — main-effects only, and the
        // CONTINUOUS knobs are densely sampled.
        let plan = SweepBuilder::new(SweepAxes::scalar_dense(), QualityGrid::Explicit(vec![50.0]))
            .with_max_deviations(1)
            .plan();
        // Isolation: one knob deviates per cell, no interaction combos.
        assert!(
            plan.cells.iter().all(|c| c.deviations <= 1),
            "scalar_dense + max_deviations(1) must be main-effects only"
        );
        // Dense COMPUTE axis: the full m0..=m6 method ladder is present —
        // the m0/m1/m3/m5 that modes_full skips are filled in, so the
        // per-method cost head sees ≥6 distinct compute tiers.
        let mut tiers: Vec<u8> = plan
            .cells
            .iter()
            .map(|c| compute_tier(&c.variant))
            .collect();
        tiers.sort_unstable();
        tiers.dedup();
        assert!(
            tiers.len() >= 6,
            "scalar_dense method ladder too thin: {} distinct compute tiers ({tiers:?})",
            tiers.len()
        );
        // Dense SCALAR axis: the SNS ladder yields ≥8 distinct strengths
        // among the lossy cells (isolated sns-only deviations + the
        // default None cell). Count distinct resolved sns spellings.
        let mut sns: Vec<Option<u8>> = plan
            .cells
            .iter()
            .filter_map(|c| match &c.variant {
                SweepVariant::Lossy(v) => Some(v.sns_strength),
                SweepVariant::Lossless(_) => None,
            })
            .collect();
        sns.sort_unstable();
        sns.dedup();
        assert!(
            sns.len() >= 8,
            "scalar_dense sns ladder too thin: {} distinct values ({sns:?})",
            sns.len()
        );
    }

    #[test]
    fn compute_tier_equals_method_and_is_monotonic() {
        // compute_tier IS the method level; a cheap (low-method) variant
        // has a strictly smaller tier than an expensive (high-method) one.
        let mk = |m: u8| {
            SweepVariant::Lossless(LosslessVariant {
                method: m,
                quality: 75.0,
            })
        };
        assert_eq!(compute_tier(&mk(2)), 2);
        assert_eq!(compute_tier(&mk(6)), 6);
        assert!(
            compute_tier(&mk(2)) < compute_tier(&mk(6)),
            "a low-method variant must be a strictly smaller compute tier"
        );
        // Lossy carries its method as the tier too.
        let lossy = |m: u8| {
            SweepVariant::Lossy(LossyVariant {
                quality: 75.0,
                method: m,
                internal: NamedLossyProbe::default_probe(),
                sharp_yuv: false,
                sns_strength: None,
                filter_strength: None,
                filter_sharpness: None,
                segments: None,
            })
        };
        assert_eq!(compute_tier(&lossy(0)), 0);
        assert!(compute_tier(&lossy(0)) < compute_tier(&lossy(5)));
    }

    #[test]
    fn with_compute_limit_caps_method_and_reports() {
        let unlimited =
            SweepBuilder::new(SweepAxes::modes_full(), QualityGrid::Explicit(vec![75.0])).plan();
        let limited = SweepBuilder::new(SweepAxes::modes_full(), QualityGrid::Explicit(vec![75.0]))
            .with_compute_limit(4)
            .plan();
        assert!(!limited.cells.is_empty(), "m ≤ 4 cells must survive");
        assert!(
            limited.cells.len() < unlimited.cells.len(),
            "compute limit must drop the m6 (and any m>4) cells"
        );
        assert!(
            limited.cells.iter().all(|c| compute_tier(&c.variant) <= 4),
            "no cell may exceed m4 under with_compute_limit(4)"
        );
        assert!(
            !limited.compute_tier_skipped.is_empty(),
            "dropped m>4 cells must be reported, never silently capped"
        );
    }
}
