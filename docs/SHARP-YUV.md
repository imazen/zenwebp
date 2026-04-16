# Sharp YUV in zenwebp

Sharp YUV improves chroma edge quality during 4:2:0 downsampling. Standard
box-filter chroma averaging blurs color transitions; sharp YUV iteratively
refines the chroma planes to minimize RGB reconstruction error.

zenwebp implements a three-mode system via `Option<SharpYuvConfig>`:

| Mode | Config | Overhead | When to use |
|------|--------|----------|-------------|
| **None** | `None` / `.with_sharp_yuv(false)` | 0% | Uniform content at low quality |
| **ChromaOnly** | `SharpYuvConfig { refine_y: false, .. }` | 0-8% | Natural photos, text, most content |
| **Full** | `SharpYuvConfig::default()` / `.with_sharp_yuv(true)` | 10-25% | High-detail textures, fine patterns |

libwebp's default is sharp YUV **off**. It's opt-in via `cwebp -sharp_yuv` or
`config.use_sharp_yuv = 1` in the C API. zenwebp follows the same default.

## Algorithm

The pipeline runs after the standard BT.601 YCbCr conversion:

1. **Y plane**: SIMD-accelerated BT.601 forward transform (zenyuv)
2. **Initial chroma**: gamma-corrected box-filter downsampling matching libwebp's decoder model
3. **Chroma refinement** (ChromaOnly/Full): Newton-step iteration on each 2x2 chroma block, minimizing `||RGB_original - RGB_reconstructed||` using the BT.601-Limited inverse matrix. 2 iterations with the correct Jacobian outperform 4 iterations of libwebp's forward-matrix gradient approach.
4. **Y refinement** (Full only): single pass adjusting each Y value to compensate for luma error introduced by the subsampled chroma. Matches libwebp's `SharpYuvUpdateY`.

The chroma refinement starts from a gamma-corrected baseline that matches the
decoder's upsampling model, so the iteration can only improve on the starting
point.

Implementation: [zenyuv::sharp](https://github.com/imazen/zenjpeg) (`refine_chroma_420_u8`, `refine_y_420_u8`).

## API

```rust
use zenwebp::{LossyConfig, Preset, SharpYuvConfig};

// Off (default, matches libwebp)
let cfg = LossyConfig::with_preset(Preset::Default, 80.0);

// Full mode (chroma + Y refinement)
let cfg = LossyConfig::with_preset(Preset::Default, 80.0)
    .with_sharp_yuv(true);

// Chroma-only (no Y refinement, near-zero overhead)
let cfg = LossyConfig::with_preset(Preset::Default, 80.0)
    .with_sharp_yuv_config(SharpYuvConfig {
        refine_y: false,
        ..Default::default()
    });

// Custom iteration count
let cfg = LossyConfig::with_preset(Preset::Default, 80.0)
    .with_sharp_yuv_config(SharpYuvConfig {
        max_iterations: 4,
        convergence_threshold: 0.05,
        refine_y: true,
    });
```

## Measurements

All data from CID22 medium corpus (15 images, 512px) at 19 quality levels
(q5-q100), measured with SSIMULACRA2, butteraugli, and zensim. Deltas are
vs the "none" baseline (standard chroma downsampling). Encode times on a
7950X, single-threaded.

Full dataset: `examples/sharp_yuv_timing.rs` in the
[coefficient](https://github.com/imazen/coefficient) repo, output at
`/mnt/v/output/coefficient/knob-eval/sharp_yuv_timing.tsv`.

### HighDetail (n=9 images)

SSIM2 is positive for both modes at every quality level tested.

| q | chroma DSSIM2 | full DSSIM2 | chroma Dzensim | full Dzensim | chroma Dms% | full Dms% |
|---|---|---|---|---|---|---|
| 5 | +0.33 | +0.54 | -0.19 | +0.11 | +8% | +25% |
| 10 | +0.35 | +0.47 | +0.27 | +0.39 | 0% | +15% |
| 15 | +0.25 | +0.69 | +0.23 | +0.17 | +8% | +23% |
| 20 | +0.28 | +0.64 | -0.20 | +0.19 | +7% | +21% |
| 25 | +0.52 | +0.73 | +0.05 | +0.23 | 0% | +13% |
| 30 | +0.28 | +0.44 | -0.20 | -0.07 | 0% | +13% |
| 35 | +0.23 | +0.39 | +0.03 | +0.14 | +7% | +13% |
| 40 | +0.38 | +0.50 | +0.01 | +0.23 | +7% | +20% |
| 45 | +0.34 | +0.46 | +0.01 | +0.09 | +7% | +20% |
| 50 | +0.32 | +0.60 | -0.04 | +0.26 | 0% | +12% |
| 55 | +0.35 | +0.59 | +0.05 | +0.21 | 0% | +12% |
| 65 | +0.53 | +0.71 | +0.10 | +0.22 | 0% | +12% |
| 70 | +0.45 | +0.79 | +0.04 | +0.24 | 0% | +12% |
| 75 | +0.59 | +0.79 | +0.08 | +0.24 | +6% | +18% |
| 80 | +0.68 | +1.00 | -0.00 | +0.27 | +6% | +11% |
| 85 | +0.56 | +0.75 | -0.01 | +0.20 | 0% | +16% |
| 90 | +0.69 | +0.93 | +0.03 | +0.25 | +5% | +15% |
| 95 | +0.67 | +0.97 | +0.05 | +0.31 | +4% | +9% |
| 100 | +0.79 | +1.07 | +0.03 | +0.29 | +4% | +8% |

Butteraugli deltas are within +/-0.15 (neutral) across the board.

### Natural (n=3 images)

All three metrics stay within +/-0.3 noise. No clear benefit or harm.

| q | chroma DSSIM2 | full DSSIM2 | chroma Dzensim | full Dzensim |
|---|---|---|---|---|
| 5 | +0.42 | +0.19 | +0.29 | +0.08 |
| 20 | -0.20 | -0.08 | -0.18 | -0.21 |
| 50 | +0.08 | -0.06 | +0.12 | +0.10 |
| 80 | +0.09 | +0.10 | -0.06 | -0.10 |
| 95 | -0.02 | -0.00 | -0.35 | -0.32 |
| 100 | -0.06 | -0.05 | -0.34 | -0.34 |

Chroma-only is 0% overhead and produces marginally better butteraugli
at mid-quality (-0.15 at q80). Full mode adds 10-25% encode time for
no measurable gain over chroma-only.

### Text (n=1 image, higher variance)

SSIM2 is positive at 16 of 19 quality levels for chroma-only. Zensim
trends negative at high quality (>= q80).

| q | chroma DSSIM2 | full DSSIM2 | chroma Dzensim | full Dzensim |
|---|---|---|---|---|
| 5 | +0.68 | +0.56 | +0.25 | +0.06 |
| 20 | +0.41 | +0.22 | -0.05 | +0.93 |
| 50 | +0.24 | +0.37 | +0.17 | -0.02 |
| 65 | +0.41 | +0.66 | +0.07 | +0.12 |
| 80 | +0.33 | +0.43 | -0.06 | +0.04 |
| 95 | +0.27 | +0.33 | -0.32 | -0.21 |
| 100 | +0.31 | +0.37 | -0.39 | -0.28 |

### Uniform (n=2 images)

All metrics within noise (+/-0.3). Slight positive SSIM2 trend at q >= 65.

| q | chroma DSSIM2 | chroma Dzensim |
|---|---|---|
| 5 | +0.05 | -0.33 |
| 25 | -0.16 | -0.14 |
| 50 | -0.20 | -0.28 |
| 65 | +0.33 | +0.08 |
| 80 | +0.01 | +0.02 |
| 95 | +0.18 | +0.02 |

## Recommended defaults by content type

Based on the "most optimistic metric" rule: enable if any trusted metric
(SSIM2, butteraugli) improves and none significantly worsen.

| Content type | Mode | Rationale |
|---|---|---|
| High-detail textures, fine patterns | **Full** | +0.3 to +1.1 SSIM2 at all quality levels, worth 10-20% overhead |
| Natural photos | **ChromaOnly** | 0% overhead; butteraugli improves at mid-q; SSIM2/zensim within noise |
| Text, diagrams, screenshots | **ChromaOnly** | 0% overhead; SSIM2 positive at most quality levels |
| Uniform gradients, q >= 65 | **ChromaOnly** | 0% overhead; slight SSIM2 positive trend |
| Uniform gradients, q < 65 | **None** | No consistent benefit from any metric |

## Where zensim disagrees

Zensim shows negative deltas at high quality (>= q90) for Natural, Text,
and Uniform content, even when SSIM2 is flat or positive. This is likely
because zensim's XYB-space perceptual model weights luma fidelity more heavily
than chroma fidelity at high quality, and the Y refinement pass trades a tiny
amount of luma precision for chroma edge improvement. For production decisions,
SSIM2 and butteraugli are more established; zensim is used as a fast triage
metric, not the final arbiter.

## History

- **v0.4.2**: sharp YUV added but broken (green color shift from y_offset sign
  inversion in zenyuv + missing gamma correction in initial chroma)
- **v0.4.3**: green shift fixed; chroma-only mode re-enabled with
  gamma-corrected initial chroma seeding
- **v0.4.4** (pending): Y refinement pass added (`refine_y_420_u8`); API
  refactored from `sharp_yuv: bool` to `Option<SharpYuvConfig>`; `SharpYuvConfig`
  re-exported from zenyuv
