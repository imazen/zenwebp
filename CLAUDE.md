# zenwebp CLAUDE.md

See global ~/.claude/CLAUDE.md for general instructions.

## Perceptual Encoder Optimizations (2026-02-04)

### Enhanced Masking Model (method 4+)

Replaced variance-based masking with modern perceptual model:
- **SATD-based AC energy** - measures texture in transform domain (more perceptually meaningful)
- **Luminance masking** - Weber's law: dark areas need finer quantization
- **Edge detection** - protects edges from coarse quantization
- **Uniformity check** - smooth gradients need protection

Files: `src/encoder/psy.rs` (compute_masking_alpha)

### JND Visibility Thresholding (method 5+)

Added Just Noticeable Difference (JND) model for coefficient quantization:
- Frequency-dependent JND thresholds for luma and chroma
- Thresholds scale with quantizer (higher q = more aggressive zeroing)
- Luminance and contrast masking adaptation
- Trellis skips psy_penalty for coefficients below JND threshold

Files: `src/encoder/psy.rs` (JND_THRESHOLD_Y/UV, is_below_jnd_y/uv)
       `src/encoder/trellis.rs` (JND-gated psy_penalty)

### Butteraugli Tuning Results

Tested perceptual features with butteraugli feedback (20 CID22 images, Q75):

| Method | Avg Size | Butteraugli | Notes |
|--------|----------|-------------|-------|
| m2 | 31.5 KB | 4.643 | Baseline |
| m3 | 31.0 KB | **4.580** | Best quality (CSF only) |
| m4 | 31.0 KB | 4.681 | Slightly worse |
| m5 | 30.2 KB | 4.820 | Smaller files |
| m6 | 29.5 KB | 4.796 | Smallest files |

**Key finding:** Psy-RD (SATD energy preservation) hurts butteraugli scores even
at low strengths because butteraugli prefers smoother reconstructions while psy-rd
penalizes smoothing. Psy-RD is now disabled.

**Recommendation:**
- Use method 3-4 for best perceptual quality
- Use method 5-6 for smaller files with slightly worse perceptual quality
- Enhanced CSF tables (method 3+) alone provide the best quality improvement

### SATD SIMD Optimization

Added `#[multiversed]` to SATD functions for autovectorization on x86-64-v3+ targets.
Clean scalar code autovectorizes well - no manual intrinsics needed.

## Current Optimization Status (2026-02-03)

### Encoder Performance vs libwebp

**Method Parameter** - Apples-to-apples comparison (792079.png, Q75, SNS=0, filter=0, segments=1):
| Method | zenwebp | libwebp | Size Ratio | Speed | Notes |
|--------|---------|---------|------------|-------|-------|
| 0 | 13,988 | 16,678 | **0.84x** | 3.7x | I16-only - we're 16% smaller! |
| 1 | 13,988 | 16,188 | **0.86x** | 3.0x | I16-only - we're 14% smaller! |
| 2 | 12,568 | 13,440 | **0.94x** | 3.7x | Limited I4 - we're 6% smaller! |
| 3 | 12,568 | 12,018 | 1.05x | 1.4x | I4 dominant, no trellis |
| 4 | 12,128 | 12,018 | 1.009x | 2.1x | Full I4, no trellis |
| 5 | 12,022 | 11,952 | **1.006x** | 2.1x | Trellis (near parity!) |
| 6 | 11,908 | 11,720 | 1.016x | 1.7x | Trellis during mode selection |

**Key finding (2026-02-02):** Method mapping now aligned with libwebp's RD optimization levels:
- m0-2: RD_OPT_NONE (fast, no RD optimization)
- m3-4: RD_OPT_BASIC (RD scoring, no trellis)
- m5: RD_OPT_TRELLIS (trellis during encoding)
- m6: RD_OPT_TRELLIS_ALL (trellis during I4 mode selection)

**CID22 corpus aggregate (248 images, Q75, SNS=0, filter=0, segments=1):**
| Method | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| 4 | 7,356,472 | 7,284,004 | 1.0099x |
| 5 | 7,199,446 | 7,197,750 | **1.0002x** |
| 6 | 7,053,912 | 7,038,104 | 1.0022x |

*Method 5 achieves near-perfect parity (0.02% larger). Method 4 is 1% larger because it
now matches libwebp's m4 (no trellis) - this is the correct apples-to-apples comparison.*

**Multi-pass removed (2026-02-02):**
Multi-pass encoding (methods 5/6 doing 2-3 passes) was removed after testing showed
it provides **no compression benefit** - in fact, files were 0.1-0.5% LARGER with
more passes. This matches libwebp's behavior where multi-pass only helps when used
with quality search (target_size convergence). All methods now use single-pass.

**Method comparison (SNS=0, filter=0, segments=1, single pass, 792079.png):**
| Encoder | m4 | m5 | m6 | m4→m5 gain | m5→m6 gain |
|---------|-----|-----|-----|-----------|-----------|
| libwebp | 12018 | 11952 | 11720 | 0.5% | 1.9% |
| zenwebp | 12128 | 12022 | 11908 | 0.9% | 0.9% |

**Key findings:**
1. **Method realignment (2026-02-02)**: Aligned method feature mapping with libwebp:
   - Moved trellis from m4 to m5 (matching RD_OPT_TRELLIS)
   - Each method now adds distinct features (see table above)
   - m5 CID22 corpus: **1.002x** (within 0.2% of libwebp)
2. **I4 flatness penalty (2026-02-02)**: Added FLATNESS_PENALTY for non-DC I4 modes
   when coefficients are flat (≤3 non-zero AC). Matches libwebp's PickBestIntra4 behavior.
3. **I4 mode context fix (2026-02-02)**: Edge blocks in I4 mode selection were using
   hardcoded DC context (0) instead of cross-macroblock context from `top_b_pred`/`left_b_pred`.
4. **Cross-macroblock non-zero context fix (2026-02-02)**: I4 mode selection was using
   hardcoded `false` for non-zero context on edge blocks instead of using cross-macroblock
   context from `top_complexity`/`left_complexity`.
5. **Multi-pass removed**: Provides no benefit without quality search (tested).

**Quality search (target_size support) - IMPLEMENTED (2026-02-02):**

The `target_size` parameter enables quality search using the secant method:
```rust
let output = EncoderConfig::new()
    .quality(75.0)
    .target_size(10000)  // Target 10KB output
    .encode_rgb(&pixels, width, height)?;
```

Implementation details:
- Uses secant method (linear interpolation) like libwebp's `ComputeNextQ()`
- Convergence threshold: |dq| < 0.4 (DQ_LIMIT from libwebp)
- Max passes: method + 3 or 6, whichever is higher
- Each pass creates a fresh encoder and does full encoding at adjusted quality
- Best result (closest to target) is returned even if not fully converged

Test results on 512x512 CID22 image:
| Target | Actual | Accuracy |
|--------|--------|----------|
| 8000 | 8054 | +0.7% |
| 10000 | 10046 | +0.5% |
| 12000 | 11778 | -1.8% |
| 15000 | 15630 | +4.2% |

**Remaining gap analysis (all methods):**
- libwebp m5 enables trellis (we already have trellis at m4), gaining ~2%
- libwebp m6 uses trellis-during-selection (we already do this at m4), gaining ~4.4%
- Our trellis brings us from m2→m4 (6.5% gain), comparable to libwebp's m4→m6 (4.4%)
- But our non-trellis baseline (m2) is worse than libwebp's (m4), so trellis
  compensates rather than outperforms
- Root cause: I4 coefficient encoding efficiency gap in the non-trellis path

### Recent SIMD Optimizations
- **Fused SIMD primitives for mode selection (2026-02-05):**
  - `ftransform_from_u8_4x4` — residual+DCT from flat u8[16] arrays, eliminates i32 subtract loop
  - `quantize_dequantize_block_simd` — quantize+dequantize in single SIMD pass (both outputs)
  - `quantize_dequantize_ac_only_simd` — AC-only variant for I16 Y1 blocks
  - `idct_add_residue_inplace` — fused IDCT+add_residue with DC-only fast path
  - Eliminated double IDCT for I4 winning mode (save best_dequantized from inner loop)
  - **Result: 586M → 259M instructions (2.26x reduction), 2.99x → 1.28x vs libwebp**
- **archmage 0.5 #[rite] conversion (2026-02-05):**
  - Converted 15 inner `_sse2` functions from `#[arcane]` to `#[rite]`
  - `#[rite]` applies `#[target_feature]` + `#[inline]` directly — no wrapper function
  - Enables full inlining when called from arcane/rite context (e.g., evaluate_i4_modes_sse2)
  - quantize_dequantize_block: 21.1M → 2.4M (inlined into I4 inner loop)
  - **Result: 259M → 183M instructions (29% reduction), 1.28x → 0.90x vs libwebp**
- **Fused tdisto_4x4 SIMD** - Port of libwebp's TTransform_SSE2: processes both blocks in parallel using
  vertical-first Hadamard (valid with symmetric weights). Now inlined, not a separate hotspot (2026-02-04)
- **I4 mode cost precompute** - Cache all 10 mode costs before inner loop (avoid 3D table lookup, 2026-02-04)
- **I4 prediction SSE SIMD** - Use sse4x4() for mode filtering instead of scalar loop (2026-02-04)
- **Row-wise copy_from_slice** - Replace nested loops with memcpy-like ops for block extraction (2026-02-04)
- **is_flat_source_16 SIMD** - Compare 16 bytes/row with CMPEQ+MOVEMASK (16x speedup, 2026-02-04)
- **TrueMotion prediction SIMD** - pred_luma16_tm/pred_chroma8_tm with packus clamping (8-16x speedup, 2026-02-04)
- **Edge density SIMD** - compute_edge_density horizontal diff scan (16x speedup, 2026-02-04)
- **Mode selection caching** - Pre-extract source blocks for TDisto/psy-rd, reuse across modes (2026-02-04)
- **SIMD quantization** - `wide::i64x4` for simple quantize path (29% speedup for methods 0-3, 2026-01-23)
- **GetResidualCost SIMD** - Precompute abs/ctx/levels with SSE2 (30% speedup, 2026-01-23)
- **FTransform2** - Fused residual+DCT for 2 blocks at once (2026-01-23)
- DCT/IDCT: SIMD i32/i16 conversion (13% speedup)
- **t_transform SIMD fix** - Previous was pseudo-SIMD (extract to scalar). Now uses _mm_madd_epi16 for Hadamard. 14%→4% runtime (2026-02-04)
- SSE4x4: SIMD distortion calculation
- **LTO + inline hints** - Added LTO and codegen-units=1 to release profile, plus #[inline] to
  hot helper functions (tdisto_*, is_flat_*, compute_filter_level). Marginal improvement (~5-8%).

### Profiler Hot Spots (method 4, 2026-02-05, after #[rite] conversion)
| Function | % Instr | M instr | Notes |
|----------|---------|---------|-------|
| evaluate_i4_modes_sse2 | 20.3% | 37.2M | I4 inner loop (inlines quantize/dequantize/SSE) |
| encode_image | 13.3% | 24.4M | Main encoding loop |
| get_residual_cost_sse2 | 12.5% | 22.9M | Coefficient cost estimation |
| choose_macroblock_info | 10.3% | 18.9M | I16/UV mode selection |
| convert_image_yuv | 5.2% | 9.5M | YUV conversion |
| ftransform2 | 4.3% | 7.8M | Fused 2-block DCT |
| record_coeff_tokens | 4.0% | 7.4M | Token recording |
| idct_add_residue_inplace | 3.7% | 6.8M | Fused IDCT+add_residue |
| pick_best_intra4 | 3.1% | 5.6M | I4 outer orchestration |
| write_bool | 2.5% | 4.6M | Arithmetic encoding |
| quantize_block (standalone) | 2.4% | 4.3M | Quantize (I16 path) |

**Speed comparison (criterion, 792079.png 512x512, Q75, default target, 2026-02-05):**
| Method | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| 0 | 7.0ms | 2.7ms | 2.6x |
| 2 | 12.3ms | 4.2ms | 2.9x |
| 4 | 15.0ms | 10.2ms | **1.47x** |
| 6 | 22.2ms | 15.5ms | **1.43x** |

*Instruction ratio is 0.90x but wall-clock is ~1.47x — gap is memory access patterns
(our stride=full image width vs libwebp's compact row cache). #[rite] conversion
eliminated archmage dispatch overhead (was ~9.7M instructions, 46% of quantize cost).*

**Performance optimization session (2026-02-04):**
- Early exit in I4 mode loop (skip flatness/spectral/psy-rd when base RD exceeds best)
- SIMD add_residue using packus saturation
- SIMD dequantization using SSE2 mul_epu32 pattern
- Fused residual+DCT for chroma blocks (ftransform2)
- Reduced I4 mode candidates (6 instead of 10 for methods 3-4)
- SIMD Y1 AC quantization in I16 mode selection
- SIMD Y2 quantize/dequantize in I16 mode selection

Note: `dct4x4`, `idct4x4`, `is_flat_coeffs`, `tdisto_4x4` are now inlined into parent functions.

### zenwebp vs libwebp Comparison (2026-02-05)

**Per-encode comparison (792079.png, Q75, M4, SNS=0):**

| Metric | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| Instructions | 183M | 203M | **0.90x** |
| Output size | 12,198 | 12,018 | 1.015x |

*Previously 586M (2.99x) → 259M (1.28x, fused SIMD) → 183M (0.90x, #[rite] inlining).*

**CID22 corpus comparison (41 images, Q75, M4, SNS=0):**

| Metric | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| Total size | 1,180,418 | 1,169,034 | 1.0097x |

**Function-level comparison (per encode, 2026-02-05, after #[rite]):**

| zenwebp | M instr | libwebp | M instr | Ratio |
|---------|---------|---------|---------|-------|
| evaluate_i4 + choose_macroblock + pick_best_i4 | 61.7M | PickBestIntra4 + quant_enc | 18.3M | 3.4x |
| get_residual_cost_sse2 | 22.9M | GetResidualCost_SSE2 | 9.7M | 2.4x |
| idct_add_residue + idct4x4 | 8.6M | ITransform_SSE2 | 15.9M | **0.54x** |
| ftransform2 + ftransform_from_u8 | 9.6M | FTransform_SSE2 | 9.8M | **0.98x** |
| record_coeff_tokens | 7.4M | VP8RecordCoeffTokens | 3.4M | 2.2x |
| quantize_block (standalone) | 4.3M | QuantizeBlock_SSE41 | 6.0M | **0.72x** |

*After #[rite] conversion: quantize/dequantize inlined into evaluate_i4_modes_sse2,
no longer visible as separate functions. Mode selection overhead reduced from 5.2x to 3.4x.*

**Remaining optimization opportunities:**
1. Mode selection still 3.4x vs libwebp — I4 inner loop orchestration overhead
2. Residual cost: 2.4x gap, tighter inner loop possible
3. record_coeff_tokens: 2.2x gap, token emission overhead
4. Wall-clock still ~1.7x despite 0.90x instructions — memory access patterns

**Profiling commands:**
```bash
# Pre-convert image for callgrind (avoids PNG decoder AVX-512 issues)
convert image.png -depth 8 RGB:image_WxH.rgb

# Profile zenwebp
valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.out \
  target/release/examples/callgrind_encode image_WxH.rgb W H 75 4

# Profile libwebp (use .libs binary, not libtool wrapper)
LD_LIBRARY_PATH=~/work/libwebp/src/.libs:~/work/libwebp/sharpyuv/.libs \
  valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.lib.out \
  ~/work/libwebp/examples/.libs/cwebp -q 75 -m 4 -sns 0 -f 0 -segments 1 -o /dev/null image.png
```

**t_transform SIMD optimization (2026-02-04):**
Previous pseudo-SIMD loaded with SIMD but extracted to scalar immediately.
New uses `_mm_madd_epi16` for horizontal Hadamard, reducing 14%→11.5% (but still 4.5x slower than libwebp).

### Mode Selection Bottleneck Analysis (2026-02-04)

The `choose_macroblock_info` function (33% of encoder time) is fundamentally expensive
because VP8 rate-distortion optimization requires full reconstruction for each mode
candidate to compute accurate SSE distortion.

**Optimizations implemented (2026-02-04):**
- Cache source block extraction for TDisto/psy-rd (extract once per MB, reuse across modes)
- Cache U/V source blocks in UV mode selection (same optimization)
- Precompute I4 mode costs (cache 10 costs per block, avoid 3D table lookup in loop)
- SIMD for I4 prediction SSE (sse4x4 instead of 160 scalar ops per block)
- Row-wise copy_from_slice for all block extractions (better vectorization)

**Remaining optimization opportunities (not yet implemented):**
1. **Defer I16 reconstruction** - Only do full IDCT for winning mode (saves ~48 IDCT/MB)
2. **Batch IDCT** - Process multiple 4x4 blocks in one SIMD operation

**Why this is fundamentally expensive:**
- VP8 has 4 I16 modes + 10 I4 modes per block + 4 UV modes
- RD optimization requires SSE(reconstructed, source), not SSE(prediction, source)
- Full reconstruction = dequantize + IDCT for accurate distortion measurement
- This is the same architecture libwebp uses

### Quality vs libwebp (2026-02-02)
- CID22 m4: **0.999x** — aggregate parity with libwebp
- CID22 m5: **1.009x** — near-parity (multi-pass refinement)
- CID22 m6: **1.034x** — remaining gap from I4 coefficient efficiency
- Screenshots m4: **1.023x**
- Token buffer (2026-02-02) improved CID22 from 1.043x → 0.999x at m4
- Multi-pass experiments (2026-02-02): Both token re-recording and re-quantization
  approaches failed to improve compression. Methods 5/6 now equivalent to method 4.

**Production settings (2026-02-03):**
With default presets (SNS=50, filter=60), zenwebp is close to libwebp:
- CID22 corpus Q75: **1.0149x** (1.5% larger)
- CID22 corpus Q90: **1.0060x** (0.6% larger - near parity!)
- 792079.png Q75: **1.027x** (2.7% larger)
- 792079.png Q90: **1.026x** (2.6% larger)

**SNS c_base fix (2026-02-03):**
Fixed bug where compute_segment_quant() computed c_base from base_quant instead
of directly from quality. This was double-applying segment alpha transformation.
Improvement: CID22 Q75 1.0210x → 1.0149x, Q90 1.0126x → 1.0060x.

**Diagnostic vs production settings:**
In diagnostic mode (SNS=0, filter=0, segments=1), zenwebp is ~1% larger than
libwebp. Production mode is similar at high quality, slightly worse at low quality.

**Preset tuning parameters now active (2026-02-01):**
| Preset | SNS | Filter | Sharp | Segs |
|--------|-----|--------|-------|------|
| Default | 50 | 60 | 0 | 4 |
| Picture | 80 | 35 | 4 | 4 |
| Photo | 80 | 30 | 3 | 4 |
| Drawing | 25 | 10 | 6 | 4 |
| Icon | 0 | 0 | 0 | 4 |
| Text | 0 | 0 | 0 | 2 |
| Auto | detected | detected | detected | detected |

**Preset::Auto content detection (2026-02-01):**
- Uniformity-based classifier: ≥0.45 block uniformity → Photo tuning, <0.45 → Default tuning
- Small images (≤128x128) → Icon tuning
- Drawing/Text ContentType variants map to Default tuning (50,60,0,4) — original presets were counterproductive
- CID22 corpus (41 images): Auto 0.987x of Default, 2 regressions <0.5% each
- Screenshots (9 images): Auto 0.954x of Default, 0 regressions

**SSIMULACRA2 quality-size tradeoff (2026-02-01):**

| Corpus | Encoder | Size ratio | Avg SSIM2 delta | SSIM2 per 1% size |
|--------|---------|------------|-----------------|-------------------|
| CID22 | zenwebp Photo | 0.985x | -0.57 | 0.38 |
| CID22 | libwebp Photo | 0.993x | -0.14 | 0.20 |
| Screenshots | zenwebp Photo | 0.954x | -0.72 | 0.16 |
| Screenshots | libwebp Photo | 0.959x | -1.19 | 0.29 |

Key finding: zenwebp's SNS implementation produces a steeper quality-size tradeoff
than libwebp's on diverse images (CID22), but is actually gentler on screenshots.
**UPDATE (2026-02-03):** The c_base bug (computing from base_quant instead of quality)
was part of this issue. After fix, Q90 production is now 1.006x (near parity).

**TokenType fix (2026-01-24):**
The TokenType enum had I16DC and I16AC values swapped compared to libwebp:
- libwebp: TYPE_I16_AC=0, TYPE_I16_DC=1
- ours (was): I16DC=0, I16AC=1 (wrong!)
This caused incorrect probability tables for Y1 AC and Y2 DC coefficients.
Fix reduced file sizes by ~2%.

**Trellis sign bit double-counting fix (2026-02-02):**
`level_cost_fast` and `level_cost_with_table` in `trellis.rs` were adding 256 (sign bit
cost) to non-zero levels, but VP8_LEVEL_FIXED_COSTS already includes this cost. This made
non-zero levels appear 1 bit more expensive, biasing trellis toward zeros.
Fix improved m4 by ~0.4% on 792079.png.

### Detailed Encoder Callgrind Analysis (2026-01-23)

**IMPORTANT: Trellis comparison clarification**
- libwebp enables trellis only for method >= 5
- Our method 4 uses trellis, making it comparable to libwebp's method 6
- Tested disabling trellis for method 4: made files 1.5% LARGER (our trellis helps!)

**Fair feature-level comparisons (2026-02-02, 5 CID22 images):**
| Comparison | Aggregate Ratio | Notes |
|------------|-----------------|-------|
| Our m2 vs libwebp m2 | **0.940x** | Same method, we're 6% smaller! |
| Our m4 vs libwebp m6 (both trellis) | **1.045x** | 4.5% larger with trellis |
| Our m2 vs libwebp m4 (both no trellis) | **1.064x** | 6.4% larger |

The 4.5% trellis gap is the main remaining encoder efficiency issue.

**Corrected comparison (our method 4 vs libwebp method 6, both with trellis):**
| Encoder | Instructions | Time | File Size |
|---------|--------------|------|-----------|
| Ours (method 4) | 2,307M | 65ms | 78KB |
| libwebp (method 6) | 2,065M | 75ms | 73KB |

We're **faster** than libwebp with trellis (65ms vs 75ms), but produce larger files.

**Non-trellis comparison (our method 2 vs libwebp method 4):**
| Encoder | Instructions | Time | File Size |
|---------|--------------|------|-----------|
| Ours (method 2) | ~1,400M | 56ms | 87KB |
| libwebp (method 4) | 817M | ~30ms | 77KB |
| Ratio | 1.7x | 1.9x slower | 1.13x larger |

**Cache behavior (D1 miss rate):**
- Ours: 0.1% (better than libwebp)
- libwebp: 0.3%

**Hotspot comparison (non-trellis, millions of instructions):**
| Our Function | Ours | libwebp Equivalent | libwebp | Ratio |
|--------------|------|-------------------|---------|-------|
| get_residual_cost | 306M | GetResidualCost_SSE2 | 126M | 2.4x |
| idct4x4 | 150M | ITransform_SSE2 | 64M | 2.3x |
| t_transform_sse2 | 134M | Disto4x4_SSE2 | 52M | 2.6x |
| dct4x4 | 79M | FTransform_SSE2 | 41M | 1.9x |

**Remaining optimization opportunities:**
1. **get_residual_cost** - 2.4x slower, SIMD loop improvements possible
2. **encode_coefficients** - Token emission overhead
3. **get_cost_luma16** - Mode cost estimation

**libwebp SIMD functions we lack:**
- `GetResidualCost_SSE2` - residual cost with SIMD
- `QuantizeBlock_SSE2` - quantization with SIMD
- `Disto4x4_SSE2` - distortion with SIMD
- `FTransform_SSE2` / `ITransform_SSE2` - faster than our SIMD

### Key Files
- `src/encoder/vp8/mod.rs` - Main encoder orchestration, single-pass token loop
- `src/encoder/vp8/residuals.rs` - TokenBuffer, coefficient token recording/emission
- `src/encoder/vp8/header.rs` - Bitstream header encoding
- `src/encoder/vp8/mode_selection.rs` - I16/I4/UV mode selection
- `src/encoder/vp8/prediction.rs` - Block prediction + transform
- `src/encoder/api.rs` - Public API, EncoderConfig, EncoderParams, Preset enum
- `src/encoder/analysis.rs` - DCT analysis, k-means clustering, auto-detection classifier
- `src/encoder/cost.rs` - Cost estimation, trellis quantization, filter level computation
- `src/common/types.rs` - Segment struct, init_matrices, quantization tables

### VP8L (Lossless) Encoder (2026-02-03)

VP8L lossless encoder implementation in `src/encoder/vp8l/`:

**Status:** Working, beats libwebp compression. CID22 50-image subset: **0.996x** of libwebp
(0.4% smaller on average). Screenshots: **0.995x** (0.5% smaller).

**Compression progression:**
| Date | Change | 10-img | 50-img | 198-img |
|------|--------|--------|--------|---------|
| Feb 2 | Initial | 1.228x | - | - |
| Feb 3 | Predictors | 1.075x | - | - |
| Feb 3 | Cross-color | 1.060x | - | - |
| Feb 3 | AnalyzeEntropy | 1.055x | - | - |
| Feb 3 | CostModel fix | 1.007x | 1.015x | - |
| Feb 3 | Huffman tree fix | 1.005x | 1.011x | - |
| Feb 3 | ExtraCost fix | **0.995x** | **0.997x** | **0.998x** |
| Feb 3 | Try-both cache | - | **0.996x** | - |

**Implemented features:**
- All 14 predictor modes with per-block entropy-based selection
- Subtract green transform
- Cross-color transform
- Color cache with auto-detection (tests sizes 0-10) + try-both strategy
- LZ77 with hash chain, left-extension, quality-dependent search depth
- TraceBackwards cost-based optimal parsing (forward DP)
- Huffman tree construction matching libwebp (count_min doubling, tie-breaking)
- Meta-Huffman spatially-varying codes with 3-phase clustering
- AnalyzeEntropy transform selection (5 modes: direct/spatial/subgreen/both/palette)
- OptimizeHuffmanForRle preprocessing
- Simple/two-symbol tree encoding
- VP8LExtraCost in histogram entropy estimation

**Key files:**
- `src/encoder/vp8l/encode.rs` - Main encoder pipeline, AnalyzeEntropy
- `src/encoder/vp8l/huffman.rs` - Huffman tree construction and encoding
- `src/encoder/vp8l/backward_refs.rs` - LZ77, cache selection, TraceBackwards
- `src/encoder/vp8l/hash_chain.rs` - Hash chain for match finding
- `src/encoder/vp8l/histogram.rs` - Symbol frequency histograms
- `src/encoder/vp8l/entropy.rs` - Entropy cost estimation, PopulationCost
- `src/encoder/vp8l/meta_huffman.rs` - Histogram clustering
- `src/encoder/vp8l/cost_model.rs` - TraceBackwards with Zopfli-style CostManager
- `src/encoder/vp8l/transforms.rs` - Image transforms
- `src/encoder/vp8l/types.rs` - Core data structures

**Cache auto-detection (2026-02-03):**
Entropy-based cache size selection uses a global histogram, but actual encoding uses
per-tile meta-Huffman codes. Larger cache expands the literal alphabet (280→408+
entries), increasing Huffman tree overhead per tile. Fix: try both auto-selected cache
and cache=0, return whichever is smaller. Improved CID22 50-image from 0.997x to 0.996x.

**Multi-config testing (2026-02-03):**
Matches libwebp's `CrunchConfig` approach. Tries multiple transform combinations
and keeps the smallest output:
- Methods 0-4: single config (no overhead)
- Method 5, quality >= 75: best guess + PaletteAndSpatial variant (if palette available)
- Method 6, quality 100: brute-force all viable modes (4-8 configs)
Six modes: Direct, Spatial, SubGreen, SpatialSubGreen, Palette, PaletteAndSpatial.
PaletteAndSpatial applies predictor transform on palette-indexed image.
Palette sorting variants (Lexicographic, MinimizeDelta) tested for palette modes at m6.

**Zopfli cost interval manager (2026-02-03):**
Replaced O(n * max_match_len) TraceBackwards DP with libwebp's interval-based
CostManager. Uses a sorted linked list of non-overlapping cost intervals to track
optimal paths without iterating over all copy lengths per pixel. Includes same-offset
optimization for constant-color regions. Compression ratio unchanged (0.997x CID22).

**Remaining potential improvements:**
- LZ77 Box strategy for non-palette structured images (currently only palette <=16)

### no_std Support (2026-01-23)

The crate supports `no_std` environments with the `alloc` crate:
- `cargo build --no-default-features` for no_std
- Both decoder and encoder work with no_std+alloc
- Decoder uses `&[u8]` slices instead of `Read` trait
- Encoder uses `Vec<u8>` instead of `Write` trait
- `Encoder.encode_to_writer()` requires `std` feature
- Dependencies: `thiserror` (no_std), `whereat` (no_std), `hashbrown` (no_std), `libm` (floating-point math)

### Decoder Performance vs libwebp (2026-02-04, after bounds check elision)

| Corpus | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| CLIC2025 (10 images) | 150-276 MPix/s | 180-280 MPix/s | **0.93-1.25x** |

*Benchmarks: 10 iterations per image, release mode with SIMD*

Our decoder averages **~1.15x slower** than libwebp, with one image now **faster** (0.93x).
Improved from ~1.28x after fixed-region bounds check elimination.

**Recent optimizations (2026-02-05):**
- **Fused IDCT + add_residue**: Combines IDCT and residual addition in one SIMD pass
  - Eliminates intermediate storage of IDCT output
  - Defers IDCT from coefficient reading to prediction time
  - DC-only fast path for blocks with only DC coefficient
- **Concrete X64V3Token types**: Replaced trait bounds for better target-feature optimization
- **32-pixel SIMD YUV conversion**: Process 32 Y pixels (16 chroma pairs) at once
  - YUV instructions dropped from 5.17% to 1.70% of total (3x improvement)
- **Fixed-size array optimization**: Eliminate bounds checks in SIMD inner loops
- **ActivePartitionReader** (commit 6327e15): Eliminated ~316K struct copy/drop per decode
  - Reader was being created/dropped once per 4x4 block, copying 32 bytes each time

**Per-decode instruction breakdown (CLIC 1360x2048, 2 iterations):**
| Category | zenwebp | % | Notes |
|----------|---------|---|-------|
| Coefficient reading | 278M | 6.96% | vs libwebp 215M GetCoeffsFast |
| Main decode loop | 171M | 4.27% | orchestration |
| YUV→RGB | 68M | 1.70% | down from 5.17% |
| Loop filters | 97M | 2.4% | 9 SIMD functions |

**Optimization attempts (2026-02-05):**
1. **SSSE3 shuffle for RGB interleave** - FAILED: archmage generates function calls for
   `_mm_shuffle_epi8` instead of inline pshufb. Made things 12% slower.
2. **Replace vec! with fill()** - Minimal impact: allocator was already efficient for
   small fixed-size allocations.

**Why YUV is 1.69x slower:** Our `planar_to_24b` does 5 passes of permutation to convert
R,G,B planes to interleaved RGB. When processing 16 pixels, we pass zeros for half the
inputs which wastes operations. A specialized 16-pixel interleave using SSSE3 shuffles
would be faster in theory, but archmage's function call overhead makes it worse.

**Recent optimizations:**
- **Unchecked coefficient reading (2026-02-05)** - `read_coefficients_inline_unchecked()` eliminates
  54 bounds checks when `unchecked` feature enabled. 4.2% instruction reduction on coefficient-heavy
  images, ~1.5% wall time improvement. Photographic images see smaller gains due to early termination.
- **Fixed-region bounds check elimination (2026-02-04)** - Add FILTER_PADDING (57KB) to pixel
  buffers, enabling fixed-size region extraction that eliminates per-access bounds checks.
  ~10% overall decode speedup. Memory overhead: ~170KB per decode.
- **Fused add_residue + coefficient clear (2026-02-04)** - Combines adding residuals with zeroing
  in one SIMD pass, eliminating separate fill(0) calls. 8.3% instruction reduction.

Previous optimizations:
- **SIMD chroma vertical loop filter** - U+V processed together as 16 pixels (10% speedup, 2026-01-23)
- **Position-indexed probability table** - eliminates COEFF_BANDS lookup (10% instruction reduction, commit 15b3771)
- **VP8HeaderBitReader for mode parsing** - replaces ArithmeticDecoder (6-8% faster, commit 9b6f963)
- **SIMD chroma horizontal loop filter** - U+V processed together as 16 rows (7.8% instruction reduction, commit c3b9051)
- **DC-only IDCT fast path** and reusable coefficient buffer (commit 9d08433)
- **SIMD horizontal loop filter** (DoFilter4/DoFilter6 for vertical edges, commit fc4c33f)
- **Inline coefficient tree** like libwebp's GetCoeffsFast (12% instruction reduction, commit ac47ba5)
- **16-bit SIMD YUV conversion** ported from libwebp (18% instruction reduction, commit b7ac4b9)
- **Row cache with extra rows** for loop filter cache locality (commit c16995f)
- **SIMD normal loop filter for vertical edges** (~10% speedup, commit 3bf30f1)
- **libwebp-rs style bit reader for coefficients** (16% speedup, commit 5588e44)
- AVX2 loop filter (16 pixels at once) - simple filter only
- **AVX2 32-pixel normal vertical filter** - normal_v_filter32_inner, normal_v_filter32_edge (commit 924e22d)
- **AVX2 32-row normal horizontal filter** - normal_h_filter32_inner, normal_h_filter32_edge (commit a08712e)

**Note on 32-pixel filter integration (2026-02-05):**
The 32-pixel AVX2 filters are implemented but NOT integrated into the decoder due to filter
order dependencies. Investigation found that MB(x)'s horizontal subblock edges (y=4,8,12)
read columns 12-15, which overlap with MB(x+1)'s left edge filter writes (columns 12-19).
This cross-MB dependency prevents restructuring to batch horizontal subblock edges across
adjacent macroblocks. The 32-pixel filters remain available for future use if the decoder
architecture changes (e.g., wider cache buffers to hold pre-filtered state).

### Decoder Profiler Hot Spots (2026-02-05, after partition reader caching)
| Category | % Time | M instr/decode | Notes |
|----------|--------|----------------|-------|
| Coefficient reading | ~10% | 13.2M | read_coefficients_inline (optimized) |
| YUV→RGB | ~16% | - | fancy upsampling with SIMD |
| Loop filter | ~12% | - | 9 SIMD filter functions |
| memset | ~5% | - | Output buffer zero-init |
| decode_frame (other) | ~20% | - | Main loop, prediction, transforms |

**Benchmark: 1024×1024 lossy WebP, 100 iterations**
- zenwebp: 3.93ms/decode (267 MPix/s)
- libwebp: 2.98ms/decode (351 MPix/s)
- **Ratio: 1.32x slower** (improved from 1.57x)

**Recent optimizations:**
- Partition reader caching (commit 6327e15): eliminated ~316K PartitionReader
  creates/drops per decode. Coefficient reading reduced from 83.7M to 13.2M
  instructions per decode on normalized basis (~6x improvement in reader overhead).
- SIMD token caching for IDCT (commit 52bd541)
- Diagnostic code removal from coefficient reader (commit 05b87ad)
- SIMD token caching for YUV→RGB (commit b8167e1)

Note: `add_residue_sse2` is now inlined via `add_residue_and_clear_sse2`.

**Bounds check analysis (cargo-asm, 2026-02-05):**
- Asserts at function entry do NOT eliminate bounds checks on individual slice accesses
- Each `try_from(&slice[a..b]).unwrap()` generates its own bounds check
- The pattern `<&[u8; 16]>::try_from(&pixels[offset..][..16]).unwrap()` generates:
  - Slice creation bounds check (verify offset <= len)
  - Slice truncation check (verify 16 <= remaining_len)
  - try_from check (verify slice.len() == 16)
- simple_v_filter16: ~50 instructions of bounds checks vs ~80 SIMD work (~40% overhead)
- Loop filters are 14.7% of decode time → bounds checks cost ~6% of total decode time
- This is the cost of safe Rust without `unsafe`; the compiler cannot prove slice
  operations are valid based on earlier asserts due to generic `try_from` internals

**Fixed-region bounds check elimination (2026-02-04, IMPLEMENTED):**
Using fixed-size arrays eliminates per-access bounds checks:
```rust
const V_FILTER_REGION: usize = 3 * MAX_STRIDE + 16;
let region: &mut [u8; V_FILTER_REGION] =
    <&mut [u8; V_FILTER_REGION]>::try_from(&mut pixels[start..start + V_FILTER_REGION]).unwrap();
// All subsequent region[offset..] accesses have NO bounds checks
```
- Assembly confirmed: interior accesses use direct memory loads with no checks
- **Solution:** Add FILTER_PADDING (57KB for normal filter, 8K support) to pixel buffers
- Updated: `ybuf/ubuf/vbuf` and `cache_y/u/v` allocations include padding
- Updated: `simple_v_filter16`, `normal_v_filter16_inner`, `normal_v_filter16_edge`
- **Result:** ~10% decode speedup, now ~1.15x vs libwebp (from ~1.28x)
- Memory overhead: ~170KB per decode (57KB × 3 planes, negligible for typical images)

**`unchecked` feature (2026-02-05, IMPLEMENTED):**

For maximum performance with trusted input, the `unchecked` feature eliminates ALL bounds
checks in loop filter hot paths using raw pointer arithmetic:

```toml
[dependencies]
zenwebp = { version = "0.2.1", features = ["simd", "unchecked"] }
```

Unchecked filter functions implemented:
- `simple_v_filter16_unchecked` - simple vertical filter
- `normal_v_filter16_inner_unchecked` - normal vertical inner filter
- `normal_v_filter16_edge_unchecked` - normal vertical edge filter
- `simple_h_filter16_unchecked` - simple horizontal filter
- `normal_h_filter16_inner_unchecked` - normal horizontal inner filter
- `normal_h_filter16_edge_unchecked` - normal horizontal edge filter

Assembly comparison (simple_h_filter16):
- Safe version: ~30 bounds check instructions (`cmp`, `ja`, `jbe`) per call
- Unchecked version: zero bounds check instructions

**WARNING:** Only enable `unchecked` if you trust your WebP input (e.g., self-generated
files, trusted sources). Malformed input may cause undefined behavior instead of clean
error returns.

### Detailed Callgrind/Cachegrind Analysis (2026-01-23)

**Per-decode instruction count (after chroma vertical SIMD):**
| Metric | Ours | libwebp | Ratio |
|--------|------|---------|-------|
| Coeff reading | 49.1M | 36.4M | 1.35x |
| Total program | 20.88B/200 | - | -10% vs previous |

*Total instructions reduced from 23.2B → 20.88B (-10%)*
| Memory reads | ~12M | 6.7M | 1.8x |
| Memory writes | ~9M | 4.9M | 1.8x |
| D1 read miss % | 0.33% | 0.70% | Better! |

*Note: 14443M / 200 decodes = 72.2M per decode. Improved from 78.7M (8% reduction since chroma SIMD).*

**Instruction breakdown comparison:**
| Category | Ours | libwebp | Extra per decode |
|----------|------|---------|------------------|
| Coeff reading | ~24M (33%) | 20.5M (44%) | +3.5M |
| Loop filter total | ~14M (19%) | ~4M (8%) | **+10M** |
| YUV conversion | ~6M (8%) | ~4M (9%) | +2M |
| Mode parsing (inlined) | ~0.2M | 0 | +0.2M |

**Root causes of the remaining 1.55x instruction gap:**
1. **Loop filter overhead** - still 2.5x more instructions than libwebp despite SIMD
2. **Coefficient reading** - ~17% more instructions than libwebp
3. **Function call overhead** - Rust abstractions vs libwebp's inline macros

Loop filter now uses SIMD for:
- Simple filter: both V and H edges (luma + chroma)
- Normal filter: both V and H edges (luma + chroma with SIMD)

### SIMD Decoder Optimizations
- `src/yuv_simd.rs` - SSE2 YUV→RGB (ported from libwebp, commit b7ac4b9)
  - **16-bit arithmetic** using `_mm_mulhi_epu16` (8 values at once vs 4)
  - **SIMD RGB interleaving** via `planar_to_24b` (6 stores vs 48)
  - Processes 32 pixels at a time (simple) or 16 (fancy upsampling)
  - `simd` feature now enabled by default
- `src/loop_filter_avx2.rs` - SSE4.1 loop filter (16 pixels at once)
  - **Normal filter SIMD for both V and H edges** (luma, commit fc4c33f)
  - **Chroma horizontal SIMD** - U+V processed together as 16 rows (commit c3b9051)
  - **Chroma vertical SIMD** - U+V packed as 16 pixels per row (2026-01-23)
  - Uses transpose technique for horizontal filtering (16 rows × 8 cols)
  - Simple filter: both V and H edges have SIMD (luma + chroma)
  - Normal filter: both V and H edges have SIMD (luma + chroma)
- `src/vp8_bit_reader.rs` - libwebp-rs style bit readers
  - **VP8HeaderBitReader** - header/mode parsing (commit 9b6f963)
  - **VP8Partitions/PartitionReader** - coefficient reading (commit 5588e44)
  - Uses VP8GetBitAlt algorithm with 56-bit buffer
  - `leading_zeros()` for normalization (single LZCNT instruction)
  - **16% speedup** in overall decode from coefficient reader

### TODO - Remaining Optimization Opportunities
Priority ordered by instruction savings potential:

1. **Loop filter overhead** (~10M savings)
   - Still 2.5x more instructions than libwebp despite SIMD
   - Per-pixel threshold checks (`should_filter_*`) are expensive
   - Consider batch threshold computation

Completed:
- [x] ~~Chroma vertical SIMD~~ (2026-01-23) - U+V packed as 16 pixels, 10% speedup
- [x] ~~Position-indexed probability table~~ (commit 15b3771) - 10% instruction reduction
- [x] ~~Replace ArithmeticDecoder for mode parsing~~ (commit 9b6f963) - 6-8% faster
- [x] ~~Loop filter SIMD for chroma horizontal~~ (commit c3b9051) - U+V together as 16 rows, 7.8% reduction
- [x] ~~DC-only IDCT and reusable coefficient buffer~~ (commit 9d08433)
- [x] ~~Row cache with extra rows for loop filter cache locality~~ (commit c16995f)
- [x] ~~Add SIMD horizontal normal filter~~ (commit fc4c33f)
- [x] ~~Inline coefficient tree like GetCoeffsFast~~ (commit ac47ba5)
**Encoder optimization opportunities (see Callgrind Analysis above):**
- [ ] trellis_quantize_block - 7.8x slower, SIMD quantization like QuantizeBlock_SSE2
- [ ] encode_coefficients - 5.3x slower, token emission overhead
- [ ] get_residual_cost - 2.6x slower, SIMD residual cost like GetResidualCost_SSE2
- [ ] get_cost_luma16 - 3.9x slower
- [ ] tdisto_16x16 - 3.0x slower (despite having some SIMD)

## VP8L (Lossless) Encoder Status (2026-02-03)

**Compression ratio vs libwebp (CID22 corpus, Q75, M4):**
- Training set (209 images): **0.996x** (0.4% smaller)
- Validation set (41 images): **0.987x** (1.3% smaller)

**Implemented features:**
- All 14 predictor modes with per-block selection
- Cross-color transform, subtract green transform
- Color indexing (palette) with sorting, delta encoding, pixel bundling
- LZ77 Standard, RLE, Box strategies with TraceBackwards DP
- Meta-Huffman spatially-varying codes with histogram clustering
- Color cache with entropy-based size selection
- imagequant integration (`quantize` feature) for near-lossless palette encoding
- Near-lossless: pixel-level preprocessing + residual-level quantization

**Near-lossless support (2026-02-03):**

Two levels matching libwebp:
1. **Pixel-level preprocessing** (`apply_near_lossless`): Quantizes non-smooth raw
   pixels before transforms. Active when predictor is NOT used (SubGreen/Direct modes).
2. **Residual-level quantization** (`near_lossless_residual`): Quantizes prediction
   residuals inside the predictor transform with forward-order processing. Active when
   predictor IS used (Spatial/SpatialSubGreen modes). Source pixels updated with
   reconstructed values for prediction consistency.

| Corpus | near_lossless | Size savings | vs libwebp |
|--------|--------------|-------------|------------|
| CID22 (250 imgs) | 60 | 30.5% | 1.003x |
| Screenshots (10 imgs) | 60 | 14.1% | 0.999x |
| CID22 | 100 | 0.0% (exact) | 0.995x |

**Outlier images:**
- 3616956: 1.128x (312KB vs libwebp 277KB) — cache selection overvalues high cache_bits
- 2079234: 1.059x (300KB vs libwebp 283KB) — same cache issue + LZ77 gap

**Cache selection bug (2026-02-03, IDENTIFIED):**
Entropy estimation picks cache_bits=8-10 for outlier images, but actual output is smaller
with cache_bits=0-2. Root cause: LZ77 refs are optimized for one cache size, and entropy
estimation at other sizes is misleading. With forced cache_bits=2, 3616956 goes from
312KB → 307KB (1.6% improvement). Even so, still 11% larger than libwebp.

**Config fix:** `Vp8lConfig.cache_bits` changed from `u8` to `Option<u8>`:
- `None` = auto-detect (default)
- `Some(0)` = disable cache
- `Some(N)` = max N bits search

**Key files:**
- `src/encoder/vp8l/encode.rs` — Main encoder pipeline
- `src/encoder/vp8l/backward_refs.rs` — LZ77 + TraceBackwards + cache selection
- `src/encoder/vp8l/histogram.rs` — Symbol frequency histograms
- `src/encoder/vp8l/meta_huffman.rs` — Histogram clustering
- `src/encoder/vp8l/entropy.rs` — Entropy estimation (matches libwebp)
- `src/encoder/vp8l/huffman.rs` — Huffman tree construction
- `src/encoder/vp8l/transforms.rs` — Predictor, cross-color, palette transforms
- `src/encoder/vp8l/near_lossless.rs` — Near-lossless pixel + residual quantization
- `src/encoder/color_quantize.rs` — imagequant integration (requires `quantize` feature)

**Diagnostic examples:** `cache_test`, `lossless_benchmark` (see examples/)

## Mux/Demux/Animation Module (2026-02-03)

**Key files:**
- `src/mux/mod.rs` — Module declarations, re-exports
- `src/mux/error.rs` — MuxError enum
- `src/mux/demux.rs` — WebPDemuxer (zero-copy chunk parser)
- `src/mux/assemble.rs` — WebPMux (container assembler)
- `src/mux/anim.rs` — AnimationEncoder (high-level animation API)
- `tests/mux_tests.rs` — 20 roundtrip/validation tests

**Features:**
- Demux: parse WebP at chunk level, iterate frames without decoding pixels
- Mux: assemble WebP containers from pre-encoded chunks + metadata
- Animation: encode animated WebP frame-by-frame using existing encoder
- All no_std + alloc compatible

**Design notes:**
- WebPDemuxer records byte ranges into original `&[u8]`, no copies
- AnimationEncoder uses timestamp-based durations (next - this)
- Frame offsets must be even (WebP spec requirement)
- assemble.rs (not mux.rs) to avoid clippy same-name-as-module lint

## Safety

This crate uses `#![forbid(unsafe_code)]` to prevent direct unsafe in source files.

**With `simd` feature:** We rely on the `archmage` crate for safe SIMD intrinsics. The
`#[arcane]` proc macro generates `unsafe` blocks internally that bypass the `forbid` lint
(due to proc-macro span handling). Soundness depends on archmage's token-based safety:
- Tokens are only created via `X64V3Token::summon()` which does runtime CPU feature detection
- No token forging patterns exist in our code
- No manual `unsafe` blocks, `transmute`, `get_unchecked`, or raw pointer derefs in source

**Without `simd` feature:** No unsafe code whatsoever - pure safe Rust.

## Known Bugs

(none currently)

### Historical Note: Method Realignment (2026-02-04, RESOLVED)

Apparent "regression" from 0.9888x to 1.0099x at method 4 was NOT a bug. Commit `47873f5`
intentionally moved trellis from m4 to m5 to align with libwebp's RD optimization levels.
The old 0.9888x compared our m4 (with trellis) to libwebp m4 (without trellis) - unfair.

Correct comparison after realignment:
- Method 4 (no trellis): 1.0099x (1% larger, matches libwebp m4 feature set)
- Method 5 (trellis): **1.0002x** (near-perfect parity with libwebp m5)

**TODO:** Bisect between e91f826..HEAD to find regression commit

## Investigation Notes

### I4 Mode Context Bug (2026-02-02, FIXED)

**Bug:** Edge blocks in I4 mode selection were using hardcoded DC (0) context instead of
cross-macroblock context from `top_b_pred` and `left_b_pred`. This caused suboptimal mode
selection for edge blocks because the mode cost tables are context-dependent.

**Root cause:** The `top_b_pred` and `left_b_pred` arrays were only being updated during
header writing (after all mode selection was complete), not during the encoding pass.
When mode selection ran for MB(x,y), the context for edge blocks came from either:
- Reset values (start of row)
- Stale values from a previous pass

**Fix (2 parts):**
1. `mode_selection.rs`: Use `self.top_b_pred[mbx * 4 + sbx]` and `self.left_b_pred[sby]`
   for edge block context instead of hardcoded 0
2. `mod.rs`: Update `top_b_pred` and `left_b_pred` immediately after `choose_macroblock_info()`
   returns, so the next macroblock sees correct context

**Results:**
- Mode distribution shifted: -257 DC modes, +93 VE, +73 HE, +922 TM
- 792079 benchmark: 12174 → 12164 bytes (-0.08%)
- CID22 corpus (248 images): 0.999x → 0.990x (zenwebp now 1% smaller than libwebp)

### Multi-pass Probability Signaling Bug (2026-02-02, FIXED)

**Bug:** Multi-pass encoding (methods 5-6) produced corrupt output that decoded incorrectly.
99.9% of Y plane pixels differed between method 4 and method 5 decoded output.

**Root cause:** Probability update signaling compared against wrong baseline.
- In multi-pass, `token_probs` was set to pass 0's updated values at start of pass 1
- `compute_updated_probabilities` compared new probabilities against pass 0 values
- Finding no savings (pass 1 ≈ pass 0), it set `updated_probs = None`
- Header signaled 0 updates (comparing against pass 0, not defaults)
- Decoder started with `COEFF_PROBS` (defaults) but encoder used pass 0 probs
- Mismatch caused garbage decoding

**Fix (2 parts):**
1. `encode_updated_token_probabilities` now compares against `COEFF_PROBS` (decoder defaults),
   not against `token_probs` (which may have intermediate pass values)
2. `compute_updated_probabilities` now always computes probabilities relative to `COEFF_PROBS`

**Results after fix:**
- All methods (4, 5, 6) now produce valid decodable output
- Method 4: 1.018x of libwebp (near parity)
- Method 5: 1.024x of libwebp (slightly larger due to no true multi-pass benefit)
- Method 6: 1.044x of libwebp (same)

### Multi-pass Re-quantization Experiment (2026-02-02, ABANDONED)

**Tested approaches and results:**

1. **Token re-recording (no re-quantization):** Re-record same tokens in pass 1+ with
   updated probability tables. **Result:** ZERO benefit - outputs are byte-identical
   because identical tokens → identical statistics → identical probabilities.

2. **Re-quantization from stored raw DCT:** Store raw DCT coefficients from pass 0,
   re-quantize in pass 1+ with level_costs derived from observed probabilities.
   **Result:** Files became LARGER (+0.82%) with WORSE quality (MSE 649 vs 511).

**Root cause analysis (revised):** The re-quantization approach was fundamentally
flawed, not just due to probability estimation issues:

1. Raw DCT coefficients depend on PREDICTION residuals
2. Predictions depend on RECONSTRUCTED pixels from previous macroblocks
3. Reconstructed pixels depend on current pass's quantized coefficients
4. If pass 1 trellis makes different decisions → different reconstructed pixels →
   different predictions for subsequent macroblocks → raw DCT from pass 0 is WRONG

Using pass 0's raw DCT for pass 1 assumes predictions would be identical, but changing
trellis decisions changes the entire reconstruction chain. This is why re-quantization
produces worse results - we're re-quantizing the wrong residuals.

**Why token re-recording provides zero benefit:** Once tokens are decided in pass 0,
the statistics are fixed. Re-recording the same tokens accumulates the same statistics,
leading to identical probability tables and identical output.

**How libwebp actually does multi-pass (confirmed from source):**

libwebp does FULL re-encoding in each pass. Key code in `frame_enc.c` and `iterator_enc.c`:

1. `VP8IteratorImport()` imports SOURCE pixels from original image to `yuv_in`
2. `VP8Decimate()` computes prediction from boundary buffers, residual, DCT, quantize
3. `VP8IteratorSaveBoundary()` saves RECONSTRUCTED pixels from `yuv_out` to boundaries

The critical insight: boundaries are saved from `yuv_out` (reconstructed), not `yuv_in`
(source). So when pass N runs, it uses boundaries from pass N's previous macroblocks,
which may differ from pass N-1. Each pass recomputes:
```
prediction = f(reconstructed_neighbors_from_THIS_pass)
residual = source - prediction
raw_dct = DCT(residual)
quantized = trellis(raw_dct, level_costs_from_previous_pass)
reconstructed = IDCT(dequantize(quantized)) + prediction
```

This is fundamentally different from our failed approach (storing raw DCT from pass 0).
Pass 0's raw DCT was `DCT(source - prediction_pass0)`, but pass 1 would have different
predictions due to different reconstructed neighbors, so the residuals differ.

**Code status:** Multi-pass implemented (2026-02-02). Previous re-quantization approach
removed. Now does full re-encoding like libwebp.

### Multi-pass Implementation (COMPLETED 2026-02-02)

Multi-pass encoding now matches libwebp's VP8EncTokenLoop behavior:

**Implementation details:**
- m5 = 2 passes, m6 = 3 passes (matching `config->pass`)
- Each pass: clear token buffer, full encode (predict → DCT → quantize → record)
- Pass 1+: apply updated probabilities → recalculate level_costs → full re-encode
- Statistics accumulate across intermediate passes (reset only on last pass, like libwebp)
- Borders come from RECONSTRUCTED pixels (already correct in our encoder)
- `reset_for_new_pass()` resets all encoder state between passes

**Results:** Multi-pass provides NO compression benefit without quality search.
Both zenwebp and libwebp produce ~0.1-0.3% LARGER files with multiple passes.
This matches libwebp's design where multi-pass is meant for `do_size_search` or
`do_psnr_search` convergence, not for probability refinement alone.

**Future improvement:** Implement quality search (binary search for target size).
This would allow m5/m6 to hit a target file size by adjusting quality between passes

### SNS Quality-Size Tradeoff Investigation (2026-02-01)

**Butteraugli corpus test results (2026-02-01):**

| Corpus | Encoder | Size Δ | SSIM2 Δ | Butteraugli Δ |
|--------|---------|--------|---------|---------------|
| CID22 | zenwebp Photo | 0.991x | -3.04 | +0.91 |
| CID22 | libwebp Photo | 0.993x | -0.14 | +0.17 |
| Screenshots | zenwebp Photo | 0.966x | -3.99 | +1.04 |
| Screenshots | libwebp Photo | 0.959x | -1.19 | +0.13 |

**Key finding:** Both SSIM2 and butteraugli agree - our SNS implementation produces
steeper quality degradation than libwebp's for equivalent size savings. The gap is
especially visible on `terminal` screenshot (-25.07 SSIM2 for zenwebp vs -1.63 for libwebp).

**Root cause analysis:** The issue is NOT the content classifier (it correctly detects
Photo-appropriate content). The issue was in our segment quantization spread calculation.
When SNS=80, we assigned larger quant deltas between segments than libwebp does.

**FIXED (2026-02-03):** Bug was in `compute_segment_quant()` computing c_base from
base_quant instead of quality directly. See SNS c_base fix in Current Status section.
Results after fix: CID22 Q90 improved from 1.0126x → 1.0060x (near parity).

### webpx Preset Bug (2026-02-01, RESOLVED)

webpx 0.1.2 had a bug where `to_libwebp()` overwrote preset-specific filter values
with defaults after `WebPConfigInitInternal` correctly set them. Fixed in webpx 0.1.3
by making `sns_strength`, `filter_strength`, `filter_sharpness` fields `Option<u8>`.
`None` = use preset default, `Some(v)` = user explicitly set.

### TokenType Fix and File Size Investigation (2026-01-24)

**Root cause found:** The TokenType enum had swapped values for I16AC and I16DC.
This caused coefficient cost estimation and trellis quantization to use incorrect
probability tables, leading to suboptimal coefficient level choices.

**After fix:** File sizes reduced from 1.07-1.17x to 1.045-1.135x of libwebp.

**Remaining 4.5% gap investigation (updated 2026-02-02):**
- Lambda calculations: Match libwebp exactly
- VP8_LEVEL_FIXED_COSTS table: Matches libwebp exactly
- VP8_FREQ_SHARPENING table: Matches libwebp exactly
- VP8_WEIGHT_TRELLIS table: Matches libwebp exactly
- RD score formula: Matches libwebp exactly
- Trellis distortion calculation: Matches libwebp (uses coeff_with_sharpen)
- **Sign bit cost in trellis: FIXED** — was double-counted (0.4% improvement)

**Diagnostic harness findings (2026-02-02):**
- I4 diagnostic harness (`tests/i4_diagnostic_harness.rs`) compares:
  - Segment quantizers: MATCH
  - Mode decisions: 74.5% match at m4 (74.8% after sign fix)
  - Coefficient levels: 48% exact match (same modes, different levels)
  - Probability tables: 86.8% match on real images
- Probability differences are largest in I4 path (plane 1), high-frequency bands
- Our probabilities for level>1 are systematically LOWER, causing more bits for higher levels
- Coefficient distribution: we have fewer level=1, more level 2-4 than libwebp

**SIMD parity verified (2026-02-02):**
Tested encoding with and without `simd` feature - outputs are **identical**.
SIMD is NOT causing bit-level differences in encoding. The mode selection and
coefficient differences are algorithmic, not SIMD-related.

**Likely root cause of remaining 4.5% trellis gap:**
Our trellis RD optimization slightly favors lower distortion over rate reduction,
resulting in more higher-level coefficients. This matches the observed coefficient
distribution pattern.

**Key insight (2026-02-02):**
The 4.5% gap in diagnostic tests (SNS=0, filter=0, segments=1) disappears in
production settings. With default presets, zenwebp actually beats libwebp by
3-5% due to effective SNS and filtering implementation. The remaining I4
coefficient efficiency gap is compensated by other features in production use.

### I4 Mode Path Investigation (2026-02-01)

**Key finding: The file size gap is in the I4 mode path, not I16.**

Method comparison for 792079 (512x512) and terminal (1646x1062):

| Image | Method | zenwebp | libwebp | Ratio |
|-------|--------|---------|---------|-------|
| 792079 | 0 (I16 only) | 13294 | 14660 | 0.91x |
| 792079 | 2 (I4, no trellis) | 12470 | 11318 | 1.10x |
| 792079 | 4 (I4 + trellis) | 12188 | 10962 | 1.11x |
| terminal | 0 (I16 only) | 77490 | 77510 | 1.00x |
| terminal | 2 (I4, no trellis) | 70092 | 64098 | 1.09x |
| terminal | 4 (I4 + trellis) | 68888 | 61612 | 1.12x |

**Analysis:**
- Method 0 (I16 only): zenwebp produces **smaller** files than libwebp (0.91-1.00x)
- Methods 2-6 (with I4): zenwebp produces **larger** files (1.09-1.14x)
- libwebp's I4 gives ~10% reduction from I16; ours gives only ~2%

**Root cause hypothesis:** Our I4 mode path doesn't achieve the same compression
benefits as libwebp's. Either:
1. We're using I4 where it doesn't help (over-selecting I4)
2. Our I4 coefficient encoding is less efficient
3. I4 mode cost estimation underestimates true cost

**First attempt (failed):** Adding sharpening and zthresh to quantize_coeff made things
*worse* in a prior session — the implementation was incorrect.

**Correct fix (2026-02-02):** `quantize_coeff()` was missing sharpen and zthresh that
libwebp's `QuantizeBlock_C` uses. The trellis path already had sharpen; only the simple
quantization path was broken. This affected both mode selection (`pick_best_intra4`) and
non-trellis encoding (methods 0-3).

Fix: `abs_coeff = |coeff| + sharpen[pos]`, then skip if `abs_coeff <= zthresh[pos]`.
Also updated SIMD `quantize()` and `quantize_ac_only()` block methods.

**Results after sharpen/zthresh fix:**
- CID22 corpus (method 4): 1.05x → **1.043x** of libwebp
- 792079 benchmark image: 1.11x → **1.024x** of libwebp
- Screenshots: unchanged at 1.060x (expected — mostly I16)

**Earlier experiments (2026-02-02):**

| Experiment | Effect on screenshot corpus size |
|------------|----------------------------------|
| Try all 10 I4 modes for method 4 | **-0.71% (better!)** - implemented |
| Fix get_cost_luma16 context tracking | **-0.2% (better!)** - implemented |
| Add spectral distortion (TDisto) to I4 | +0.06% (worse) |
| Add flatness penalty for I4 | +0.03% (worse) |
| Disable trellis for method 4 (match libwebp) | +1.5% (much worse, trellis helps us) |

**Token buffer resolved the ~4% two-pass mismatch gap (2026-02-02):**
Token buffer (commit 9625d4e) replaced two-pass with single-pass recording.
Multi-pass experiments abandoned (see Investigation Notes). Methods 5/6 now equivalent to m4.
CID22 aggregate at m4: 1.043x → **0.999x**.

**Remaining m5/m6 gap (2026-02-02):**
Our trellis compensates for weaker non-trellis I4 efficiency. libwebp adds
trellis as a NEW capability at m5/m6 on top of an already-efficient m4 baseline.
We add trellis at m4 to REACH that baseline. So m5/m6 have nothing new to add.
Root cause: I4 coefficient pipeline efficiency. See decomposition plan below.

### I4 Over-Selection Investigation (2026-02-03)

**Summary:** Investigated why zenwebp chooses I4 mode for ~5% more macroblocks than
libwebp (73.4% vs 68.3% on 792079.png), resulting in 1-2% larger files.

**Key findings from MB_DEBUG analysis:**

1. **When we choose I4 but libwebp chooses I16:** We produce MORE nonzero coefficients
   - MB(3,0): zen I4 has 7 nonzeros, lib TM has 2 (+5)
   - MB(10,2): zen I4 has 8 nonzeros, lib TM has 0 (+8)
   - MB(27,19): zen I4 has 10 nonzeros, lib DC has 0 (+10)

2. **When we choose I16 but libwebp chooses I4:** We produce FEWER nonzeros
   - MB(30,25): zen V has 0 nonzeros, lib I4 has 13 (-13)
   - MB(6,14): zen DC has 0 nonzeros, lib I4 has 6 (-6)

**Conclusion:** Our I16 is MORE efficient than libwebp's I4, but our I4 is LESS
efficient than libwebp's I16. The mode selection isn't wrong given our cost
estimation - I4 really does produce lower RD cost during selection. But the
final encoded output shows libwebp's I16 would have been better.

**Root cause analysis:**
- Quantizers and mode costs (FIXED_COSTS_I16) match libwebp exactly
- First 3 MBs of row 0 have identical mode decisions (borders match)
- Divergence starts at MB(3,0) with same borders → different cost estimation
- BMODE_COST sweep (211 to 3000) showed NO monotonic improvement
- Issue is NOT a simple threshold - it's in the RD cost balance

**BMODE_COST sweep results (792079.png):**
| BMODE_COST | File Size | I4 Usage |
|------------|-----------|----------|
| 211 (orig) | 12,128 | ~70% |
| 500 | 12,384 | 70.2% |
| 1000 | 12,424 | 69.9% |
| 2000 | 12,400 | 73.6% |
| 3000 | 12,418 | 69.9% |

**Current status (2026-02-03):**
- CID22 corpus (248 images): 1.0101x of libwebp (1.01% larger)
- BMODE_COST kept at 211 (libwebp's value) - higher values don't help
- Root cause is subtle coefficient cost estimation difference, not threshold

**Diagnostic examples added:**
- `analyze_mb.rs` - Compare disputed MB coefficient counts
- `verify_cost_estimation.rs` - Compare I4 usage and savings per MB

### I4 Encoding Efficiency Decomposition Plan (2026-02-02)

**Goal:** Identify WHERE in the I4 pipeline we lose efficiency vs libwebp, so
we can verify each stage independently instead of treating it as a black box.

**Approach:** For a single macroblock (or small image), compare intermediate
outputs at each stage between zenwebp and libwebp. Build a test harness that
feeds identical input and compares stage-by-stage.

**Pipeline stages to verify (for a single I4 macroblock):**

1. **Prediction residuals** — Given identical pixels + prediction mode + reference
   pixels, do we produce identical residual coefficients before quantization?
   - zenwebp: `transform_luma_block()` → `y_block_data`
   - libwebp: `VP8EncIterator.yuv_in/yuv_out` → `VP8FTransform()`
   - Compare: 16×[16 coefficients] per I4 MB

2. **Quantization** — Given identical residuals + quantization matrix, do we
   produce identical quantized levels?
   - zenwebp: `VP8Matrix::quantize()` / `quantize_ac_only()` (non-trellis path)
   - libwebp: `QuantizeBlock_C()` — includes sharpen and zthresh
   - Compare: quantized levels for each 4x4 block
   - Also compare: sharpen[] and zthresh[] arrays per matrix

3. **Mode selection decisions** — Given identical cost tables, do we choose
   the same I4 prediction modes?
   - zenwebp: `pick_best_intra4()` → `bpred[16]`
   - libwebp: `PickBestIntra4()` → `modes_i4[16]`
   - Compare: per-block mode choices, per-block RD scores

4. **Token/coefficient encoding** — Given identical quantized levels + probability
   tables, do we produce the same token stream?
   - zenwebp: `record_coeff_tokens()` → token buffer
   - libwebp: `VP8RecordCoeffTokens()` → token buffer
   - Compare: token count per block, total bit cost estimate

5. **Probability table updates** — Given identical token statistics, do we
   compute the same updated probability tables?
   - zenwebp: `compute_updated_probabilities()` → `updated_probs`
   - libwebp: `FinalizeTokenProbas()` → `proba.coeffs`
   - Compare: all 4×8×3×11 probability values

6. **Arithmetic encoding** — Given identical tokens + probabilities, does our
   arithmetic encoder produce identical byte output?
   - zenwebp: `TokenBuffer::emit_tokens()` → ArithmeticEncoder
   - libwebp: `VP8EmitTokens()` → VP8BitWriter
   - Compare: final byte stream length and content

**Implementation:** Build a diagnostic test that:
- Takes a small test image (e.g., 16x16 or 32x32 crop of 792079)
- Encodes with both zenwebp and libwebp via FFI
- Dumps intermediate state at each stage via debug hooks
- Reports first divergence point

**Key insight:** If stages 1-2 match but stage 3 diverges, the problem is in
mode selection cost estimation. If stages 1-3 match but stage 4 diverges,
the problem is in token encoding. This narrows the search space dramatically.

**Files involved:**
- `src/encoder/vp8/prediction.rs` — stage 1 (transforms)
- `src/encoder/quantize.rs` — stage 2 (quantization)
- `src/encoder/vp8/mode_selection.rs` — stage 3 (mode decisions)
- `src/encoder/vp8/residuals.rs` — stage 4 (token recording)
- `src/encoder/cost.rs` — stage 5 (probability updates)
- `src/encoder/arithmetic.rs` — stage 6 (arithmetic encoder)

### VP8BitReader Success (2026-01-22)

Initial attempt to replace ArithmeticDecoder was slower. Second attempt using
libwebp-rs's exact algorithm succeeded with **16% speedup**.

Key differences from initial failed attempt:
1. Used libwebp's VP8GetBitAlt algorithm exactly (not our modified version)
2. Stored `range - 1` internally (127-254 interval) matching libwebp
3. Used `leading_zeros()` which compiles to single LZCNT instruction
4. Simpler split calculation: `split = (range * prob) >> 8`

Verified algorithms produce identical bit streams (0 differences across all probabilities).

Results:
- Micro-benchmarks: 24-37% faster bit reading
- Full decoder: 55 → 62 MPix/s (16% speedup)
- Gap vs libwebp: 2.5x → 2.15x slower

The ArithmeticDecoder is still used for header/mode parsing (self.b field).
Only coefficient reading uses the new VP8Partitions/PartitionReader.

### Loop Filter Optimization (Resolved 2026-01-23)

Normal filter now uses SIMD for horizontal edges (both luma and chroma).
Chroma is handled by processing U and V planes together as 16 rows, reusing
the existing 16-row infrastructure. See commit c3b9051.

Remaining loop filter overhead (~6M extra instructions vs libwebp) appears
to come from per-pixel threshold checks rather than the filter math itself.

### AVX2 32-Pixel Loop Filter Batching (2026-02-05, BLOCKED)

**Goal:** Process 32 columns/rows at once using AVX2 for 2x throughput.

**Implemented filters (commits 64a6847, 924e22d, a08712e):**
- `simple_v_filter32` - 32-column simple vertical filter
- `normal_v_filter32_inner`, `normal_v_filter32_edge` - 32-column normal vertical filters
- `normal_h_filter32_inner`, `normal_h_filter32_edge` - 32-row normal horizontal filters
- All filters tested and verified correct, but NOT integrated due to dependency issues.

**Why 32-column batching (vertical filters) is blocked:**

Filter order for each macroblock:
```
1. Left edge (horizontal filter on vertical edge)
2. Vertical subblock edges at x=4,8,12 (horizontal filters)
3. Top edge (vertical filter on horizontal edge)
4. Horizontal subblock edges at y=4,8,12 (vertical filters)
```

**Dependency 1: Cross-MB write interference (2026-02-05)**
- MB(mbx+1)'s left edge filter writes to columns 13-18 (columns 13-15 overlap MB(mbx))
- MB(mbx)'s horizontal subblock filters (step 4) read all 16 columns including 13-15
- Original order: MB(mbx) complete → MB(mbx+1) left edge writes → ...
- Two-pass attempt: All horizontal filters for all MBs, then all vertical filters
- Result: MB(mbx+1)'s left edge modifies columns before MB(mbx) finishes reading them
- **15 test failures** with two-pass approach

**Dependency 2: Top edge vs horizontal subblock overlap**
- Top edge filter at y=extra_y_rows writes to rows (extra_y_rows-3) to (extra_y_rows+2)
- Horizontal subblock at y=4 reads rows (extra_y_rows+1) to (extra_y_rows+7)
- Overlap: rows extra_y_rows+1 to extra_y_rows+2 (2 rows)
- Cannot defer top edge without changing what horizontal subblock reads

**Why 32-row batching (horizontal filters) requires wider cache:**

32-row horizontal filters process left edges of MB(mbx, mby) and MB(mbx, mby+1) together.
This requires 32 rows of decoded data in cache simultaneously.

**Current cache:** `extra_y_rows + 16` rows (one MB row + overlap for edge filtering)
**Required cache:** `extra_y_rows + 32` rows (two MB rows)

**Implementation complexity for wider cache:**
1. Double cache allocation in `read_frame_header()`
2. Add `cache_row_offset` parameter to `intra_predict_luma/chroma`
3. Restructure main decode loop to process 2 MB rows before filtering
4. Create `filter_two_rows_in_cache()` using 32-row horizontal filters
5. Adjust `output_row_from_cache()` and `rotate_extra_rows()` for paired rows

**Expected benefit:** Loop filter is ~15% of decode time. Even 30% reduction in filter
overhead yields only ~5% total improvement. High complexity for modest gain.

**Conclusion:** The 32-pixel filters are implemented and tested but integration is
blocked by fundamental filter order dependencies. The only viable path is wider cache
buffers for 32-row horizontal batching, which requires significant restructuring.
Consider if ~5% improvement justifies the implementation complexity.

### libwebp-rs Mechanical Translation Comparison (2026-01-22)

Profiled `~/work/webp-porting/libwebp-rs` - a direct Rust port of libwebp:

| Decoder | Arith Decoder | Loop Filter | Overall |
|---------|--------------|-------------|---------|
| image-webp (ours) | 24% | ~11% | 2.5x slower |
| libwebp-rs (mechanical) | 11% | 24% | 2.6x slower |
| libwebp C | baseline | baseline | 1.0x |

Key finding: **Both Rust decoders have similar ~2.5x slowdown** despite different
bottleneck distributions. The mechanical translation uses rayon parallelism and
libwebp's exact bit reader algorithm, yet achieves similar performance to our
hand-written decoder.

libwebp-rs uses:
- libwebp's VP8GetBitAlt algorithm with `range - 1` representation
- `leading_zeros()` for normalization (not lookup table)
- SIMD for macroblock edge filters but scalar for inner edge (`filter_loop24`)
- rayon parallelism (adds ~18% overhead)

The inner edge filter `filter_loop24` is 24% of their time - scalar loop over
16 pixels. Same bottleneck pattern as our decoder.

Conclusion: The ~2.5x gap may be inherent to Rust vs C for this workload, or
requires deeper investigation into codegen/memory access patterns.

### Cache/Memory Layout Analysis (2026-01-22)

**Root cause of 18x more L1 cache misses identified.**

#### libwebp Architecture
```
1. Decode macroblock → 832-byte yuv_b working buffer (BPS=32, cache-aligned)
2. Copy via WEBP_UNSAFE_MEMCPY → row cache (cache_y/u/v)
   - cache_y_stride = 16 * mb_w (contiguous row of macroblocks)
3. Loop filter operates on row cache (contiguous data)
4. At row completion, FinishRow() outputs to final destination
```

Code from `frame_dec.c:195`:
```c
for (j = 0; j < 16; ++j) {
    WEBP_UNSAFE_MEMCPY(y_out + j * dec->cache_y_stride, y_dst + j * BPS, 16);
}
```

#### Our Architecture
```
1. Decode macroblock → 544-byte working buffer (same stride=32)
2. Copy byte-by-byte → final ybuf/ubuf/vbuf (full image size)
   - stride = full image width (e.g., 768 for 768x512 image)
3. Loop filter operates on final buffers (scattered access)
4. No row cache - immediate output
```

Code from `vp8.rs:889`:
```rust
for y in 0usize..16 {
    for (ybuf, &ws) in self.frame.ybuf[(mby * 16 + y) * mw * 16 + mbx * 16..][..16]
        .iter_mut()
        .zip(ws[(1 + y) * stride + 1..][..16].iter())
    {
        *ybuf = ws;
    }
}
```

#### Problems Identified

1. **No row cache** - We write directly to final buffers (MAIN ISSUE)
   - When filtering top edge of MB(x, y), we need data from row y-1
   - By then, we've processed 48 macroblocks (full row) since touching row y-1
   - With ~1KB touched per MB, that's 48KB of intervening data
   - Exceeds L1 cache (32KB), causing eviction and cache misses

2. **libwebp's clever "extra rows" solution**:
   - `cache_y - extra_rows` to `cache_y`: holds last N rows from PREVIOUS mb row
   - After each row: `memcpy(cache_y - ysize, bottom_of_current_row, ysize)`
   - Filter reads from extra_rows area (still hot) + current row (hot)
   - Only keeps what's needed: 2 rows for simple filter, 8 for normal filter

3. ~~Byte-by-byte copy~~ Fixed with `copy_from_slice()` (~1-2% gain)

#### Solutions Implemented

1. ~~**Quick fix**: `copy_from_slice()`~~ Done, ~1-2% gain

2. **Row cache with extra rows** - IMPLEMENTED (commit c16995f):
   - Added `cache_y/u/v` buffers: `(extra_rows + 16) * mb_w` bytes
   - `extra_y_rows` = 8 for normal filter, 2 for simple, 0 for none
   - Prediction writes to cache, not final buffer
   - Filter operates on cache (tighter stride, better cache locality)
   - Delayed output: filter modifies pixels above AND below edge
   - `rotate_extra_rows()` copies bottom rows for next iteration

   **Outcome**: Performance essentially unchanged (~1.92x). The extra copy
   overhead may be offsetting cache locality gains. The remaining 2x gap
   vs libwebp likely comes from instruction count (4.5x more) rather than
   cache efficiency alone.

## User Feedback Log

(none currently)

## Diagnostic Examples

Encoder debugging tools in `examples/`:

| Example | Usage |
|---------|-------|
| `corpus_test [dir]` | Batch file size comparison vs libwebp |
| `compare_all_methods` | Per-method size comparison on test image |
| `compare_coefficients` | Compare quantized levels for same-mode blocks |
| `compare_i4_modes` | Per-block I4 mode choice comparison |
| `compare_i4_sse` | Coefficient counts for disputed blocks |
| `compare_quantizers` | Verify quantizer values match libwebp |
| `compare_rd_costs` | Macroblock type agreement stats |
| `compare_trellis_block` | Single-block trellis comparison |
| `debug_block` | Debug single I4 block with BLOCK_DEBUG env |
| `debug_mode_decision` | MB_DEBUG env for mode selection |
| `force_i16_comparison` | File sizes with I4 threshold forcing |
| `test_i4_vs_i16` | I4 vs I16 decision analysis |
| `test_lambda` | Lambda value verification |
| `test_no_trellis` | Test encoding without trellis |
| `test_tdisto` | Spectral distortion testing |

Run with: `cargo run --release --example <name> [args]`

### I4 Coefficient Level Analysis (2026-02-02)

**Findings from `compare_coefficients` example:**

For same-mode I4 blocks (both encoders chose same prediction mode):
- Exact coefficient match: 57.9% of blocks
- Total |level| sum: zenwebp 2.7% higher
- Non-zero coefficient count: zenwebp 1.3% higher

**Root cause:** Our trellis is slightly less aggressive at zeroing coefficients
than libwebp's. For example:
- `zen=3 lib=1` — we kept level 3 where libwebp chose 1
- `zen=-7 lib=-6` — we kept level -7 where libwebp chose -6

This is consistent with the trellis RD balance favoring slightly lower distortion
over rate reduction. The effect is small (~1-2% more bits) but compounds across
many coefficients.

**Mitigation:** The I4 flatness penalty fix (see commit 98b6c85) addresses the
mode selection aspect. The remaining coefficient-level difference is inherent
to our trellis tuning. Corpus results show overall parity or better, so this
may be acceptable or even beneficial for perceptual quality.


### I4 Over-Selection Investigation (2026-02-03)

**Problem:** zenwebp selects I4 mode for ~5% more macroblocks than libwebp (73.4% vs 68.3%
on 792079.png), resulting in ~1% larger files despite I4 being theoretically more efficient.

**Key finding:** Our I16-only encoding is 16-19% smaller than libwebp's, but once I4 is
enabled (methods 3+), we become 1% larger. This means:
1. Our I16 encoding is MORE efficient than libwebp's
2. Our I4 encoding is LESS efficient than libwebp's
3. We're selecting I4 for blocks where it actually produces larger output

**Method comparison (792079.png, SNS=0, filter=0, segments=1, 2026-02-03):**
| Method | zenwebp | libwebp | Ratio | Notes |
|--------|---------|---------|-------|-------|
| 0 | 14,522 | 17,912 | **0.811x** | I16-only |
| 4 | 12,398 | 12,158 | 1.020x | I4 enabled |

**Verified components that MATCH libwebp:**
- Lambda values (lambda_i4, lambda_i16, lambda_mode, tlambda)
- BMODE_COST penalty (211 in 1/256 bits)
- VP8_FIXED_COSTS_I4 mode cost table
- VP8_LEVEL_FIXED_COSTS coefficient fixed costs
- LevelCosts calculation from probability tables
- RD score formula: (R + H) * lambda + 256 * (D + SD)
- SSE calculation (sse4x4_with_residual)
- Context tracking (top_nz, left_nz)
- SIMD vs scalar parity (outputs identical with and without SIMD)

**Spectral distortion (TDisto) added to I4 (2026-02-03):**
Added spectral distortion calculation to I4 mode selection matching libwebp:
- `rd_tmp.SD = tlambda ? MULT_8B(tlambda, VP8TDisto4x4(src, tmp_dst, kWeightY)) : 0`
- Only has effect when tlambda > 0 (requires SNS > 0)
- With SNS=0, tlambda=0 so SD=0 (matches libwebp behavior)

**Remaining hypothesis - coefficient encoding efficiency:**
Our I4 coefficient encoding produces more bits than libwebp's for the same modes.
Diagnostic harness (same-mode I4 blocks):
- Exact coefficient match: 57.9% of blocks
- Total |level| sum: zenwebp 2.7% higher
- Non-zero coefficient count: zenwebp 1.3% higher

This suggests our quantization or trellis produces slightly different (worse) coefficients.
The cost ESTIMATION matches libwebp, but the actual ENCODING produces more bits.

**Investigation progress (2026-02-03):**

**Step 2 - Simple quantization (VERIFIED):** Sharpen/zthresh/bias calculations match libwebp.
For same-mode I4 blocks: 79.2% exact match, total |level| sum 0.997x libwebp (slightly better),
non-zero count 0.998x libwebp. Simple quantization is NOT the cause.

**Step 3 - Probability tables (VERIFIED with fix):**
- Mid-stream probability update helps compression (1.0111x → 1.0101x without it)
- Level_costs recalculation HURTS compression (1.0101x → 1.0114x with it)
- Fix: Update probabilities mid-stream but don't recalculate level_costs
- This differs from libwebp which does both, but our level_costs calculation may
  introduce slight bias when recalculated. Keeping probabilities synced with
  mode selection while level_costs stays at initial values works best for us.

**Step 1 - Trellis/I4 penalty (INVESTIGATED 2026-02-03):**
- Investigation found I4 mode over-selection was due to insufficient I4 penalty
- libwebp uses `i4_penalty = 1000 * q²` with fixed lambdas (λ_i16=106, λ_i4=11)
- Our scoring uses q-dependent lambdas, so we need a different penalty formula
- **Fix:** Use `i4_penalty = 3000 * lambda_mode` (where lambda_mode = q²/128)
- Result: CID22 corpus 1.0101x → **1.0099x** (small improvement)

**Trellis coefficients:** Same-mode I4 blocks have 79.2% exact coefficient match.
Our total |level| sum is 0.997x of libwebp (slightly more aggressive zeroing).
The trellis implementation itself is correct; the issue was mode selection threshold.

**Current status:** Near parity with libwebp (1.0099x on CID22 corpus)
