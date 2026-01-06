# Lossy Encoder Roadmap: Matching libwebp

This document captures our learnings and outlines a plan to match libwebp's encoding efficiency.

---

## ⚠️ STOP: No Benchmarking Until Full Port Complete

**DO NOT run size/quality benchmarks until ALL libwebp algorithms are fully ported and integrated.**

Premature benchmarking wastes time because:
1. Individual optimizations interact non-linearly
2. Partial implementations may regress when other features are added
3. Tuning parameters before the full system exists leads to re-tuning later

**Remaining work before benchmarking:**
- [x] ~~Enable and tune segment-based quantization~~ - **DONE**: Full DCT-based analysis ported from libwebp
- [x] ~~Implement optimal loop filter level selection~~ - **DONE**: `compute_filter_level()` ported from libwebp
- [x] ~~Tune probability update thresholds~~ - **DONE**: Changed to `savings > 0` matching libwebp exactly
- [x] ~~Fix Intra4 mode with proper cost estimation~~ - **DONE**: VP8GetCostLuma4 with remapped_costs ported

**Ready for benchmark:** All core features implemented. Current results:
- File sizes: 1.17-1.41x of libwebp (down from 2.5x before I4 was enabled)
- PSNR gap: ~1.35 dB behind at equal BPP (improved from 2.27 dB)

---

## Current State

### Kodak Corpus Benchmark (24 images, 768x512) - Updated with Intra4 enabled

**Rate-Distortion Comparison at Equal Bits Per Pixel:**

| BPP | Our PSNR | libwebp PSNR | Difference | Our Q | libwebp Q |
|-----|----------|--------------|------------|-------|-----------|
| 0.25 | 30.18 dB | 30.57 dB | **-0.39 dB** | Q20 | Q21 |
| 0.50 | 31.04 dB | 32.18 dB | **-1.15 dB** | Q29 | Q41 |
| 0.75 | 32.57 dB | 33.80 dB | **-1.23 dB** | Q47 | Q57 |
| 1.00 | 33.93 dB | 35.26 dB | **-1.34 dB** | Q58 | Q68 |
| 1.50 | 36.18 dB | 37.54 dB | **-1.36 dB** | Q75 | Q82 |
| 2.00 | 37.76 dB | 39.20 dB | **-1.44 dB** | Q84 | Q88 |
| 3.00 | 39.65 dB | 41.12 dB | **-1.47 dB** | Q91 | Q93 |
| 4.00 | 40.41 dB | 41.69 dB | **-1.27 dB** | Q93 | Q94 |

**Average PSNR difference at equal BPP: -1.35 dB (improved from -2.27 dB)**

### Size Comparison at Same Quality Setting

| Quality | Our Size | libwebp Size | Size Ratio | PSNR Δ |
|---------|----------|--------------|------------|--------|
| Q20 | 678 KB | 480 KB | 141.1% | -0.23 dB |
| Q50 | 1,120 KB | 851 KB | 131.5% | -0.12 dB |
| Q75 | 1,444 KB | 1,178 KB | 122.6% | -0.14 dB |
| Q90 | 2,794 KB | 2,372 KB | 117.8% | -0.62 dB |

**Key Observations (after Intra4 enabled):**
1. File sizes now 1.17-1.41x of libwebp (improved from 2.5x before)
2. PSNR gap reduced to ~1.35 dB at equal BPP (improved from 2.27 dB)
3. At same quality setting, PSNR difference is now small (-0.12 to -0.62 dB)
4. Remaining gap is in encoding efficiency, not mode selection

## What's Implemented

### Core Encoding
- [x] VP8 bitstream generation (valid, decodable by libwebp)
- [x] Boolean arithmetic encoder
- [x] DCT transform and inverse
- [x] Quantization with DC_QUANT/AC_QUANT tables

### Mode Selection
- [x] RD-based Intra16 mode selection (DC, V, H, TM)
- [x] RD-based chroma mode selection (DC, V, H, TM)
- [x] SSE distortion calculation
- [x] Coefficient cost estimation

### Quantization
- [x] VP8Matrix with biased quantization (`quantdiv(coeff, iq, bias)`)
- [x] Proper dequantization for reconstruction
- [x] Y1, Y2, UV matrix separation

### Optimizations
- [x] Skip detection for zero macroblocks
- [x] Loop filter (fixed level 63)

### Recently Implemented
- [x] Two-pass token counting with adaptive probability updates (quality < 85 only)
- [x] Quality-to-quant mapping ported from libwebp
- [x] ProbaStats for coefficient statistics tracking
- [x] Segment infrastructure (4 segments with quant deltas, encoding ready)
- [x] **DCT-based segment analysis** - Full port from libwebp:
  - [x] BPS=32 work buffer layout matching libwebp
  - [x] VP8DspScan block offsets for histogram collection
  - [x] Forward DCT (VP8FTransform) for coefficient analysis
  - [x] DC and TM mode prediction (VP8EncPredLuma16, VP8EncPredChroma8)
  - [x] Histogram collection with proper binning (abs(coeff) >> 3)
  - [x] Alpha calculation from histogram (ALPHA_SCALE * last_non_zero / max_value)
  - [x] Proper border handling (VP8IteratorImport): left with corner, top row
  - [x] Final alpha mix: (3 * luma_alpha + uv_alpha + 2) >> 2
  - [x] Alpha inversion: MAX_ALPHA - alpha
  - [x] K-means clustering for segment assignment (AssignSegments)
  - [x] Weighted average from final cluster centers (SetSegmentAlphas)
  - [x] Per-segment quantization computation (compute_segment_quant)

### Intra4 Mode - ENABLED
- [x] i4_penalty = 1000 * q² matching libwebp
- [x] Early-exit when I4 score exceeds I16 score
- [x] Header bit limiting
- [x] VP8GetCostLuma4 with remapped_costs for accurate coefficient cost estimation
- [x] LevelCosts struct with probability-dependent cost tables
- [x] Proper context tracking (top_nz/left_nz) for accurate coefficient costs

---

## Feature Inventory: Ours vs libwebp

### ✅ Fully Implemented and Used

| Feature | Our Implementation | libwebp Equivalent | Status |
|---------|-------------------|-------------------|--------|
| Boolean arithmetic encoder | `vp8_arithmetic_encoder.rs` | `vp8l_enc.c` | ✅ Working |
| DCT/IDCT transforms | `transform.rs` | `dsp/enc.c` | ✅ Working |
| Biased quantization | `VP8Matrix::quantdiv()` | `VP8Quantize*()` | ✅ Working |
| Two-pass token probability updates | `encode_with_updates()` | `VP8EncTokenLoop()` | ✅ Working |
| DCT-based segment analysis | `setup_segment_quantization()` | `VP8MBAnalyze()` | ✅ Full port |
| K-means segment assignment | `assign_segments_kmeans()` | `AssignSegments()` | ✅ Full port |
| SNS segment quantization | `compute_segment_quant()` | `SetSegmentAlphas()` | ✅ Full port |
| Loop filter level computation | `compute_filter_level()` | `VP8FilterStrengthFromDelta()` | ✅ Full port |
| Quality-to-quant mapping | Non-linear curve | Same | ✅ Ported |
| RD-based I16 mode selection | `pick_best_luma_mode()` | `VP8MakeIntra16Preds()` | ✅ Working |
| RD-based chroma mode selection | `pick_best_chroma_mode()` | `VP8MakeChroma8Preds()` | ✅ Working |
| Intra4 mode selection | `pick_best_intra4()` | `VP8MakeLuma4Preds()` | ✅ Working |
| VP8GetCostLuma4 coefficient costs | `get_cost_luma4()` | `VP8GetCostLuma4()` | ✅ Full port |
| LevelCosts with remapped tables | `LevelCosts` struct | `VP8CalculateLevelCosts()` | ✅ Full port |
| Skip detection | `check_all_coeffs_zero()` | `VP8SetSkip()` | ✅ Working |

### ⚠️ Exists But NOT Used

| Feature | Our Code | Problem | Impact |
|---------|----------|---------|--------|
| Trellis quantization | `trellis_quantize_block()` in `vp8_cost.rs` | Function exists but never called from encoder | HIGH - ~5-10% size savings |

### ❌ Hardcoded / Suboptimal

| Feature | Our Implementation | libwebp Implementation | Gap |
|---------|-------------------|----------------------|-----|
| Skip probability | Hardcoded `200` | Computed from actual skip count: `CalcSkipProba(nb_skip, nb_mbs)` | LOW - minor bitstream overhead |

### ❌ Not Implemented

| Feature | libwebp Implementation | Impact | Priority |
|---------|----------------------|--------|----------|
| **Chroma error diffusion** | `CorrectDCValues()` for quality ≤ 98 | MEDIUM - reduces banding in gradients | Medium |
| **Autofilter** | `lf_stats` + `VP8AdjustFilterStrength()` | LOW - auto-tunes filter strength | Low |
| **Multi-pass rate control** | `config->pass > 1` | LOW - for target size encoding | Low |
| **Target size encoding** | `config->target_size` | LOW - specific use case | Low |
| SmoothSegmentMap | Optional when `preprocessing & 1` | LOW - smooths segment boundaries | Very Low |
| FastMBAnalyze | Method <= 1 fast path | N/A - speed optimization only | Very Low |

---

## Priority Action Items

### HIGH Priority: Enable Trellis Quantization
**Status**: Code exists in `vp8_cost.rs` but is never called!

The `trellis_quantize_block()` function is fully implemented but not integrated into the encoding loop. This is likely the single biggest missing optimization.

**To enable:**
1. Call trellis quantization after initial quantization in `encode_block()`
2. Use for both I16 and I4 modes
3. Estimated impact: 5-10% size reduction at equal quality

### MEDIUM Priority: Compute Skip Probability from Data
**Status**: Hardcoded to 200

libwebp computes optimal skip probability:
```c
proba->skip_proba = CalcSkipProba(nb_events, nb_mbs);
proba->use_skip_proba = (skip_proba < SKIP_PROBA_THRESHOLD);  // threshold = 250
```

**To fix:**
1. Count actual skipped macroblocks during first pass
2. Compute probability: `255 * (total - skipped) / total`
3. Only enable if probability < 250

### MEDIUM Priority: Chroma Error Diffusion
**Status**: Not implemented

libwebp uses Floyd-Steinberg-style error diffusion for **chroma DC coefficients only** at quality ≤ 98:

```
        | top[0] | top[1]
--------+--------+---------
left[0] | c[0]   | c[1]   ->  err0 err1
left[1] | c[2]   | c[3]       err2 err3
```

- Spreads quantization error from each chroma block's DC to neighbors
- Reduces visible banding in smooth color gradients
- **Affects perceptual quality more than PSNR** (error redistributed, not eliminated)
- Important for images with smooth gradients (sky, skin tones)

**Implementation notes:**
- Track `top_derr[mb_w][2][2]` (per-column, per-channel, top/left)
- Track `left_derr[2][2]` (per-channel, top/left)
- Apply correction to UV DC coefficients before quantization
- Store resulting errors for next macroblock

---

### Not Implemented (Optional/Minor - Very Low Priority)
- [ ] SmoothSegmentMap - Optional post-processing when `preprocessing & 1` is set
- [ ] FastMBAnalyze - Fast path for low-quality encoding (method <= 1)
- [ ] Mode pre-selection storage - libwebp stores best modes during analysis for reuse in encoding

## Analysis of the Gap

The -2.27 dB gap breaks down into several components:

### Quality Curve Analysis

At Q20:
- Our size: 229 KB (47.8% of libwebp's 480 KB)
- Our PSNR: 26.83 dB vs libwebp's 30.41 dB (-3.58 dB)

At equal BPP (0.25):
- We need Q33, libwebp needs Q21 to produce same file size
- Our quant index: `127 - 33*127/100 = 85`
- libwebp's quant index (via their formula): ~61

**Key insight**: At Q33 we use quant 85 (aggressive), libwebp at Q21 uses quant 61 (less aggressive). Yet we both produce 0.25 BPP. This means:
1. We're spending more bits on overhead (mode signaling, probability encoding)
2. Less of our bits go to actual coefficient data
3. Even with more aggressive quantization, our files aren't proportionally smaller

### Encoding Efficiency Problem
At Q75+, we produce larger files with slightly worse quality. This suggests:
1. Coefficient encoding overhead (token probabilities)
2. Mode signaling overhead
3. Possibly suboptimal mode decisions

### What to Investigate
1. **Dump quantization values** - Compare our quant indices vs libwebp at same Q
2. **Dump mode decisions** - Compare I16 vs I4 choices
3. **Dump coefficient statistics** - Compare coefficient distributions
4. **Check reconstruction** - Verify our dequant/IDCT matches libwebp

## What's Missing (Root Causes of Inefficiency)

### 0. Quality-to-Quant Mapping (CRITICAL - INVESTIGATE FIRST)

**The Problem**: Our Q20 produces 47.8% of libwebp's file size. This is not just encoding efficiency - we're quantizing much more aggressively.

**Investigation Needed**:
```rust
// At Q75, what quant index do we use vs libwebp?
// Our formula: quant = 127 - (75 * 127 / 100) = 32
// What does libwebp use?
```

**libwebp's Formula** (from `src/enc/quant_enc.c`):
```c
const double Q = quality / 100.;
const double linear_c = (Q < 0.75) ? Q * (2./3.) : 2. * Q - 1.;
const double c = pow(linear_c, 1./3.);
const int q = (int)(127. * (1. - c));
// Plus segment alpha adjustments
```

At Q75: `linear_c = 0.5`, `c = 0.794`, `q = 26`
Our formula gives: `q = 32`

So we're using **higher quantization** (32 vs 26) = more compression = smaller files at low Q. But we're still getting worse quality!

### 1. Adaptive Token Probability Updates (HIGH IMPACT)

**The Problem**: VP8 uses context-adaptive arithmetic coding with 11 token probability tables × 8 bands × 3 contexts × 11 tokens = 2,904 probabilities. libwebp updates these based on actual coefficient statistics; we use the default tables.

**Impact**: Estimated 15-25% of our size overhead. Higher quality = more coefficients = more impact.

**libwebp Implementation** (`src/enc/frame_enc.c`):
```c
// Two-pass approach:
// Pass 1: Count token occurrences per context
// Pass 2: Compute optimal probabilities, encode with updates
VP8EncTokenLoop(enc);  // Counts tokens
VP8WriteProbas(enc);   // Writes probability updates to bitstream
```

**What We Need**:
1. Token counting infrastructure (count occurrences per context)
2. Probability optimization (compute best prob from counts)
3. Cost-benefit analysis (only update if savings > signaling cost)
4. Two-pass encoding OR streaming probability estimation

**Files to Study**:
- libwebp `src/enc/token_enc.c` - token recording
- libwebp `src/enc/frame_enc.c` - `VP8EncTokenLoop()`
- libwebp `src/enc/webp_enc.c` - `VP8WriteProbas()`

### 2. Segment-Based Quantization (MEDIUM IMPACT)

**The Problem**: VP8 supports 4 segments with independent quantization. libwebp uses these to apply different quality to different image regions (edges get more bits, flat areas fewer).

**Impact**: Estimated 5-15% improvement, especially on complex images.

**libwebp Implementation**:
```c
// Segment assignment based on local complexity
void VP8SetSegmentParams(VP8Encoder* enc, float quality) {
  // Computes per-segment quant based on:
  // - Local activity (variance)
  // - Perceptual importance
  // - Target bitrate allocation
}
```

**What We Need**:
1. Complexity analysis pass (compute variance per macroblock)
2. Segment assignment algorithm (cluster macroblocks into 4 groups)
3. Per-segment quantization parameters
4. Segment map encoding

**Current State**: We have `segments: [Segment; 4]` and `segments_enabled` field but always use segment 0.

### 3. Intra4 Mode Selection (MEDIUM IMPACT)

**The Problem**: Our Intra4 implementation causes file bloat. The RD cost estimation doesn't match actual encoding cost.

**Root Cause Analysis**:
- Mode signaling cost: 4 bits/mode × 16 modes = 64 bits overhead
- Our cost estimation uses `get_i4_mode_cost()` but this doesn't account for:
  - Context-dependent mode coding
  - Actual token costs after quantization
  - Correlation between adjacent mode choices

**What We Need**:
1. Better coefficient cost estimation for 4x4 blocks
2. Context-aware mode cost (depends on left/above mode)
3. Threshold tuning (when is Intra4 actually beneficial?)

**libwebp Approach**: Uses `VP8MakeLuma4Preds()` and `VP8GetCostLuma4()` with accurate context modeling.

### 4. Rate Control / Quality Curve (LOW IMPACT)

**The Problem**: Our Q75 doesn't produce the same file size as libwebp's Q75.

**libwebp's Quality Mapping**:
```c
// From src/enc/quant_enc.c
double QualityToCompression(double q) {
  double linear_c = (q < 0.75) ? q * (2./3.) : 2. * q - 1.;
  return pow(linear_c, 1./3.);
}
// Then: quant = 127 * (1 - c) with segment alpha adjustment
```

**Decision**: This is cosmetic. Users can adjust Q to get desired size. Not worth the complexity.

### 5. Loop Filter Optimization (DONE)

**Status**: Implemented via `compute_filter_level()` in `vp8_cost.rs`.

The function computes filter level based on:
- Quantizer index (via AC quantizer step from VP8_AC_TABLE)
- Sharpness setting (0-7)
- User filter strength (0-100, default 50)

Formula matches libwebp: `f = base_strength * level0 / 256` with proper clamping and cutoff.

## Implementation Roadmap

### Phase 1: Adaptive Token Probabilities (Target: 20% size reduction)

**Step 1.1: Token Counting Infrastructure**
```rust
struct TokenStats {
    // counts[type][band][ctx][token]
    counts: [[[[u32; 12]; 3]; 8]; 4],
}

impl TokenStats {
    fn record_token(&mut self, plane: Plane, band: usize, ctx: usize, token: Token);
    fn get_probability(&self, ...) -> u8;
}
```

**Step 1.2: Two-Pass Encoding**
```rust
// Pass 1: Encode to count tokens (don't write to output)
let stats = self.count_tokens_pass();

// Compute updated probabilities
let new_probs = stats.compute_optimal_probs(&DEFAULT_PROBS);

// Pass 2: Encode with updated probabilities
self.encode_with_probs(&new_probs);
```

**Step 1.3: Probability Update Signaling**
- For each probability, compute: `savings = bits_saved - update_cost`
- Only signal update if `savings > 0`
- Update cost = 1 bit (flag) + 8 bits (new prob value)

**Validation**: Compare token distributions with libwebp using debug output.

### Phase 2: Segment-Based Quantization (Target: 10% size reduction)

**Step 2.1: Complexity Analysis**
```rust
fn compute_macroblock_complexity(&self, mbx: usize, mby: usize) -> f32 {
    // Compute variance of 16x16 block
    // Higher variance = more complex = needs more bits
}
```

**Step 2.2: Segment Assignment**
```rust
fn assign_segments(&mut self) {
    // K-means clustering of macroblocks into 4 groups by complexity
    // Or simpler: quartile-based assignment
}
```

**Step 2.3: Per-Segment Quantization**
```rust
// Segment 0: Low complexity (flat areas) - higher quant
// Segment 3: High complexity (edges/texture) - lower quant
```

### Phase 3: Fix Intra4 Mode (Target: 5% quality improvement)

**Step 3.1: Accurate Cost Estimation**
- Port libwebp's `VP8GetCostLuma4()` exactly
- Include context-dependent mode costs

**Step 3.2: Threshold Tuning**
- Only use Intra4 when: `i4_cost < i16_cost - threshold`
- Threshold should account for mode signaling overhead

**Step 3.3: Validation**
- Compare mode decisions with libwebp on test images
- Ensure Intra4 is only chosen when it actually helps

### Phase 4: Polish (Target: Match libwebp within 5%)

- Tune lambda values for RD optimization
- ~~Implement optimal loop filter selection~~ (DONE)
- Profile and optimize hot paths

## Key Learnings

### 1. Biased Quantization Matters
Using `quantdiv(coeff, iq, bias)` instead of simple division improved quality significantly. The bias pushes small coefficients toward zero, improving compression.

### 2. Skip Detection is Essential
Detecting all-zero macroblocks and signaling skip saves significant bits on simple images.

### 3. RD Mode Selection Works
Our SSE + lambda × rate scoring produces good mode decisions. The modes we choose are reasonable.

### 4. The Gap is in Entropy Coding
We make similar decisions to libwebp but encode them less efficiently. The coefficient encoding overhead is the main issue.

### 5. Quality Curve Doesn't Matter Much
Trying to match libwebp's quality-to-quant curve was a distraction. The real issue is encoding efficiency, not the mapping.

## Testing Strategy

### Regression Tests
- `size_comparison_vs_libwebp`: Ensure we stay within 2x of libwebp's size
- `quality_comparison_at_same_quality_setting`: Ensure PSNR within 80% of libwebp

### Progress Metrics
Track these on a standard test corpus (e.g., Kodak images):

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Target |
|--------|---------|---------|---------|---------|--------|
| Size ratio @ Q75 | 1.42x | 1.15x | 1.05x | 1.00x | ≤1.05x |
| Size ratio @ Q90 | 1.66x | 1.30x | 1.15x | 1.05x | ≤1.10x |

### Comparison Tool
```bash
# Compare our output vs libwebp at same quality
cargo run --example compare_encoders -- input.png --quality 75
```

## References

- [VP8 Bitstream Specification (RFC 6386)](https://datatracker.ietf.org/doc/rfc6386/)
- [libwebp source](https://chromium.googlesource.com/webm/libwebp)
- [WebP Compression Techniques](https://developers.google.com/speed/webp/docs/compression)

## Files Reference

### Our Implementation
- `src/vp8_encoder.rs` - Main encoder, mode selection, coefficient encoding
- `src/vp8_cost.rs` - VP8Matrix, cost estimation tables
- `src/vp8_common.rs` - Segment struct, quant tables
- `src/vp8_arithmetic_encoder.rs` - Boolean arithmetic encoder

### libwebp Equivalents
- `src/enc/vp8l_enc.c` - Main encoder loop
- `src/enc/quant_enc.c` - Quantization, segment setup
- `src/enc/token_enc.c` - Token recording
- `src/enc/frame_enc.c` - Frame encoding, probability updates
