# Lossy Encoder Roadmap: Matching libwebp

This document captures our learnings and outlines a plan to match libwebp's encoding efficiency.

## Current State

### Performance Summary (128x128 test image, Q75)

| Metric | Our Encoder | libwebp | Ratio |
|--------|-------------|---------|-------|
| File size | 10,088 bytes | 7,124 bytes | 1.42x larger |
| PSNR | 16.89 dB | 16.91 dB | 99.9% |
| Bits per pixel | 4.93 bpp | 3.48 bpp | 1.42x |

**Key insight**: At equal file sizes, we achieve ~99% of libwebp's quality. The gap is purely encoding efficiency, not quality decisions.

### Size Ratio vs Quality Setting

| Quality | Our Size | libwebp Size | Ratio |
|---------|----------|--------------|-------|
| Q50 | 4,800 bytes | 5,884 bytes | 0.82x (smaller!) |
| Q75 | 10,088 bytes | 7,124 bytes | 1.42x |
| Q90 | 17,316 bytes | 10,416 bytes | 1.66x |

**Observation**: We're actually more efficient at low quality (Q50) but progressively less efficient at high quality. This suggests our overhead is per-coefficient, not per-macroblock.

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

### Disabled/Incomplete
- [ ] Intra4 mode (implemented but disabled - causes file bloat)
- [ ] Segment-based quantization (infrastructure exists, not used)
- [ ] Adaptive token probabilities (writes "no update" for all)

## What's Missing (Root Causes of Inefficiency)

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

### 5. Loop Filter Optimization (LOW IMPACT)

**The Problem**: We use fixed filter_level=63 (maximum). libwebp computes optimal filter level.

**Impact**: Minimal on file size, some impact on decoded quality.

**What We Need**: Filter level selection based on quantization and image characteristics.

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
- Implement optimal loop filter selection
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
