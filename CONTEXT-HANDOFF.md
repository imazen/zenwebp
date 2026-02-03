# Context Handoff: I4 Encoding Parity Work

**Session Date:** 2026-02-02
**Objective:** Achieve exact I4 encoding parity with libwebp

## Summary of Work Done

### 1. I4 Flatness Penalty Fix (COMPLETED)
**Commit:** 98b6c85

Added FLATNESS_PENALTY (140) for non-DC I4 modes when coefficients are flat (≤3 non-zero AC coefficients). This matches libwebp's `PickBestIntra4` behavior.

**Results:**
- Mode match rate: 66% → 69%
- Method 4 ratio: 1.012x → 1.006x (single image)
- CID22 corpus: 0.993x (we're 0.7% smaller!)
- Screenshots: 0.996x (we're 0.4% smaller!)

**File changed:** `src/encoder/vp8/mode_selection.rs` lines 561-577

### 2. Verified Matching Components
- Quantization matrices (q, iq, bias, zthresh, sharpen) ✓
- Lambda calculations (lambda_i4, lambda_mode, lambda_trellis_i4) ✓
- VP8_LEVEL_FIXED_COSTS table ✓
- VP8_WEIGHT_TRELLIS table ✓
- BMODE_COST = 211 ✓
- FLATNESS_LIMIT_I4 = 3 ✓

### 3. Coefficient Level Analysis
From `compare_coefficients` example on test image:
- Same-mode blocks: 57.9% exact coefficient match
- zenwebp total |level| sum: 2.7% higher
- zenwebp non-zero count: 1.3% higher

Our trellis is slightly less aggressive at zeroing, but corpus results show we're smaller overall.

## REMAINING GAP FOUND: Cross-Macroblock Non-Zero Context

### The Problem
In `src/encoder/vp8/mode_selection.rs` lines 405-406:
```rust
// Get non-zero context from neighboring blocks
let nz_top = if sby == 0 { false } else { top_nz[sbx] };
let nz_left = if sbx == 0 { false } else { left_nz[sby] };
```

For I4 edge blocks (sby==0 or sbx==0), the non-zero context is hardcoded to `false`. 
But in the actual encoding path (`src/encoder/vp8/prediction.rs` lines 321-333), we correctly initialize from `top_complexity` and `left_complexity`:
```rust
let mut top_nz = [
    self.top_complexity[mbx].y[0] != 0,
    self.top_complexity[mbx].y[1] != 0,
    self.top_complexity[mbx].y[2] != 0,
    self.top_complexity[mbx].y[3] != 0,
];
```

### The Impact
- Mode selection uses wrong context for edge I4 blocks → potentially suboptimal mode choices
- Coefficient cost estimation differs between mode selection and actual encoding
- May account for some of the remaining 0.6% gap on individual images

### The Fix
In `pick_best_intra4()`, initialize `top_nz` and `left_nz` from `self.top_complexity` and `self.left_complexity` at the start of the function (before the sby/sbx loops), similar to how `prediction.rs` does it.

Change lines ~365-370 in mode_selection.rs from:
```rust
let mut top_nz = [false; 4];
let mut left_nz = [false; 4];
```

To:
```rust
let mut top_nz = [
    self.top_complexity[mbx].y[0] != 0,
    self.top_complexity[mbx].y[1] != 0,
    self.top_complexity[mbx].y[2] != 0,
    self.top_complexity[mbx].y[3] != 0,
];
let mut left_nz = [
    self.left_complexity.y[0] != 0,
    self.left_complexity.y[1] != 0,
    self.left_complexity.y[2] != 0,
    self.left_complexity.y[3] != 0,
];
```

Then remove the `if sby == 0 { false }` and `if sbx == 0 { false }` checks on lines 405-406.

## Current Status

| Metric | Value |
|--------|-------|
| Method 4 (792079.png) | 1.006x |
| CID22 corpus (m4) | 0.993x |
| Screenshots corpus (m4) | 0.996x |
| Mode match rate (I4) | 68.8% |
| Coefficient match rate | 57.9% |

## Key Files
- `src/encoder/vp8/mode_selection.rs` - I4 mode selection, flatness penalty
- `src/encoder/vp8/prediction.rs` - Actual encoding, context tracking
- `src/encoder/trellis.rs` - Trellis quantization
- `src/encoder/cost.rs` - Level costs, mode costs
- `CLAUDE.md` - Full investigation notes

## Verification Commands
```bash
# Quick benchmark
cargo run --release --example compare_all_methods

# Corpus test
cargo run --release --example corpus_test -- /home/lilith/work/codec-corpus/CID22/CID22-512/validation

# Mode comparison
cargo run --release --example compare_i4_modes

# Coefficient comparison
cargo run --release --example compare_coefficients
```

## Next Steps
1. Apply the cross-macroblock non-zero context fix (described above)
2. Re-run compare_all_methods and corpus tests
3. If gap remains, investigate trellis tuning (our trellis is 2.7% less aggressive)
