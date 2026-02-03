# Context Handoff: I4 Over-Selection Investigation

**Session Date:** 2026-02-02
**Objective:** Find why zenwebp over-selects I4 mode compared to libwebp

## Summary of Findings

### The Problem
On certain images, zenwebp chooses I4 mode for ~15% more macroblocks than libwebp, resulting in 1-2% larger files. Example on 844297.png:
- Mode agreement: 85.1% (871/1024 MBs)
- We choose I4 for 153 MBs where libwebp chooses I16
- Our I4 coefficient encoding is efficient (0.997x of libwebp)
- **Root cause: I4 mode SELECTION, not encoding**

### Affected Images (worst cases at m5)
| Image | Ratio | Notes |
|-------|-------|-------|
| 844297.png | 1.013x | 153 extra I4 MBs |
| 6078297.png | 1.012x | |
| 5055743.png | 1.011x | |
| 1418519.png | 1.009x | |
| 1420710.png | 1.000x | Parity achieved |

### m6 Regression
Some images get WORSE at m6 (trellis during mode selection) than m5:
- 844297: 1.013x → 1.022x
- 6078297: 1.012x → 1.023x

This confirms the issue is in I4 mode selection scoring, not coefficient encoding.

## Investigation Plan

### Step 1: Compare I4 vs I16 RD Scores Directly
For macroblocks where we disagree (zen=I4, lib=I16), compare the actual RD scores.

**Add debug output to `choose_macroblock_info()` in `mode_selection.rs`:**
```rust
// Around line 917, when pick_best_intra4 returns Some but libwebp chose I16:
if debug_mb {
    eprintln!("I4 vs I16 decision:");
    eprintln!("  I16 score: {}", i16_score);
    eprintln!("  I4 score:  {}", i4_score);
    eprintln!("  Margin:    {} ({}%)", i16_score - i4_score,
              100.0 * (i16_score - i4_score) as f64 / i16_score as f64);
}
```

**Question to answer:** How close are the scores when we choose I4 but libwebp chooses I16?

### Step 2: Compare Component Costs
Break down the I4 and I16 scores into components:
- Mode header cost (H)
- Coefficient rate cost (R)
- Distortion (D)
- Spectral distortion (SD)

**Files to modify:**
- `src/encoder/vp8/mode_selection.rs`: Add component tracking to `pick_best_intra4` return value

**Libwebp reference:** `quant_enc.c` lines 1058-1114 (PickBestIntra16) and 1128-1220 (PickBestIntra4)

### Step 3: Verify Lambda Values Match
Compare our lambda values against libwebp for the same segment/quality:

| Lambda | Our Variable | libwebp Variable |
|--------|--------------|------------------|
| I16 mode selection | `segment.lambda_i16` | `dqm->lambda_i16` |
| I4 mode selection | `segment.lambda_i4` | `dqm->lambda_i4` |
| I4 vs I16 comparison | `segment.lambda_mode` | `dqm->lambda_mode` |

**Verification command:**
```bash
cargo run --release --example compare_quantizers
```

### Step 4: Check BMODE_COST Application
Our I4 starts with `211 * lambda_mode`. Verify this matches libwebp:
- libwebp: `rd_best.H = 211; SetRDScore(dqm->lambda_mode, &rd_best);`
- Ours: `running_score = 211 * lambda_mode`

These should be equivalent, but verify the accumulation logic matches.

### Step 5: Verify Mode Cost Accumulation
For each I4 sub-block, verify:
1. We use the correct context (top/left mode) for `get_i4_mode_cost()`
2. The mode costs match `VP8FixedCostsI4[top][left][mode]`
3. We accumulate costs the same way libwebp does in `AddScore()`

**Key file:** `mode_selection.rs` lines 652-666

### Step 6: Create Diagnostic Example
Create `examples/debug_i4_decision.rs` that:
1. Takes a specific MB coordinates as input
2. Prints full I16 and I4 scoring breakdown
3. Compares against libwebp's decision for same MB

```rust
// Usage: MB_X=3 MB_Y=0 cargo run --release --example debug_i4_decision
```

## Key Code Locations

### I16 Score Calculation
`src/encoder/vp8/mode_selection.rs` lines 34-211:
- `pick_best_intra16()` - returns `(LumaMode, u64)` where u64 is the RD score
- Final score: `(mode_cost + coeff_cost) * lambda_mode + RD_DISTO_MULT * (sse + spectral_disto)`

### I4 Score Calculation
`src/encoder/vp8/mode_selection.rs` lines 312-700:
- `pick_best_intra4()` - returns `Option<([IntraMode; 16], u64)>`
- Initial: `running_score = 211 * lambda_mode`
- Per-block: `running_score += (mode_cost + coeff_cost) * lambda_mode + RD_DISTO_MULT * sse`
- Early exit: `if running_score >= i16_score { return None; }`

### I4 vs I16 Decision
`src/encoder/vp8/mode_selection.rs` lines 916-936:
```rust
match self.pick_best_intra4(mbx, mby, i16_score) {
    Some((modes, i4_score)) => (LumaMode::B, Some(modes)),
    None => (luma_mode, None),
}
```

### libwebp Reference
- `quant_enc.c:1058` - PickBestIntra16
- `quant_enc.c:1128` - PickBestIntra4
- `quant_enc.c:1419` - VP8Decimate (orchestrates mode selection)
- `quant_enc.c:563` - SetRDScore formula

## Hypotheses to Test

### H1: I16 Coefficient Cost Overestimated
Our `get_cost_luma16()` may return higher values than libwebp's `VP8GetCostLuma16()`.
**Test:** Compare coefficient costs for same quantized levels.

### H2: I4 Mode Costs Underestimated
Our I4 mode costs from `VP8_FIXED_COSTS_I4` may differ from libwebp.
**Test:** Already verified tables match - unlikely.

### H3: Distortion Calculation Differs
Our SSE calculation may differ slightly.
**Test:** Compare `sse_16x16_luma()` vs `VP8SSE16x16()`.

### H4: Lambda Values Differ
Our segment lambda values may be computed differently.
**Test:** Run `compare_quantizers` and verify all lambda values match.

### H5: Spectral Distortion (TDisto) Difference
I16 uses TDisto but I4 doesn't (by design). This asymmetry may affect the decision.
**Test:** Temporarily disable TDisto in I16 and compare results.

## Reproduction Commands

```bash
# Set test image
cp /home/lilith/work/codec-corpus/CID22/CID22-512/validation/844297.png /tmp/CID22/original/792079.png

# Compare modes
cargo run --release --example compare_rd_costs

# Compare I4 modes
cargo run --release --example compare_i4_modes

# Compare coefficients
cargo run --release --example compare_coefficients

# Full method comparison
cargo run --release --example compare_all_methods

# Restore original test image
cp /home/lilith/work/codec-corpus/CID22/CID22-512/validation/792079.png /tmp/CID22/original/
```

## Session Changes Made

1. **Aligned method features with libwebp** (commit 47873f5)
   - Moved trellis from m4 to m5
   - m4→m5 now gives 0.9% improvement

2. **Fixed cross-macroblock non-zero context** (commit e91f826)
   - I4 mode selection now uses correct context from `top_complexity`/`left_complexity`

## Next Session Priority

1. Create `debug_i4_decision.rs` example
2. Test H1 (I16 coefficient cost) first - most likely culprit
3. If H1 fails, test H4 (lambda values)
4. Document findings in CLAUDE.md
