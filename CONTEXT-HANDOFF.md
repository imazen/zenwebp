# Context Handoff: I4 Mode Selection Investigation (2026-02-02)

## Summary

Investigating why zenwebp produces files ~1.3% larger than libwebp at method 4 (1.013x).
Root cause is **MODE SELECTION** - we pick I4 for 26 more macroblocks than libwebp.

## Key Discovery: Quantizers Are Identical

**CONFIRMED**: Both zenwebp and libwebp use identical quantizers at Q75:
- ydc=24, yac=30, y2dc=48, y2ac=46, uvdc=24, uvac=30
- q_i4=30, lambda_mode=7

Since quantizers and lambdas are the same, the mode decision difference must come from:
1. **SSE (distortion) calculation** - different reconstruction error
2. **Coefficient cost estimation** - different rate estimation

## Mode Selection Stats

| Category | Count |
|----------|-------|
| Both I4 | 260 |
| Both I16 | 678 |
| Zen I4, Lib I16 | **56** |
| Zen I16, Lib I4 | 30 |
| Net extra I4 | 26 |

## Detailed Example: MB(25,4)

Debug output with `MB_DEBUG=25,4 --features mode_debug`:

```
I16: mode=V, score=757738
  lambda_mode=7, lambda_i4=21, lambda_i16=6348
I4: score=695770 (beats I16)
  modes=[TM, TM, TM, TM, TM, VE, VE, DC, VE, TM, VE, DC, VE, VE, VE, VE]
  RESULT: I4 wins by 61968 points
```

**Coefficient analysis:**
- zenwebp I4: 21 nonzero coefficients, all ±1
- libwebp I16: 23 total (15 Y1 AC + 8 Y2 DC), Y2 has values like 5, -3, -2

**Bits used:**
- zenwebp I4: ~95 bits (rough estimate)
- libwebp I16: ~48 bits

**Key insight:** I4 uses ~2x MORE bits but still wins the RD comparison. This means the SSE
must be significantly lower for I4 to compensate for the higher rate cost.

## RD Score Formula

```
score = (mode_cost + coeff_cost) * lambda_mode + 256 * sse
```

For MB(25,4):
- I16 V mode: score = 757738
- I4 total: score = 695770 = 1477 (BMODE penalty) + 694293 (sum of block scores)

The 61968 point difference could come from:
- SSE reduction of ~242 units (61968/256)
- Or rate savings of ~8853 units (61968/7)

Since I4 uses MORE bits, the SSE must be lower.

## Hypotheses for SSE Difference

1. **I4 predictions are better for this content**
   - I4 uses per-block prediction modes (TM, VE, etc.)
   - I16 uses single prediction for entire 16x16 block
   - Per-block predictions may capture local detail better

2. **Our I16 SSE is higher than libwebp's**
   - Different reconstruction path?
   - Different prediction implementation?

3. **Our I4 SSE is lower than libwebp's**
   - More accurate IDCT?
   - Different rounding?

## Files to Investigate

1. `src/encoder/vp8/mode_selection.rs` - Mode selection with debug output
   - `pick_best_intra16()` - I16 mode selection
   - `pick_best_intra4()` - I4 mode selection with early exit

2. `src/encoder/residual_cost.rs` - Coefficient cost estimation
   - `get_cost_luma16()` - I16 coefficient cost
   - `get_cost_luma4()` - I4 coefficient cost

3. `src/common/transform.rs` - DCT/IDCT transforms
   - May affect reconstruction SSE

## Test Commands

```bash
# Run with mode debug for specific macroblock
MB_DEBUG=25,4 cargo run --release --features mode_debug --example compare_rd_scores -- 25 4

# Run I4 vs I16 comparison
cargo run --release --example test_i4_vs_i16

# Compare quantizers
cargo run --release --example compare_quantizers
```

## Major Finding: I4 Efficiency Gap

**The problem is NOT mode selection (I4 vs I16). It's I4 encoding efficiency.**

### I16-only Comparison (m0)
| Encoder | Size | vs libwebp |
|---------|------|------------|
| zenwebp | 13988 | **0.84x** (16% smaller!) |
| libwebp | 16678 | 1.00x |

### Full I4 Comparison (m4)
| Encoder | Size | vs libwebp |
|---------|------|------------|
| zenwebp | 12174 | 1.013x (1.3% larger) |
| libwebp | 12018 | 1.00x |

### I4 Benefit (m0 → m4)
| Encoder | Reduction |
|---------|-----------|
| zenwebp | -1814 bytes (13%) |
| libwebp | -4660 bytes (28%) |

**Key insight:** Our I16 is 16% better than libwebp's, but our I4 only provides 13% benefit
vs libwebp's 28%. The gap comes from **I4 encoding efficiency, not mode selection**.

## Per-Block I4 Mode Analysis

For macroblocks where BOTH chose I4:
- **Only 67% of per-block modes match**
- We use **+172 more DC modes**, +65 more TM modes
- We use **-161 fewer VE modes**, -42 fewer HE modes

This suggests our per-block RD scoring favors DC/TM when VE/HE would compress better.

## Next Steps

1. **Investigate per-block mode selection** - Why do we choose DC over VE so often?
2. **Check VE/HE prediction accuracy** - Are our VE/HE predictions correct?
3. **Compare coefficient counts per mode** - Does DC really compress better for those blocks?

## Files Modified This Session

- `Cargo.toml` - Added `mode_debug` feature
- `src/encoder/vp8/mode_selection.rs` - Added conditional debug output
- `examples/compare_rd_scores.rs` - New diagnostic tool
- `examples/compare_quantizers.rs` - New diagnostic tool
- `examples/debug_mode_decision.rs` - New diagnostic tool
- `examples/test_i4_vs_i16.rs` - Enhanced with nonzero coefficient info

## Git State

All changes committed. Examples directory is untracked (intentional).
