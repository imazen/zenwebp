# Multi-Pass Encoding Handoff (2026-02-02)

## What Was Attempted

I tried implementing "true multi-pass re-quantization" where pass 1+ re-quantizes
raw DCT coefficients using level_costs derived from pass 0's probability statistics.
This made files **larger** (+0.29%), not smaller.

## Why It Failed

The approach was fundamentally wrong. I was using `compute_updated_probabilities()`
which only updates probabilities where signaling cost is recouped. This means the
level_costs in pass 1 were nearly identical to pass 0's level_costs, so trellis
made the same decisions.

## How libwebp Actually Does Multi-Pass

Looking at libwebp's `token_enc.c` (VP8EncTokenLoop):

```c
// Simplified flow
for (pass = 0; pass < num_passes; ++pass) {
    ResetTokenStats(enc);  // Clear statistics
    
    for (each macroblock) {
        VP8Decimate(it, ...);  // Trellis quantization using current level_costs
        RecordTokens(...);     // Record tokens and collect statistics
    }
    
    // After all macroblocks: compute new probabilities from observed statistics
    FinalizeTokenProbas(&enc->proba_);  // <-- KEY DIFFERENCE
    
    // Then recalculate level_costs for next pass
    VP8CalculateLevelCosts(&enc->proba_);
}
```

**The key insight:** `FinalizeTokenProbas()` computes probabilities DIRECTLY from
the observed token statistics, not "signaling-optimized" probabilities. Even if
a probability won't be signaled (because the default is close enough), libwebp
still uses the observed probability for level_cost calculation.

## What We Do Wrong

Our `compute_updated_probabilities()` does this:
```rust
for each probability position:
    if should_update(stats, default_prob, update_prob):
        updated[t][b][c][p] = observed_prob
    else:
        updated[t][b][c][p] = default_prob  // <-- WRONG FOR LEVEL_COSTS
```

We use the **signaling-optimized** probabilities for level_costs. But signaling
optimization only cares about "will updating save bits in the header?" - it doesn't
care about accurate rate estimation for trellis.

## Correct Approach

Need TWO probability tables:
1. **Signaled probabilities** - for header encoding (what decoder will use)
2. **Observed probabilities** - for level_cost calculation (accurate rate estimation)

### Implementation Plan

#### Step 1: Add `compute_observed_probabilities()` in `src/encoder/vp8/mod.rs`

ProbaStats already has `calc_proba()` that computes observed probability from counts!
We just need a method that builds a full probability table using it:

```rust
/// Compute observed probabilities from token statistics.
/// Unlike compute_updated_probabilities(), this uses the RAW observed
/// frequencies for ALL positions, not just where signaling is beneficial.
/// This is essential for accurate level_cost calculation in multi-pass.
fn compute_observed_probabilities(&self) -> TokenProbTables {
    let mut observed = COEFF_PROBS;  // Start from defaults

    for t in 0..4 {
        for b in 0..8 {
            for c in 0..3 {
                for p in 0..11 {
                    let stats = self.proba_stats.stats[t][b][c][p];
                    let total = stats >> 16;  // upper 16 bits = total count

                    if total > 0 {
                        // Use the existing calc_proba method
                        observed[t][b][c][p] = self.proba_stats.calc_proba(t, b, c, p);
                    }
                    // If no samples, keep the default (already set)
                }
            }
        }
    }

    observed
}
```

NOTE: `calc_proba()` returns `255 - (nb * 255 / total)` which matches libwebp's GetProba().

#### Step 2: Update multi-pass loop in `src/encoder/vp8/mod.rs`

The key change: use **observed** probabilities for level_costs, not signaling-optimized ones.

```rust
for pass in 0..num_passes {
    self.token_buffer = Some(residuals::TokenBuffer::with_estimated_capacity(num_mb));
    self.proba_stats.reset();

    if pass > 0 {
        // KEY FIX: Use OBSERVED probabilities for level_costs (accurate rate estimation)
        // NOT the signaling-optimized updated_probs!
        let observed_probs = self.compute_observed_probabilities();
        self.level_costs.mark_dirty();
        self.level_costs.calculate(&observed_probs);
        self.reset_for_second_pass();
    } else {
        // Pass 0: prepare storage
        self.stored_mb_info.clear();
        self.stored_mb_info.reserve(num_mb);
        self.stored_mb_coeffs.clear();
        self.stored_mb_coeffs.reserve(num_mb);
        if num_passes > 1 {
            self.stored_raw_coeffs.clear();
            self.stored_raw_coeffs.reserve(num_mb);
        }
    }

    for each macroblock {
        if pass == 0 {
            // Full mode selection + transform
            let macroblock_info = self.choose_macroblock_info(...);
            let y_block_data = self.transform_luma_block(...);
            let (u_block_data, v_block_data) = self.transform_chroma_blocks(...);

            // Store raw DCT for pass 1+ re-quantization
            if num_passes > 1 {
                self.store_raw_coeffs(&macroblock_info, &y_block_data, &u_block_data, &v_block_data);
            }

            // Quantize and record
            let stored_coeffs = self.record_residual_tokens_storing(...);
            self.stored_mb_coeffs.push(stored_coeffs);
            self.stored_mb_info.push(macroblock_info);
        } else {
            // Pass 1+: Re-quantize from raw DCT with updated level_costs!
            let mb_info = self.stored_mb_info[mb_idx];
            if !mb_info.coeffs_skipped {
                let raw_coeffs = self.stored_raw_coeffs[mb_idx].clone();
                let new_quantized = self.requantize_and_record(&mb_info, mbx, &raw_coeffs);
                self.stored_mb_coeffs[mb_idx] = new_quantized;
            }
        }
    }

    // compute_updated_probabilities is for HEADER SIGNALING only (what decoder will use)
    // It's computed at the end but NOT used for level_costs!
    self.compute_updated_probabilities();
}

// Final emit uses updated_probs (signaling-optimized, matches decoder expectations)
let final_probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);
token_buf.emit_tokens(&mut self.partitions[0], final_probs);
```

#### Step 4: Verify with diagnostics

Add temporary logging to verify:
```rust
#[cfg(debug_assertions)]
{
    let observed = self.compute_observed_probabilities();
    let mut diff_count = 0;
    for t in 0..4 {
        for b in 0..8 {
            for c in 0..3 {
                for p in 0..11 {
                    if observed[t][b][c][p] != COEFF_PROBS[t][b][c][p] {
                        diff_count += 1;
                    }
                }
            }
        }
    }
    eprintln!("Pass {}: {} observed probabilities differ from defaults", pass, diff_count);
}
```

## Files to Modify

1. `src/encoder/vp8/mod.rs`:
   - Re-add `RawMbCoeffs` struct (I removed it when reverting)
   - Re-add `stored_raw_coeffs: Vec<RawMbCoeffs>` field to Vp8Encoder
   - Add `compute_observed_probabilities()` method
   - Update multi-pass loop: store raw DCT in pass 0, re-quantize in pass 1+
   - Use OBSERVED probabilities for level_costs (not signaling-optimized)

2. `src/encoder/vp8/residuals.rs`:
   - Re-add `store_raw_coeffs()` function
   - Re-add `requantize_and_record()` function

NOTE: No changes needed to `cost.rs` - `ProbaStats::calc_proba()` already exists and does what we need!

## Code to Add

### RawMbCoeffs struct (in `src/encoder/vp8/mod.rs`, after QuantizedMbCoeffs)

```rust
/// Raw DCT coefficients for a macroblock, stored for true multi-pass encoding.
/// Unlike QuantizedMbCoeffs (which stores post-quantization values), this stores
/// raw DCT coefficients BEFORE quantization, enabling re-quantization in pass 2+
/// with updated level_costs derived from observed probability tables.
#[derive(Clone)]
struct RawMbCoeffs {
    /// Raw Y2 DC transform coefficients (16 values, post-WHT), only used for I16 mode.
    y2_dct: [i32; 16],
    /// Raw Y1 block DCT coefficients (16 blocks × 16 values), natural order.
    y1_dct: [[i32; 16]; 16],
    /// Raw U block DCT coefficients (4 blocks × 16 values), natural order.
    u_dct: [[i32; 16]; 4],
    /// Raw V block DCT coefficients (4 blocks × 16 values), natural order.
    v_dct: [[i32; 16]; 4],
}
```

Also add `stored_raw_coeffs: Vec<RawMbCoeffs>` field to Vp8Encoder and initialize in `new()`.

### store_raw_coeffs function (in `src/encoder/vp8/residuals.rs`)

```rust
/// Store raw DCT coefficients for multi-pass re-quantization.
pub(super) fn store_raw_coeffs(
    &mut self,
    macroblock_info: &MacroblockInfo,
    y_block_data: &[i32; 16 * 16],
    u_block_data: &[i32; 16 * 4],
    v_block_data: &[i32; 16 * 4],
) {
    let is_i4 = macroblock_info.luma_mode == LumaMode::B;

    // For I16 mode, compute Y2 DC coefficients (WHT of DC values from Y1)
    let y2_dct = if !is_i4 {
        let mut coeffs0 = get_coeffs0_from_block(y_block_data);
        transform::wht4x4(&mut coeffs0);
        coeffs0
    } else {
        [0; 16]
    };

    // Y1 blocks: copy to natural order arrays
    let mut y1_dct = [[0i32; 16]; 16];
    for (block_idx, dct_block) in y1_dct.iter_mut().enumerate() {
        dct_block.copy_from_slice(&y_block_data[block_idx * 16..][..16]);
    }

    // U blocks
    let mut u_dct = [[0i32; 16]; 4];
    for (block_idx, dct_block) in u_dct.iter_mut().enumerate() {
        dct_block.copy_from_slice(&u_block_data[block_idx * 16..][..16]);
    }

    // V blocks
    let mut v_dct = [[0i32; 16]; 4];
    for (block_idx, dct_block) in v_dct.iter_mut().enumerate() {
        dct_block.copy_from_slice(&v_block_data[block_idx * 16..][..16]);
    }

    self.stored_raw_coeffs.push(super::RawMbCoeffs {
        y2_dct, y1_dct, u_dct, v_dct,
    });
}
```

### requantize_and_record function (in `src/encoder/vp8/residuals.rs`)

This is a longer function - it's essentially `record_residual_tokens_storing` but operates
on raw DCT from `RawMbCoeffs` instead of computing new DCT. The key parts:

```rust
pub(super) fn requantize_and_record(
    &mut self,
    macroblock_info: &MacroblockInfo,
    mbx: usize,
    raw: &super::RawMbCoeffs,
) -> super::QuantizedMbCoeffs {
    // Get matrices and trellis lambda (same as record_residual_tokens_storing)
    let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
    let y1_matrix = segment.y1_matrix.clone().unwrap();
    let y2_matrix = segment.y2_matrix.clone().unwrap();
    let uv_matrix = segment.uv_matrix.clone().unwrap();

    let is_i4 = macroblock_info.luma_mode == LumaMode::B;
    let y1_trellis_lambda = if self.do_trellis {
        Some(if is_i4 { segment.lambda_trellis_i4 } else { segment.lambda_trellis_i16 })
    } else {
        None
    };

    // Storage for newly quantized coefficients
    let mut stored = super::QuantizedMbCoeffs { ... };

    // Y2 block (I16 mode only) - re-quantize from raw.y2_dct
    // Y1 blocks - re-quantize from raw.y1_dct using trellis with updated level_costs
    // U/V blocks - re-quantize from raw.u_dct/v_dct

    // Record tokens to token_buffer and update complexity tracking
    // (same logic as record_residual_tokens_storing)

    stored
}
```

The full implementation is ~200 lines. Copy the structure from `record_residual_tokens_storing`
but read from `raw.y1_dct[block_idx]` instead of `y_block_data[block_idx * 16..][..16]`.

## Do We Need Re-Quantization?

YES - the re-quantization approach was correct! The bug was using the wrong probabilities.

libwebp's VP8EncTokenLoop does re-run VP8Decimate() (which includes trellis) in each pass.
The reason is that trellis decisions depend on level_costs, which depend on probabilities.
With better probabilities, trellis makes better rate-distortion tradeoffs.

So we need BOTH:
1. Store raw DCT coefficients in pass 0 (the code I wrote was correct)
2. Re-quantize in pass 1+ using level_costs from OBSERVED probabilities (not signaling-optimized)

The previous implementation stored raw DCT and re-quantized, but used signaling-optimized
probabilities for level_costs, which were nearly identical to defaults. The fix is to use
observed probabilities instead.

## Expected Outcome

- Methods 5/6 should produce smaller files than method 4
- The gain should be ~1-2% (matching libwebp's multi-pass benefit)
- Pass 1 trellis makes DIFFERENT decisions because level_costs are based on
  accurate observed probabilities, not defaults

## Why This Should Work

The issue before was that level_costs barely changed between passes because
`compute_updated_probabilities()` only updates where signaling is beneficial.
Many positions keep the default probability even when the observed frequency
differs significantly.

With observed probabilities:
- If pass 0 shows that level>1 is rare for some position, level_costs will
  increase for that position in pass 1
- Trellis in pass 1 will then prefer level=1 more often at that position
- This improves the rate-distortion tradeoff

## Testing

After implementation:
```bash
# Create test binary
rustc --edition 2021 -O /tmp/test_multipass.rs -L target/release/deps \
    --extern zenwebp=target/release/libzenwebp.rlib \
    -o /tmp/test_multipass

# Run - should show m5 < m4 and m6 < m5
/tmp/test_multipass
```

Expected output:
```
Method 4: XXXXX bytes
Method 5: YYYY bytes (0.5-1% smaller)
Method 6: ZZZZ bytes (1-2% smaller)
```

## Reference: libwebp's FinalizeTokenProbas

From `enc/token_enc.c`:
```c
static void FinalizeTokenProbas(VP8EncProba* const proba) {
  for (int t = 0; t < NUM_TYPES; ++t) {
    for (int b = 0; b < NUM_BANDS; ++b) {
      for (int c = 0; c < NUM_CTX; ++c) {
        for (int p = 0; p < NUM_PROBAS; ++p) {
          const VP8ProbaCount* const cnt = &proba->stats[t][b][c][p];
          const int nb = cnt->counts[0] + cnt->counts[1];
          if (nb > 0) {
            // Direct computation from observed counts
            proba->coeffs[t][b][c][p] = GetProba(cnt->counts[0], cnt->counts[1]);
          }
        }
      }
    }
  }
}
```

Note: It uses ALL observed frequencies, not just where signaling is beneficial.
