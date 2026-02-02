# Context Handoff: I4 Diagnostic Harness Implementation

**Date:** 2026-02-02
**Session focus:** Implementing I4 encoding efficiency decomposition harness

## What Was Done

Implemented the full diagnostic harness as planned in CLAUDE.md's "I4 Encoding Efficiency Decomposition Plan":

### Files Created/Modified

1. **`src/decoder/vp8.rs`** — Added diagnostic infrastructure:
   - `BlockDiagnostic` - raw quantized levels (pre-dequant) for a 4x4 block
   - `MacroblockDiagnostic` - luma/chroma modes, segment ID, I4 bpred modes, coefficient blocks
   - `DiagnosticFrame` - complete frame diagnostic with segment quantizers, all MBs, probability tables
   - `TreeNode` made public for probability table export
   - `Vp8Decoder::decode_diagnostic()` - new entry point returning `(Frame, DiagnosticFrame)`
   - Instrumented `read_coefficients_to_block()` and `read_coefficients()` to capture raw levels

2. **`src/decoder/mod.rs`** — Re-exports for diagnostic types (doc-hidden)

3. **`src/common/types.rs`** — Made `LumaMode`, `ChromaMode`, `IntraMode` public (doc-hidden)

4. **`tests/i4_diagnostic_harness.rs`** — Test harness with 5 test cases:
   - `single_mb_diagnostic` - 16x16 single MB, method 2
   - `small_image_diagnostic` - 64x64, method 2
   - `small_image_m4_diagnostic` - 64x64, method 4 (with trellis)
   - `benchmark_image_diagnostic` - 792079.png if available
   - `probability_table_comparison` - compares token prob tables

5. **`justfile`** — Added `just diag` target

### How It Works

1. Encodes same image with both zenwebp and libwebp (matched settings: Q75, SNS=0, filter=0, segments=1)
2. Extracts VP8 chunks from WebP containers
3. Decodes both with diagnostic capture, collecting:
   - Segment quantizer values (ydc, yac, y2dc, y2ac, uvdc, uvac)
   - Per-MB: luma_mode, chroma_mode, segment_id, coeffs_skipped
   - Per-MB (I4 only): bpred_modes[16], y_blocks[16], uv_blocks[8]
   - Per-block: levels[16] (raw quantized, zigzag order), eob_position
   - Final token probability tables
4. Compares and reports divergences

### Initial Results (Synthetic Test Images)

```
=== Single MB Diagnostic (16x16, method 2, Q75) ===
File sizes: zenwebp=50 bytes, libwebp=50 bytes, ratio=1.000x
Segment quantizers: [MATCH]
Mode decisions: 1/1 match (100.0%)

=== Small Image Diagnostic (64x64, method 2, Q75) ===
File sizes: zenwebp=100 bytes, libwebp=136 bytes, ratio=0.735x
Mode decisions: 16/16 match (100.0%)
Mode breakdown: zenwebp 0 I4 / 16 total, libwebp 0 I4 / 16 total

=== Probability Table Comparison ===
Token probabilities: 1046/1056 match (99.1%)
```

Synthetic checkerboard patterns produce mostly I16 modes, so coefficient comparison isn't triggered. Need real CID22 images to see I4 differences.

## Next Steps

1. **Run on real images** — Test with CID22 corpus at `/mnt/v/cid22_pngs/cid22/`
2. **Identify divergence stage** — The harness will show whether differences are in:
   - Mode selection (different I4 vs I16 decisions)
   - Coefficient levels (same mode, different quantized values)
   - Probability tables (different probability updates)
3. **Trace root cause** — Once the divergence stage is identified, focus investigation on that specific file:
   - Modes differ → `src/encoder/vp8/mode_selection.rs`
   - Levels differ → `src/encoder/quantize.rs` or `src/encoder/cost.rs` (trellis)
   - Probs differ → `src/encoder/cost.rs` (ProbaStats)

## Key Commands

```bash
just diag                    # Run diagnostic harness
just check                   # fmt + clippy + test
cargo build --no-default-features  # Verify no_std
```

## Commits This Session

- `308451a` feat: add I4 encoding diagnostic harness for encoder comparison
- `74fb655` chore: add 'just diag' target for I4 diagnostic harness

## Files to Read First in New Session

1. `CLAUDE.md` - Current optimization status and investigation notes
2. `tests/i4_diagnostic_harness.rs` - The diagnostic test harness
3. `src/decoder/vp8.rs` lines 38-95 - Diagnostic data structures
