# Context Handoff — Token Buffer Implementation

## What Just Happened

Completed a 9-step refactoring plan splitting `vp8.rs` (3335 lines) and `cost.rs` (2779 lines) into focused modules. Zero functional changes, all tests pass. Commits `1fc5a14` through `88207ee`.

### Final file layout:
```
encoder/vp8/mod.rs            1045 lines  (structs, orchestration, segments)
encoder/vp8/header.rs          268 lines  (bitstream header encoding)
encoder/vp8/residuals.rs       659 lines  (coefficient encoding + stats) ← TOKEN BUFFER TARGET
encoder/vp8/mode_selection.rs  792 lines  (I16/I4/UV mode selection)
encoder/vp8/prediction.rs      631 lines  (block prediction + transform)

encoder/cost.rs               1676 lines  (lambdas, RD, ProbaStats, LevelCosts + tests)
encoder/quantize.rs            416 lines  (VP8Matrix, quantdiv)
encoder/trellis.rs             317 lines  (trellis quantization)
encoder/residual_cost.rs       420 lines  (SIMD residual cost + block costs)
```

## Next Task: Token Buffer Implementation

The token buffer is the #1 priority for closing the ~4% file size gap vs libwebp. Plan exists at `PLAN-token-buffer.md`.

### Why it matters — the architectural mismatch

**libwebp method ≥ 3** uses `VP8EncTokenLoop` (file: `~/work/libwebp/src/enc/frame_enc.c:795`):
1. Record all macroblock coefficient decisions as compact tokens (4 bytes each)
2. Periodically refresh probabilities every ~N/8 macroblocks mid-stream
3. At end: finalize probabilities, emit all tokens to arithmetic coder

**zenwebp currently** (file: `src/encoder/vp8/mod.rs:455-614`):
1. Pass 1: Disable trellis, run full encode, record stats to `ProbaStats` — then DISCARD all work
2. Compute updated probabilities from Pass 1 stats
3. Pass 2: Re-enable trellis, redo everything from scratch with updated probabilities

### Three specific gaps this causes:

1. **Re-encoding waste** (~2-3%): Pass 2 redoes `choose_macroblock_info` + transforms + quantization. Mode decisions may differ between passes because trellis is off in Pass 1 but on in Pass 2, so the probabilities don't match the actual output.

2. **No mid-stream probability refresh** (~1%): libwebp calls `FinalizeTokenProbas` + `VP8CalculateLevelCosts` every N/8 macroblocks during token recording (frame_enc.c:840-843). Cost tables get progressively more accurate. We compute once and freeze.

3. **Trellis mismatch**: Pass 1 stats reflect non-trellis quantization (mod.rs:469 sets `self.do_trellis = false`). Pass 2 uses trellis. Coefficient distributions differ. libwebp's token loop uses same RD level throughout — recorded tokens ARE the final output.

### Implementation order:
1. Token buffer struct + recording (Steps 1-2 of PLAN-token-buffer.md)
2. Token-based probability update (Step 3)
3. Token-based arithmetic encoding / replay (Step 4)
4. Mid-stream probability refresh during token recording
5. Enable by default for method ≥ 2 (Step 5)

### Key files to read:
- `PLAN-token-buffer.md` — full implementation plan with Token struct layout
- `src/encoder/vp8/residuals.rs` — `encode_coefficients()` (line ~252) and `record_residual_stats()` (line ~479) are the two paths to unify
- `src/encoder/vp8/mod.rs:455-614` — current two-pass loop to replace
- `src/encoder/cost.rs` — `ProbaStats`, `record_coeffs`, `LevelCosts`
- `~/work/libwebp/src/enc/frame_enc.c:795-906` — libwebp's `VP8EncTokenLoop`
- `~/work/libwebp/src/enc/token_enc.c` — libwebp's token buffer implementation

### Current quality metrics:
- CID22 corpus (method 4): **1.043x** of libwebp file size
- Screenshots: **1.060x** of libwebp
- Method 0 (I16 only): **0.91-1.00x** (better than libwebp — I16 path is good)
- Methods 2-6 (with I4): **1.09-1.14x** (I4 path is the problem)

### Verification commands:
```
cargo clippy --lib --all-features -- -D warnings
cargo test --release
cargo build --no-default-features
```

Note: `--all-targets` has pre-existing test file warnings; use `--lib` for clippy.

### SNS investigation (separate from token buffer):
Our segment quantization spread is more aggressive than libwebp's. CLAUDE.md has details under "SNS Quality-Size Tradeoff Investigation". This is independent work that doesn't block the token buffer.
