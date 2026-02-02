# Token Buffer Implementation Plan

## Goal
Replace direct arithmetic encoding with a two-pass token buffer approach (matching libwebp's `VP8EncTokens`). This enables optimal probability updates between passes, improving compression ratio.

## Background
Currently, zenwebp encodes coefficients directly to the arithmetic coder in a single pass. libwebp instead:
1. **Pass 1**: Records all coefficient decisions as tokens in a buffer
2. Uses token statistics to compute optimal probability updates
3. **Pass 2**: Emits tokens from the buffer using updated probabilities

The token buffer is the key remaining architectural difference causing our ~4% file size gap on I4-heavy content.

## Architecture

### Token Representation
```rust
/// Compact token: 4 bytes each
/// - bits 0-7: value (coefficient level, clamped to 67)  
/// - bits 8-9: context for next coefficient (0, 1, or 2)
/// - bit 10: sign
/// - bit 11: is_eob (end of block marker)
/// - bits 12-15: position (0-15, coefficient index in zigzag order)
/// - bits 16-19: coefficient type (0-3: I16AC, I16DC, Chroma, I4)
/// - bits 20-22: band (0-7, from VP8_ENC_BANDS)
struct Token(u32);
```

### Buffer Structure
```rust
struct TokenBuffer {
    tokens: Vec<Token>,
    /// Per-macroblock start indices for random access
    mb_starts: Vec<u32>,
}
```

### Memory Budget
- ~400 tokens per macroblock (16 luma + 1 DC + 8 chroma blocks × ~16 coeffs)
- 1920×1080 image: 8100 macroblocks × 400 × 4 bytes ≈ 12.3 MB
- Acceptable for a quality-focused encoder

## Implementation Steps

### Step 1: Add TokenBuffer struct to `vp8/residuals.rs`
- Define `Token` and `TokenBuffer` types
- Implement `push_token()`, `push_eob()`, `clear()`
- Add `token_buf: Option<TokenBuffer>` field to `Vp8Encoder`
- Unit tests for token packing/unpacking

### Step 2: Record tokens during encoding
- In `encode_coefficients()`: when token buffer is active, record tokens instead of (or in addition to) emitting to arithmetic coder
- Keep the existing direct-encode path as fallback (method 0-1)
- Token recording should capture: level, context, position, type, band, sign

### Step 3: Token-based probability update
- After all macroblocks are tokenized, compute optimal probabilities from token statistics
- Compare with `ProbaStats`-based approach (current) — token buffer should give identical or better statistics
- Write updated probabilities to bitstream header

### Step 4: Token-based arithmetic encoding (pass 2)
- Emit tokens from buffer using updated probabilities
- This replaces the current `encode_coefficients()` → arithmetic coder path
- Verify bitstream correctness by comparing decoded output

### Step 5: Enable by default for method ≥ 2
- Methods 0-1: direct encode (fast, no token buffer)
- Methods 2+: token buffer for better compression
- Benchmark file size improvement and speed impact

## Verification
- Decoded images must be bit-identical before/after token buffer (same quality target)
- File sizes should decrease (better probability estimates)
- Speed regression should be < 10% (extra pass over tokens)

## Files Affected
- `src/encoder/vp8/residuals.rs` — token buffer struct, recording, emission
- `src/encoder/vp8/mod.rs` — encoder state, enable/disable logic
- `src/encoder/cost.rs` — probability update from tokens (reuse ProbaStats)
