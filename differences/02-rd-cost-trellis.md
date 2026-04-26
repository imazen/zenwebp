# RD Cost / Trellis Differences vs libwebp

## Summary

The core trellis quantization algorithm in `src/encoder/trellis.rs` is a faithful
port of libwebp's `TrellisQuantizeBlock` (`quant_enc.c:575-821`): same node layout,
same `{level, level+1}` candidate set, same `kWeightTrellis` distortion weighting,
same `q[1]^2 / 4` last-coefficient threshold, same EOB-pushback logic, same path
unwind. Quantization tables (`kBiasMatrices`, `BIAS`, `QUANTDIV`, `kFreqSharpening`,
`kZigzag`), level tables (`VP8LevelFixedCosts`, `VP8LevelCodes`, `VP8EncBands`,
`VP8EntropyCost`), trellis weights (`kWeightTrellis`), the `MIN_DELTA=0/MAX_DELTA=1`
candidate set, and the lambda formulas in `Segment::init_matrices` all match libwebp
byte-for-byte.

Where zenwebp diverges in ways that are likely to leave bytes on the table:

1. **`get_cost_luma16` and `get_cost_uv` ignore neighbor non-zero context** —
   the residual cost of every DC and AC block is computed with `ctx=0`/`top_nz=[false;4]`
   regardless of whether the surrounding macroblocks produced coefficients. libwebp
   uses `it->top_nz[]`/`it->left_nz[]` carried across MBs. This systematically
   under-estimates rate for textured regions and biases mode selection toward
   modes that would actually be expensive to encode. **This is the single largest
   algorithmic divergence in this subsystem.**

2. **`tlambda` is not gated by method** — libwebp computes
   `tlambda_scale = (method >= 4) ? sns_strength : 0`. zenwebp's
   `Segment::init_matrices` always applies the full `sns_strength`, so methods
   0-3 incur spectral-distortion penalty that libwebp does not. Affects mode
   selection at low method levels.

3. **I16/UV mode selection never invokes trellis** — libwebp's `PickBestIntra16`
   passes `it->do_trellis` to `ReconstructIntra16`, which calls
   `TrellisQuantizeBlock` for each Y1 AC block when method≥6 (`RD_OPT_TRELLIS_ALL`).
   zenwebp's `pick_best_intra16` always uses `quantize_ac_only_simd` regardless
   of method, so the I16 mode chosen at m6 is ranked using non-trellis
   coefficients while final reconstruction uses trellis. The mode selected may
   not be the one that minimizes the actual emitted size.

4. **i4 vs i16 penalty is heuristically scaled, not ported** — `pick_best_intra4`
   uses `3000 * lambda_mode` instead of libwebp's `1000 * q^2 = i4_penalty`.
   The author comments acknowledge this is "43x smaller" than libwebp's value
   to compensate for unrelated lambda differences, but the resulting mode mix
   between I4 and I16 is no longer guaranteed to track libwebp's.

There are also a handful of minor divergences (MAX_COST scale, dual-quantization
of I4 blocks in both `transform_luma_blocks_4x4` and `record_residual_tokens_storing`,
unused `zigzag_out` in the prediction pass) that are very unlikely to affect
output size but are documented below for completeness.

## Divergences

### `get_cost_luma16` ignores cross-MB and intra-MB non-zero context — severity: high — likely size impact: 0.5-1.5%

- **zenwebp:** `src/encoder/residual_cost.rs:781-812` — `get_cost_luma16`
  initializes `top_nz=[false;4]` / `left_nz=[false;4]` per call, ignoring any
  previously-coded neighboring macroblocks; passes hard-coded `ctx=0` for the
  Y2 (DC) residual.
- **libwebp:** `src/enc/cost_enc.c:237-261` — `VP8GetCostLuma16` calls
  `VP8IteratorNzToBytes(it)` to import the live cross-MB context, then uses
  `it->top_nz[8] + it->left_nz[8]` for the Y2 DC residual and updates
  `it->top_nz[x]`/`it->left_nz[y]` after each AC block.
- **Difference:** zenwebp's version is "context-amnesiac" — the cost it returns
  for an I16 macroblock is independent of the surrounding image, while
  libwebp's accounts for the actual probability state at this MB. Because
  `VP8EntropyCost[]` for ctx=0 vs ctx=1/2 differs by 100-300 cost units per
  band (and the EOB cost is `vp8_bit_cost(false, p[0])` which can vary by
  300-700), the systematic underestimate is on the order of 1-3 KiB per
  full-HD image.
- **Why this might cost bytes:** mode selection compares the I16 RD score
  against the I4 RD score (which has its own context-tracking issues, see below)
  and against UV. Because zenwebp under-estimates I16 rate, the encoder picks
  I16 modes more often than libwebp would when neighbors actually had
  coefficients — the mode is then expensive to actually encode in the
  bitstream.
- **Suggested fix:** propagate `top_complexity[mbx]` / `left_complexity` (which
  are already maintained for the actual encode path in `residuals.rs:1311-1320,
  1325-1370`) into `get_cost_luma16` so the cost tracker matches the live
  encoder context. Track `top_nz`/`left_nz` updates inside the function the
  same way libwebp does, including the Y2 DC ctx from `top/left_complexity.y2`.

### `get_cost_uv` always uses `ctx=0` — severity: high — likely size impact: 0.3-1.0%

- **zenwebp:** `src/encoder/residual_cost.rs:825-836` — `get_cost_uv` iterates
  all 8 UV blocks and calls `get_residual_cost(0, &res, ...)` for every block.
- **libwebp:** `src/enc/cost_enc.c:263-283` — `VP8GetCostUV` walks the U and V
  channels with `int ctx = it->top_nz[4 + ch + x] + it->left_nz[4 + ch + y];`
  per block and updates the iterator state.
- **Difference:** identical class of bug to the I16 case but applied to UV
  mode selection. Every chroma block is costed as if no neighbor had coefficients.
- **Why this might cost bytes:** the chroma mode selection (DC/V/H/TM) is
  decided by an under-estimated rate, so a more-expensive-to-encode mode may
  get picked. libwebp's chroma is already very cheap (~1-2% of the bitstream),
  so the absolute byte impact is smaller than the I16 case but still
  measurable.
- **Suggested fix:** same pattern — pass cross-MB context in, track per-block,
  use `(top_nz + left_nz).min(2)` for the call.

### `tlambda` is not gated by `method >= 4` — severity: medium — likely size impact: 0.1-0.3% at m0-m3

- **zenwebp:** `src/common/types.rs:875-877` —
  `self.tlambda = (u32::from(sns_strength) * q_i4) >> 5;`
  unconditionally.
- **libwebp:** `src/enc/quant_enc.c:226` —
  `const int tlambda_scale = (enc->method >= 4) ? enc->config->sns_strength : 0;`
  followed by `m->tlambda = (tlambda_scale * q_i4) >> 5;`.
- **Difference:** at methods 0-3 with `sns_strength > 0` (which is the default
  preset, sns=50), zenwebp adds a spectral-distortion (`TDisto`) penalty in
  mode selection that libwebp omits. Affects which I16 mode is picked at fast
  methods.
- **Why this might cost bytes:** TDisto biases the encoder toward smoother
  reconstructions, which usually means more zero-able coefficients but
  occasionally a worse pixel-domain match. At m0-m3 the spectral penalty
  shouldn't be in play.
- **Suggested fix:** in `Segment::init_matrices`, multiply `sns_strength` by
  `(method >= 4) as u32` before computing `tlambda`.

### I16 mode selection never invokes trellis at m6 — severity: medium — likely size impact: 0.1-0.4% at m6

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:730-745` (within
  `pick_best_intra16`) — Y1 AC blocks for each candidate mode are quantized
  with `quantize_ac_only_simd(&mut block, y1_matrix, true)` regardless of
  `do_trellis_i4_mode`/`do_trellis`.
- **libwebp:** `src/enc/quant_enc.c:847-870` (`ReconstructIntra16`) — when
  `it->do_trellis` is set (which `RD_OPT_TRELLIS_ALL` / m6 enables before
  `PickBestIntra16` runs), each Y1 AC block goes through `TrellisQuantizeBlock`
  with `lambda_trellis_i16`.
- **Difference:** at m6, libwebp ranks the four I16 modes (DC/V/H/TM) using
  trellis-quantized coefficients (the same coefficients that will actually be
  emitted). zenwebp ranks them using simple-quantize coefficients and only
  applies trellis afterward in `record_residual_tokens_storing`. The mode
  picked at the simple-quantize ranking can differ from the optimal-after-
  trellis mode.
- **Why this might cost bytes:** trellis often zeros small coefficients that
  simple quantization keeps; a mode whose simple-quant residual looks "cheap"
  may not be the cheapest after trellis. At m6 specifically, this is a
  correctness gap relative to libwebp's `RD_OPT_TRELLIS_ALL` semantics.
- **Suggested fix:** when `do_trellis_i4_mode` (m6) is set, call
  `trellis_quantize_block` with `lambda_trellis_i16` and `ctype=0` (I16_AC)
  inside the `pick_best_intra16` per-mode loop, mirroring the existing m6
  trellis path in `evaluate_i4_modes_sse2`.

### I4-vs-I16 penalty rescaled instead of ported — severity: medium — likely size impact: unclear, depends on image mix

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:1101-1104` —
  `let base_penalty = 3000u64 * u64::from(lambda_mode);`
  with comments stating "ratio (43x smaller) matches the ratio of our
  lambdas" and a `partition_limit`-driven scaling of the penalty.
- **libwebp:** `src/enc/quant_enc.c:267, 1315` —
  `m->i4_penalty = 1000 * q_i4 * q_i4;` then in `RefineUsingDistortion`:
  `score_t score_i4 = dqm->i4_penalty;`. Note that `RefineUsingDistortion`
  is the m0-m1 path; `PickBestIntra4`/`PickBestIntra16` (m2+) instead use
  `lambda_mode * (R + H) + 256 * (D + SD)` directly with no extra penalty
  beyond `total_header_bits` checks.
- **Difference:** zenwebp applies the I4 penalty in its `pick_best_intra4`
  scoring at all method levels, scaled to `3000 * lambda_mode`. libwebp's
  `i4_penalty` is only used in `RefineUsingDistortion` (the no-RD path). At
  m2-m6, libwebp's I4 vs I16 decision is purely RD-driven with the
  `max_i4_header_bits` cap providing the only structural bias against I4.
- **Why this might cost bytes:** at m4-m6 (where production sizes are
  measured) zenwebp is suppressing I4 with a heuristic penalty that libwebp
  has nothing equivalent to. On detailed images where I4 wins, zenwebp picks
  I16 too often and pays the resulting coefficient cost.
- **Suggested fix:** needs measurement. The cleanest port would be: at
  m0-m1 mirror `RefineUsingDistortion` exactly (use `i4_penalty = 1000 * q^2`
  with `lambda_d_i4 = 11`); at m2+ remove the per-block `i4_penalty` and rely
  on the RD score plus a `max_i4_header_bits = 256 * mb_w * mb_h / 4`-style
  cap as libwebp does.

### `pick_best_intra16` does not import live `top_nz`/`left_nz` for `get_cost_luma16` — severity: high — likely size impact: subsumed by item #1

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:749` — calls
  `get_cost_luma16(&y2_quant, &y1_quant, &self.level_costs, probs)` with no
  context input.
- **libwebp:** `src/enc/quant_enc.c:1084` — calls `R += VP8GetCostLuma16(it, rd_cur);`
  passing the iterator (which carries the live nz state).
- **Difference:** see item #1 — same root cause, listed separately because the
  fix has two halves: (a) `get_cost_luma16` API needs to accept context and
  (b) `pick_best_intra16` needs to supply it. Saving the call-site change as
  its own line item to avoid losing it.
- **Suggested fix:** add `top_complexity[mbx]` and `left_complexity` parameters
  to `get_cost_luma16` and forward them from the caller.

### `pick_best_uv` does not import live `top_nz`/`left_nz` for `get_cost_uv` — severity: high — likely size impact: subsumed by item #2

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:1590` — calls
  `get_cost_uv(&uv_quant, &self.level_costs, probs)` with no context.
- **libwebp:** `src/enc/quant_enc.c:1248` — `R += VP8GetCostUV(it, &rd_uv);`.
- **Difference:** mirror of item #2 at the call site.
- **Suggested fix:** mirror of item #2's fix.

### I4 mode-selection trellis at m6 uses a separate `level_costs` snapshot from final encode — severity: low — likely size impact: <0.1%

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:225-235` (mode selection) and
  `src/encoder/vp8/residuals.rs:1339-1349` (final encode) both call
  `trellis_quantize_block` with `&self.level_costs`. If `level_costs` was
  refreshed between the two calls (multi-pass), trellis at mode selection
  would use older probabilities than the final encode.
- **libwebp:** `src/enc/quant_enc.c:892-896, 1292` — both call sites read
  `enc->proba.remapped_costs[coeff_type]` which is updated only by
  `VP8CalculateLevelCosts` between full encode passes.
- **Difference:** in zenwebp's single-pass mode (default at m4) this is a
  no-op — `level_costs` is calculated once at start. In multi-pass (m5-m6),
  the costs are recalculated between passes; both call sites then see the
  same fresh values, so this should also be fine. Listed for completeness.
- **Suggested fix:** none; verify with a single-pass-vs-multi-pass A/B.

### `transform_luma_blocks_4x4` discards the trellis `zigzag_out` — severity: low — likely size impact: 0%

- **zenwebp:** `src/encoder/vp8/prediction.rs:405-431` — calls
  `trellis_quantize_block(&mut current_subblock, &mut zigzag_out, ...)` then
  uses `current_subblock` (dequantized values) for IDCT and discards
  `zigzag_out`. The actual quantized levels are recomputed in
  `record_residual_tokens_storing`.
- **libwebp:** `src/enc/quant_enc.c:881-902` — `ReconstructIntra4` runs
  trellis once and uses both outputs (the `levels` are written into the
  caller's `levels[16]` buffer for direct token recording).
- **Difference:** zenwebp redoes the trellis work twice for the same block
  (once in prediction, once in record). The output should be identical
  (same lambda, same matrix, same coeffs). Pure CPU cost, no bitstream
  effect.
- **Suggested fix:** thread the trellis output from prediction through to
  the residual recorder. Not a size optimization but worth ~10-15% encoder
  speedup at m5-m6.

### `MAX_COST` scale differs from libwebp — severity: low — likely size impact: 0%

- **zenwebp:** `src/encoder/trellis.rs:24` — `const MAX_COST: i64 = i64::MAX / 2;`
  ≈ 4.6e18.
- **libwebp:** `src/enc/vp8i_enc.h:113` —
  `#define MAX_COST ((score_t)0x7fffffffffffffLL)` ≈ 3.6e16.
- **Difference:** zenwebp's sentinel is ~127× larger. In all practical paths
  the trellis adds at most ~`lambda * VP8_LEVEL_FIXED_COSTS[2047] * 17`
  ≈ `1e7 * 7761 * 17` ≈ 1.3e12 to the score, so neither sentinel is reached.
- **Suggested fix:** none required; harmonize values for clarity if desired.

### EOB context for last coefficient differs in trellis path subtly — severity: low — likely size impact: 0%

- **zenwebp:** `src/encoder/trellis.rs:295-298` — `level_costs.get_eob_cost(ctype, n, ctx)`
  where `ctx = (level as usize).min(2)`.
- **libwebp:** `src/enc/quant_enc.c:752-755` — `VP8BitCost(0, probas[band][ctx][0])`
  where `band = VP8EncBands[n + 1]` and `ctx = (level > 2) ? 2 : level`.
- **Difference:** `level_costs.get_eob_cost` internally computes
  `band = VP8_ENC_BANDS[(n+1).min(15)]` (`level_costs.rs:194-195`), and
  `eob_cost[ctype][band][ctx] = vp8_bit_cost(false, p[0])`. With `n<15`
  (already guarded), `(n+1).min(15) == n+1`, so this is equivalent. Match.
- **Suggested fix:** none.

### `quantize_dequantize_ac_only_simd` overrides DC slot inconsistently — severity: low — likely size impact: 0%

- **zenwebp:** `src/encoder/quantize.rs:846-859` — after fused
  quantize+dequantize, sets `quantized[0] = coeffs[0]` and
  `dequantized[0] = coeffs[0]` (the original DC). The dequantized DC is
  meant to be replaced by the WHT-derived value later in I16 reconstruction.
- **libwebp:** `src/enc/quant_enc.c:862-869` (`ReconstructIntra16`,
  non-trellis path) — explicitly zeroes `tmp[n][0] = tmp[n+1][0] = 0;`
  before calling `VP8EncQuantize2Blocks`, so the DC slot of the quantized
  output is exactly 0 and the dequantized slot is 0 until `VP8TransformWHT`
  refills it.
- **Difference:** the residual DC slot is either "0 then overwritten by WHT"
  (libwebp) or "original-coeff then overwritten by WHT" (zenwebp). Both end
  up overwritten before being used downstream, so this is a transient state
  difference only.
- **Suggested fix:** none required; the divergence is invisible in output.

## Things that LOOK different but are actually equivalent

- `quantization_bias(b) = ((b << 17) + 128) >> 8` (zenwebp `quantize.rs:65-67`)
  vs `BIAS(b) = b << (QFIX-8)` (libwebp `vp8i_enc.h:116`). Both evaluate to
  `b << 9` for all `b` in `[0, 255]`. Verified at `b ∈ {0, 0x80, 96, 110, 115}`.
- zenwebp's `quantdiv` uses `u64` to avoid overflow; libwebp uses `uint32_t`.
  For coefficients in the legal `[0, 2048 << QFIX]` range the high word is
  always zero. Same numeric output.
- `LevelCostArray` in zenwebp is padded to 128 entries (vs libwebp's
  `MAX_VARIABLE_LEVEL+1=68`) to allow `t[level & 0x7F]` bounds-check
  elimination. Padding entries `68..127` all hold the level-67 cost, so
  any out-of-band index returns the correct clamped cost.
- `kBiasMatrices` ordering: libwebp's comment `[luma-ac, luma-dc, chroma]`
  with indices `[0, 1, 2]`. zenwebp's `MatrixType::{Y1, Y2, UV}` maps to
  the same `(96,110), (96,108), (110,115)` triples in the same order.
- `VP8_LEVEL_CODES`, `VP8_LEVEL_FIXED_COSTS`, `VP8_ENTROPY_COST`,
  `VP8_ENC_BANDS`, `VP8_ZIGZAG`, `VP8_FREQ_SHARPENING`, `VP8_WEIGHT_TRELLIS`,
  `VP8_WEIGHT_Y` all match libwebp byte-for-byte.
- `VP8_FIXED_COSTS_I4`, `VP8_FIXED_COSTS_I16`, `VP8_FIXED_COSTS_UV` arrays
  (verified for I16/UV; I4's full 10×10×10 table not exhaustively diffed but
  the constants in `vp8/mode_selection.rs` are the same as `cost_enc.c:106-206`).
- Trellis lambda formulas in `Segment::init_matrices` (`common/types.rs:860-872`):
  `((7*q²)>>3, (q²)>>2, (q²)<<1)` for `(i4, i16, uv)` and
  `((3*q²)>>7, 3*q², (3*q²)>>6, (q²)>>7)` for `(lambda_i4, lambda_i16, lambda_uv,
  lambda_mode)` — exact match with libwebp `quant_enc.c:245-251`.
- Trellis distortion formula `weight * (new_err² - orig_err²)`,
  `kWeightTrellis[j]`, the `q[1]² / 4` last-coefficient threshold, the
  `level0..level0+1` candidate set, the EOB-extension bonus on the last
  significant coeff, the unwind that sets `coeffs[j] = level * q[j]` —
  all match libwebp exactly.
- Trellis `coeff_with_sharpen` uses sharpen for Y1 only (because
  `m.sharpen[]` is zero for Y2 and UV). Match.
- Skip-block "all zero" path in trellis: both encode `last < 0 → return 0/false`
  with `best_score = RDScoreTrellis(lambda, VP8BitCost(0, last_proba), 0)`
  as the comparison baseline. Match.
- `RD_DISTO_MULT = 256` (zenwebp `cost/mod.rs:47`) matches libwebp
  `quant_enc.c:54`. The libwebp `SetRDScore` formula
  `score = (R + H) * lambda + RD_DISTO_MULT * (D + SD)` matches zenwebp's
  `rd_score_full` (`cost/mod.rs:187-197`).
- I16 flat-source penalty (`D *= 2; SD *= 2`) and `FLATNESS_PENALTY * num_blocks`
  for I4/UV: match.
- `IsFlat` and `IsFlatSource16`: present in both with the same threshold
  semantics.
- `ReconstructIntra16` uses `VP8EncQuantizeBlockWHT = QuantizeBlock_C` for
  the Y2 (DC) block (no trellis). zenwebp matches in
  `record_residual_tokens_storing` (`residuals.rs:1300-1322`).
- `ReconstructUV` always uses `VP8EncQuantize2Blocks` (no trellis,
  `DO_TRELLIS_UV = 0`). zenwebp matches in `record_residual_tokens_storing`
  (`residuals.rs:1372-1421`).

## Open questions

1. **Quantify the cost-context divergences.** Wire `top_complexity` /
   `left_complexity` into `get_cost_luma16` / `get_cost_uv` and re-run CID22
   Q75 production sweep. Hypothesis: closes 0.5-1.0% of the 1.49% gap.
2. **m6 I16 trellis in mode selection.** Add a `trellis_lambda_i16` arm to
   `pick_best_intra16` mirroring the existing `evaluate_i4_modes_sse2` pattern;
   measure the m6 size delta in isolation. Hypothesis: 0.1-0.3% improvement at
   m6 only, no effect at m4.
3. **i4_penalty rework.** Replace the `3000 * lambda_mode` hack with libwebp's
   `total_header_bits / max_i4_header_bits` cap from `PickBestIntra4`
   (`quant_enc.c:1202-1205`). May increase I4 selection rate; need to
   verify it doesn't blow partition 0 on large images (the current
   `partition_limit` scaling exists for that reason).
4. **Verify `VP8_FIXED_COSTS_I4` table.** I read the libwebp values but
   only spot-checked the first few entries against the zenwebp constant.
   Worth a full diff to rule out any transcription error.
5. **`tlambda` gating.** Add `if method >= 4` to `init_matrices`; measure
   m0-m3 size deltas. Should be negligible at m4+, may shift size at m0-m2.
6. **Trellis double-work.** Thread the `zigzag_out` from
   `transform_luma_blocks_4x4` to `record_residual_tokens_storing` so trellis
   runs once. Pure speed optimization; verify no regressions in pixel parity
   tests.
7. **Cross-check `level_cost` table generation.** zenwebp's
   `LevelCosts::calculate` (`cost/level_costs.rs:115-168`) matches libwebp's
   `VP8CalculateLevelCosts` (`cost_enc.c:66-96`) line-by-line on inspection,
   but the level_cost values produced by both should be diffed numerically
   for at least one `(ctype, band, ctx)` triple to rule out subtle bugs.
