# Token Coding / Bitstream Differences vs libwebp

## Summary

The boolean arithmetic encoder (`ArithmeticEncoder`), per-coefficient token tree
(`TokenBuffer::record_coeff_tokens` / `emit_tokens`), zigzag/band/category
tables, fixed mode-cost tables, and frame-header bit-ordering are byte-for-byte
faithful ports of libwebp's `VP8PutBit` / `VP8RecordCoeffTokens` /
`VP8EmitTokens` / `VP8WriteProbas` / `PutQuant` / `PutFilterHeader`. The 2026-03-05
fixes for `write_optional_signed_value` sign and the missing UV quant deltas are
in place and correct (`arithmetic.rs:156-164`, `header.rs:152-165`).

The remaining divergences are architectural: zenwebp uses a token-buffer pipeline
(matching libwebp's `VP8EncTokenLoop`) but layers libwebp's *non-token-loop*
skip-flag mechanism on top of it, always emits the per-MB skip bit even when it
costs more than it saves, never refreshes `level_costs` mid-stream (so trellis
and mode selection use stale costs in m5-6), runs only `num_passes = 1` (no
stat-collection pass for `pass>1` or size search), and always emits a single
partition (no `num_parts ∈ {2,4,8}` support). The probability-update gate uses
COEFF_PROBS as the comparison baseline, not the previous pass's probabilities,
which is correct for single-pass but disables progressive refinement.

The big size-cost items, in approximate order of impact at CID22 m4 Q75:

1. `use_skip_proba`-style threshold gate is missing (zenwebp always emits the
   per-MB skip bit). On low-skip photos this wastes ~0.3-0.5 bits/MB.
2. Multi-pass / stat-collection (`StatLoop` / `pass>=2`) is absent. The 8-times-
   per-pass mid-stream `compute_updated_probabilities` is dead work because
   `level_costs` is never recalculated.
3. Per-segment quantizer is signaled in **delta** mode rather than libwebp's
   **absolute** mode — bitstream is valid and decoder produces identical
   matrices, but the encoded byte counts differ by a few bits per frame depending
   on the deltas chosen.

## Divergences

### Skip-coeff flag has no `SKIP_PROBA_THRESHOLD` gate — severity: high — likely size impact: 0.3-0.8% on photos with low skip rates

- **zenwebp:** `src/encoder/vp8/mod.rs:1093-1097` and `src/encoder/vp8/mod.rs:1540`
  — `setup_encoding` initializes `self.macroblock_no_skip_coeff = Some(200)` and
  `encode_image` overwrites it with `((255*non_skip + total/2)/total).clamp(1,254)`,
  so the field is *always* `Some(_)`. `encode_compressed_frame_header`
  (`src/encoder/vp8/header.rs:63-71`) therefore always emits
  `mb_no_skip_coeff = 1` followed by the 8-bit skip_proba, and
  `write_macroblock_header` (`src/encoder/vp8/header.rs:221-224`) always emits a
  per-MB skip bit.
- **libwebp:** `src/enc/frame_enc.c:111` defines
  `SKIP_PROBA_THRESHOLD = 250`; `FinalizeSkipProba`
  (`src/enc/frame_enc.c:118-132`) sets
  `proba->use_skip_proba = (skip_proba < 250)`. When `use_skip_proba == 0`,
  `VP8WriteProbas` (`src/enc/tree_enc.c:432-434`) emits a single `0` flag and no
  per-MB skip bits in `VP8CodeIntraModes` (`src/enc/tree_enc.c:293-295`). The
  token-loop path even *asserts* `use_skip_proba == 0`
  (`src/enc/frame_enc.c:816`).
- **Difference:** zenwebp pays ~0.3-0.5 bits per MB for the skip flag
  unconditionally. libwebp turns the flag off entirely when fewer than ~2 % of
  MBs are skip candidates — saving the per-MB cost AND the 8-bit skip_proba in
  the header.
- **Why this might cost bytes:** for typical lossy photo content the skip rate
  at Q75 is well below 2 %; under those conditions zenwebp pays ~1 bit per MB
  for a flag that is almost always 0 with a flat-ish prior, while libwebp pays
  zero. CID22-style photo corpora at Q75 sit in exactly this regime.
- **Suggested fix:** when computed `prob >= 250` (i.e. skip rate < ~2 %), set
  `self.macroblock_no_skip_coeff = None` and stop emitting the per-MB
  `coeffs_skipped` bit. Verify against libwebp by encoding a no-skip image and
  comparing partition-0 sizes.

### `num_passes = 1`, no stat-collection pass, no progressive `level_costs` refresh — severity: high — likely size impact: 0.5-1.0% at m5-6, smaller at m4

- **zenwebp:** `src/encoder/vp8/mod.rs:840-845` hard-codes
  `let num_passes = 1`; `mod.rs:931-935` does call
  `compute_updated_probabilities` mid-stream every `~mb_count/8` MBs, but that
  function only writes to `self.updated_probs` for header signaling — it does
  *not* recompute `self.level_costs`. The comment at lines 928-930 says this
  was tested and "hurts compression," which is consistent with the fact that
  zenwebp recomputes neither `level_costs` nor anything else mid-stream, so the
  refresh has no effect.
- **libwebp:** `src/enc/frame_enc.c:626-684` (`StatLoop`) runs `config->pass`
  passes of `OneStatPass` *before* the real encode, accumulating token stats
  into `proba->stats` and calling `FinalizeTokenProbas` /
  `FinalizeSkipProba` between iterations to derive the probas the next pass
  encodes against. `VP8EncTokenLoop` (`src/enc/frame_enc.c:795-906`) then runs
  `pass` more passes, calling `FinalizeTokenProbas` AND
  `VP8CalculateLevelCosts` mid-pass every `max_count = max(mb/8, 96)` MBs
  (`frame_enc.c:840-844`). The token loop's output is therefore quantized and
  mode-selected against costs derived from the actual coefficient distribution,
  not the static defaults.
- **Difference:** zenwebp's mode-selection (`PickBestIntra4` cost equivalents)
  and trellis quantization (m5-6) always run against
  `level_costs.calculate(&COEFF_PROBS)` (line 820). libwebp's same paths run
  against costs that converge toward the empirical distribution of the image.
- **Why this might cost bytes:** wrong cost weights bias mode selection toward
  options that look cheap under defaults but turn out expensive under the real
  proba updates. The effect is largest at m6 (trellis-during-I4) and at very
  low/high quantizers where the empirical distribution diverges from the spec
  defaults the most.
- **Suggested fix:** either (a) wire up `num_passes >= 2` and exercise the
  `record_residual_tokens_storing` / `record_from_stored_coeffs` two-phase
  path that already exists, or (b) at minimum recompute `level_costs` mid-pass
  alongside each `compute_updated_probabilities` call and re-run mode selection
  for subsequent MBs. The "tested, hurt compression" comment may be the result
  of recomputing probas without ALSO recomputing matrices/lambdas — match
  libwebp's full mid-stream refresh sequence (`FinalizeTokenProbas` +
  `VP8CalculateLevelCosts`) in lockstep.

### Single-partition only — severity: low — likely size impact: 0% size, but blocks `cwebp -partitions N>0` parity

- **zenwebp:** `src/encoder/vp8/mod.rs:559` initializes
  `partitions: vec![ArithmeticEncoder::new()]` and `mod.rs:695` always
  resets to a single-partition `vec![ArithmeticEncoder::with_capacity(...)]`.
  `header.rs:53` emits
  `partitions_value = self.partitions.len().ilog2() = 0`. There is no codepath
  that creates 2/4/8 partitions.
- **libwebp:** `src/enc/syntax_enc.c:280-285` writes `log2(num_parts)` (0..3)
  and `EmitPartitionsSize` (`syntax_enc.c:245-262`) emits `(num_parts-1) * 3`
  bytes of partition sizes. `cwebp -partitions N` selects 1, 2, 4, or 8
  partitions, with MB rows interleaved across them.
- **Difference:** zenwebp cannot honor `EncoderParams::partitions`. With one
  partition the bitstream is identical-shaped to libwebp's `partitions=0`
  default.
- **Why this might cost bytes:** none under the default `partitions=0`. With
  multiple partitions the framing overhead is `(N-1)*3` bytes plus a small
  per-row interleaving cost; libwebp users rarely set this.
- **Suggested fix:** if multi-partition support is wanted, plumb through
  `params.partitions`, allocate `ArithmeticEncoder::with_capacity` per
  partition, route MB row `r` to `partitions[r % num_parts]`, and let
  `write_partitions` already-existing length-prefix path handle the rest.
  Otherwise document that this knob is intentionally a no-op.

### Segment header uses delta mode (libwebp uses absolute) — severity: low — likely size impact: ±10-30 bits per frame

- **zenwebp:** `src/encoder/vp8/header.rs:106` writes
  `self.encoder.write_flag(false)` for `segment_feature_mode` (delta), then
  emits each segment's `quantizer_level` (computed as
  `seg_quant_index - base_quant_index`, `mod.rs:1402,1427`) via the standard
  optional-signed-7 path.
- **libwebp:** `src/enc/syntax_enc.c:196` writes
  `VP8PutBitUniform(bw, 1)` for `segment_feature_mode = 1` (absolute), then
  emits `enc->dqm[s].quant` (the absolute 0..127 quantizer index) via
  `VP8PutSignedBits(bw, ..., 7)`.
- **Difference:** both modes are spec-legal and reconstruct identical
  per-segment quantization tables. Only the byte cost differs slightly: in
  delta mode segment 0's delta is exactly 0 (saving ~8 bits); in absolute mode
  every segment's `quant` is non-zero so all 4 entries cost the full 9 bits
  each.
- **Why this might cost bytes:** marginal — at most a few tens of bits per
  frame, and zenwebp's delta mode is usually slightly *cheaper* when the base
  segment carries delta=0. Not a regression. Listed for completeness because
  the bit pattern at the segment header bytes will differ from libwebp's even
  when settings match exactly, which can confuse byte-comparison tests.
- **Suggested fix:** none required. If absolute-mode parity is desired for
  bit-exact comparison, switch the flag and the encoded values together.

### `loop_filter_adjustments` flag wired but `i4x4_lf_delta` always 0 — severity: low — likely size impact: 0%

- **zenwebp:** `src/encoder/vp8/mod.rs:526` initializes
  `loop_filter_adjustments: false` and the field is never set true; the gated
  `encode_loop_filter_adjustments` (`src/encoder/vp8/header.rs:146-150`) only
  emits a single `write_flag(false)` (no per-mode/per-ref deltas).
- **libwebp:** `src/enc/syntax_enc.c:217-232` computes
  `use_lf_delta = (hdr->i4x4_lf_delta != 0)` and emits the full delta block
  when set. `i4x4_lf_delta` is initialized to 0 in `webp_enc.c:56` and never
  touched elsewhere, so libwebp also always emits `use_lf_delta = 0`.
- **Difference:** none in practice — both encoders emit a single 0 bit. The
  zenwebp scaffolding for nonzero `i4x4_lf_delta` is dead code that wouldn't
  match libwebp's emission shape if exercised (the inner block writes a
  single flag, not the 4 ref bits + signed mode delta + 3 trailing zeros that
  libwebp emits).
- **Why this might cost bytes:** none today.
- **Suggested fix:** either delete `loop_filter_adjustments` /
  `encode_loop_filter_adjustments`, or finish porting the inner block to
  libwebp's exact emission (`PutBits(0, 4)` for ref deltas,
  `PutSignedBits(i4x4_lf_delta, 6)` for mode delta, `PutBits(0, 3)` for trailing
  unused). Until then, leave the flag forced to false.

### Mid-stream `compute_updated_probabilities` is a no-op for emission — severity: low — likely size impact: 0%, ~5% CPU waste

- **zenwebp:** `src/encoder/vp8/mod.rs:931-935` decrements
  `refresh_countdown` and calls `compute_updated_probabilities` every
  `max(mb_count/8, 96)` MBs. That function (`mod.rs:613-650`) overwrites
  `self.updated_probs`. But token emission only reads the FINAL value of
  `updated_probs` (`mod.rs:1136-1138`), and `level_costs` is never refreshed.
- **libwebp:** the corresponding mid-stream call
  (`src/enc/frame_enc.c:840-844`) does both `FinalizeTokenProbas(proba)` AND
  `VP8CalculateLevelCosts(proba)`, so trellis and mode selection see the
  refreshed costs immediately.
- **Difference:** zenwebp's mid-stream refresh has no effect on the output.
- **Why this might cost bytes:** doesn't directly cost bytes, but means the
  refresh logic is misleading future readers and burning CPU.
- **Suggested fix:** either delete the mid-stream call, or actually rebuild
  `level_costs` and let trellis/mode selection benefit. See the multi-pass
  divergence above — this is the same root cause.

## Things that LOOK different but are actually equivalent

- **`write_optional_signed_value` bit ordering vs `VP8PutSignedBits`.** zenwebp
  emits `flag, magnitude_bits (MSB-first via write_literal), sign_flag`;
  libwebp emits `flag, (mag<<1)|sign_lsb` as `nb_bits+1` bits MSB-first via
  `VP8PutBitUniform`. Both produce identical bit sequences because the
  magnitude bits land in the same positions and the sign bit is the last bit
  in both encoders. The 2026-03-05 sign-fix corrected a real inversion;
  current code is correct.
- **`encode_segment_updates` always emits `update_data = 1`.** Same as libwebp
  which hard-codes `const int update_data = 1` (`syntax_enc.c:191`).
- **Segment ID prob `255 = no update`.** Both encoders treat 255 as the "skip
  this entry" sentinel (zenwebp `header.rs:137`, libwebp `tree_enc.c:206`).
- **`record_coeff_tokens` v==1 / v>=2 ctx update.** zenwebp `residuals.rs:297`
  computes `ctx = if v <= 1 { 1 } else { 2 }` post non-zero handling, which
  matches libwebp's three-way switch on AddToken returns
  (`token_enc.c:134/139/187`).
- **`base_id = TOKEN_ID(coeff_type, n, ctx)` at first iteration.** zenwebp
  uses `VP8_ENC_BANDS[n]` for the band lookup (`residuals.rs:171`); libwebp
  uses raw `n` with the comment "should be stats[VP8EncBands[n]], but it's
  equivalent for n=0 or 1" (`token_enc.c:122-124`). Both produce the same
  result because `VP8_ENC_BANDS[0]=0` and `VP8_ENC_BANDS[1]=1`.
- **`KEYFRAME_BPRED_MODE_PROBS` table layout vs libwebp's `kBModesProba`.**
  zenwebp's i4 mode integers differ from libwebp's
  (`B_LD_PRED=4` vs libwebp's `=6`, `B_RD_PRED=5` vs `=4`, `B_VR_PRED=6` vs
  `=5`), and `KEYFRAME_BPRED_MODE_PROBS` rows 4-6 are reordered to match.
  Spot-checked: zenwebp row 4 (B_LD context) matches libwebp row 6 (B_LD
  context); row 5 (B_RD) matches libwebp row 4; row 6 (B_VR) matches libwebp
  row 5. The wire format is identical because the i4 mode tree
  (`KEYFRAME_BPRED_MODE_TREE`) was also reordered to keep the same path
  structure with the new mode integers.
- **`PutCoeffs` return = `top_nz/left_nz`.** zenwebp's
  `record_coeff_tokens` returns `bool` (any nonzero), encoder converts to
  0/1 for the next ctx — same as libwebp's PutCoeffs returning 0 or 1.
- **`write_macroblock_header` order: segment_id → skip → ymode → bmodes →
  uvmode.** Matches libwebp `VP8CodeIntraModes` (`tree_enc.c:283-313`).
- **`KEYFRAME_YMODE_TREE` / `KEYFRAME_UV_MODE_TREE` and probs (145, 156, 163, 128, 142, 114, 183).**
  Match libwebp's hard-coded constants in `PutI16Mode`/`PutUVMode`
  (`tree_enc.c:262-276`).
- **Lookup-table normalization (`K_NORM` / `K_NEW_RANGE`) and run-length carry
  in `ArithmeticEncoder::flush`.** Byte-identical port of libwebp's
  `bit_writer_utils.c` `kNorm`, `kNewRange`, and `Flush`.
- **`flush_and_get_buffer` pads with `9 - nb_bits` zero bits.** Matches
  `VP8BitWriterFinish` (`bit_writer_utils.c:174-178`).
- **`VP8_LEVEL_CODES`, `VP8_ENC_BANDS`, `VP8_ZIGZAG`, `VP8_DC_TABLE`,
  `VP8_AC_TABLE`, `VP8_AC_TABLE2`.** All byte-identical to libwebp's
  `VP8LevelCodes`, `VP8EncBands`, `kZigzag`, `kDcTable`, `kAcTable`,
  `kAcTable2`.
- **`PROB_DCT_CAT` (Cat3-Cat6 extra-bit probs) and the constants 159, 165,
  145, 128.** Match libwebp's `VP8Cat3..6` and the inline constants in
  `VP8RecordCoeffTokens`.

## Open questions

- The CLAUDE.md notes a 2026-03-28 measurement "Production settings (SNS=50,
  filter=60): CID22 Q75: **1.0149x** | Q90: **1.0060x**." The Q75-vs-Q90 split
  suggests the gap shrinks as quantizers get smaller. That fits the
  skip-flag-overhead hypothesis (low Q → more zeros → more skip MBs → libwebp
  enables `use_skip_proba`, narrowing the gap; high Q at the same filter →
  fewer zero MBs → libwebp turns skip OFF, widening libwebp's lead). Worth
  measuring whether *just* applying the `SKIP_PROBA_THRESHOLD` gate closes the
  CID22 Q75 gap, before tackling multi-pass.
- libwebp's `VP8WriteProbas` uses `VP8CoeffsProba0` as the "old" baseline for
  the update gate; zenwebp's `compute_updated_probabilities`
  (`src/encoder/vp8/mod.rs:613-650`) and `encode_updated_token_probabilities`
  (`src/encoder/vp8/header.rs:173-211`) both use COEFF_PROBS (the same
  defaults). For single-pass this is correct and matches libwebp. If
  multi-pass is wired up, the FIRST pass should use COEFF_PROBS as the
  baseline, but subsequent passes should still compare against COEFF_PROBS in
  the *header* (since the decoder always starts from defaults). zenwebp's
  current code already does the right thing in the header path; verify the
  multi-pass `token_probs` propagation path (`mod.rs:878-883`) doesn't drift.
- The `update_proba` table at `header.rs:177` (`COEFF_UPDATE_PROBS`) wasn't
  spot-checked against libwebp's `VP8CoeffsUpdateProba`. Worth a 30-second
  diff to confirm.
- `record_coeff_tokens` at `residuals.rs:215-235` uses constant probs `159,
  165, 145` for the 5..10 magnitude tail. Matches libwebp's inline constants.
  But the v∈{5,6} arm only emits `add_constant_token(v == 6, 159)` (no
  AddToken for the band/ctx node 7); libwebp does
  `AddToken(v == 6, ..., s + 7)` followed by no AddConstantToken. Re-check:
  zenwebp lines 226-235:
  ```
  if v <= 6 { add_constant_token(v == 6, 159) }
  else { add_constant_token(v >= 9, 165); add_constant_token((v & 1) == 0, 145) }
  ```
  vs libwebp `token_enc.c:147-152`:
  ```
  if (!AddToken(v > 6, base_id+7, s+7)) AddConstantToken(v == 6, 159);
  else { AddConstantToken(v >= 9, 165); AddConstantToken(!(v & 1), 145); }
  ```
  zenwebp emits `add_token(v > 6, base_id + 7)` ABOVE on line 227 (`s[7]`),
  then drops into the `if v <= 6 ... else ...` branch. So zenwebp emits the
  v>6 token via `add_token`, then the constant 159/165/145 — same sequence as
  libwebp. **Equivalent, no bug.** Including this here only because the
  refactor moved the AddToken outside the if/else and it's worth verifying
  during code review.
