# Analysis / SNS / Segments Differences vs libwebp

## Summary

Audit of zenwebp's macroblock analysis, SNS modulation, segment assignment, per-segment quantizer setup, and filter-level derivation. Eight divergences from libwebp identified. The single biggest one is that zenwebp computes `Y2 AC quant` as `kAcTable[q] * 155/100` (capped at 8) instead of using libwebp's dedicated `kAcTable2[q]` lookup — this systematically over-quantizes the WHT pass and changes every encoded I16 block. Two other high-impact divergences: (a) zenwebp always smooths the segment map when `num_segments > 1`, while libwebp only smooths when `config->preprocessing & 1` (off by default); (b) zenwebp never calls libwebp's `VP8AdjustFilterStrength` post-encode pass, so per-segment filter level is never bumped up based on observed `max_edge` of reconstructed coefficients. Several medium-impact differences in alpha-range handling, base_quant choice, and gating thresholds. Some pre-existing perceptual extensions (SATD masking blend, enhanced CSF, JND) are zenwebp-specific knobs that are correctly gated by `method >= 4/5` but still diverge from libwebp at production settings (m4 default).

## Divergences

### Y2 AC quantizer uses kAcTable*155/100 instead of kAcTable2 — severity: high — likely size impact: 0.5–1.5%
- **zenwebp:** `src/encoder/vp8/mod.rs:1424` and `src/encoder/vp8/mod.rs:1555` — `y2ac: ((i32::from(AC_QUANT[seg_quant_usize]) * 155 / 100) as i16).max(8)`
- **libwebp:** `src/enc/quant_enc.c:236` — `m->y2.q[1] = kAcTable2[clip(q + enc->dq_y2_ac, 0, 127)];`
- **Difference:** zenwebp synthesises Y2 AC quantization steps by scaling the regular `kAcTable` by 1.55 (with a floor of 8). libwebp uses a completely separate hard-coded table `kAcTable2` (zenwebp has it as `VP8_AC_TABLE2` in `src/encoder/tables.rs:283` but it is **never read**). Spot-checks: `kAcTable[64]*155/100 = 133`, `kAcTable2[64] = 120` (≈10% larger); `kAcTable[80]*155/100 ≈ 158`, `kAcTable2[80] = 145` (≈9% larger); `kAcTable[40]*155/100 ≈ 65`, `kAcTable2[40] = 68` (4% smaller). Direction of error varies with q.
- **Why this might cost bytes:** The Y2 (WHT) block carries the 16 luma DC coefficients of every I16 macroblock. A coarser Y2 AC quant (zenwebp at mid-q) shifts more DC residual energy out of Y2 into Y1 DC (which is unused, since Y2 holds it) — but more importantly produces visibly different reconstructions and the encoder's RD model is built around the wrong table. Mode selection sees the wrong score and trellis sees the wrong cost. At Q75 this is exactly the regime where the discrepancy is largest.
- **Suggested fix:** Use `VP8_AC_TABLE2[clip(seg_quant + dq_y2_ac, 0, 127)]` for `y2ac`. Also ensure the decoder side uses the `kAcTable2` derivation (it does, via the spec-defined Y2 AC formula `2 * VP8AcTable[q]` — wait, that's still wrong). Actually inspecting: kAcTable2 ≠ 2*kAcTable. The relationship is documented in libwebp's `dec/quant_dec.c` via the dec-side `dqm_factor` for Y2. **Verify decoder Y2 AC dequant matches encoder before changing.**

### Segment-map smoothing always-on instead of preprocessing-gated — severity: high — likely size impact: 0.3–0.8%
- **zenwebp:** `src/encoder/vp8/mod.rs:1336-1342` — `if self.num_segments > 1 { super::cost::smooth_segment_map(...) }` (unconditional when multi-segment)
- **libwebp:** `src/enc/analysis_enc.c:217-218` — `const int smooth = (enc->config->preprocessing & 1); if (smooth) SmoothSegmentMap(enc);`
- **Difference:** libwebp's default `config->preprocessing = 0` (`src/enc/config_enc.c:48`), so SmoothSegmentMap is **off** unless the user opts in. zenwebp always smooths for multi-segment frames.
- **Why this might cost bytes:** The 3×3 majority filter forces isolated MBs into the surrounding segment. On natural images this typically merges small textured patches into a coarser segment that uses a higher quantizer than what the alpha histogram chose, increasing distortion in those patches. It also reduces the effective number of segments after `simplify_segments`, which can collapse a 4-segment encoding into 2 segments and lose the bit-savings from differential quantization.
- **Suggested fix:** Gate smoothing on a config flag (mirroring libwebp's `preprocessing & 1`). Default OFF. Compare CID22 Q75 before/after.

### `VP8AdjustFilterStrength` not implemented — severity: medium — likely size impact: 0.0–0.3% (file-size neutral; affects quality at fixed q)
- **zenwebp:** no equivalent — `max_edge` is never tracked per segment; filter strength is set once during analysis (`src/encoder/vp8/mod.rs:1404-1411`) and never refined.
- **libwebp:** `src/enc/filter_enc.c:198-237` — `VP8AdjustFilterStrength` is called from `PostLoopFinalize` (`src/enc/frame_enc.c:730`). For each segment it computes `delta = (dqm->max_edge * dqm->y2.q[1]) >> 3`, derives a strength via `VP8FilterStrengthFromDelta`, and **bumps `dqm->fstrength` upward** if the edge-derived strength is larger. `dqm->max_edge` is updated during `VP8Decimate` from the maximum encoded coefficient magnitude.
- **Difference:** zenwebp's per-segment loop filter is purely a function of (quant, sharpness, filter_strength, beta). libwebp adds an "edge-aware" upgrade based on observed coefficient peaks.
- **Why this might cost bytes:** Higher filter level reduces blocking artifacts in segments with strong edges, which can let mode selection get away with cheaper modes. Direct effect on file size is small (only the 4 per-segment filter delta values are written), but indirect through next-pass token cost in multi-pass. Real cost at single-pass is mostly quality, not size.
- **Suggested fix:** Track `max_edge_per_segment` during the encode loop (max of `|coeff|` across Y1 AC of MBs in that segment), then before writing the frame header re-compute per-segment `fstrength = max(orig_fstrength, FilterStrengthFromDelta(sharp, (max_edge*y2_q1)>>3))`. Wire into the per-segment filter delta.

### Base-quantizer for delta encoding differs from libwebp — severity: medium — likely size impact: 0.05–0.2%
- **zenwebp:** `src/encoder/vp8/mod.rs:1499` and `1402` — `base_quant_index = quality_to_quant_index(quality)` (the un-modulated quant), then per-segment delta = `seg_quant_index - base_quant_index`.
- **libwebp:** `src/enc/quant_enc.c:404` — `enc->base_quant = enc->dqm[0].quant;` (segment 0's quant, which is the SNS-modulated quant of the lowest-alpha segment).
- **Difference:** With libwebp, segment 0 always has `delta == 0` and writes a single bit (the "no delta" flag) for its quantizer entry. With zenwebp, segment 0's delta is `seg0_quant - quality_to_quant_index(quality)`, which is non-zero whenever SNS modulation > 0. zenwebp burns ~9 extra bits per non-zero delta (1 flag + 7-bit magnitude + 1 sign).
- **Why this might cost bytes:** Each unnecessary non-zero segment delta costs ~9 bits in the header. Across 4 segments × 2 (Y quant + LF) = up to ~72 bits per frame. Trivial in absolute terms (~10 bytes) but trivial to fix.
- **Suggested fix:** After computing per-segment quant_indices, set `self.quantization_indices.yac_abs = segments[0].quant_index` and recompute deltas as `seg.quant_index - segments[0].quant_index`. Same for filter-level deltas relative to `segments[0].fstrength` instead of `compute_filter_level(base_quant)`.

### Segment analysis gated off below 256 macroblocks — severity: medium — likely size impact: image-dependent, large for small images
- **zenwebp:** `src/encoder/vp8/mod.rs:1574` — `let use_segments = self.num_segments > 1 && total_mbs >= 256;` (256 MB ≈ 256×256 image)
- **libwebp:** `src/enc/analysis_enc.c:434-436` — `do_segments = enc->config->emulate_jpeg_size || (enc->segment_hdr.num_segments > 1) || (enc->method <= 1);` (no min-size threshold)
- **Difference:** zenwebp skips segmentation entirely for images smaller than 256×256, while libwebp segments any multi-segment-configured image. zenwebp also never runs analysis when `method <= 1` (libwebp does, to populate `preds[]`).
- **Why this might cost bytes:** For 128×128 or smaller images, zenwebp uses a single segment with the unmodulated base quant — losing the alpha-driven quant differentiation that helps mid-range images. Method 0/1 path additionally lacks any mode hints.
- **Suggested fix:** Drop the `>= 256` gate (or align with libwebp's "always when num_segments>1"). For method ≤ 1, port `FastMBAnalyze` (DC-variance-based intra16/intra4 decision in `analysis_enc.c:260`) so the path remains.

### Alpha-range floor of 64 changes per-segment alpha & beta — severity: medium — likely size impact: 0.1–0.4% on flat images
- **zenwebp:** `src/encoder/vp8/mod.rs:1324-1325` and `1388,1394` — `const MIN_ALPHA_RANGE: i32 = 64; let effective_range = range.max(MIN_ALPHA_RANGE);`
- **libwebp:** `src/enc/analysis_enc.c:92` — `if (max == min) max = min + 1;` (only handles exact equality, no floor)
- **Difference:** zenwebp clamps the alpha range to ≥64 to "prevent extreme deltas on uniform images". libwebp accepts any positive range, only nudging the divisor when min == max.
- **Why this might cost bytes:** On gradients/skies where center alphas are tightly clustered (range ~10), libwebp produces large normalized alphas (close to ±127 → larger SNS modulation → bigger per-segment q spread → bigger compression savings on the larger segments). zenwebp dampens the spread by 6.4× (64/10), losing most of the SNS benefit on flat content. The same dampening is also applied to `beta`, which then affects `compute_filter_level_with_beta`.
- **Suggested fix:** Remove the floor or replace it with a less aggressive scheme (e.g., a smaller constant like 8, or only apply when `range == 0`). Measure on a flat-content corpus.

### Multi-pass `StatLoop` not implemented — severity: medium — likely size impact: 0.2–0.6% for `config->pass > 1`, 0% at default
- **zenwebp:** `src/encoder/vp8/mod.rs:845` — `let num_passes = 1;` (hard-coded comment: "Multi-pass without quality search provides no benefit (tested), so we use 1 pass for all methods.")
- **libwebp:** `src/enc/frame_enc.c:626-684` — `StatLoop` runs `enc->config->pass` passes (default 1, but increases to 6 when target_size search is on). Each pass calls `OneStatPass` which does a full encode-and-record without writing the bitstream, then `FinalizeTokenProbas` updates `enc->proba` and `VP8CalculateLevelCosts` refreshes the cost tables before the next pass / before the final emit pass.
- **Difference:** libwebp does at least one stats-only pass before the final emit pass. zenwebp does mid-stream probability refresh during the single pass but never refreshes `level_costs` (per the comment in mod.rs), so RD scoring uses default-prob costs throughout instead of image-tuned ones.
- **Why this might cost bytes:** libwebp's default `config->pass = 1` already runs StatLoop for one pass before the emit loop, giving the emit pass image-tuned token probabilities and level costs. zenwebp's single combined pass has the cost tables drift behind the actual probabilities. The mid-stream refresh helps but does not refresh level_costs (per the inline note).
- **Suggested fix:** Add a stats-only `OneStatPass` pre-pass for `method >= 3` (matching libwebp's `rd_opt = (method >= 3 || do_search) ? RD_OPT_BASIC : RD_OPT_NONE`). After the stats pass, call `compute_updated_probabilities` and recalculate level_costs before the real emit pass.

### Perceptual masking alpha blend changes alpha histogram at method ≥ 4 — severity: medium — likely size impact: ±0.2–0.5% (measured: production CID22 Q75 is 1.49% above libwebp at sns=50/filter=60, m4)
- **zenwebp:** `src/encoder/analysis/mod.rs:228-233` and `src/encoder/psy.rs:527-548` — when `method >= 4 && sns_strength > 0`, `blend_masking_alpha` adds `(masking_alpha - 128) * 64/256` (or `*96/256` at method≥5) as an additive delta to the DCT-derived alpha.
- **libwebp:** `src/enc/analysis_enc.c:317-342` — pure DCT-histogram alpha, no masking blend at any method level.
- **Difference:** zenwebp's per-MB alpha at default `method = 4` is shifted by up to ±32 by a SATD/luminance/edge masking model, before being inserted into the alpha histogram and going into k-means. libwebp uses the unmodified DCT-histogram alpha.
- **Why this might cost bytes:** The blend changes which MBs land in which segment. At default settings this is part of the m4 baseline, so the 1.49% overhead vs libwebp at production settings reflects (among other things) this divergence. Note CLAUDE.md states m4 production was tuned with this enabled and SNS=50; turning it off would re-baseline against libwebp m4. Could be either direction (zenwebp may be smaller on some images).
- **Suggested fix:** Provide a config flag to disable the blend (`compute_masking_alpha`/`blend_masking_alpha`) so libwebp parity can be tested cleanly. If the blend is genuinely improving butteraugli/SSIMULACRA but increasing size, that is a quality–size tradeoff to surface to the user, not a bug.

### Enhanced CSF tables (PSY_WEIGHT_Y/UV) at method ≥ 3 diverge from libwebp's kWeightY — severity: low — likely size impact: 0.05–0.2% on m3+
- **zenwebp:** `src/encoder/psy.rs:35-52,148-154` — at `method >= 3`, replaces `VP8_WEIGHT_Y` (which matches libwebp's `kWeightY`) with `PSY_WEIGHT_Y` (steeper HF rolloff), and uses `PSY_WEIGHT_UV` for chroma TDisto (libwebp uses kWeightY for both).
- **libwebp:** `src/enc/quant_enc.c:495` — `kWeightY[16] = {38, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2}` is used for both luma and chroma TDisto (`quant_enc.c:1082,1170`).
- **Difference:** zenwebp's CSF tables emphasise low-frequency error more than libwebp's, especially on chroma. Affects TDisto weight in mode selection.
- **Why this might cost bytes:** Steeper rolloff → mode selection prefers modes that put error in HF bins (which are easier to code) → smaller files in some cases, larger in others.
- **Suggested fix:** Document explicitly that this is intentional. Provide a `--strict-libwebp-parity` mode that uses kWeightY for both, for parity testing.

### JND coefficient zeroing at method ≥ 5 — severity: low — likely size impact: 0.1–0.3% on m5/m6
- **zenwebp:** `src/encoder/psy.rs:167-188` — at `method >= 5`, sets `jnd_threshold_y/uv` and `psy_trellis_strength` based on quant. Coefficients below the JND threshold get zeroed during trellis (per `is_below_jnd_y/uv`).
- **libwebp:** no equivalent. Trellis only zeros coefficients based on rate-distortion score.
- **Difference:** zenwebp can be more aggressive at zeroing perceptually small coefficients at m5/m6.
- **Why this might cost bytes:** Reduces non-zero coeff count, which directly reduces token stream bits. Could reduce file size at m5/m6.
- **Suggested fix:** Already gated to m5+, so the m4 production parity case is unaffected. Document.

### `init_matrices` uses approximate average q for lambda derivation — severity: low — likely size impact: <0.05%
- **zenwebp:** `src/common/types.rs:856-858` — `q_i4 = (self.ydc + 15 * self.yac + 8) >> 4;` (assumes Y1 matrix has q[0]=ydc, q[1..]=yac, average is `(q_dc + 15*q_ac + 8) >> 4`)
- **libwebp:** `src/enc/quant_enc.c:191-218` — `ExpandMatrix` returns `(sum + 8) >> 4` over the actual `m->q[0..16]` array, where `q[0]` is `kDcTable[q]` and `q[1..16]` are filled with `kAcTable[q]`.
- **Difference:** Algebraically the same when DC/AC are pure table lookups. **But** zenwebp's `Segment` stores `ydc` from `DC_QUANT[seg_quant_usize]` and `yac` from `AC_QUANT[seg_quant_usize]` directly — no `dq_y1_dc` offset applied. libwebp's m->y1.q[0] = `kDcTable[clip(q + dq_y1_dc, 0, 127)]`. Both encoders have `dq_y1_dc = 0` at default, so this is currently equivalent, but zenwebp would silently drop any non-zero `dq_y1_dc` (currently dead-code anyway since the field doesn't exist).
- **Why this might cost bytes:** Zero impact at default settings.
- **Suggested fix:** Cosmetic — when (if) Y1 DC offset is added, ensure both the matrix and the average-q computation pick it up.

## Things that LOOK different but are actually equivalent

### Per-MB boundary handling
- libwebp uses `tmp_32` per MB call (allocated on stack, written from source). zenwebp uses a global `y_top` row buffer. Both functionally store source pixels for the row above. The local-vs-global storage is irrelevant since each MB only reads `[x*16..x*16+16]`.

### `final_alpha_value` (invert+clip)
- `final_alpha_value` in `src/encoder/analysis/mod.rs:202-205` matches `FinalAlphaValue` in `analysis_enc.c:111-114` exactly (post the 2026-03-05 sign-bug fix).

### Histogram bin computation
- `(coeff.unsigned_abs() >> 3).min(MAX_COEFF_THRESH)` matches `clip_max(abs(out[k]) >> 3, MAX_COEFF_THRESH)` exactly. `MAX_COEFF_THRESH = 31` matches.

### k-means init/iteration
- Initial center spread `(1 + 2k) * range / (2 * nb)`, near-center walk `while abs(a - centers[n+1]) < abs(a - centers[n])`, displaced threshold `< 5`, weighted-average computation — all match `analysis_enc.c:135-222`.

### `SimplifySegments` predicate
- zenwebp checks `quant_index && loopfilter_level (delta)` ; libwebp checks `quant && fstrength (absolute)`. Equivalent because both deltas/absolutes share the same base.

### UV alpha → DQ_UV mapping
- `(uv_alpha - 64) * (6 - (-4)) / (100 - 30)` then `* sns/100`, clamped — matches `quant_enc.c:414-419` exactly. `MID_ALPHA=64`, `MIN_ALPHA=30`, `MAX_ALPHA=100`, `MAX_DQ_UV=6`, `MIN_DQ_UV=-4` constants verified.

### Lambda formulas
- `lambda_i4 = (3*q²) >> 7`, `lambda_i16 = 3*q²`, `lambda_uv = (3*q²) >> 6`, `lambda_mode = q² >> 7`, `lambda_trellis_i4 = (7*q²) >> 3`, `lambda_trellis_i16 = q² >> 2`, `lambda_trellis_uv = q² << 1`, `tlambda = (sns_strength * q) >> 5` — all match `quant_enc.c:245-253`.

### Bias matrices and zthresh
- `kBiasMatrices[type][is_ac]` constants `(96,110)`, `(96,108)`, `(110,115)` and `bias = (bias << QFIX) >> 7` (i.e. `BIAS()` macro), `zthresh = ((1 << QFIX) - 1 - bias) / iq` — all match `quant_enc.c:176-208`.

### Sharpening (kFreqSharpening)
- `VP8_FREQ_SHARPENING` table and `(freq_sharpen * q) >> SHARPEN_BITS=11`, applied only to Y1 — matches `quant_enc.c:184,210-214`.

### `compute_filter_level_with_beta`
- `level0 = 5*filter_strength`, `qstep = kAcTable[q] >> 2`, `f = base_strength * level0 / (256 + beta)`, `FSTRENGTH_CUTOFF = 2`, clamp to 63 — matches `SetupFilterStrength` in `quant_enc.c:278-296`.

### Final segment-tree probability computation
- `GetProba(a, b) = (255*a + (a+b)/2) / (a+b)` and tree probas indexing — matches `SetSegmentProbas` in `frame_enc.c:202-235`.

## Open questions

1. **Verify the encoder/decoder Y2 AC handshake before changing `y2ac`.** The decoder's Y2 AC dequant table must match the encoder's quantizer to avoid a mismatch. libwebp encoder uses `kAcTable2`; libwebp decoder for Y2 uses `kAcTable2` as well (via `dec_dquant`). Zenwebp's decoder uses `2 * AC_QUANT[q]` somewhere — check `src/decoder/vp8/quant.rs` (or v2 equivalent) before touching the encoder. If the decoder is also wrong but in a self-consistent way with the encoder, the encoder's "wrong" table actually produces the "right" reconstruction, and the apparent bug is benign at the bitstream level (but still produces non-libwebp-compatible files).

2. **Effect of removing the `MIN_ALPHA_RANGE = 64` floor on natural images.** The floor was added presumably because tiny ranges produce extreme deltas that segment 0 quantizes hard. Re-running CID22 without it would tell us whether the dampening is helping or hurting compression on real content. May need a different heuristic (e.g. minimum spread between segments, not minimum range).

3. **Should zenwebp port `FastMBAnalyze` for method ≤ 1?** The current path runs the full DCT-based analysis even at m0/m1, which is correct algorithmically but expensive. libwebp's FastMBAnalyze uses DC variance to decide intra4 vs intra16, which is much cheaper. This is a speed (not size) question.

4. **Is the 1.49% CID22 Q75 production gap dominated by the Y2 AC table bug or the smoothing-always-on?** Need a controlled A/B with each fix in isolation.

5. **`compute_segment_quant` recomputes `c_base` per segment via `quality_to_compression`.** Trivial perf nit (4× redundant work). libwebp computes `c_base` once outside the loop. Move it out for cleanliness.

6. **`fast_math::pow` 1% tolerance.** Tests show ~1% relative error; that translates into ±1 quantizer index occasionally. Probably acceptable, but if Y2 AC is fixed and we want bit-identical-to-libwebp output, we'd need higher-precision pow (or libm).
