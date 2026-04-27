# Mode Selection Differences vs libwebp

## Summary

Catalog of 14 algorithmic divergences between zenwebp and libwebp (commit `b8dad2f`,
libwebp v1.x in `/home/lilith/work/third-party/libwebp`). Severity breakdown:

- **High (>=0.5% bytes likely):** 4 — IntraMode enum renumbering vs un-remapped
  `VP8_FIXED_COSTS_I4` table; `FIXED_COSTS_I16` / `FIXED_COSTS_UV` indexed by
  zenwebp `MODES` array order rather than VP8 mode IDs (V/H costs swapped for I16,
  all UV costs except DC misaligned); `i4_penalty` 14x larger than libwebp at
  partition_limit=0 (3000\*lambda_mode vs 211\*lambda_mode); `get_cost_uv` uses
  `ctx=0` for every UV block (no top/left non-zero tracking at all).
- **Medium (0.1-0.5%):** 5 — `get_cost_luma16` initializes top_nz/left_nz to all
  zero each call (no cross-MB context import) and uses `ctx=0` for the Y2 DC
  block (libwebp uses `top_nz[8]+left_nz[8]`); zenwebp skips I16 V/H/TM modes at
  MB borders rather than using border values 127/129; `should_try_i4` skip-gate
  has no equivalent in libwebp PickBestIntra4 (libwebp always tries I4 when
  method >= 2); zenwebp uses a different (`PSY_WEIGHT_Y`) CSF table for tdisto
  at method >= 3; method 2 evaluates I4 with full-RD pipeline rather than
  libwebp's distortion-only `RefineUsingDistortion` path.
- **Low / unknown:** 5 — at method 6 zenwebp does not trellis-quantize Y2 DC or
  Y1 AC during I16 mode selection (libwebp does); zenwebp at methods 0-2 only
  evaluates 3 sorted I4 modes (method 2 only — methods 0-1 skip I4 entirely);
  flatness penalty applied differently for I4 (zenwebp folds into the early-exit
  lower bound, libwebp puts it into `R` before first SetRDScore — same end
  result but slightly different pruning); UV flatness penalty checks all 8
  blocks, libwebp checks only block 0 (`rd_uv.uv_levels[0]`); I16 final-score
  recomputes `D*RD_DISTO_MULT` after using a doubled-flat distortion in the
  per-mode score (zenwebp keeps `d_final` in `best_sse` and uses it again with
  `lambda_mode`; libwebp keeps the already-doubled `D` in the score and just
  recomputes via SetRDScore — equivalent but worth flagging).

## Divergences

### 1. `VP8_FIXED_COSTS_I4` table indexed by zenwebp's renumbered B_*_PRED IDs but never remapped — severity: high — likely size impact: noticeable mode-cost noise on every I4 sub-block

- **zenwebp:** `src/encoder/tables.rs:399-500` — table is byte-for-byte identical
  to libwebp's `VP8FixedCostsI4` (no permutation applied) yet
  `src/common/types.rs:70-79` redefines the constants as
  `B_LD_PRED=4, B_RD_PRED=5, B_VR_PRED=6, B_VL_PRED=7` (libwebp uses
  `B_RD=4, B_VR=5, B_LD=6, B_VL=7`). Lookup at
  `src/encoder/vp8/mode_selection.rs:1172-1174`:
  `mode_costs[mode_idx] = VP8_FIXED_COSTS_I4[top_ctx][left_ctx][mode_idx]` where
  `mode_idx` is the position in the zenwebp `MODES` array (which is
  `[DC, TM, VE, HE, LD, RD, VR, VL, HD, HU]` — the zenwebp enum order).
  `top_ctx`/`left_ctx` come from previously-chosen `IntraMode` values, also in
  zenwebp's enum.
- **libwebp:** `src/enc/cost_enc.c:106-206` — same numeric table; indexed at
  `quant_enc.c:1125` (`return VP8FixedCostsI4[top][left]`) where `top`/`left`
  are libwebp `B_*_PRED` IDs and the inner `mode_costs[mode]` index is also
  libwebp's `B_*_PRED` mode ID.
- **Difference:** Note that the **probability** table in zenwebp
  (`KEYFRAME_BPRED_MODE_PROBS` at `src/common/types.rs:226-237`) WAS deliberately
  remapped to zenwebp's enum order (verified by spot-checking: zenwebp
  `[0][4]` = libwebp `[0][6] (LD)`). The cost table was not. Net effect: a sub-block
  in zenwebp choosing LD (zenwebp ID 4) gets charged whatever cost libwebp had
  for RD at the same numerical context. Most cells differ by tens to hundreds
  of cost units (e.g., libwebp `[0][0][4]=2103` for RD vs `[0][0][6]=1628` for
  LD). With per-MB scoring at `lambda_i4 = (3*q²)>>7 ≈ 21 at q≈30`, a cost
  delta of ~500 units shifts an RD score by ~10000, comparable to the SSE of
  modest 4x4 reconstruction error. Affects mode picks in nontrivial fraction
  of I4 blocks.
- **Why this might cost bytes:** Mode selection in I4 controls 16 sub-blocks
  per MB plus sub-block mode header bits. Wrong cost weighting picks
  suboptimal modes whose actual coded size is larger. This is one of the
  costliest single bugs.
- **Suggested fix:** Either un-remap the prob table back to libwebp ordering
  AND keep the cost table as-is, or remap the cost table the same way the prob
  table is remapped. Decide based on which simplifies the rest of the codebase.

### 2. `FIXED_COSTS_I16` indexed by `MODES = [DC, V, H, TM]` instead of [DC, TM, V, H] — severity: medium — likely size impact: small per-MB but every MB

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:649` — `const MODES: [LumaMode; 4] = [LumaMode::DC, LumaMode::V, LumaMode::H, LumaMode::TM]`
  iterated by index, then `src/encoder/vp8/mode_selection.rs:838`
  `let mode_cost = FIXED_COSTS_I16[mode_idx]`. Cost table at
  `src/encoder/tables.rs:386` is `[663, 919, 872, 919]` copied verbatim from
  libwebp and commented "(DC, V, H, TM)" but the libwebp values are actually
  ordered (DC, TM, V, H).
- **libwebp:** `src/enc/cost_enc.c:105` — `VP8FixedCostsI16[4] = {663, 919, 872, 919}`,
  indexed at `quant_enc.c:1083` by `mode` which iterates in libwebp's enum
  order: `DC=0 (663), TM=1 (919), V=2 (872), H=3 (919)`.
- **Difference:** zenwebp's V mode pays cost 919 (libwebp says 872), zenwebp's
  H mode pays cost 872 (libwebp says 919). DC and TM happen to be 663/919 in
  both. Net: V/H mode-cost values are swapped vs the bitstream's actual cost.
- **Why this might cost bytes:** The wrong costs may flip mode picks between V
  and H on a few percent of MBs. Real cost delta is small (47 cost units at
  `lambda_i16 = 3*q²`), but persistent across the frame.
- **Suggested fix:** Reorder either `MODES` to `[DC, TM, V, H]` or
  `FIXED_COSTS_I16` to `[663, 872, 919, 919]`.

### 3. `FIXED_COSTS_UV` indexed by `MODES = [DC, V, H, TM]` instead of [DC, TM, V, H] — severity: high — likely size impact: significant

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:1476-1477` —
  `const MODES: [ChromaMode; 4] = [ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM]`,
  cost lookup at `src/encoder/vp8/mode_selection.rs:1691`
  `let mode_cost = FIXED_COSTS_UV[mode_idx]`. Cost table at
  `src/encoder/tables.rs:390` is `[302, 984, 439, 642]`.
- **libwebp:** `src/enc/cost_enc.c:103` — `VP8FixedCostsUV[4] = {302, 984, 439, 642}`,
  indexed at `quant_enc.c:1247` by `mode` in libwebp's `DC=0, TM=1, V=2, H=3`
  order.
- **Difference:** zenwebp gives V mode cost 984 (libwebp: 439), H mode cost
  439 (libwebp: 642), TM mode cost 642 (libwebp: 984). All non-DC UV modes get
  wrong costs. With `lambda_uv = (3*q²)>>6` (substantial), a cost delta of
  ~545 units (e.g., V) shifts the score significantly and biases away from V
  picks toward H.
- **Why this might cost bytes:** Chroma mode selection uses these costs in
  every MB. Picking the wrong UV prediction mode means worse predictions →
  more residual coding → more bytes.
- **Suggested fix:** Same as I16 — align the table order with the MODES array.

### 4. I4 `i4_penalty` ~14x higher than libwebp, biases away from I4 — severity: high — likely size impact: large at images where I4 wins

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:1101-1104`:
  ```
  let base_penalty = 3000u64 * u64::from(lambda_mode);
  let i4_penalty = base_penalty + base_penalty * limit_scale * limit_scale / 400;
  ```
  At default `partition_limit=0`: `i4_penalty = 3000 * lambda_mode`.
- **libwebp:** `src/enc/quant_enc.c:1144-1145`:
  ```
  rd_best.H = 211;  // '211' is the value of VP8BitCost(0, 145)
  SetRDScore(dqm->lambda_mode, &rd_best);
  ```
  Initial running score = `211 * lambda_mode` (just the bit-cost of the
  is_intra4 flag).
- **Difference:** Note the source-comment in zenwebp references libwebp's
  `RefineUsingDistortion` (`i4_penalty = 1000 * q² * lambda_d_i4=11`), but that
  is the method-0/1 path; for `PickBestIntra4` (method >= 2) libwebp uses just
  `H = 211`. zenwebp's choice penalizes I4 by ~14x more bits than libwebp at
  the start of the I4 evaluation, so the early-exit `running_score >= i16_score`
  fires far more often even when I4 would actually win.
- **Why this might cost bytes:** I4 mode is critical for textured/edge regions
  and complex content. Rejecting I4 too aggressively forces I16 with worse
  prediction → more residual bytes per MB. Likely 0.5%-2% of file size on
  textured photographs at default settings.
- **Suggested fix:** Set `running_score = 211 * lambda_mode` (or `211 * lambda_mode + 0`
  to mirror libwebp's `H=211, R=D=SD=0` initial state).

### 5. `get_cost_uv` uses ctx=0 for all 8 UV blocks (no top/left non-zero tracking) — severity: high — likely size impact: nontrivial

- **zenwebp:** `src/encoder/residual_cost.rs:825-836`:
  ```
  for block in uv_levels.iter() {
      let res = Residual::new(block, 2, 0);
      total_cost += get_residual_cost(0, &res, costs, probs);
  }
  ```
  Initial context is `0` for every U and V block in the macroblock.
- **libwebp:** `src/enc/cost_enc.c:263-283` — iterates `ch` over `{0, 2}`
  (U then V), then 2x2 grid, computing
  `ctx = it->top_nz[4 + ch + x] + it->left_nz[4 + ch + y]` per block, and
  updates `top_nz`/`left_nz` after each block based on whether `res.last >= 0`.
  Cross-MB context comes from `VP8IteratorNzToBytes(it)` at the start.
- **Difference:** zenwebp drops ALL UV non-zero context. The first
  coefficient's probability prefix (which dominates the EOB cost when the
  block is empty) is computed against `ctx0=0` rather than the true context.
  This systematically misestimates the cost of UV blocks by a substantial
  amount.
- **Why this might cost bytes:** The cost feeds the UV mode RD score. With
  every mode equally mis-priced, the wrong UV modes get picked. Also affects
  the mode that wins overall when costs would have differed.
- **Suggested fix:** Implement proper top_nz/left_nz tracking for the 4 UV
  blocks (matching libwebp's loop), passing in cross-MB context from the
  encoder's `top_complexity[mbx].uv` / `left_complexity.uv` if those exist.

### 6. `get_cost_luma16` does not import cross-MB nz context and uses ctx=0 for Y2 DC block — severity: medium — likely size impact: moderate

- **zenwebp:** `src/encoder/residual_cost.rs:792, 796-808`:
  ```
  total_cost += get_residual_cost(0, &dc_res, costs, probs);  // Y2 DC ctx=0
  let mut top_nz = [false; 4];
  let mut left_nz = [false; 4];
  // ... AC loop using ctx = (top_nz[x] as usize) + (left_nz[y] as usize)
  ```
  Top/left nz arrays start at all-false every call.
- **libwebp:** `src/enc/cost_enc.c:243-260` — `VP8IteratorNzToBytes(it)` is
  called first to import cross-MB nz context (see
  `iterator_enc.c:238-269`), then Y2 DC uses
  `ctx = it->top_nz[8] + it->left_nz[8]` (a special slot reserved for Y2 DC
  carry-over), and the AC loop reads/writes the live `it->top_nz`/`left_nz`
  with cross-MB carry.
- **Difference:** zenwebp loses cross-MB context entirely for I16 cost
  estimation. Inside the MB the local nz tracking is correct, but at the
  upper/left edges every block sees `ctx=0` even when the neighbor MB had
  non-zero coefficients.
- **Why this might cost bytes:** The `ctx` heavily influences the EOB
  probability and thus the cost of empty/sparse blocks. Misestimating it
  affects I16-vs-I4 RD comparison and the I16 mode pick.
- **Suggested fix:** Wire the encoder's `top_complexity[mbx].y` /
  `left_complexity.y` arrays into `get_cost_luma16` (similar to how I4 already
  does at `mode_selection.rs:1124-1135`), and add a `top_nz[8]`/`left_nz[8]`
  pair for Y2 DC.

### 7. zenwebp skips I16 V/H/TM modes at MB borders instead of using border values 127/129 — severity: medium — likely size impact: small but consistent

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:695-711`:
  ```
  if mode == LumaMode::V && mby == 0 { continue; }
  if mode == LumaMode::H && mbx == 0 { continue; }
  if mode == LumaMode::TM && mbx == 0 && mby == 0 { continue; }
  ```
- **libwebp:** `src/enc/quant_enc.c:1072-1100` — never skips a mode based on
  position; the prediction generator (`VP8MakeLuma16Preds` →
  `VP8EncPredLuma16`) uses border values 127 (above) and 129 (left) when the
  neighbors are out of frame, and lets RD scoring decide.
- **Difference:** At top edge zenwebp won't even consider V; at left edge
  won't consider H; at top-left corner won't consider TM. libwebp lets the
  border-value-padded prediction compete; sometimes it wins.
- **Why this might cost bytes:** Some flat-image MBs at the top row are
  better coded with V using border 127 than with DC. Skipping the candidate
  prevents that pick.
- **Suggested fix:** Remove the skip checks; ensure border-value (127/129)
  prediction paths exist for all modes.

### 8. `should_try_i4` skip gate has no equivalent in libwebp `PickBestIntra4` — severity: medium — likely size impact: depends on content

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:1758-1769`:
  ```
  let limit_boost = 211u64 + 211u64 * u64::from(self.partition_limit) * 5 / 100;
  let skip_i4_threshold = limit_boost * u64::from(segment.lambda_mode);
  let should_try_i4 = (self.method >= 5 && self.partition_limit < 50)
      || i16_score > skip_i4_threshold
      || luma_mode != LumaMode::DC;
  ```
  At method 2-4 with partition_limit=0, if I16 picked DC and `i16_score <= 211*lambda_mode`,
  I4 evaluation is skipped entirely.
- **libwebp:** `src/enc/quant_enc.c:1419-1437` — when `rd_opt > RD_OPT_NONE`
  and `method >= 2`, `PickBestIntra4` is always called; no flat-DC skip.
- **Difference:** zenwebp adds an early-out gate that suppresses I4
  evaluation on flat DC blocks. The condition `i16_score <= 211 * lambda_mode`
  triggers when D and SD are very small AND the chosen mode is DC.
- **Why this might cost bytes:** For flat areas, I16 DC is usually best, so
  the skip is reasonable. But the check is conservative and may also fire on
  borderline cases where I4 would have won by a small margin.
- **Suggested fix:** Measure: drop the gate at method >= 3 and see if size
  changes. Keep partition_limit-based gating (that prevents partition 0
  overflow on huge images, a separate concern).

### 9. zenwebp uses `PSY_WEIGHT_Y` CSF table for tdisto at method >= 3, libwebp always uses `kWeightY` — severity: medium — likely size impact: hard to predict (perceptual tuning)

- **zenwebp:** `src/encoder/psy.rs:152` — at method >= 3,
  `config.luma_csf = PSY_WEIGHT_Y` (a different table with steeper HF rolloff,
  defined at `psy.rs:35-66`). Used in tdisto calls inside
  `mode_selection.rs:798, 1649, 1964`. Default (method 0-2) uses `VP8_WEIGHT_Y`
  matching libwebp.
- **libwebp:** `src/dsp/enc.c:615-648` (`TTransform`) called with
  `kWeightY` from `quant_enc.c:495-496` always; never uses an alternative
  table.
- **Difference:** Intentional perceptual tuning per CLAUDE.md; biases mode
  selection toward different distortion balance at method >= 3.
- **Why this might cost bytes:** Different tdisto values change which mode
  wins; could be smaller or larger than libwebp depending on content. Likely
  the dominant per-method-3+ size delta vs libwebp on specific image classes.
- **Suggested fix:** None requested; flagging because it's a legit divergence
  from "match libwebp" intent.

### 10. Method 2 uses full RD pipeline (Pick* with coeff costs), libwebp uses distortion-only `RefineUsingDistortion` — severity: medium — likely size impact: small

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:1753-1803` — at method 2 the
  code path enters `pick_best_intra16` (full RD with coeff costs and SD) and
  may call `pick_best_intra4` if the gate passes. zenwebp's method-to-rd_opt
  mapping per CLAUDE.md says "m0-2: RD_OPT_NONE" but the implementation
  actually uses full-RD scoring at method 2.
- **libwebp:** `src/enc/quant_enc.c:1419-1448` — `rd_opt = RD_OPT_NONE` for
  methods 0-2, sending control to
  `RefineUsingDistortion(it, method >= 2, method >= 1, rd)` — this is
  distortion-only scoring with hard-coded lambdas
  (`lambda_d_i16=106, lambda_d_i4=11, lambda_d_uv=120`) and uses raw SSE on
  `prediction` (not reconstructed) blocks, no coefficient cost.
- **Difference:** At method 2 zenwebp uses a richer scoring than libwebp.
  Likely produces slightly different mode picks (sometimes better, sometimes
  worse on size depending on content).
- **Why this might cost bytes:** Could go either direction. Probably not a
  major contributor to the production-settings 1.0149x gap (production
  default method is 4, where both use full RD).
- **Suggested fix:** Either implement a `RefineUsingDistortion`-style path at
  method 0-2 to match libwebp, or accept the divergence and note in CLAUDE.md
  that method 2 deviates intentionally.

### 11. At method 6 (RD_OPT_TRELLIS_ALL) zenwebp does not trellis-quantize Y2 DC or Y1 AC during I16 mode selection — severity: low/medium — likely size impact: noticeable at method 6 only

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:730-744` —
  `quantize_block_simd(&mut y2_quant, y2_matrix, true)` and
  `quantize_ac_only_simd(&mut block, y1_matrix, true)`. No trellis path.
- **libwebp:** `src/enc/quant_enc.c:830-879` (`ReconstructIntra16`) — when
  `DO_TRELLIS_I16 && it->do_trellis` (set at `quant_enc.c:1432` for
  `RD_OPT_TRELLIS_ALL`), uses `TrellisQuantizeBlock` for the Y1 AC blocks
  during the I16 RD evaluation.
- **Difference:** At method 6 libwebp picks the I16 mode using
  trellis-quantized residuals (closer to final coding); zenwebp picks using
  greedy quantization then trellis-quantizes only at the final encode pass
  (if at all in the I16 path).
- **Why this might cost bytes:** Mode pick at method 6 may be worse-aligned
  with the trellis-coded result. Method 6 is rarely the production default,
  so impact is small.
- **Suggested fix:** Wire trellis into the I16 reconstruction path when
  `do_trellis` is on.

### 12. Methods 0-2 evaluate only 3 (sse-presorted) I4 modes — severity: low — likely size impact: only matters at method 2

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:1205-1208`:
  ```
  let max_modes_to_try = match self.method {
      0..=2 => 3,
      _ => 10,
  };
  ```
  Combined with the SSE pre-sort at `mode_selection.rs:1226`, only the
  3 lowest-prediction-SSE modes are RD-evaluated.
- **libwebp:** Always evaluates 10 modes inside `PickBestIntra4`
  (`quant_enc.c:1159`); RefineUsingDistortion (used at method 0-1) also
  evaluates all 10 modes (`quant_enc.c:1360`).
- **Difference:** zenwebp explicitly culls 7 candidates at method 0-2.
  Method 0-1 don't reach this code (I4 disabled), so only method 2 is
  affected. Note method 2 is also the level where zenwebp uses full RD
  while libwebp uses distortion-only — so direct comparison is moot.
- **Why this might cost bytes:** Fewer candidates → worse picks → larger
  residuals.
- **Suggested fix:** Bump method-2 to 10 modes or accept the speed/size
  trade.

### 13. UV flatness penalty checks all 8 blocks; libwebp checks only block 0 — severity: low — likely size impact: small

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:1671-1687` — flattens all 8
  UV blocks into `all_levels_uv[128]` and calls
  `is_flat_coeffs(&all_levels_uv, 8, FLATNESS_LIMIT_UV)`.
- **libwebp:** `src/enc/quant_enc.c:1249-1251`:
  ```
  if (mode > 0 && IsFlat(rd_uv.uv_levels[0], kNumBlocks, FLATNESS_LIMIT_UV)) {
    rd_uv.R += FLATNESS_PENALTY * kNumBlocks;
  }
  ```
  `kNumBlocks = 8`, but the call passes `rd_uv.uv_levels[0]` (a single 16-element
  block) and the function's signature is `IsFlat(int16_t* levels, int num_blocks, ...)`
  — so it scans `num_blocks=8` blocks starting at `uv_levels[0]`, which is
  contiguous storage for all 8 UV blocks. Functionally equivalent to zenwebp.
- **Difference:** Cosmetic — both paths examine 128 coefficients across the 8
  UV blocks. No actual divergence; flag as "looks different but equivalent."
- **Suggested fix:** None.

### 14. I16 score recomputation with `lambda_mode` may double-apply flat penalty — severity: low — likely size impact: small

- **zenwebp:** `src/encoder/vp8/mode_selection.rs:818-871`. Per-mode score uses
  `d_final = sse * 2` and `sd_final = spectral_disto * 2` when flat. The
  doubled values are stored in `best_sse` / `best_spectral_disto`. The final
  recompute at line 868-871 multiplies these doubled values by
  `RD_DISTO_MULT * lambda_mode`, applying the doubling consistently.
- **libwebp:** `src/enc/quant_enc.c:1085-1105` — `rd_cur->D *= 2` and
  `rd_cur->SD *= 2` modify the score struct in place; `SetRDScore(lambda, ...)`
  uses the doubled values; on copy to best, the doubled D/SD persist; final
  `SetRDScore(dqm->lambda_mode, rd)` at line 1105 uses the doubled values.
- **Difference:** zenwebp's behavior matches libwebp here — both keep the
  doubled distortion through the final lambda_mode rescore. No bug.
- **Suggested fix:** None — included only because earlier inspection raised
  the question.

## Things that LOOK different but are actually equivalent

- **`MODES` array iteration order in I4** — zenwebp's `[DC, TM, VE, HE, LD, RD, VR, VL, HD, HU]`
  corresponds to its B_*_PRED enum values 0-9; libwebp iterates `for mode in 0..NUM_BMODES`.
  At max_modes_to_try=10 (method 3+) the eval set is the same, just visited in
  different order; pre-sort by SSE also doesn't affect the final pick. (Mode-cost
  table indexing is the bug — see divergence #1.)
- **Flatness penalty applied via `R` field vs folded into `total_rate_cost`** —
  same RD score; (R + H) * lambda is identical whether flatness lives in R or H.
- **Source block extraction layout** — zenwebp reads from raw frame buffer with
  stride; libwebp pre-imports into `yuv_in` (16x16 contiguous). Same source
  pixels.
- **`lambda_*` values** — `src/common/types.rs:860-872` formulas match
  `quant_enc.c:245-252` exactly (same shifts, same `max(1)` clamp).
- **`RD_DISTO_MULT = 256`, `FLATNESS_PENALTY = 140`, `FLATNESS_LIMIT_*`** — all
  match libwebp's `quant_enc.c:47-54`.
- **`VP8_ENC_BANDS` table** — `src/encoder/tables.rs:223` matches
  `dsp/cost.c:230-233` exactly.
- **`KEYFRAME_BPRED_MODE_PROBS` table** — confirmed remapped to zenwebp's enum
  IDs (verified by spot-checking row [0][4-9] which permutes to libwebp's
  [0][6,4,5,7,8,9]). Self-consistent with zenwebp's enum.
- **I4 sub-block `top_nz`/`left_nz` initialization** — zenwebp imports cross-MB
  context from `top_complexity` / `left_complexity` (mode_selection.rs:1124-1135),
  matching libwebp's `VP8IteratorNzToBytes`. Bug exists in I16/UV cost paths
  only.
- **I4 mode cost inner-block `top_ctx`/`left_ctx`** — `mode_selection.rs:1149-1162`
  correctly tracks cross-MB I4 modes via `top_b_pred`/`left_b_pred`, matching
  libwebp's `GetCostModeI4` logic.
- **`MULT_8B(a,b) = (a*b + 128) >> 8`** — zenwebp uses
  `(tlambda as i32 * td + 128) >> 8` directly (see mode_selection.rs:799,
  1652, 1965). Match.
- **WHT for I16 Y2 DC block** — both apply WHT before quantizing, both use
  `y2_matrix` with `q[0]` (DC) and `q[1]` (AC) per
  `quant_enc.c:235-236` and `types.rs:850-862`.
- **I4 sub-block iteration order** — both iterate raster (sby outer, sbx
  inner). zenwebp uses `i = sby * 4 + sbx`, libwebp uses VP8Scan-derived
  iteration; same sub-blocks visited in same order.
- **Top-right pixel replication for last-column I4 sub-blocks** —
  `prediction.rs:51-56, 113-116` does the equivalent of libwebp's
  `iterator_enc.c:418-426, 453-457`.

## Open questions

- **`updated_probs` vs default `token_probs`** — `mode_selection.rs:658, 1138, 1485`
  use `self.updated_probs.as_ref().unwrap_or(&self.token_probs)`. libwebp uses
  the live (per-pass-updated) `enc->proba.coeffs` always. Need to check
  whether zenwebp's first pass also has this populated equivalently.
- **`SimpleQuantize` finalization at method 5** — libwebp calls
  `it->do_trellis = 1; SimpleQuantize(it, rd);` after Pick* when
  `rd_opt == RD_OPT_TRELLIS` (method 5). Does zenwebp re-trellis-quantize
  before final emission, and with the same lambda values? Briefly looked at
  `vp8/mod.rs:1010-1078` and `do_trellis` is set at `method >= 5`, but the
  `record_residual_tokens_storing` path may or may not honor that. Worth
  verifying with a focused trace.
- **Y2 / Y1 quant matrix `sharpen` field** — libwebp applies
  `kFreqSharpening` only to AC luma (type 0) at `quant_enc.c:209-216`. Need
  to confirm zenwebp's `VP8Matrix::new(MatrixType::Y1)` does the same and
  that `is_first` distinguishes Y1 from Y2/UV during quant.
- **PickBestUV ordering uniqueness** — libwebp's PickBestUV iterates DC, TM,
  V, H and *only* swaps `dst <-> tmp_dst` when a new best is found; zenwebp
  doesn't have a swap-out concept (it generates all reconstructions per mode
  and keeps the one whose RD wins). Functionally equivalent unless something
  in zenwebp's reconstruction has order-dependent state — should be fine.
- **`min_disto` / `max_edge` recording** — libwebp's `PickBestIntra16` updates
  `dqm->max_edge` based on `(rd->nz & 0x100ffff) == 0x1000000 && rd->D > dqm->min_disto`
  at `quant_enc.c:1111-1113`, used later by `SetupFilterStrength`. Need to
  confirm zenwebp tracks this; couldn't see it during this audit.
- **`PickBestIntra4` `max_i4_header_bits` vs zenwebp's `max_header_bits`** —
  zenwebp uses `256 * 16 * 16 / 4` (mode_selection.rs:1117); libwebp uses
  `enc->max_i4_header_bits` set by `webp_enc.c` from the partition_limit
  control. Should compare formulas precisely.
