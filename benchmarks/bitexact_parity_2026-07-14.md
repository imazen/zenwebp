# Bit-exact libwebp parity (#38) — feasibility + divergence inventory

**Date:** 2026-07-14
**Config under test:** CID22 382297 (512×512) and a 32×32 synthetic gradient,
q75, `CostModel::StrictLibwebpParity`, vs libwebp-sys 0.14.4 (`webpx`).
**Harness:** `dev/bitexact_diff.rs` (mode streams) + an instrumented libwebp
`cwebp-trace` build and a zen `MBDUMP` dump comparing per-MB coefficient
signatures `(signed-sum, abs-sum, nonzero-count)` over the quantized levels.

## Feasibility: YES — the forward transform + quantization is bit-identical

Where zen and libwebp pick the **same** intra mode, the quantized coefficient
levels are **identical**. On the 32×32 gradient at m0:

| MB | mode (both) | zen luma sig | libwebp luma sig |
|----|-------------|--------------|------------------|
| (0,0) | V  | −169 / 169 / 37 | −169 / 169 / 37 |
| (0,1) | H  | −78 / 102 / 37  | −78 / 102 / 37  |

MB(0,0) UV also matches (0 / 36 / 12 both). Three independent sums over the
272 luma coefficients agreeing exactly is a strong hash — the FTransform +
QuantizeBlock path reproduces libwebp bit-for-bit. **Bit-exact output is
reachable by aligning the DECISION logic; no transform/quant rewrite is
needed.**

## Remaining divergences (all decision-level), m0 segs1, 382297

1. **Luma mode selection ~95% agree** (was 0% — every MB collapsed to I16-DC
   until the single-segment FastMBAnalyze-hint fix, 331f386). I4-vs-I16 split
   now matches libwebp exactly (136 = 136 I4 MBs). Residual ~5% is
   SSE/cost tie-breaking in `pick_intra16_sse`.
2. **UV mode selection — FIXED for m0** (45.5% → 96.2% agree). libwebp m0 uses
   `refine_uv_mode=0` — the UV mode is the analysis pass's `MBAnalyzeBestUVMode`
   alpha pick (DC or TM; `MAX_UV_MODE=2`), left untouched by
   RefineUsingDistortion. The analysis UV mode is now stored per-MB
   (`fast_mb_uv_hints`) and used verbatim at m0 under `StrictLibwebpParity`.
   Residual ~4% is analysis-iterator reconstruction context, not the pick.
3. **Coefficient probabilities.** libwebp ships DEFAULT probas at m0-m2 with a
   single segment: `OneStatPass` returns `size_p0 = ΣH + segment_hdr.size = 0`
   (H is 0 at RD_OPT_NONE, segment header is 0 at 1 segment), StatLoop treats 0
   as failure and returns before `FinalizeTokenProbas`. zen updates probas
   (396 updates). This is the dominant byte gap at m0-m2 segs1 (zen 56 KB @
   29.38 dB vs libwebp 76 KB @ 29.38 dB — identical PSNR, libwebp just codes
   the same pixels with worse probabilities). Parity needs a
   StrictLibwebpParity gate that ships defaults under the same condition.
4. **I4 sub-mode agreement — characterized, not yet closed.** All-16-match is
   3/136 MBs; per-sub-block is ~59%, and even **sub-block 0** (corner, default
   context) is only ~69%. The scoring is verified faithful: `pick_intra4_sse_
   body!` uses libwebp's exact `VP8SSE4x4(src, prediction) * RD_DISTO_MULT +
   VP8FixedCostsI4[top][left][mode] * lambda_d_i4` over all 10 modes, strict
   `<` in mode order, reconstructing the winner. The I16-neighbor context is
   also correct: `LumaMode::into_intra()` maps I16 modes to the same B-mode
   indices libwebp's `VP8SetIntra16Mode` stores (DC→0, TM→1, V→2, H→3). So the
   divergence is in the **I4 predictions themselves** (`I4Predictions::compute`
   vs `MakeIntra4Preds` — 10 modes with subtle top-right/edge handling) or the
   `VP8SSE4x4` kernel, cascading through the 16-sub-block reconstruction chain.
   Next chunk: per-prediction instrumentation on both sides for one shared-I4
   MB's sub-block 0.
5. **Skip decisions + segment analysis (alpha/beta/k-means)** at segs>1
   (seg-map agreement only 38.1% at segs4 — the k-means cluster assignment
   diverges, which also perturbs luma agreement 94.8% → 87.9% at segs4).

## Root cause found: forward WHT rounding (2026-07-14 part 4)

Traced the first divergence on the photo to mb(4,0): identical mode (DC),
identical left border, identical per-mode I16 SSEs — yet one Y2 coefficient
differed (libwebp −1, zen 0). The forward Walsh-Hadamard transform
(`wht4x4`) finalized with `(x + (x>0?1:0))/2` (round-half-up / truncate
toward zero) where libwebp's `FTransformWHT` uses arithmetic `x >> 1` (floor
toward −∞). They diverge on **every odd intermediate** (−1 >> 1 = −1, but
−1 / 2 = 0). MBs 0-3 matched only because their Y2 coefficients were even;
mb(4,0)'s odd coefficient crossed a quant boundary and seeded a whole-frame
mode-selection cascade.

Fixed to `>> 1`. Impact:
- mb(4,0) Y2 now byte-identical to libwebp.
- m0/m1 luma-mode agreement 95.2% → **96.5%**, sub-block-0 69.1% → 72.1%.
- Corpus lossy size (12 files) ±0.02% (noise); zensim matrix unchanged
  (14 ok). Spec-correct and quality-neutral, so applied unconditionally
  (default path too), not gated on the parity flag.

This is one cascade root; per-subblock I4 agreement is still ~58%, so
further roots remain (other odd-boundary quantization cases, UV/I4 ties).

## What shipped from this investigation

- `fix(vp8): run FastMBAnalyze hints at m0/m1 with a single segment` (331f386)
  — a real quality bug (all-DC collapse) affecting every m0/m1 single-segment
  encode, not just parity. PSNR at m0 went from degraded to matching libwebp's
  exactly (29.38 dB).
- **Proba bail under `StrictLibwebpParity`** (item 3): at method ≤ 2 with one
  segment, ship default coefficient probabilities instead of the image-adapted
  update, matching libwebp's `StatLoop` early return. m0 size vs libwebp went
  **0.7393× → 0.9954×** (within 0.5%); m0/m1/m2 proba-update counts now both 0.
  The tuned default (`ZenwebpDefault`) is unaffected — it keeps the smaller,
  image-adapted proba coding. Remaining m2 gap (1.06×) is mode divergence, not
  probas.

## Byte-ratio snapshot vs libwebp (382297, q75, segs1, StrictLibwebpParity)

| method | before this work | after hint-fix + proba-bail |
|--------|------------------|-----------------------------|
| m0 | all-DC (broken) | **0.9954×** @ equal PSNR |
| m1 | all-DC (broken) | 1.0119× |
| m2 | 0.9193× (wrong probas) | 1.0645× (probas now match; modes differ) |

## Root cause #2: I4 mode array in wrong order (2026-07-14 part 5) — a real quality bug

Traced the mb(9,0) sub-block-4 divergence to the end: the m0-m2 I4 pick
(`pick_intra4_sse_body!`) had its `MODES` lookup array in libwebp's INTERNAL
B-mode numbering `[…RD, VR, LD…]` (indices 4/5/6) while `preds.data` and
`mode_costs` — indexed by the same loop variable — are in zenwebp's IntraMode
declaration order `[…LD, RD, VR…]`. So `MODES[best_idx]` mapped a winning
LD/RD/VR *prediction* to the WRONG `IntraMode`: the encoder scored one mode as
best, then emitted and reconstructed a different one. On mb(9,0) sub-block 4 it
scored VR (SSE 331, near-perfect) but emitted LD (SSE 107441), turning a
zero-residual block into `[-24, 7, 8, …]`.

Fixed by deriving the mode from the index via the single canonical
`IntraMode::from_i8` (removing the duplicate array entirely). The m3-m6 path
(`pick_best_intra4`) already used the correct order.

**Impact — this is a quality bug, not just parity.** Default-path lossy corpus
(12 files, equal PSNR): **m0 −2.14%, m1 −2.14%, m2 −7.88%** bytes; m3-m6
unchanged. Parity (382297, StrictLibwebpParity): I4 per-sub-block agreement
**58% → 92.6%**, all-16-match **2/136 → 86/136**, m0 byte ratio **1.011× →
0.9996×** at matched PSNR. zensim matrix unchanged (14 ok).

## Forward DCT proven bit-exact — chroma divergence is decision-level (part 6)

The remaining chroma-DC divergence (mb(1,0) bottom U blocks quantized DC off by
+1, seen while feeding zen libwebp's exact YUV via the `yuvenc` harness) was
suspected to be a SIMD-vs-scalar rounding difference in the forward transform.
It is **not**. Two committed differential tests
(`src/common/transform.rs::tests`) port libwebp's `FTransform_C` exactly and
compare it against zen's actual production kernels:

- `ftransform_single_matches_libwebp_c` — luma path `ftransform_from_u8_4x4`
  vs `FTransform_C`, 500k random (src, ref) pairs: **identical**.
- `ftransform2_matches_libwebp_c` — chroma path `ftransform2_from_u8` (the
  paired 2-block SIMD DCT, exactly the kernel that produced the +1) vs
  `FTransform_C` on each block, 500k random inputs: **identical**.
- `ftransform_edge_cases_match_libwebp_c` — uniform-213-prediction / near-flat
  source (the literal chroma-DC scenario) + odd/even `>>9`/`>>16` boundary
  probes: **identical**.

Combined with the already-verified quantize constants (QFIX=17,
kBiasMatrices, kDcTable) this proves the **forward transform + quantization
is bit-exact** for any matched input. Therefore the chroma DC +1 cannot be
arithmetic: given identical source YUV, the only way a bit-exact DCT+quant
emits a different DC is a different **prediction** (`ref_`). The chroma DC
prediction of mb(1,0) reads mb(0,0)'s reconstructed chroma right edge; a
hair's divergence in an upstream chroma mode pick or reconstruction cascades
into mb(1,0)'s prediction and the transform faithfully reproduces it. This
reclassifies the last chroma gap from "unresolved numerical contradiction" to
the **same decision-level cascade** as the luma I4 work — closable by aligning
chroma mode selection + the RGB→YUV source, not by touching the transform.

These tests are permanent regression gates: if any future SIMD refactor breaks
forward-transform bit-exactness, they fail loudly.

## Chroma downsampling made byte-exact — U/V now identical to libwebp (part 7)

The chroma prediction cascade traced to its true source: zen's gamma-corrected
2×2 downsampling was rounding the gamma-averaged R/G/B to a **byte** before the
U/V matrix, where libwebp keeps them at **YUV_FIX+2 (×4) precision** through the
matrix and only clamps the final U/V (`SUM4`/`SUM2` → `VP8ClipUV`,
`picture_csp_enc.c` + `dsp/yuv.h`). That sub-byte precision loss produced the
±1 chroma divergences that cascaded into chroma mode selection.

Fixed by porting `Interpolate` / `LinearToGamma(base,shift)` / `VP8ClipUV`
exactly (`src/decoder/yuv.rs`): the averaged channel stays at ×4 precision, the
matrix runs on ×4 values, and `VP8ClipUV` descales by 18 with the `(1<<17)`
rounder + `(128<<18)` bias and clamps. The gamma tables were already correct
(`kGammaToLinearTab`, `kLinearToGammaTab` verified). A key algebraic identity
(`linear_to_gamma_fx(2·S, 0) ≡ linear_to_gamma_fx(S, 1)`, both reduce to
`Interpolate(2·S)`) means the odd-width / odd-height / corner edge cases, which
pass replicated pixels, match libwebp's `SUM2` / `rgb_stride=0` handling with no
special-casing. The production SIMD kernel (`gamma_chroma_rows_generic`) was
converted to the same ×4 path via a 16384-entry sum→×4 LUT; two tests
(`gamma_chroma_simd_matches_scalar`, `gamma_chroma_grayscale_is_128`) lock SIMD
== scalar and the grayscale invariant.

**Measured (382297, q75, libwebp's YUV extracted via `WebPPictureImportRGB` +
`WebPPictureARGBToYUVA` — exactly what `WebPEncode` runs):**
- **U plane: 0 / 65536 pixels differ (max |Δ| = 0)** — byte-identical.
- **V plane: 0 / 65536 pixels differ (max |Δ| = 0)** — byte-identical.
- **UV mode agreement m0 segs1: 96.2% → 100.0%** (every chroma mode now matches).
- Remaining: **Y plane 572 / 262144 differ (max |Δ| = 1)** — zenyuv's SIMD
  maddubs Y kernel vs libwebp's exact `VP8RGBToY` (16-bit coeffs). zen's scalar
  `rgb_to_y` already matches libwebp exactly; the production path uses zenyuv's
  approximate Y for speed. This is now the *only* source-level parity gap.

**Gated on `StrictLibwebpParity`, not unconditional.** The byte-exact precision
*regresses* the zensim floor on two synthetic low-q cells (gradient q10
70.76→69.29, noise q10 43.37→41.57): zenwebp's byte-rounded chroma is
*measurably better* than libwebp-exact there. So the tuned default keeps the
byte-rounded path (`ChromaPrec::TunedByteRound`) and only
`StrictLibwebpParity` uses `ChromaPrec::LibwebpExact` — exactly the
"libwebp-exact vs zenwebp-tuned" split #38 is about. Both modes share one SIMD
kernel and differ only by a 16384-entry LUT, because `byte×4` through the ×4
matrix `>>18` reproduces the historical `byte` matrix `>>16` (the clamp added
for both only fixes the old cast-wrap at out-of-range U/V — a strict
correctness improvement that never changes an in-range value). The
`ChromaPrec` is selected in `prepare_input_for_encoding` from
`params.cost_model`. Result: tuned floors stay green (zensim matrix 14 ok),
parity chroma is byte-identical to libwebp.

## Exact Y + full RGB→YUV byte-identity; m0 modes 100% (part 8)

zen's scalar `rgb_to_y` is already a bit-exact port of `VP8RGBToY`, but the
production path used zenyuv's maddubs SIMD Y (±1: 572/262144 pixels on 382297).
Under `StrictLibwebpParity` we now fill Y with `rgb_to_y`. With that, the
**entire RGB→YUV conversion is byte-identical to libwebp** (`WebPPictureImportRGB`
+ `WebPPictureARGBToYUVA`): **Y 0/262144, U 0/65536, V 0/65536 differ**, and
`zen-from-RGB` == `zen-from-libwebp-YUV` bitstreams.

The exact Y also closed the residual I4 tie divergence — the ±1 Y differences
had been flipping I4 SSE ties. Post-fix, **m0 segs1**: `y_same 100%`,
`uv_same 100%`, `b4_same 136/136` (every luma mode, chroma mode, and all 16
sub-blocks of all 136 I4 MBs match). m1/m2 luma likewise 100%; **partition 0
(modes + proba updates) is byte-identical** (both 1619 bytes).

**Remaining m0 gap — coefficient reconstruction cascade (~40 bytes, 0.05%).**
The VP8 payload matches through partition 0 and the first 68 bytes of the token
partition, then a *coefficient value* diverges. With byte-exact YUV, matching
modes, bit-exact forward transform + quant, matching default probas
(`COEFF_PROBS` == `VP8CoeffsProba0`, verified) and a verified-equivalent inverse
WHT (`iwht4x4` vs `TransformWHT_C`, +3 rounder placement is algebraically
identical), this must be a ±1 encoder-reconstruction difference in some MB that
doesn't flip a mode but crosses a quant boundary in a downstream MB's residual.
Harness: `scratchpad/webp-ll-compare/src/bin/chromacmp.rs` (libwebp YUV via FFI,
per-plane diff, VP8-payload alignment). This is the last m0 parity gap; m1–m6
UV (88.8%), m3–m6 modes (~74%), and segs>1 (k-means) remain beyond it.

## Chroma DC error diffusion + the residual m0 gap (part 9)

**libwebp does chroma DC error diffusion (`CorrectDCValues`) for any quality
≤ `ERROR_DIFFUSION_QUALITY` (98)** — i.e. at essentially all normal qualities,
not just low-q dithering. `top_derr` is allocated whenever `quality ≤ 98 ||
pass > 1` (`webp_enc.c`). So it is ON at q75. zen's
`apply_chroma_error_diffusion` is a faithful port (verified: the C1=7/C2=8/
DSHIFT=4/DSCALE=1 constants, the block 0→1→2→3 propagation using err0/err1/err2,
`QuantizeSingle`'s `err = |V| − level·q` with the DSCALE shift, and
`StoreDiffusionErrors`' `left=[err1, 3·err3>>2]`, `top=[err2, err3−left1]` all
match). It must stay ON under `StrictLibwebpParity` (an earlier attempt to gate
it OFF was wrong and made parity *worse* — zen-ED-off 76376 vs zen-ED-on 76470
vs libwebp 76430; ED-on is closer, and libwebp does ED).

**Remaining m0 chroma gap is NOT error diffusion.** Traced MB(1,0)'s
bottom-right U sub-block: zen reconstructs it flat to 174, libwebp to 177. But
ED on/off both leave zen at level −13 (174) — the FDCT DC (−307 pre-ED, −311
post-ED) quantizes to −13 with uvDCq=24 either way. For libwebp's 177 it needs
level −12, i.e. an FDCT-DC magnitude ~278–301 vs zen's ~307. With source
byte-exact, uvDCq=24 shared (luma is byte-exact so the base quant matches), and
the left border from the byte-identical MB(0,0) giving DC pred = 213 in both,
every traced input matches yet the coded coefficient differs by one level. This
is the same sub-±1-level chroma divergence that has resisted exhaustive
inspection; resolving it requires an instrumented-libwebp dump of MB(1,0)'s
chroma prediction + pre-quant DC to find where the ~10-magnitude FDCT-DC
difference originates (candidate: a UV-mode/prediction edge case at the top row,
mby=0, where V/TM are unavailable). Harness + per-MB dumps:
`scratchpad/webp-ll-compare/src/bin/chromacmp.rs` (ZEN_CDUMP/ZEN_EDUMP env).

**Net m0 state:** luma byte-identical, RGB→YUV byte-identical, all modes match,
partition 0 byte-identical; a handful of chroma DC coefficients differ by one
quant level (zen 76470 vs libwebp 76430 bytes, 99.95% identical).

## m0 BYTE-IDENTICAL — chroma DC diffusion inter-MB store (part 10)

The final m0 gap was the chroma DC error diffusion's **inter-MB store**. Root
cause traced against an instrumented libwebp (built from libwebp-sys 0.14.4's
vendored source, byte-identical output to the linked lib): libwebp calls
`StoreDiffusionErrors` **only inside `PickBestUV`**, which runs when
`rd_opt_level > RD_OPT_NONE` (method ≥ 3). At method 0-2 libwebp runs the
intra-MB `CorrectDCValues` correction with **zero** neighbor errors but never
stores them, so `top_derr`/`left_derr` stay 0 for the whole frame. zenwebp's
`apply_chroma_error_diffusion` stored unconditionally → over-propagated the DC
diffusion error → shifted MB(1,0)'s block-3 DC one quant level (174 vs 177).

Evidence: the store fires **0× at m0-2, 2× at m3-6**; patching libwebp to store
at m0 (`ZEN_MIMIC=1`) reproduced zen's exact wrong state (`left_derr=[-1,4]`,
`errs=[3,4,-7,0]`, level −13). Fixed by gating the store on `method >= 3 ||
cost_model != StrictLibwebpParity` (`residuals.rs::apply_chroma_error_diffusion`).

**Result — m0 segs1 `StrictLibwebpParity` is now BYTE-IDENTICAL to libwebp**
(382297: 76430 bytes both, VP8 payload 100% identical, 0 decoded-pixel diff).
m1/m2 UV mode agreement 88.8% → 95.6%. The tuned default is unchanged (it keeps
inter-MB propagation, which measurably reduces banding on smooth low-q content —
zensim gate stays green, 14 ok).

**Full m0 parity chain (all verified byte-exact):** RGB→YUV (Y+U+V) → I16/I4/UV
modes → forward DCT/WHT + quant → partition 0 → token coefficients → total
bitstream. Remaining tail: m1/m2 UV residual 4.4% (method-specific, minor),
m3-6 (~74-90%, RD-path I4 tie-break), segs>1 (k-means).

### What became a tuned-default quality question
Two m0-2 behaviors were found where zenwebp's deviation from libwebp is
*measurably better on synthetic low-q* (gradient/noise): (1) byte-rounded chroma
downsampling vs libwebp's ×4 precision, and (2) inter-MB DC diffusion
propagation vs libwebp's m0-2 no-store. Both are kept in the tuned default and
gated OFF only under parity. Whether they help *real* content (vs synthetic
gradient/noise) is the open question a CID22 + imazen-26 chroma sweep answers.

### Chroma-precision sweep result (real content) — keep the tuned default
A/B on 3 CID22 + 12 imazen-26 color-rich images (imazen resized ≤1024 Lanczos3),
Y held constant, exact ×4 vs tuned byte-round chroma, ZenwebpDefault method 4,
q ∈ {20,35,50,75,90}, scored with zensim (`benchmarks/chroma_precision_sweep_2026-07-14.tsv`):
**overall mean Δzsim(exact−tuned) = −0.038, Δsize = −0.05%** — both in the noise.
Tuned keeps a small edge only at q20 (−0.195), consistent with the synthetic
banding finding; q35–90 are dead even (±0.03). So byte-exact chroma should NOT
become the default: it's neutral-to-marginally-worse on real content and would
regress the synthetic floor gate. Keep tuned byte-round as default; parity uses
exact. (Scope: single size regime, method 4 — an A/B decision, not a calibration
constant, so the reduced grid is adequate for a "no meaningful difference" call.)

## Byte-identity ladder + segs>1 alignment (part 11)

**Byte-identical to libwebp (382297, StrictLibwebpParity):**
- **segs1: m0, m1, m2** (76430 / 75608 / 61190) — the whole RD_OPT_NONE range.
- **segs4: m2** (51164).

Fixes that got here (all on main, tuned default unchanged / parity-gated):
- **m1/m2**: `pick_uv_sse` evaluated all 4 UV modes at edges (was skipping V/H/TM;
  libwebp uses the default 127/129 borders the decoder shares). → m1/m2 segs1
  byte-identical.
- **segs>1 m0/m1**: use FastMBAnalyze alpha (0) at method ≤ 1 — libwebp's
  `MBAnalyze` only runs the full susceptibility analysis at method ≥ 2; zen ran
  it always, diverging the m0/m1 segmentation histogram → centers → seg map +
  quantizers. → seg_same 38.9% → 100%, seg_q / seg_tree_probs now match.
- **segs>1 filter**: gate the edge-based loop-filter bump (`StoreMaxDelta`) on
  the RD path — libwebp calls it only from `PickBestIntra16` (method ≥ 3); zen
  bumped at all methods, over-filtering segments 2/3 at m0-2. → m2 segs4
  byte-identical; m0/m1 segs4 seg_lf now matches.

**Remaining tail (precisely characterized):**
- **m0/m1 segs4**: everything matches (modes, segmentation, seg_q, seg_lf) except
  the coefficient-proba *update count* (zen 296 vs libwebp 210). The
  `should_update` decision is a verified-faithful port of `FinalizeTokenProbas`
  (branch_cost, `savings>0`, the 8·256 signalling term all match), so the
  divergence is in the token-stat histogram feeding it — the same proba residual
  as m3-6.
- **m3-6 segs1**: RD-path cascade (y_same ~92%, uv_same ~84.5% after the zigzag-
  cost / UV-diffusion / tie-break fixes). Next primary pieces: the tlambda
  `CheckLambdaValue(≥1)` clamp coupled with an I4 ctx0 nz cascade (must land
  together), then the proba residual above. m5/m6 additionally need trellis
  alignment.
- **segs>1 m3-6**: the segmentation is aligned (seg_same 100%); the residual is
  the same m3-6 RD cascade on top.

## Non-RD tier CLOSED + m3-6 RD cascade collapsed (part 12, 2026-07-14)

**Byte-identical to libwebp now (382297, StrictLibwebpParity):**
- **segs1: m0, m1, m2** (unchanged).
- **segs4: m0, m1, m2, AND m6** (57008 / 55284 / 51164 / 47406).

The whole RD_OPT_NONE range is byte-identical for both single- and
multi-segment. Five fixes (all on main, all parity-gated, tuned default
unchanged, lib 323 + zensim 14 + chroma 3 green throughout):

1. **VP8 even-pad inside the chunk** (`vp8/mod.rs`, 44b6a38). libwebp
   (`syntax_enc.c` `VP8EncWrite`) pads the whole VP8 bitstream to even and
   counts the pad byte *inside* the `VP8 ` chunk size; zenwebp emitted the
   odd size and let `write_chunk` add a RIFF pad *after* the chunk. Same file
   size, different chunk-size field (byte 16) on every odd-length stream. → m1
   segs4 (the isolated 1-byte trailing-0x00 case) byte-identical.

2. **fast_probe m0 stat subset** (`vp8/mod.rs`, a899036). libwebp's method-0
   `StatLoop` finalizes the emitted probas from only the first `nb_mbs>>2` (or
   50) MBs; zenwebp records the whole frame. Snapshot `proba_stats` at the
   fast_probe boundary under parity. → m0 segs4 byte-identical (295 == 295
   proba updates; was zen 431).

3-5. **m3-6 RD I4/I16 cost alignment** (`types.rs` + `mode_selection.rs` +
   `vp8/mod.rs`, 4d41a33 + defebc5). Four root causes, largest last:
   - **tlambda `CheckLambdaValue(≥1)` clamp** — libwebp keeps the I16/I4
     spectral-distortion term active at m0-m3/sns=0. (The documented pair.)
   - **I4 running-total flatness penalty** — libwebp folds `FLATNESS_PENALTY`
     (140) for flat non-DC sub-blocks into `rd_i4.R`; zenwebp applied it in
     per-mode selection but dropped it from the I4-vs-I16 running total.
     **CORRECTION to part 11:** the part-11 note blamed "an I4 ctx0 nz
     cascade / missing ctx0==0 bit". That was a **misdiagnosis** — per-sub-block
     ctx/nz dumps confirmed ctx is identical on both sides; the 140/blk gap is
     purely the flatness penalty (it correlates exactly with non-DC mode, not
     with ctx). The clamp + flatness fix land together.
   - **`max_i4_header_bits`** — libwebp uses `256*16*16*(limit²)/100²`,
     `limit = 100 - partition_limit` (default → 65536); zenwebp hardcoded 16384
     (the limit=50 point), 4× low, bailing high-detail MBs to I16 early
     (mb(17,0): total_mode_cost 17509 > 16384 but < 65536).
   - **mid-pass `level_costs` refresh** (the dominant one) — libwebp's token
     loop rebuilds `level_costs` (`VP8CalculateLevelCosts`) alongside the proba
     every `max_count` MBs; zenwebp refreshed `updated_probs` but left
     `level_costs` on pass-start defaults, so p0 came from refreshed probs and
     level costs from default probs. The mismatch ran chroma-UV coeff costs
     ~1.5× high on later rows and flipped DC/TM picks. Rebuild under parity.

   Cumulative on m3 segs1: y-mode 92.0→**99.7%**, UV-mode 84.5→**100%**, I4
   sub-block 133→**865/874**, proba-update count zen **296→209** (lib 210).

**KEY FINDING — the "Task 1 recording bug" does NOT exist.** The proba-stat
histogram recording is **byte-exact**: instrumenting both `FinalizeTokenProbas`
(libwebp) and `compute_updated_probabilities` (zen) to dump all 1056
`(nb,total,newp,use)` cells, m1 segs4 (100% modes) diffs in **0/1056 cells**.
The 296-vs-210 count from part 11 split into two unrelated causes — fast_probe
(m0, fix #2) and the RD mode cascade (m3-6, fixes #3-5) — neither a recording
bug. Harness: `scratchpad/lwsrc` (instrumented libwebp, `DUMPSTATS`/`TARGX`) +
`scratchpad/webp-ll-compare` (`methodcmp`, `bitex5`, `dumpstat` with
`ZEN_DUMPSTATS`/`MB_DEBUG`).

**Remaining tail (precisely characterized, all cascade- or trellis-dominated):**
- **m3/m4**: down to a coefficient-level cascade. m3 segs1 = 99.7% modes,
  proba count off by **1** (209 vs 210), 3 mode-flip MBs in the bottom-right
  corner (mb(30,28)/(31,28)/(30,30)). Those flips are genuine RD ties tipping
  on a ±1 reconstruction drift from an upstream *matching-mode* MB whose
  coefficients differ slightly — so the byte stream desyncs at the part-0
  proba-update section (one different update flag) and cascades (99% of bytes
  differ despite 99.7% mode agreement). Closing it needs coefficient-level
  (not mode-level) tracing to find the first ±1-coefficient MB; the transform +
  quant are already proven bit-exact (parts 6/8), so the root is a downstream
  quant-boundary crossing, not arithmetic.
- **m5, m6 segs1**: trellis (`RD_OPT_TRELLIS`/`_ALL`) — m5 segs1 y_same 94.0%,
  m6 segs1 93.5%. Task 3 (trellis alignment); untouched, and gated on m3/m4
  reaching byte-identity first per the task plan.

## 11 of 14 cells BYTE-IDENTICAL — I4 sub-block tie-break (part 13, 2026-07-14)

**Byte-identical to libwebp now (382297, StrictLibwebpParity):**
- **segs1: m0, m1, m2, m3, m4, m6** (all but m5).
- **segs4: m0, m1, m2, m3, m6** (all but m4, m5).

**11 of 14 cells.** One fix got here from part 12's "m3 modes 99.7%":

**I4 sub-block mode tie-break** (`mode_selection.rs`, e6ee888). libwebp's
`PickBestIntra4` iterates the 10 sub-block modes in index order 0..9 and keeps
the first strict-minimum, so score ties resolve toward the lower mode index.
zenwebp presorts modes by prediction SSE (for its RD early-exit), which
resolves ties toward whichever tied mode sorts first by SSE — usually not the
lowest index. On flat/low-detail sub-blocks two predictions often tie exactly:
382297 m3 mb(27,23) sub-block 9 is an exact tie (blkscore 142133) between RD(5)
and VR(6); libwebp picks RD, zen picked VR, and the different reconstruction
cascaded pixels into downstream MBs — which is what had desynced the whole
m3/m4 token stream (one different part-0 proba-update flag → 99% of bytes
differ despite 99.7% mode agreement). At m3+ all 10 modes are evaluated, so
index order changes only the tie-break; zen's `>= best` early-exit skips line
up with libwebp's `score >= best` early `continue` in the same order (verified:
the early-exit SD/coeff omissions only ever skip provably-can't-win modes, so
the winner is unaffected). Applied to the sse2/wasm/scalar I4 paths under
parity. → m3 segs1, m4 segs1, m3 segs4, m6 segs1 all byte-identical.

**Remaining 3 cells (precisely characterized):**
- **m5 segs1, m5 segs4** — trellis (`RD_OPT_TRELLIS`). Task 3; untouched, gated
  on m4 reaching byte-identity first.
- **m4 segs4** — a spectral-distortion (tdisto/SD) precision cascade. At m4
  with sns>0, `tlambda_scale = sns_strength` makes `tlambda` large (42 here vs
  the `.max(1)` floor of 1 at m3/sns=0), so the SD term (`tlambda·VP8TDisto`)
  dominates near-ties. The first divergence is a **±1 pixel** reconstruction at
  mb(4,10) (a matching-mode MB): sub-blocks 8/9 tie between VR(6) and LD(4) with
  identical winner D/SD/H, but zen's *loser*-mode score differs from libwebp's
  by a sub-unit amount that a `tlambda=42` multiplier turns into a mode flip.
  That ±1 cascades into mb(5,10)'s I4-vs-I16 pick and onward (468/924 MBs
  eventually differ, max |Δ|=84). Root is a sub-±1 numerical difference in one
  I4 sub-block mode's SD or coeff cost, only observable when `tlambda` is large
  — i.e. a `VP8TDisto4x4` (vertical-first Hadamard) bit-exactness question that
  m3/sns=0 (SD≈0) never exercises. Needs per-mode SD/tdisto instrumentation on
  both sides for mb(4,10) sub-block 8 to localize; the same class will gate m5
  (trellis also weights SD). Harness ready (`MB_DEBUG`, `DIVERGE_SEGS` in
  `bitex5`; `TARGX/TARGY` in the instrumented libwebp).

**Session net (382297 StrictLibwebpParity): 3/14 → 11/14 byte-identical cells.**
Seven fixes landed (all parity-gated, tuned default unchanged, lib 323 +
zensim 14 + chroma 3 + pixel-perfect 14 green throughout): even-pad chunk,
fast_probe m0 subset, tlambda clamp + I4 flatness penalty + max_i4_header_bits,
mid-pass level_costs refresh, and the I4 sub-block tie-break.

## 14 of 14 cells BYTE-IDENTICAL — trellis + chroma + filter roots (part 14)

**ALL 14 (method × segment) cells are byte-identical to libwebp** on 382297
(q75, `StrictLibwebpParity`): segs1 m0–m6 and segs4 m0–m6. Four parity-gated
fixes closed the last three cells (m4-segs4, m5-segs1, m5-segs4). Harness:
`scratchpad/webp-ll-compare` (`methodcmp`, `bitex5`, `i4dump`, `decodediff`) +
instrumented libwebp `scratchpad/lwsrc` (`LIBI16m`/`LIBI4blk`/`LIBMODE`/
`LIBTRELLIS`/`LIBCDC`/`LIBSTORE`, `TARGX`/`TARGY`/`TARGI`/`TDUMP`).

### 1. m4-segs4 — I16 flat-source penalty must be a stateful latch (8256bec)

Not a `VP8TDisto` precision cascade as part 13 guessed. libwebp's I16
flat-source penalty (`quant_enc.c:1044-1058`) is a **stateful latch**:
`is_flat` starts as `IsFlatSource16`, then is refined to
`is_flat && IsFlat(mode_coeffs)` for each mode **in DC→TM→V→H order**, and once
it goes false it never re-arms — so a mode's D/SD is doubled only if the source
AND every earlier-order mode's coefficients are flat. zenwebp checked each
mode's coeff-flatness independently. At mb(5,10) TM (mode 1) has non-flat
coeffs, latching `is_flat` off, so libwebp does NOT double V's distortion
(D=96); zen doubled it (D=192, exactly 2×, because V's coeffs happen to be
flat). The inflated I16-V score (92100 vs 62660) made zen keep I4 (88248 <
92100) where libwebp bailed to I16 (67706 > 62660) — the I4 blocks were
byte-identical through the bail point. Fix: under parity iterate the I16 modes
in libwebp's DC,TM,V,H order (MODES indices `[0,3,1,2]`) and carry the single
`is_flat` latch. Iteration order does not change the winner (the tie-break is
rank-based). The `tlambda`/`VP8TDisto` term was a red herring — same for
m5-segs4, which is *filter*, not SD (item 4).

### 2. m5 I4 trellis uses a static per-MB context (8b60a62)

m5 = `RD_OPT_TRELLIS`: pick modes with regular quant, then re-quantize the
chosen mode with trellis in `SimpleQuantize`. `SimpleQuantize`→
`ReconstructIntra4` (`quant_enc.c:844`) reads the trellis rate context from the
inter-MB neighbour nz and **never updates `it->top_nz`/`left_nz` between
sub-blocks**, so all 16 I4 sub-blocks of an MB share one static context. m6 =
`RD_OPT_TRELLIS_ALL` instead runs the I4 trellis inside `PickBestIntra4`, which
DOES update per sub-block (that is why m6 was already byte-identical). zenwebp's
final reconstruction pass updated the context per sub-block. Per-call
`LIBTRELLIS`/`ZTRELLIS` dumps at mb(0,0): identical DCT inputs/lambda, but ctx0
differed (zen `0,1,0,1,1,0,1,2,…` vs libwebp all `0`), flipping sub-block 15
from a kept `-1` coeff to a skip. Fix: under parity at m5
(`do_trellis && !do_trellis_i4_mode`), take the I4 trellis ctx0 from a frozen
snapshot of the inter-MB neighbour nz. This made mb(0,0) luma byte-identical
but did not close m5 alone — the chroma cascade (item 3) remained.

### 3. m5 chroma DC is corrected TWICE — the second pass reads self errors (c96f767)

m5 reconstructs chroma twice. `PickBestUV` corrects the chroma DC with the
neighbour diffusion errors and calls `StoreDiffusionErrors` (`quant_enc.c:1349`),
then `SimpleQuantize`→`ReconstructUV` corrects AGAIN (`quant_enc.c:988`) — but
`StoreDiffusionErrors` has already overwritten `top_derr[x]`/`left_derr` with
**this MB's own** errors, so the final coded chroma DC uses the self errors, not
the neighbour errors. `LIBCDC`/`LIBSTORE` at mb(0,0): the 4-mode PickBestUV pass
reads `top=[0,0] left=[0,0]`; `StoreDiffusionErrors` writes `left=[-1,4]
top=[2,2]`; the SimpleQuantize pass then reads `top=[2,2] left=[-1,4]` and bumps
a U DC level 32→33. zen corrected once (neighbour errors), so its final chroma
DC diverged and cascaded into downstream chroma prediction + UV mode flips
(mb(5,0) UV zen=DC vs lib=H). m4 (`RD_OPT_BASIC`) and m6 (`RD_OPT_TRELLIS_ALL`)
reconstruct chroma once, so they were already correct. Fix: under parity at m5,
after the neighbour correction + store, re-correct the FRESH FTransform DC with
the just-stored self errors; keep the neighbour-correction errors as the stored
propagation (libwebp does not re-store). Closed m5-segs1; brought m5-segs4 to
size + full mode/segmentation parity (one seg_lf field remaining, item 4).

### 4. m5-segs4 blocky-I16 filter delta from mode-selection (non-trellis) nz (6b4fa0c)

Last field: `seg_lf[3]` zen=4 vs lib=2 (all modes/coeffs matched). libwebp's
`StoreMaxDelta` (per-segment loop-filter `max_edge` bump for blocky DC-only I16
MBs) is called from `PickBestIntra16` (`quant_enc.c:1120`) with `do_trellis=0`
at m5, so its `(nz & 0x100ffff)==0x1000000` test (all 16 Y1 AC blocks zero, Y2
nonzero) reads the **mode-selection (non-trellis)** quant. zenwebp tested it
against `stored_coeffs`, which at m5 are the final TRELLIS quant — trellis zeros
more AC, so more MBs counted as blocky, inflating `max_edge` for segment 3 and
its derived filter level. Fix: under parity at m5 recompute the all-Y1-AC-zero
test from the non-trellis quant of the I16 DCT (`y_block_data.coeffs` ==
PickBestIntra16's `luma_blocks`, via `quantize_ac_only_simd`); Y2 DC is never
trellised so its nonzero test is unchanged. m6 (libwebp runs the I16 trellis in
PickBestIntra16, so its StoreMaxDelta nz is already trellis-based) and the tuned
default keep the stored-coeffs test.

**Session net (382297 StrictLibwebpParity): 11/14 → 14/14 byte-identical
cells.** All four fixes parity-gated, tuned default unchanged, lib 323 + zensim
14 + chroma 3 + pixel-perfect 14 green throughout. The two part-13 "remaining"
guesses were both corrected: m4-segs4 was the I16 flat-latch (not a `VP8TDisto`
cascade), and m5 was I4-trellis-context + chroma-double-correction +
blocky-nz-source (not a single trellis-DP bug — the DP was already correct, as
m6's byte-identity proved).

## Scope caveat (added same day)

The 14/14 above is **q75 on 382297** for the two `methodcmp` configs. It does
NOT generalize: a broad grid (13 images, q5–95, 4 configs, m0–6 = 4004 cells)
is only **972/4004 (24%)** byte-identical — two configs never match and identity
oscillates 12–34% across q. The fixes here are correct where traced (q75/382297)
but bit-exactness across q/config/content is open work. Measurement + next-step
breakdown: `byteparity_scope_2026-07-14.md`. #38 stays OPEN.
