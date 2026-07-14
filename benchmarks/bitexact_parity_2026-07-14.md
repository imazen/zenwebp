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
