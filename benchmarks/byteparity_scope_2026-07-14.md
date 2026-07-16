# StrictLibwebpParity byte-identity — actual scope (2026-07-14)

## TOOLING NOTE (2026-07-15, read first)

The harness that produced every number below lived in `/tmp` and **was wiped**
mid-session. It is now committed as `dev/byteparity_sweep.rs` (the score),
`dev/mbpixdiff.rs` (first-EMITTED-divergence finder), and the pre-existing
`dev/bitexact_diff.rs` (header fields + mode stream) — all wired as
`__expert` examples. Never rebuild this in `/tmp` again.

**The committed grid is COMPLETE: 4004/4004 = 100% byte-identical
(2026-07-16).** The path: 3578 (89.4%, harness committed) → +189 Cat5/Cat6
stat-node `44ae3a0` → +62 StoreMaxDelta I16-candidate `c9abe85` → +93 m0-m2
skip-proba StatLoop `46e2a2c` → +67 m5/m6 trellis-skip `a9fc2da` → +5
segment-quant libm-pow `9a6a289` → **+10 I4 tie-break in libwebp's enum order
(this commit) = 4004/4004**. The gate
(`tests/libwebp_byte_parity.rs`, CI `__expert` step) now pins regression
anchors for all four 2026-07-16 roots on top of the q75/tiny/q90 pins.

**Claim discipline:** what is proven is byte-exactness across THIS grid —
13 images × q{5..95} × 4 (sns,flt,segs) configs × m0-m6. Settings outside it
(filter_sharpness ≠ 0, partitions > 1, alpha, target_size, exotic content)
are unswept; widen the grid before widening the claim.
Neither number is comparable to the 3488/4004 = 87.1% below: 10 of
the 13 images are synthetic and their generator was lost with the /tmp wipe and
reconstructed differently, so the synthetic cells are simply different content
(the encoder was byte-identical across that particular pair of runs). **3767/4004
is the durable baseline going forward** — the grid is committed now; re-run the
tool for any before/after, and don't compare across grids.

### Setting-permutation validation (2026-07-16, later): every shared knob axis at 100%

After the base grid completed, `dev/byteparity_sweep.rs` gained a phase-2 axis
sweep over the previously-unswept settings both encoders expose. Six more
parity roots fell (each dumped live before fixing, all parity-gated except
the two marked BOTH):

| axis | before → after | root |
|---|---|---|
| filter_sharpness 1-7 | 196/1960 → **1960/1960** | libwebp's `SetupFilterStrength` READS `filter_hdr.sharpness` at the top and assigns it from config at the BOTTOM — the strength derivation always sees the previous call's value (0 at the default single pass). Dumped: config sh=1 still gives `base = qstep` (row 0). zen derived with the real sharpness; parity now derives with 0. Tuned keeps read-through (it filters with the sharpness the decoder will use). |
| quality edges q0/q1/q99/q100 | 864/1456 → **1456/1456** | Three roots: (a) **BOTH MODELS** — the single-segment base init looked up `DC_QUANT[qi]` unclipped for UV DC; only q0 reaches qi>117, where the encoder then quantized UV DC with step 157 while every decoder (incl. zen's own, which caps at 132) dequantizes with 132 — a real encoder/decoder mismatch. (b) libwebp only allocates diffusion buffers at quality ≤ 98 (`webp_enc.c:167`); zen diffused at q99/q100 too, flipping ~14% of UV picks (parity now gates; tuned keeps always-on, effect is noise-level up there). (c) the `StoreMaxDelta` gate compares libwebp's `rd->D`, which carries the flat-latch DOUBLING; zen gated on raw SSE (synth 17x17 q1 m5: 2304 vs 1152 straddled min_disto 1560). |
| partition_limit 30/60/100 | 123/252 → **252/252** | Two zenwebp-only mechanisms leaked into parity: a pl²-scaled I4 seed penalty (`base + base·pl²/400`; libwebp's partition_limit feeds ONLY `max_i4_header_bits`) and `partition_limit >= 100` gates that reroute m0-m2 onto entirely different paths (libwebp's RD_OPT_NONE never consults partition_limit — `RefineUsingDistortion` caps header bits with the partition_limit-independent `mb_header_limit`). |
| segments-3 + sns/filter extremes (incl. sns>0 with segs1) | → **1120/1120** | libwebp applies the uv_alpha-derived UV quant deltas at EVERY segment count; when its analysis pass doesn't run (m2+ / one segment) `uv_alpha` keeps the 0 default (`analysis_enc.c:372`), still yielding `uvac_delta=-4`-class deltas at sns>0. zen's single-segment path skipped the deltas entirely (dumped: q50 m4 sns80 segs1, lib `q_uvac=-4`, zen none, 24% UV picks flipped). Parity computes them; tuned adoption is a candidate (finer UV at high SNS). |
| alpha (RGBA) | 0/192 → 64/192 → **192/192** (2026-07-16) | Six roots, in landing order. (1) **BOTH MODELS**: zen keyed the ALPH chunk on the input LAYOUT; libwebp scans the pixels (`WebPPictureHasTransparency`) — opaque RGBA now encodes as a bare `VP8 ` file. (2) The full libwebp alpha pipeline port (filters × full VP8L × raw fallback, `src/encoder/alpha.rs`). (3) **BOTH MODELS**: the Huffman equal-count tie-break — zen's BinaryHeap left equal-count INTERNAL-node order to sift order; libwebp's `GenerateOptimalTree` sorted-array merge inserts a fresh internal node before every equal-count entry. Exact port in `generate_tree_with_min_count`; same-cost trees, different symbol assignment (dumped: green lens 269-272 swapped on the checker while the REFS were identical — the earlier "LZ77 length distribution" reading was wrong, as was "meta-huffman group count": the 15-vs-10 TREE stores were a SECOND CRUNCH CONFIG, disambiguated by `img=`/STREAM tags). (4) **BOTH MODELS**: `red_and_blue_always_zero` was computed but never consumed — libwebp skips cross-color entirely when R/B are constant-zero (every alpha-in-green plane); zen spent ~5 trees + a transform image on it and flipped the filter choice downstream. (5) **BOTH MODELS**: `WebPCleanupTransparentArea` (YUV flavor) was documented on `exact` but never implemented for lossy — mixed-alpha 8×8 blocks get visible-average luma smoothing, fully-transparent blocks flatten Y/U/V to the run's first pixel, run POST-conversion like libwebp; the MB padding must be re-replicated after (stale pre-cleanup edges broke every non-MB-aligned size). Checker VP8 layer: 1554 → 1058 B. (6) parity-gated: `WebPAccumulateRGBA` alpha-WEIGHTED linear chroma averaging for mixed-alpha 2×2 blocks (`kInvAlpha` verified = floor((1<<19)/a), computed not baked), and libwebp's hash-chain iteration accounting (heuristics consume iterations, pre-decrement chain walk, no stall budget, row-above gated `!low_effort`) via `Vp8lConfig::parity` — zen's tuned lossless keeps its stall-budget + always-on row-above (measured m0 wins). CI: `transparent_rgba_matches_libwebp`. |
| sharp_yuv | 0/96 → **96/96** (2026-07-16) | Closed by porting libwebp's SharpYUV library exactly (`src/encoder/sharpyuv.rs`): 10-bit fixed-point W/RGB-delta iteration, linear-light targets via baked 16-bit sRGB tables (dumped from an instrumented build — immune to libm `pow` drift), decoder-matched 9-3-3-1 filter, ≤4 passes with the padded-dims `3·w·h` exit, `kSharpYuvMatrixWebp` 16.16 final. Converter verified IDENTICAL vs `SharpYuvConvert` on 17 shapes + the full webpx ARGB flow; tiny images (<4 px) fall back to standard conversion like libwebp (`kMinDimensionIterativeConversion`). Also ADOPTED for the tuned `.sharp_yuv(true)` (+1.0..+1.8 zsim vs +0.18..+0.32 from zenyuv's converter; 1.5× faster than libwebp's SSE2 build). See `sharpyuv_port_2026-07-16.md`. |

Out of scope (no matched knob / architecturally different — documented in the
sweep header): target_size/target_PSNR (zen searches q in an outer loop,
libwebp inside StatLoop), pass>1, autofilter, filter_type, preprocessing
bits, multi-partition output, low_memory, emulate_jpeg_size, qmin/qmax.

The CI gate gained `permutation_axis_anchors_byte_identical` (sharpness,
q-edges, plim, segs1-SNS anchors) and `opaque_rgba_matches_libwebp_bare_vp8`.
`bitexact_diff`/`mbpixdiff` accept `[sharp] [plim]` args 7-8;
`zen38_driver` gained `[sharpness] [partition_limit]` args 9-10; new dump
hooks: `I16DBG`, `UVDBG`, `SMDBG`, `UVQDBG`/`ZUVQ` (chroma-DC diffusion
state), plus the earlier `REFRESHDBG2`/`LEVFINAL`/`TRELDBG`/`SKIPDBG`.

### Alpha-plane pipeline ported; remaining distance is VP8L-stream parity (2026-07-16, later)

`src/encoder/alpha.rs` now carries byte-faithful ports of libwebp's whole
`EncodeAlpha` pipeline — `QuantizeLevels` (f64 k-means, libwebp's
`alpha_levels = (q<=70) ? 2+q/5 : 16+(q-70)*8` mapping), the
horizontal/vertical/gradient prediction filters, `WebPEstimateBestFilter`,
`GetFilterMap`, and the `ApplyFiltersAndEncode` trial loop with per-trial
raw fallback and libwebp's exact header byte — wired in under
`StrictLibwebpParity` (`encode_alpha_libwebp_pipeline`, api.rs). The trial
payloads go through zen's VP8L encoder at libwebp's alpha operating point
(alpha in GREEN with R=B=0, `method = effort`, `quality = 8·effort`, or 100
at m6+aq100).

**Measured with the new `dev/alphadiff.rs` layer-isolating harness:** the
pipeline stages agree (try-maps match, preprocessing bits match, the
checker-alpha filter choice matches), but the PAYLOADS diverge at byte 0 and
libwebp's are 6-10× smaller (64×64 gradient alpha: lib 23 B vs zen 137 B;
checker: 32 vs 222). Re-encoding libwebp's OWN filtered plane through zen's
VP8L reproduces the gap — so the remaining distance is pure **VP8L-stream
divergence at the alpha operating point** (libwebp's low-quality VP8L path
picks radically better modes for these planes), and the filter-choice
differences are just each side's VP8L preferring different planes. Chunk B
of the alpha work = the VP8L parity loop (instrument `VP8LEncodeStream`'s
decisions — entropy mode, palette, cache_bits, huffman — in the trace tree
and converge zen's, same method as the lossy campaign).

### Alpha Chunk B part 1: full VP8L behind the ALPH payload (2026-07-16, later)

Root of the 6-10× payload gap found: zen's ALPH payloads NEVER used the full
VP8L pipeline — `encode_frame_lossless`'s `implicit_dimensions` branch (the
alpha plane) falls to a **literal-only** encoder (subtract-green + plain
literals, "fast but produces larger files"). libwebp's `EncodeLossless` runs
the complete `VP8LEncodeStream` (palette / predictors / LZ77 / meta-huffman)
and only skips the container header bits (signature + dimensions + version
live in `VP8LEncodeImage`, not in the stream).

Fix: `Vp8lConfig::omit_headers` starts the stream at the transform bits, and
the parity trial payload (`alpha_vp8l_payload_inner`) now calls the full
`encode_vp8l` with `exact=1, method=effort, quality=8·effort`. Measured
(alphadiff): gradient-alpha payload 137 → **26 B** (lib 23); checker 222 →
**32 B (size-identical to lib's 32)**; ALPH header/filter/preprocessing all
match; and zen's VP8L analysis decisions now line up with the instrumented
libwebp's (`VP8LDBG`/`ZVP8LDBG` dumps: same palette_size, histo_bits=5,
transform_bits=5, same entropy mode — Spatial on the raw gradient plane,
Palette on filtered — same MinimizeDelta sorting, same rb_zero). Remaining:
bit-level divergence from ~byte 4-5 inside the entropy-coded region
(huffman code-length coding / palette-image literals) — the next loop
iterations live inside zen's VP8L stream writer vs libwebp's
`StoreHuffmanCode`/`EncodeImageNoHuffman`.

**Tuned-adoption candidate surfaced (large):** the TUNED default's ALPH
payloads still use the literal-only fallback — 5-7× larger than the full
pipeline produces for the same lossless plane (137 vs 26 B on the probe).
Routing tuned alpha through full VP8L is a strict size win with identical
decoded pixels.

### Alpha bit-level loop: state after the A=0 fix (2026-07-16, later still)

`BITDBG` (both bit writers dump `(nbits, value)` records; `TREE` markers with
full code-length dumps at every huffman store) drove two iterations:

1. **A=0 alpha-in-green (fixed, `e535ac21`)**: `WebPDispatchAlphaToGreen`
   leaves A/R/B zeroed — the trial pixels are `0x0000aa00` with ALPHA=0, not
   opaque (hence libwebp's `exact=1`). zen built opaque RGB, flipping the
   palette image's alpha tree from `{0}` to `{0,255}`. With A=0 the palette
   image matches libwebp **bit-for-bit** and the checker payload is 31 vs 32
   bytes, first-diff at byte 5.
2. **Remaining two roots, precisely located** (checker cell, main
   packed-index image):
   * **Meta-huffman group count**: libwebp keeps TWO huffman groups for the
     main image (histo_bits=5 → 1×2 tiles on the 8×64 packed image); zen's
     clustering merges them into ONE. Different combine-cost decision —
     the known "histogram clustering differences" residual from the
     lossless work, now with a minimal reproducer.
   * **LZ77 length distribution**: green-tree lengths differ only on
     length-codes 269-272 (lib `3,3,4,4` vs zen `4,4,3,3` — swapped
     frequencies), i.e. the match LENGTHS zen's LZ77 emits differ slightly
     from libwebp's on this pattern. Distance trees identical.

**RESOLVED (2026-07-16, later): the axis is 192/192.** The image-boundary
tags (`img=` on TREE, `STREAM` markers, `REFDBG` op dumps) answered both
open questions in one pass: the 15-vs-10 TREE stores were NOT an
entropy-image/group difference — libwebp's `VP8LEncodeStream` runs a
SECOND CRUNCH CONFIG inside one stream (15 = palette 5 + main 5 + second
config's main 5), and the "LZ77 length distribution" divergence was NOT
in the refs at all (`REFDBG`: byte-identical op streams) but in the
Huffman equal-count tie-break assigning the same {3,3,4,4} length
multiset to different symbols. The clustering was never wrong. See the
alpha row in the axis table above for all six roots and where each
landed.

### Failure shape: NONE — 4004/4004 (2026-07-16)

#### SOLVED: I4 tie-break must follow libwebp's ENUM order (this commit, +10)

The last 10 cells (all real photos, all q80+, in 5 apparent "clusters") were
ONE root: exact I4 RD-score ties broken toward different modes. zenwebp's
`B_*` constants use the VP8 spec order (`LD=4, RD=5, VR=6`); libwebp's
internal enum permutes them (`RD=4, VR=5, LD=6`, `common_dec.h`). libwebp
keeps the FIRST minimum (strict `<`) iterating ITS order, so on an exact tie
between LD and VR libwebp picks VR while zen's "iterate 0..9" parity
tie-break picked LD. Traced at 382297 q80 m3 sns0 mb(26,16) blk3: LD
`(1564+6909)·12 + 256·171` == VR `(1282+7447)·12 + 256·159` == **145452**,
an exact tie; 264 MBs cascaded from that one sub-block. (The trace first
proved probas/level-costs identical via the REFRESHDBG2 block diff — blocks
1-4 matched, the divergence sat between refreshes — then I16DBG showed all
four I16 candidates byte-equal, isolating the I4 loop; the per-sub-block R
delta of exactly 140 = FLATNESS_PENALTY was a red herring, zen folds it
into the score differently but equivalently.) Fix: `LIBWEBP_I4_ORDER =
[0,1,2,3,5,6,4,7,8,9]` — libwebp's visit order in zen indices — used by all
three evaluator paths (sse2/wasm/scalar) under parity. Exact ties concentrate
at high q (fine quant → many near-equal candidates), which is why the tail
was all q80+.

Earlier the same day (see below): +93 m0-m2 skip-proba, +67 m5/m6
trellis-skip, +5 segment-quant libm-pow.

#### SOLVED: segment-quant pow approximation off-by-one (this commit, +5)

The whole synth_33x17 q90 sns50/segs4 cluster (m2-m6, the only m2 cell) was
ONE header field found by `bitexact_diff` in a single run: `seg_q zen=[12,12,
9,6] lib=[12,11,9,6]` (seg_lf followed). `compute_segment_quant` evaluated
`pow(QualityToCompression(Q), expn)` with zenwebp's fast polynomial
`pow`/`cbrt` approximations (~1e-10 relative error); libwebp uses the
platform libm, and the result feeds a truncation `(int)(127.*(1.-c))` — a
hair of error at an integer boundary flips the quant index by one. Fix
(parity-gated): `compute_segment_quant_libm` + `quality_to_compression_libm`
mirror libwebp's exact expressions via `libm::pow` (including the base
`pow(linear_c, 1./3.)` — libwebp does NOT call cbrt). The parity-only
`quality_to_quant_index_trunc` (segs1 base quant) now also routes through the
libm chain. Tuned default keeps the fast path. `mbpixdiff`/`bitexact_diff`
gained `synth:WxH:SEED` image specs reproducing the sweep's synthetic cells.

#### SOLVED: m5/m6 skip decision from simple quant, not trellis (this commit, +67)

At m5/m6 libwebp's skip is `is_skipped = (rd->nz == 0)` (`VP8Decimate`) — the
nz of the FINAL trellis quantization. zen decided the skip with
`check_all_coeffs_zero`, which re-quantizes the raw DCT coefficients with the
SIMPLE quantizer. The trellis derives its level0 candidates with a NEUTRAL
bias plus the `sharpen` term and then RD-scores keep-vs-drop, so it keeps
borderline coefficients the simple bias drops — on those MBs zen skipped what
libwebp coded. Traced at 1025469 q20 m5 sns0 mb(14,4) (the ONLY differing MB
of that cell): both sides' trellis produce the identical single AC level=1
(`TRELDBG`: in, ctx=1, lam=3249, out all match), but zen's simple-quant skip
test said all-zero and discarded it. Fix (parity-gated): record the actual
levels via `record_residual_tokens_storing` and derive the skip from
`stored_coeffs.is_all_zero` — for agreeing MBs the recorded tokens are
byte-identical EOB streams, so nothing else moves. `StoreMaxDelta` now also
fires BEFORE the skip test on this path (libwebp stores it at the end of
`PickBestIntra16`, before the skip decision). m5 39→4, m6 35→3.

Note the tuned default keeps the simple-quant skip test — switching it to the
trellis-derived skip is a tuned-adoption CANDIDATE (it fixes a real
encoder-side reference mismatch: the encoder reconstructs WITH the kept
coefficient, then codes a skip, so its intra predictions drift from the
decoder's pixels) but needs a size/zensim A/B before adoption.

Diagnosis tooling committed for the next tail: `REFRESHDBG2` (full per-finalize
stats+proba table dump, both sides, diffable line-by-line), `LEVFINAL` (final
recorded levels for one MB), `TRELDBG` (per-block I16-AC trellis in/out) — all
`mode_debug` + `TARGX`/`TARGY`-gated in zen, env-gated in
`libwebp--zen38trace`. The FDUMP diff proved the mid-pass refresh state was
IDENTICAL at the diverging MB (block 1 of 8 matched; blocks 2-8 diverged only
as the downstream cascade), which pinned the root inside the MB's own
quantization in one step.

#### SOLVED: m0-m2 `use_skip`/`skip_prob` (this commit, +93)

The single biggest divergence class of the 175: parity forced
`macroblock_no_skip_coeff = None` for EVERY method, justified by libwebp's
`assert(use_skip_proba == 0)` — but that assert is at `VP8EncTokenLoop` entry
(`frame_enc.c:816`) and only covers m3-m6. m0-m2 run plain `VP8EncLoop`, whose
`StatLoop` DOES finalize the skip probability. Three mechanisms had to be
reproduced exactly (`SKIPDBG` dumps from the instrumented tree, q5/m1/382297):

1. **`nb_skip` is counted over the stats subset, not the emission frame.**
   `OneStatPass` counts `VP8Decimate()` skips; m0 `fast_probe` shortens that
   pass to `total>200 ? total>>2 : 50` MBs (m1/m2 cover the whole frame). The
   stats-pass skip decisions equal the emission-pass ones at RD_OPT_NONE
   (`RefineUsingDistortion` uses fixed mode costs, never adapted level costs),
   so zen freezes its per-MB skip count at the existing
   `fast_probe_stat_limit` boundary (`fast_probe_skip_count`).
2. **`FinalizeSkipProba` divides by the FULL frame** (`mb_w*mb_h`), truncated
   integer division, NO clamp (`CalcSkipProba`), then
   `use_skip_proba = (skip_proba < 250)`. zen's tuned formula rounds and
   clamps to 1..254 — either alone flips cells near the threshold.
3. **The `size_p0 == 0` StatLoop bailout gates the skip finalize too.** With a
   single effective segment (sns0, or collapsed segs>1), `OneStatPass` returns
   `size_p0 = ΣH + segment_hdr.size = 0` at RD_OPT_NONE and StatLoop returns
   BEFORE `FinalizeSkipProba` — `use_skip_proba` keeps its `VP8DefaultProbas`
   value 0. Dumped: q5/m1/sns0/segs1 counts `nb_skip=53` but never finalizes
   (emits use_skip=0); q5/m1/sns50/segs4 finalizes `nb_skip=46/1024 →
   skip_proba=243, use=1`. Same quirk `compute_updated_probabilities` already
   reproduces for the coefficient probas — the skip arm now gates on
   `!segments_enabled` identically.

The earlier attempt (reverted at 3778/4004) enabled the flag with zen's
full-frame rounded/clamped count — the sns configs improved but every sns0
cell broke, exactly the (1)+(3) mechanisms. Verified per-cell with
`bitexact_diff` on q5 m0/m1/m2 across all four configs before sweeping.
`SKIPDBG` hooks (`FinalizeSkipProba`, `OneStatPass`, `VP8EncLoop` emit-skip
counter) are committed in `~/work/zen/libwebp--zen38trace`.

#### SOLVED: StoreMaxDelta gated on the final mode (`c9abe85`, +62)

libwebp runs `StoreMaxDelta` at the END of `PickBestIntra16`
(`quant_enc.c:1035`) — BEFORE `PickBestIntra4` can override the mode and before
the skip decision — so a macroblock whose I16 candidate is blocky-and-high-D
feeds `max_edge` even when finally emitted as I4 or skipped. zenwebp gated on the
FINAL mode and dropped `intra16_d` for I4 macroblocks, under-counting `max_edge`,
so `VP8AdjustFilterStrength` failed to bump the segment's loop-filter level and
the segmentation header shipped a wrong `seg_lf`.

Traced at 382297 q5/m4/sns50/flt60/segs4, where EVERY mode decision already
matched (y 100%, uv 100%, b4 714/714, seg 100%) and the only divergence was one
header field:

```
seg_lf:   zen=[50, 29, 63, 60]   lib=[50, 55, 63, 60]
max_edge: zen=[0, 1, 3, 4]       lib=[0, 2, 3, 4]
```

ONE missed macroblock: `max_edge[1]` 1 vs 2 (delta 27 → no bump, vs 55 → bump).
Fixed by carrying the I16 candidate's Y2 + blocky flag + `intra16_cand_d` out of
`pick_best_intra16`. That also retired the m5 trellis workaround, since the
blocky test now comes from the same place libwebp's does.

**Method note:** `dev/mbpixdiff.rs` and `dev/bitexact_diff.rs` were BOTH
hardcoded (sns0/flt0/segs1, and q75 respectively) and could not reach this
cluster at all; `bitexact_diff` was additionally unbuildable (it `include!`s a
`coeff_update_probs.rs` that did not exist). Both are now parameterised
`[image] [q] [m] [sns] [flt] [segs]`. `bitexact_diff` found this root in ONE run
by printing the header-field diff — reach for it whenever `mbpixdiff` reports a
small `1st-diff@` offset, which means a header field rather than content.

What this says now. (1) The **sns=0 configs are essentially closed** (18/1001
each, ~98% identical) — the remaining 201 of 237 live in the **SNS + filter +
multi-segment** configs, which is now overwhelmingly the dominant axis. So the
next work is segment-dependent quantisation / filter-strength interaction, not
coefficient rate. (2) The m3/m4 cluster collapsed (70→23, 68→21) with the
Cat5/Cat6 fix; m5/m6 (52/47) retain a trellis-flavoured residue. (3) All 10
synthetic images are now byte-identical except 6 cells on synth_256x255 — the
remainder is real-photo content.

**(SOLVED — see "Cat5/Cat6 stat-node mismatch" below; kept for the trace method.)**
The high-q I4 coefficient *rate* divergence. At q90
m3 (sns0/segs1, 382297) the first emitted divergence is mb(28,7); I4 sub-blocks
0-3 match exactly, then blk4 diverges with the SAME mode (VR), SAME D (36) and
SAME mode-cost H (1667) but R=16752 (zen) vs 16730 (lib) — a 22-unit
coefficient-rate delta that then cascades (blk11 flips RD vs DC). Since D
matches exactly and dequantisation is injective, the *levels match* — so this is
a rate-computation divergence, not quantisation. blk4 is the first sub-block
carrying a large level (12 → cat3, range 11-18), which is why q75 never sees it:
higher quant keeps levels small enough to avoid the category-coded range.
**Two candidates already REFUTED (2026-07-15) — do not re-check:**

1. **`VP8_LEVEL_FIXED_COSTS` is byte-identical to libwebp's
   `VP8LevelFixedCosts`** (`libwebp/src/dsp/cost.c`). Entry 12 = 901 in both;
   the whole leading run matches (`0, 256, 256, 256, 256, 432, 618, 630, 731,
   640, 640, 828, 901, 948, 1021, 1101, 1174, 1221, 1294, 1042, …`).
2. **zen's level-cost precomputation matches `VP8CalculateLevelCosts`**
   (`libwebp/src/enc/cost_enc.c`) line for line:
   `cost0 = ctx>0 ? BitCost(1,p[0]) : 0`; `cost_base = BitCost(1,p[1]) + cost0`;
   `table[0] = BitCost(0,p[1]) + cost0`; `table[v] = cost_base +
   VariableLevelCost(v,p)` for `v in 1..=MAX_VARIABLE_LEVEL`; band remap via
   `VP8EncBands`. zen's `variable_level_cost` matches `VariableLevelCost`
   including the `VP8LevelCodes` pattern/bits walk from proba index 2 (zen's
   `.min(MAX_VARIABLE_LEVEL)` clamp is equivalent — libwebp's caller never
   exceeds it). zen's padding of `MAX_VARIABLE_LEVEL+1..127` with the max-level
   cost is a bounds-check optimisation, not a semantic difference.

So the rate *formula* and its tables are right. Since blk0-3 match with the same
`level_costs` and the same ctx=2, the **probabilities also match at that MB** (a
proba divergence would move every sub-block's R, not just blk4's).

**That leaves the levels themselves.** The "D matches ⇒ levels match" inference
is the weak link: D is an SSE, so two different level sets *can* coincide at
D=36. If the levels do differ, this is a **quantisation-boundary divergence**
at m3 (simple quant: libwebp `VP8EncQuantizeBlock` vs zen's
`quantize_block_simd`) that only shows on blk4's residual — the round-vs-
truncate class of bug that already bit the base quantiser (`52cf96f2`).

### CORRECTION (2026-07-15, later): the StatLoop claim below was WRONG

An earlier revision of this file (commit `f1844f4`) claimed the root cause was
"zen never runs libwebp's StatLoop" and that zen scores RD against DEFAULT
probabilities. **That is false and the commit message is wrong too.** zen's
mid-row refresh (`mod.rs:1632-1645`) does fire and does adapt, and its
`max_count = (num_mb/8).max(96)` already matches libwebp's `(mb_w*mb_h)>>3`
floored at `MIN_COUNT 96` (both = 128 for a 32×32-MB frame).

Measured with a `REFRESHDBG` probe, zen's refresh at mb(0,4) (MB 128) produces
`prob[3][1][2] = [17, 51, 85, 127, 172, 180, 152, 123, 231, 155, 128]` against
libwebp's in-effect `[17, 51, 85, 127, 172, 180, 152, 123, 231, 164, 128]`.
**Nine of eleven entries are identical; only index 9 differs (155 vs 164).**

That single entry explains everything: index 9 is one of the two probability
slots that ONLY a level-12-class coefficient reaches (level 10 stops at 7), which
is why exactly one `t` moved and by only 22. zen's next refresh is mb(1,8) =
MB **257**, i.e. *after* the divergent mb(28,7) = MB 252, so blk4 is scored with
the MB-128 values.

### SOLVED (2026-07-15): libwebp's Cat5/Cat6 stat-node mismatch — fixed in `44ae3a0`

**Everything below in this subsection is SUPERSEDED.** The "schedule difference"
conclusion was wrong, as was the StatLoop theory before it. Kept only as a record
of the dead ends.

The real cause: **libwebp has TWO coefficient recorders that disagree, and picks
between them by which encode loop runs** (`use_tokens = rd_opt >= RD_OPT_BASIC`):

* **m3-m6** → `VP8EncTokenLoop` → `VP8RecordCoeffTokens` (`token_enc.c`), which
  codes Cat5/Cat6 with proba index `base_id + 10` but records the STATISTIC into
  `s + 9`: `AddToken(tokens, 0, base_id + 10, s + 9)`. Cat3/Cat4 pair
  `base_id + 9` with `s + 9` consistently — only Cat5/Cat6 mismatch. Upstream
  consequence: `stats[..][10]` is never populated at all, so node 10's
  probability can never adapt, and node 9's count merges Cat3/4/5/6.
* **m0-m2** → plain `VP8EncLoop` → `RecordResiduals` → `VP8RecordCoeffs`
  (`cost_enc.c`, `USE_LEVEL_CODE_TABLE`), whose `VP8LevelCodes` pattern records
  node **10** for v>=35.

zenwebp recorded node 10 for every method — the natural reading, and arguably
more correct; the upstream split looks like a slip, not intent. But these counts
feed `FinalizeTokenProbas`, so byte-exactness requires reproducing libwebp's
accounting **per path**. Fixed via `TokenBuffer::cat56_stat_node9`
(`parity && method >= 3`); tuned default keeps node-10 and is byte-unchanged.

**Result: 3578/4004 (89.4%) → 3767/4004 (94.1%), +189.** m3 70→23, m4 68→21,
m5 99→52, m6 95→47; m0-m2 unchanged. Applying the token-path rule to ALL methods
first gave 3702 and REGRESSED m0-m2 (20/38/36 → 36/69/54) — that regression is
what exposed the per-path split.

Also settled here: **StatLoop's `FinalizeTokenProbas` never runs at m3** — it
early-outs on `if (size_p0 == 0) return 0;`. All 8 finalizes come from the token
loop (7 mid-refreshes + 1 pre-emit), confirmed by tagging each call site. So the
StatLoop port is NOT needed for this; #27 stands on its own merits.

Gate: `tests/libwebp_byte_parity.rs` (+ a CI step with `--features __expert`;
CI previously never ran `__expert`, so the claim had no gate at all).

---

#### (superseded) "It is a SCHEDULE difference, not a code difference"
Dumping `stats[3][1][2][9]` as `(nb,total)` at every `FinalizeTokenProbas`,
libwebp at q90/m3 calls it **8 times with CUMULATIVE, never-reset stats**:

```
(30,84) → (77,198) → (146,367) → (207,538) → (260,710) → (270,741) → (276,763) → (278,772)
```

`CalcTokenProba(nb,total) = 255 - nb*255/total`, so:
* call #1 → `255 - 30*255/84` = **164** — and 164 is exactly libwebp's LIVE value
  at blk4. So blk4 is scored with libwebp's **first** finalize, which is
  **StatLoop's**, computed over the m3 `fast_probe` subset (`nb_mbs >> 1` = 512
  MBs), *before* the encode loop starts.
* call #2 → `255 - 77*255/198` = 156 ≈ **zen's 155**.

So zen's first refresh (MB 128, mid-encode) lands near libwebp's *second*
finalize, while libwebp is still using its *first* (StatLoop's). The two
encoders adapt on different schedules over different MB subsets — the recorders
and the math agree. (The recorders are provably equivalent for v=12: libwebp's
`USE_LEVEL_CODE_TABLE` path records `s+3(1), s+6(1), s+8(0), s+9(0)`, and zen's
explicit-tree `record_coeffs` (`cost/stats.rs:149`) records the same.)

**So StatLoop does matter — but for its SCHEDULE, not because zen "uses
defaults".** The port must reproduce: (a) a pre-encode stats pass over the right
subset (`fast_probe`: m3 → `nb_mbs>>1`|100, m0 → `nb_mbs>>2`|50, m4-m6 → full;
disabled by `do_search`) at `rd_opt = RD_OPT_BASIC` for m3+, (b) one
`FinalizeTokenProbas` + `VP8CalculateLevelCosts` from it, and (c) stats that
then CONTINUE accumulating into the encode loop's periodic refreshes rather than
starting from zero. Point (c) is the surprise: the observed totals only grow, so
the encode loop's refreshes finalize from StatLoop's counts plus their own.
Resolve how that squares with `ResetTokenStats` in `VP8EncTokenLoop` before
porting — it is the one piece that reading the source did not settle.

**Tool:** `REFRESHDBG=1` (mode_debug) prints zen's per-refresh probs + raw
`(nb,total)` stats; the matching libwebp probe is `LIBFINALIZE` in
`~/work/zen/libwebp--zen38trace`.

The StatLoop material below is retained because it is accurate ABOUT LIBWEBP and
the port constraints are real — but it is NOT the root cause of this gap.

### (superseded) StatLoop analysis

**libwebp calls `StatLoop(enc)` UNCONDITIONALLY as the first statement of
`VP8EncLoop`** (`frame_enc.c`), for every method: it sweeps the frame collecting
token statistics, finalises the coefficient probabilities, and rebuilds the
level-cost tables before the encode loop. zen has no equivalent pre-pass; it
reaches adapted probabilities only via the mid-row refresh (which, per the
correction above, does work).

**Evidence chain (q90/m3, 382297, mb(28,7), I4 sub-block 4):**
1. Levels are byte-identical: `[-10,12,5,9,-9,-2,0,2,-6,2,0,-1,0,0,-1,-1]` in
   both → quantisation is correct, this is purely a RATE divergence.
2. Per-coefficient dump: every `t` matches (1251, 861, 740, 1023, 586, 479, …)
   EXCEPT n=1 (level 12, band 1, ctx 2): **lib t=1064 vs zen t=1086 = the Δ22**,
   which then propagates to the block total (16730 vs 16752).
3. Level 12 maps to `VP8LevelCodes[11] = {0x0d3, 0x013}`, whose pattern is the
   only one reaching probability indices **8 and 9** — level 10 stops at 7. So a
   single wrong pair of probabilities moves exactly one `t` and nothing else,
   which is why the divergence looked impossibly narrow.
4. Live probability dump at that moment, `prob[band1][ctx2]`:
   * libwebp `[17, 51, 85, 127, 172, 180, 152, 123, 231, 164, 128]` (adapted)
   * zen     `[39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128]` (**the
     shipped defaults** — `src/common/types.rs:668`, byte-identical to
     libwebp's `VP8CoeffsProba0[3][1][2]`)

**Everything else on this path is already correct and was verified, not assumed:**
`VP8_LEVEL_FIXED_COSTS` == `VP8LevelFixedCosts`; `VP8_LEVEL_CODES` ==
`VP8LevelCodes`; the level-cost precomputation == `VP8CalculateLevelCosts`;
`get_residual_cost` == `GetResidualCost_SSE2` (same clamps, same
`t = costs[n+1][ctx]`, same EOB tail); `ExpandMatrix`/`zthresh`/`QFIX`/`BIAS`;
the default probability table itself.

**Why this is THE root of the remaining ~426 cells**, and consistent with the
whole failure shape: m0-m2 barely use cost-driven decisions and are nearly
closed (94/426); m3-m6 are exactly the RD-optimising methods and carry 332/426.
High-q hurts more because there are more (and larger) coefficients for a wrong
rate to mis-score. This is the **#27 StatLoop** area — the skip-proba half was
already closed (`91c96168`), but the probability-adaptation half was never
ported.

### Porting StatLoop: one attempt MEASURED AND REJECTED (2026-07-15)

zen already owns most of the parts — `compute_updated_probabilities()` IS
`FinalizeTokenProbas` (including the m0 `fast_probe` subset snapshot), and a
two-pass mechanism exists at m4 whose pass 1 applies `updated_probs` and rebuilds
`level_costs` (`mod.rs:1266-1278`). **Naively enabling that two-pass under parity
for every method regressed the grid 3578/4004 → 2299/4004** (m3-m6: ~70-99
failures each → ~400, i.e. nearly every cell). Reverted; do not re-attempt by
flipping the pass count.

**Why it fails — zen's two-pass is not libwebp's StatLoop:**
* libwebp's stats pass runs at `rd_opt = (method >= 3 || do_search) ?
  RD_OPT_BASIC : RD_OPT_NONE` — so m5/m6 collect stats with **BASIC, not
  trellis** — and it never emits.
* `FinalizeTokenProbas` always writes either the stats-derived `new_p` or
  `VP8CoeffsProba0[t][b][c][p]` — it **never carries a previously-adapted value
  forward**. So "adapted" state is recomputed from whatever stats currently
  exist, not accumulated.
* `VP8EncTokenLoop` then calls `ResetTokenStats` **only on the last pass**, and
  re-finalizes probs + level costs every `max_count` MBs from *that pass's own*
  stats.

**The live puzzle to solve first (cheap, do this before porting anything):**
zen's `max_count = (num_mb / 8).max(96)` (`mod.rs:1335`) already matches
libwebp's `(mb_w*mb_h)>>3` floored at `MIN_COUNT 96` — for 32×32 MBs both are
**128**. So zen's mid-row refresh *should* have fired at ~MB 129 and left
mb(28,7) (MB 252) with adapted probabilities. It didn't: the dump shows the
shipped defaults. **Find out why that refresh is a no-op at m3 before building
StatLoop on top of it** — either it isn't firing, or
`compute_updated_probabilities` is rejecting every candidate and falling back to
`COEFF_PROBS`. That single answer may be worth more than the whole port.

Groundwork landed: `self.do_search` now mirrors libwebp's
`enc->do_search = (target_size > 0 || target_PSNR > 0)` (`webp_enc.c:118`), which
is what gates `fast_probe`.

**Constraint for any port:** it must be parity-gated. zen's tuned default
deliberately skips mid-row level-cost rebuilds because they measurably regressed
compression (1.0101x→1.0114x, `mod.rs:1577-1580`), so a StatLoop that changes the
tuned default's bytes is NOT acceptable without a fresh A/B.

**Tooling:** the instrumented libwebp used for this lives at
`~/work/zen/libwebp--zen38trace` (a copy — the reference tree at
`/home/lilith/work/webp-porting/libwebp` is READ-ONLY and was not modified).
`ZTRACE`/`RCDBG` env hooks + `zen38_driver.c` + `build_zen38.sh` are in it. Note
`cargo test` captures stderr — pass `-- --nocapture` or debug prints vanish.

## STATE AT THE TIME OF THE ANALYSIS BELOW: 3488/4004 = 87.1% byte-identical

Five parity-gated fixes this session took the grid **24% → 87.1%**: base-quant
`52cf96f2`, segmentation-collapse `41923466`, trailing-slots `7acdd775`,
**skip-proba `91c96168`**, **I16-AC-trellis nz-context seed (this commit, +81
cells)**. The skip-proba one was NOT a StatLoop rearchitecture (I wrongly called
it deep and stopped — it was a one-line gate): instrumented libwebp always writes
`use_skip_proba = 0` (there's an unconditional `assert(use_skip_proba == 0)` at
`VP8EncTokenLoop` entry; the flag is never enabled in the shipping encoder), so
parity just forces `macroblock_no_skip_coeff = None`. Closed the whole low-q
cluster (+256 cells).

**I16-AC-trellis nz-context seed (the m6 root cause).** At m6
(RD_OPT_TRELLIS_ALL) zen's `pick_best_intra16` trellis-quantizes each I16
candidate's AC blocks (`mode_selection.rs:793`). It seeded the per-block nz
context `top_nz_t`/`left_nz_t` to **all-false**; libwebp's `ReconstructIntra16`
calls `VP8IteratorNzToBytes` (`quant_enc.c:826`) FIRST, so its trellis context
`ctx = it->top_nz[x] + it->left_nz[y]` (`:829`) uses the REAL neighbouring-MB
coefficients. For an MB whose neighbours carry coefficients, the top-row /
left-column blocks get ctx0=0 in zen vs 1-2 in libwebp → different trellis
level-costs → different keep/drop → a candidate's D **and** R both shift. **Traced
end-to-end (q40 m6, 382297, first emitted-pixel divergence = mb(11,8)):** zen and
libwebp agreed exactly on I16 DC/TM/V, but the H candidate diverged (zen
D=3062/R=22825 vs lib D=2734/R=25473) with a byte-identical H_PRED left column
`[44,44,44,44,71,50,36,39,40,40,40,40,41,41,41,41]` — proving the prediction
matched and only the trellis differed. The wrong H candidate won zen's I16
(raw score 401464640 < DC 429540099) but lost I4-vs-I16 in FINAL terms
(1212032 > 1172862), so zen emitted I4 where libwebp emitted I16-DC. Seeding the
context from the real neighbour nz (zen's `top_complexity`/`left_complexity`,
same source the I4 path already used) makes that cell byte-identical. Parity-gated
(`seed_ctx = cost_model == StrictLibwebpParity`); the tuned default keeps the
all-false seed and is byte-unchanged by construction.

**Method to find it (reusable): `mbpixdiff`** — decode BOTH bitstreams and diff
per-MB pixels to find the first EMITTED divergence, instead of chasing per-MB
debug prints (which mix the ~4 non-emission probe calls per MB and mislead — my
earlier mb(3,0) trace was a probe, not the emission). The first differing MB is
the clean root; trace only that one.

**Remaining ~15% is luma I4/I16 mode-RD**, in two overlapping clusters:
- **m6, all q (~207 cells).** The m5→m6 delta is trellis-during-I4-mode-selection
  (RD_OPT_TRELLIS_ALL): q40 m5 has `b4_same` 832/832 (all I4 sub-modes match) but
  q40 m6 has 377/768 + `y_same` 90.8% + i4-count off by ~4. **Traced (q40 m6,
  382297):** it's a per-MB coefficient-RATE cascade, NOT a lambda/DP bug. The
  I4-trellis lambdas match (`(7·q²)>>3`), block-score lambda matches
  (`lambda_mode=18`), D/SD/H/modes match — only R diverges, driven by an nz-CONTEXT
  difference (mb(11,8) sub-block 0: zen `ctx=1 t1+l0`, lib `ctx=2 t1+l1`). The R
  drift starts BEFORE the first mode flip (mb(10,8), not a flip, already has zen
  I16 R≈56210 vs lib 56470), so it cascades MB-by-MB. **Root traced to mb(4,0)**
  (an all-MB I16-D/R diff found it as the first diverging MB in raster order).
  At mb(4,0) sub-block 0, zen and libwebp have the SAME incoming context (`ctx=0
  t0+l0`) and matching lambdas — yet zen's I4 sub-mode selection picks HU while
  libwebp picks TM. Both winning sub-modes are **nz=0** (empty block), so the
  coefficient rate should be mode-independent, yet zen's R=89 vs libwebp's R=229.
  So the divergence is the **per-candidate-mode I4 *trellis rate* at m6** — the
  deepest layer: lambda (`lambda_i4=(3·q²)>>7=56`, matches), context, trellis DP
  (byte-identical at m5), and D are ALL ruled out. Next: per-mode trellis-rate
  dump at mb(4,0)#0 (same-mode zen-vs-lib) to pin the rate-accounting difference;
  it's mode-independent-looking (nz=0 R differs), so likely one systematic fix
  closes the cluster. Instrumented libwebp `LIBI16`/`LIBI4blk`/`LAMDBG`/`A16`/
  `LIBMODE`/`CTXDBG` hooks + zen `MB_DEBUG` are in place in the scratchpad.

  **Exhaustive trace update (correcting the above):** the "nz=0 R=89 vs 229" was a
  MISREAD — libwebp's 229 = flatness-penalty 140 + `VP8GetCostLuma4(empty)=89`,
  which MATCHES zen's 89. So the empty-block rate is NOT the divergence. The real
  root is the **I4-vs-I16 decision at mb(3,0)** (index 3, row 0): zen picks I4 (I4
  score 863058 < I16 869604, margin **6546**), libwebp picks I16 (its I4 running
  exceeds I16). zen's mb(3,0) I4 win makes sub-block 3 = HU, which becomes mb(4,0)'s
  left mode-context (`left_ctx=9` where libwebp has `0`), cascading. So the divergence
  is a **tiny-margin I4/I16 RD tie at m6**, from small per-sub-block RD deltas — NOT
  a systematic lambda/table/rate bug (all of those are verified matching:
  `lambda_i4=56`, the permuted `VP8_FIXED_COSTS_I4` (DC/HU costs match at default
  ctx), `into_intra` I16→context mapping, and the empty-block rate). The FLIPS tool
  missed mb(3,0) because its EMITTED mode may still match (multi-pass: the
  context-building pass picks I4, emission may differ) — a subtlety to resolve next.
  **RESOLVED (this commit) for the dominant m6 mechanism:** the m6 divergence was
  the **I16-AC-trellis nz-context seed** (see the CURRENT STATE section above) —
  zen seeded the mode-selection trellis context all-false where libwebp uses the
  real neighbour nz. Fixing that took m6 from part of the 15% tail to a much
  smaller remainder. The earlier "bit-level ~16-unit coefficient-rate at mb(3,0)"
  trace below was a **NON-emission probe call** (zen calls `pick_best_intra*` ~4×
  per MB with evolving state; `MB_DEBUG` mixes them), NOT the emission root — the
  real first emitted divergence was mb(11,8). Lesson kept for the next tail:
  **use `mbpixdiff` (decode both, diff per-MB) to find the first EMITTED
  divergence; trace only that MB.** The stale probe-call trace is retained below
  only as a cautionary example.

  **Stale probe-call trace (cautionary — NOT the emission root):** mb(3,0)
  sub-block scores showed zen running=96252 vs libwebp 95966. This was a probe
  call; zen actually emits I16 at mb(3,0) matching libwebp. Do not chase per-MB
  `MB_DEBUG` output without first confirming (via `mbpixdiff`) that the MB is a
  real emitted divergence.
- **High-q q80–95, m3–m5.** Milder luma mode flips (`y_same` ~97%) + `n_proba_updates`
  off by a few. The n_proba is DOWNSTREAM of modes (when modes match, n_proba
  matches — verified at q40 m5), so the mode-RD is the root.

Both are per-MB RD-score matching (the genuinely harder tail), but NOT assumed
deep — trace each flip against instrumented libwebp before concluding.

## TL;DR

**(2026-07-16: COMPLETE — the grid below is now 4004/4004 = 100%; see the top
of this doc.)** Historical framing kept for provenance: the #38 parity work
initially made `CostModel::StrictLibwebpParity` byte-identical to libwebp only
at the specific operating point it was traced against (q75, CID22 382297, two
configs), across all 7 methods. The broad grid below started at **972/4004
(24%)** byte-identical; everything from here down records the climb.

**Update (2026-07-15): trailing-segment-slot fix → 3151/4004 (78.7%).** The VP8
segment header always carries 4 quant+filter slots; libwebp leaves the slots
beyond the configured count (`[config..4]`) at base/seg0 (a 2-segment encode has
`dqm[2]==dqm[3]==dqm[0]`), zen held seg1's values. Fix (parity-gated,
`vp8/mod.rs`): point `[config..4]` at seg0 using the pre-simplify count (so
`SimplifySegments`' correct `[num_final..config]` replication isn't clobbered).
sns30/segs2 **30.4% → 78.8%**; all four configs now converged at **78–79%** — the
residual is now a COMMON cluster (low-q skip/quant, high-q, m6 mode-RD),
config-independent. Commit `7acdd775`.

**Update (2026-07-15): segmentation-collapse fix → 2666/4004 (66.6%).** Biggest
single jump. libwebp writes `segmentation_enabled = (num_segments > 1)` *after*
`SimplifySegments` merges equivalent segments; at sns=0 the SNS quantizer spread
is 0, so all segments are uniform → libwebp collapses to 1 and turns segmentation
OFF. zen set `segments_enabled` unconditionally, emitting a full 4-segment header
where libwebp emits none — so the entire sns0/segs>1 config diverged. Fix
(parity-gated, `vp8/mod.rs`): `segments_enabled = num_segments > 1`. Per-config:
sns0/segs4 **0% → 78.3%** (now identical to sns0/segs1), sns50/segs4 **48.6% →
79.3%**, sns30/segs2 **0% → 30.4%**, sns0/segs1 78.3% (unchanged). q75 14/14 held;
tuned byte-unchanged. Commit `41923466`. Tuned-adoption candidate (strict
byte-saving when it fires) pending a sweep.

**Update (2026-07-15): base-quant round→truncate fix → 1270/4004 (32%).**
First generalization step: `setup_encoding` computed the segs1 base quant with
`quality_to_quant_index` (which **rounds** `127*(1-c)`), but libwebp truncates
(`VP8SetSegmentParams`). They diverge by +1 at q10/30/50/80 (frac ≥ 0.5), which
q75 (26.20, rounds==truncates) could never expose. Parity now uses
`quality_to_quant_index_trunc`. Effect on the clean **sns0/segs1** config:
**48.6% → 78.3%** identical (q30 0→6/7, q50 0→7/7; q10 and q80 still carry a
*second* divergence). Tuned default byte-unchanged, q75 still 14/14. The
remaining open axes below are unchanged (segs4 mechanism, q5/q10/q80 second
divergence, high-q q90/q95).

## Why the 14/14 was misleading

`methodcmp` (the gate that reported 14/14) fixes **q=75** and **one image**
(382297) and tests exactly two configs: (SNS=0, filter=0, segs1) and (SNS=50,
filter=60, segs4). Those are the settings every differential trace this session
used (`TARGX`/`TARGY` at q75 on 382297). So the fixes are tuned to make *that
point* byte-exact; nothing forced the other points to converge.

## The broad sweep

`dev`-style bin `byteparity_sweep` (in the scratchpad harness): 13 images
(3 CID22 512² + 10 synthetic incl. 1×1, 2×2, 3×3, 17×17, 33×17 odd-chroma,
edge-partial MBs) × q ∈ {5,10,20,30,40,50,60,70,80,90,95} × 4 configs
{(0,0,1),(50,60,4),(0,0,4),(30,20,2)} × m0–6 = **4004 cells**. The 3032-line
raw fail list (196 KB) is reproducible via the `byteparity_sweep` bin and is not
committed (>30 KB); the breakdown tables below are the durable record.

**972 / 4004 = 24.3% byte-identical.**

### By config (of 1001 each) — the dominant axis
| config (sns/flt/segs) | % identical |
|---|---|
| (0, 0, 1)   | 48.6% |
| (50, 60, 4) | 48.6% |
| (0, 0, 4)   | **0.0%** |
| (30, 20, 2) | **0.0%** |

The two `methodcmp` configs sit at ~49%; the two it never tested are **0%**.
`(SNS=0, segs4)` and `(SNS=30, segs2)` never produce a byte-identical file at any
q/method/image — a concrete segmentation/SNS-interaction divergence to trace.

### By quality (of 364 each)
q5 31% · q10 12% · q20 32% · q30 13% · q40 33% · q50 12% · q60 32% · q70 34% ·
q80 13% · q90 29% · q95 25%. Clear odd/even oscillation (odd steps ~12%, even
~32%) — a quantizer-index parity effect. **No q exceeds 34%** (q75 itself, the
traced point, is not in this grid — it is the outlier at 100% for the two tested
configs).

### By method (of 572 each)
m0 29% · m1 29% · m2 30% · m3 22% · m4 22% · m5 21% · m6 17%. The RD tiers (m3–6)
are worse — more places to diverge.

### By image (of 308 each)
382297 24% (the traced image, best of the reals) · 1025469 8% · 1418519 11% ·
tiny synth 16% · larger synth 27–40%. Other CID22 images are far worse than the
one that was traced — direct evidence the fixes are 382297-specific.

### First divergence
2462 of 3032 diffs first differ in the header (byte < 8 → the RIFF/VP8 size
field, i.e. total size differs); 570 match a prefix then diverge in content.

## What this means

- The parity fixes are correct **where traced** and buy real value there (a
  user encoding at ~q75 with those configs gets byte-identical output). But the
  mode "byte-identical to libwebp" claim only holds at that point.
- Generalizing is open work along three axes, most tractable first:
  1. **Config:** `(SNS=0, segs4)` and `(SNS=30, segs2)` are 0% — a specific
     segment-quantizer / SNS-derivation divergence. Trace like the q75 work but
     at those configs.
  2. **Quality:** the odd/even q oscillation points at quantizer-table index
     rounding (`VP8SetSegmentParams` / `q` → `qi` mapping). One fix likely lifts
     many cells.
  3. **Content:** once config+q converge on 382297, re-sweep the other CID22 +
     synthetics.
- **Do not re-report 14/14 as "parity complete."** Use this grid's 24% as the
  headline number and drive it up.

## Default config (sns50/filter60/segs4) — diagnosis (2026-07-15)

Per the scope decision, the Default preset was investigated first.

### Two facts established with a real-libwebp-C oracle

The vendored libwebp source (`scratchpad/lwsrc/`, `trace_driver`) was rebuilt
with instrumentation and used as ground truth:

1. **`webpx` == real libwebp C, byte-for-byte.** At q30 m4 sns50 segs4, zen ==
   webpx == `trace_driver` output (all 27944 bytes identical). So the
   `webpx`-based byteparity baseline *is* canonical libwebp — the grid numbers
   are trustworthy.
2. **The base-quant fix (`52cf96f2`) closed far more than the 4004-grid +8%
   implied for this config.** The Default preset on 382297 is now **47/77 (61%)**
   byte-identical, not ~49%.

### METHODOLOGY WARNING — rebuild every harness binary after a lib change

The first diagnosis pass (committed then corrected here) chased a **phantom**:
`segfielddiff` was built *before* the base-quant fix, so its zen encode still
rounded the q30 base quant (53 vs the fixed 52), producing a fake `seg_lf`
divergence at q30 **m4**. The 3-way oracle proved q30 m4 is fully
byte-identical. The harness (`webp-ll-compare`) builds zenwebp into its *own*
target, so rebuilding the lib in the zenwebp repo does **not** refresh a harness
binary — you must rebuild the binary. Always rebuild ALL harness bins
(methodcmp, byteparity_sweep, segfielddiff, …) after any lib edit before trusting
their output. (Also: `bitexact_diff`'s field diffs print *before* their method's
summary line — attribute each diff to the method whose summary follows it.)

### Actual current divergence map (382297, post base-quant fix)

```
q\m  0 1 2 3 4 5 6      . = byte-identical
  5  X X . X X X X      m2 is 100% identical across all q
 10  X X . . . X X      m0/m1 identical except low-q (skip_prob)
 20  . X . . . X X      m3/m4 identical mid-q
 30  . . . . . X X      divergences cluster at LOW-q and HIGH-q,
 40  . . . . . . .        consistent with parity traced at q75
 50  . . . . . . .
 60  . . . X . . .
 70  . . . . X . X
 80  . . . X X X X
 90  . . . X X X X
 95  . . . X X X X
```

### Real remaining causes (distinct, q-regime-clustered)

- **m0/m1 low-q `use_skip`/`skip_prob`** (q5/q10): zen sets `use_skip=1`
  (`skip_prob=248`) where libwebp uses `use_skip=0`. NOT a shallow threshold fix:
  the decision is `use_skip_proba = (skip_proba < 250)` on both sides and the
  formula is the same modulo rounding (zen rounds `255*non_skip/total`; libwebp's
  `CalcSkipProba` truncates — worth aligning, but the wrong direction to explain
  248<250 alone), so the divergence is the **`nb_skip` count**. libwebp counts
  `nb_skip` inside `StatLoop` via `VP8Decimate` per MB (`frame_enc.c:625`),
  entangling this with the multi-pass `StatLoop` architecture — i.e. issue #25
  (SKIP_PROBA) **and** #27 (full multi-pass StatLoop), both major open items, not
  a one-line gate.
- **m5 `seg_lf`** (low-mid q): the trellis-path StoreMaxDelta (`6b4fa0c` fixed it
  at q75; still diverges off-q75). NOTE the earlier "seg_lf/max_edge" narrative
  applied to m5-trellis, **not** the non-trellis m4 path — m4 is identical.
- **m6 mode-RD** (`y_same` 90–99%) + `n_proba_updates` — I4/mode RD divergence.
- **High-q (q80+) across m3–m6** — proba/RD divergence at fine quant.

**Next step:** the m0/m1 low-q `skip_prob` threshold is the most tractable;
trace with the now-instrumented `trace_driver` (env dumps `LWSMD`/`LWSFS`/`LWADJ`
added for filter work; add a skip-proba dump). The `seg_lf`/`max_edge` machinery
(`adjust_filter_strength`, `store_max_delta` in `vp8/mod.rs`) was verified
CORRECT on the non-trellis path (zen and libwebp both compute seg2 max_edge=2,
final=20 at q30 m4) — the m5 divergence is trellis-specific.

## Method note

`methodcmp` should be extended to sweep q and multiple images (or replaced by
`byteparity_sweep`) so a single-point 14/14 can't again be mistaken for general
byte-exactness. The zensim/pixel gates are unaffected — this is purely about the
*byte-identical* claim.
