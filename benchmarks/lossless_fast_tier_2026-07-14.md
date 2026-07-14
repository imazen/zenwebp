# Lossless m0 fast tier — before/after (2026-07-14)

m0 lossless is now a true low-effort tier mirroring libwebp's `low_effort`
(method 0) shortcuts. Raw grid: `lossless_fast_tier_2026-07-14.tsv`.
Follow-up from `crabmagick_lossless_claims_2026-07-13.md` finding #1
("we have no competitive fast lossless tier").

- Host: 7950X WSL2, single-threaded, release, no `-C target-cpu=native`
- Timing: median of adaptive iterations (≥5 or ~1.2 s), warmup 1
- libwebp via webpx 0.4.0 (`lossless(true).exact(true)`); every output
  verified pixel-exact through BOTH zenwebp and libwebp decoders
- crab = crabmagick@0f1196a encoder (verbatim copy), for reference

## What m0 now does (libwebp low_effort parity)

Skips `analyze_entropy` (palette if available, else SubtractGreen+Predictor);
fixed Select predictor for every tile (no per-tile search); no cross-color;
no color cache; single plain-LZ77 pass (no RLE trial, no cost comparison, no
TraceBackwards); hash-chain warm-start heuristics skipped; histogram
clustering = 4 literal-entropy bins with unconditional same-bin merges, no
stochastic/greedy. m1+ unchanged (byte-identical on the whole grid).

## Results (encode ms / bytes)

| image | old m0 | new m0 | libwebp m0 | crab eff0 | zenwebp m1 |
|---|---|---|---|---|---|
| photo CID22 512² | 71.3 / 193,808 | **29.3 / 230,464** | 27.2 / 229,790 | 15.7 / 220,120 | 76.8 / 178,846 |
| alpha dice 800×600 | 73.7 / 179,040 | **15.2 / 233,394** | 8.9 / 238,462 | 12.2 / 208,476 | 93.6 / 173,848 |
| line art frymire | 65.4 / 211,210 | **56.1 / 323,348** | 41.5 / 323,816 | 34.9 / 319,802 | 117.7 / 172,590 |
| screenshot 314×365 | 2.1 / 21,472 | **2.4 / 29,722** | 1.4 / 29,700 | 1.4 / 29,988 | 5.8 / 18,320 |
| gradients_png 256×128 | 5.8 / 27,840 | **1.7 / 32,898** | 1.3 / 32,898 | 2.3 / 27,128 | 7.4 / 22,048 |
| synth gradient 800×600 | 60.0 / 6,024 | **7.9 / 11,718** | 5.0 / 11,740 | 2.8 / 12,500 | 67.5 / 5,418 |
| synth gradient 256² | 7.4 / 136 | **0.6 / 96** | 0.3 / 92 | 0.9 / 906 | 7.6 / 134 |

Instructions (callgrind, photo 512², per encode): 1.01B → 0.29B (−71%).
Old-m0 phase costs eliminated: per-tile predictor search 36.2%, cross-color
search+apply 27.5%, entropy analysis ~3%, RLE trial + cache-size analysis ~2%.

## Reading

- New m0 tracks libwebp m0 closely on both axes (sizes within ±2% except
  dice, where we are 2.1% smaller; time 1.1–1.7× libwebp's — remaining gap
  is `HashChain::new`, now 67% of the profile).
- vs crabmagick eff0: comparable-to-2× slower, but smaller output on 5 of 7
  images (they beat us on photo/frymire size — their eff0 runs a color
  cache, which libwebp-m0 semantics omit).
- **Old m0 was a near-duplicate of m1** — same search pipeline, only
  histo_bits differed (7 vs 6) — and m1 dominates it on size (photo:
  178.8 KB vs old-m0's 193.8 KB at similar time). Its operating point
  remains available via m1; nothing was lost by re-purposing m0.

## Bug found and fixed along the way

`build_final_histograms` used to emit Huffman tree groups for clusters that
the post-remap tile mapping no longer referenced. The decoder sizes the group
list from the entropy image's max symbol, so an unreferenced *trailing*
cluster made the decoder parse the extra trees as pixel data — a silent
whole-image corruption. Latent for any method; easiest to hit with m0's
unconditional bin merging. Now compacted + dense-renumbered; guarded by
`tests/lossless_fast_tier.rs` (m0 roundtrips through both decoders on photo,
palette, alpha, multi-region, and tiny images).

## Round 2 (same day): hash-chain parity + fixed-bits color cache

Follow-up commits close the mechanical gap and buy back compression:

1. **Dedicated fixed-Select residual pass** (m0, exact-lossless only): forward
   streaming with a one-row history buffer, no per-pixel tile lookup or mode
   dispatch. 31.1M → ~3M instructions per photo-512² encode (libwebp's
   PredictorSub11_SSE2 equivalent is 2.1M).
2. **Hash-chain instruction parity with libwebp confirmed** (photo: ours
   ~182M vs libwebp ~150M per encode; dice: 250M vs 256M — at parity).
   `vector_mismatch` rides 4-pixel fixed-array compares; the walk hoists the
   next-link load and folds the quick-reject range+load checks via `get`;
   `hash_to_first` is a boxed `[i32; HASH_SIZE]` so the shift-derived index
   is provably in range.
3. **Row-above heuristic kept ON at m0** (deviation from libwebp, which
   skips it): seeds distance==width matches for ~1 compare/pixel and is a
   pure size win — dice −7%, frymire −2 to −3%.
4. **Fixed-bits color cache at m0** (deviation from libwebp, which never
   caches at m0): size-heuristic bits (8/9/10 by pixel count; palette-derived
   cap for palette images), accepted only when one A/B histogram-cost check
   says it wins — no 0..=10 search. Photo −6.4%, dice −11%, frymire −9%,
   gradients_png −14%; synthetics correctly declined (bytes unchanged).

Final m0 (this file's TSV): smaller than libwebp m0 on ALL 7 images
(photo −6.1%, dice −19.0%, frymire −12.4%, screenshot −5.2%) and smaller
than crabmagick eff0 on 6 of 7. Instructions per encode: photo 279M
(vs libwebp 210M, 1.33×; wall tracked ~1.0–1.2× in same-run comparisons),
dice 170M (vs 120M, 1.41×, −19% bytes). Session start → now:
photo m0 went 1,010M/193.8KB (near-dupe of m1) → 279M/215.7KB (true fast
tier), with m1 (77ms/178.8KB) holding the high-compression slot.

**Measured and rejected:** capping chain iterations at m0 (libwebp-style
iters stay quality-derived = 51 at q75). iters=16: photo 2.3× faster but
smooth-gradient content +75% bytes; iters=32: gradient still +8.5%. A
content-class size regression fails the tier's "still saves space" bar;
revisit only with a cheap content gate.

## Round 3: no-progress chain pruning at m0 (d8c9e3d)

Search-tree pruning per the "cut stalled walks, not the toolbox" principle:
abandon a chain walk after P consecutive non-improving candidates
(quick-rejects included); improvements refill the budget. P sweep:

| P | photo ms/bytes | dice | frymire | synth-diag-gradient 800×600 |
|---|---|---|---|---|
| none | ~28-34 / 215,688 | 16.5 / 193,254 | 70.0 / 283,652 | 11,716 |
| 8 | 13.5 / 212,140 | 10.3 / 194,778 | 38.8 / 288,296 | 20,080 (+71%) |
| 16 | 17.5 / 213,418 | 11.6 / 193,004 | 45.0 / 286,032 | 13,642 (+16%) |
| **24 (shipped)** | 24.0 / 214,096 | 14.4 / 193,056 | 49.0 / 284,938 | 12,342 (+5.3%) |

P=24 keeps every real-content size within ±0.5% (photo −0.7%, real-gradient
gradients_png −0.1%) while cutting wall ~15-30%. The only regression is the
synthetic diagonal-gradient stress case (+5.3%), whose productive candidates
hide behind >24-candidate stall runs — flat iteration caps measured far
worse there (+8.5% at 32 iters, +75% at 16). m1+ byte-identical.

## Round 4: imazen-26 corpus validation + lossy m0 investigation

Representative production content: 12-image stratified sample (every 7th) of
`/mnt/v/input/imazen-26-screenshots-2026-05-28` (106 web-viewport screenshots).
Raw grid: `imazen26_m0_2026-07-14.tsv`. All lossless roundtrips pixel-exact.

**Lossless m0 vs libwebp m0: 0.95× wall, 0.6396× bytes (−36%).** The fast
tier is *faster than libwebp and a third smaller* on real screen content
(best case: whitehouse briefing, 108 KB vs 1,019 KB — 9.4× smaller at equal
speed; typical: −10..−45%). The palette path, fixed-bits cache, and
entropy-cluster compaction all land on exactly this content class.

**Lossy m0: libwebp's low-effort shortcuts MEASURED AND REJECTED.** zen lossy
m0 runs 1.8–2.2× libwebp m0 wall at −4.3% bytes (q75, corpus aggregate).
Ported libwebp's two m0 gates and measured:
- `FastMBAnalyze` alpha=0 (skip the 4-mode Intra16 histogram analysis,
  analysis_enc.c): −9.6% instructions but **+8.5% bytes and −0.49 dB PSNR on
  screen content** (noaa homepage) — segmentation collapses when luma alpha
  is constant. libwebp's m0 tuning absorbs this; ours measurably should not.
- `refine_uv_mode = method >= 1` (use analysis UV mode at m0, skip the RD
  pick): only ~2% of instructions (`pick_best_uv` = 1.6M of 96M/encode —
  earlier wall deltas were machine noise), and +7..9% bytes on smooth
  gradients even after extending the analysis UV search to 4 modes — trips
  the vs_libwebp_matrix 1.3× size gate. Not worth it.

The real 1.8× is architectural: the token-buffer stats pass (record 6.6M +
emit 3.6M per encode vs libwebp m0's direct VP8EncLoop with a 25%-MB
probability probe) — and it is exactly what buys the −4.3% bytes. Left as-is
deliberately. Known follow-up: zen m0 trails libwebp m0 by ~0.9 dB PSNR
aggregate on screens at −4.3% bytes (pre-existing, RD-point difference plus
screen-content tuning; photo content is at parity).

## Round 5: lossy m0/m1 — direct-loop verdict + SSE-scored decimation (landed)

**Direct emission loop: measured and rejected, by wall clock this time.**
History: zenwebp's pre-2026-02 two-pass loop was replaced by the token buffer
(9625d4e); libwebp's m0-2 direct `VP8EncLoop` (single pass + 25%-MB
probability probe) was never tried. Phase timers show why it never needs to
be: at m0 the emit phase is **0.43 ms of ~4.5 ms** — the token round-trip is
not the wall cost, and a probe pass would cost more than emission saves. No
Rust constraint; a sink-generic serializer (~300 lines) could do it — it
just doesn't pay.

**The real m0-2 lever: SSE-scored decimation (libwebp RefineUsingDistortion
shape), landed for m0/m1.** zenbench (interleaved, ±0.1 ms CIs — wall, not
instruction counts):

| tier | before | after | libwebp |
|---|---|---|---|
| zen m0 (diagnostic) | 4.52 ms | **4.4 ms** | 2.60 ms |
| photo-512 m0 bytes | 13,536 | **13,172 (−2.7%)** | 14,660 |
| imazen-26 lossy q75 aggregate | 6,656,292 B / 37.54 dB | **6,508,166 B (−2.2%) / 37.56 dB** | 6,955,106 B / 38.40 dB |

I16-hint MBs: all four modes SSE-scored (lambda_d_i16=106, incl. the libwebp
bug-#432 flat-border guard); I4-hint MBs: all ten sub-block modes SSE-scored
(lambda_d_i4=11) with a single winner-only quantize for reconstruction — the
RD path quantized every candidate. No I16 bail (libwebp try_both==0
semantics). Two synthetic zensim floors re-baked for the new m0 mode
decisions (gradient q75: 81.71→79.28, floor 80→78; mandelbrot q10:
42.59→41.93, floor 42→41 — sub-point deltas, user-approved) while corpus
bytes and PSNR both improve. Remaining m0 wall gap vs libwebp is ~1.7×;
next candidates: single-arcane hoisting of the SSE pick loops (~14 dispatch
boundaries per I4 sub-block today) and extending hints+SSE decimation to m2
(worst tier at 2.9×).

## Round 6: arcane hoist + full RefineUsingDistortion at m2

- **Hoist** (739bc14 recipe): the SSE I4 pick loop now runs in a single
  `#[arcane]` region via a shared macro body instantiated with `#[rite]`
  SSE2 kernels (was ~14 dispatch boundaries per sub-block). Bit-exact;
  wall-neutral at m0 on photo (I16-hinted MBs dominate) but load-bearing
  for m2 where the I4 loop runs on every MB.
- **m2 = full RefineUsingDistortion(try_both=1, refine_uv=1)**: SSE-scored
  4-mode I16 pick, SSE-scored 10-mode I4 with libwebp's accumulate-and-bail
  (score_i4 from i4_penalty = 1000·q_i4², header bits capped by
  mb_header_limit = 256·510·8·1024/mbs), SSE-scored 4-mode UV refine
  (lambda_d_uv=120). m1 also gets the SSE UV refine (libwebp
  refine_uv_mode = m ≥ 1); m0 keeps the RD UV pick (libwebp m0 skips UV
  refinement; the SSE pick there pushed gradient q90 to 1.317× — gate max
  1.3×). m3+ keep the full RD path (libwebp RD_OPT_BASIC+).

zenbench (interleaved, ±0.1 ms):

| tier | before | after | libwebp | ratio |
|---|---|---|---|---|
| m0 | 4.4 ms | 4.4 ms (byte-stable) | 2.5 ms | 1.76× |
| m1 (ad-hoc) | ~5.6 ms | ~4.6 ms | 3.2 ms | 1.45× |
| **m2** | **9.6 ms** | **4.1 ms** | 3.1 ms | **1.30× (was 3.0×)** |
| m4 / m6 | 11.7 / 19.9 | unchanged | 8.9 / 14.0 | 1.31× / 1.42× |

photo-512 ladder after: m0 13,172 B / 37.05 dB → m1 13,198 / 37.06 →
m2 12,772 / 36.91 → m4 11,444 / 37.05. m2's PSNR dip below m1 mirrors
libwebp's own m2 (36.99 < their m1's 37.19) — inherent to RD_OPT_NONE
semantics. m2 is +12.8% bytes vs libwebp m2 (their direct-loop probas and
mode mix differ); all vs_libwebp_matrix gates (≤1.3× size, score deltas,
cross-decode) and zensim floors pass. m2 is now absolutely faster than m0
(no hint pass, bail cuts I4 work).

## Round 7: "regression vs historical" investigated + large-image validation

**m6 wall bisected**: 17.4 → 20.3 ms landed inside the libwebp-parity audit
(PR #37, 2026-04-26; #44 gate innocent — 19.9 ≈ 20.3). Dominated by #22
(I4 penalty 3000 → 211: deliberately shifts macroblocks into the I4 path,
which costs zenwebp 3.4× libwebp) and #29 (trellis quantization inside I16
mode selection at m6 — RD_OPT_TRELLIS_ALL parity, spot-measured −0.7% bytes
in its own commit). The audit bought m6 from 1.0022× to 0.9948× bytes. m4's
ratio never regressed (1.36× then, 1.36–1.39× now); absolute times today are
faster than the 2026-03-28 table on both sides (toolchain/box drift) — the
March table is superseded in CLAUDE.md.

**Large-image A/B (interleaved zen/lib pairs, medians of 7):**

| tier | 2MP screen (1280×1558) | 8MP mixed (1440×5653) |
|---|---|---|
| m0 | 2.01× wall / 0.933× bytes | 2.02× / 0.927× |
| m2 | **1.20× / 1.078×** | **1.22× / 1.080×** |
| m4 | 1.38× / 1.001× | 1.39× / 0.998× |
| m6 | 1.41× / 0.992× | — |

The m2 RefineUsingDistortion port scales (1.30× at 512² → 1.20× at 2-8MP).
m0 degrades to ~2.0× at scale and is slower than m2 in absolute terms — its
full-alpha analysis + hint collection + RD UV pick grow with area while its
cheap decimation saves less; it still buys −7% bytes vs libwebp m0. Named
follow-ups: I4 selection cost (pulls m4/m6 AND makes the audit's I4 shift
cheap); m0 analysis diet at scale.
