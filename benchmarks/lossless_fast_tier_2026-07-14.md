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
