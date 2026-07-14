# crabmagick "fast-webp" lossless encoder claims — verification (2026-07-13)

**Scope:** claims-verification comparison, not a source-informing calibration
sweep (no constants land in source from this). Raw data:
`crabmagick_lossless_claims_2026-07-13.tsv`.

- zenwebp @ efddb6c3 (includes the VP8L decoder refill fix this work surfaced)
- crabmagick @ 0f1196a (github.com/UsernameN0tAvailable/crabmagick), encoder
  under test: `rust/crates/crabmagick-core/src/webp_decode/encoder.rs`
  (1,194 lines; the `fast-webp` crate's copy is a stale earlier revision)
- libwebp via webpx 0.4.0 (libwebp-sys 0.14), `lossless(true).exact(true)`
- Host: 7950X WSL2, single-threaded, release, **no `-C target-cpu=native`**
  (their own `bench-remote.sh` uses `-C target-cpu=native`)
- Timing: median of adaptive iterations (≥5 or ~1.2 s budget), warmup 1
- Every output verified pixel-exact by decode (zenwebp decoder)

## Their claim (README)

> "WebP lossless uses a custom pure-Rust encoder with an effort-aware LZ77
> chain (eff=0–6). It is dramatically faster than libwebp at any effort level
> because libwebp serializes its lossless passes. Our encoder parallelizes
> them across all cores. At higher efforts the chain depth increases (eff=4 →
> depth 64, eff=6 → depth 256), matching libwebp's quality while remaining
> 30-80× faster."

## Verdict

1. **"Parallelizes them across all cores" is false.** The lossless encoder
   (`encoder.rs`) contains no rayon/threads at all — it is fully serial. The
   rayon in the crate is in the *decoder* (loop filter, YUV→RGB, forked from
   image-webp) and the lossy encoder's row conversion.
2. **"Matching libwebp's quality" (= size, for lossless) is false on real
   content.** Measured at eff=4 vs libwebp m4: **+25% bytes** on a CID22
   photo, **+65%** on a screenshot, **+78%** on line art, **+30%** on an
   alpha graphic. Their encoder is image-webp's minimal encoder plus a basic
   LZ77 (single-pixel hash chain, depth 16–256, 1-step lazy) + heuristic
   color cache. No palette, no per-tile predictors (fixed TOP for the whole
   image), no cross-color, single Huffman group, no entropy-strategy search.
3. **The 30-80× numbers reproduce only on degenerate synthetic content** (its
   README sizes — 8-15 KB "lossless" files from 256×256 through HD — imply a
   near-flat test image). There, libwebp burns its full analysis pipeline on
   trivial pixels; an RLE-ish encoder is naturally 20-40× faster. Amusingly,
   on our 800×600 synthetic gradient their eff=6 output (3,472 B) is *smaller*
   than libwebp m6 (5,568 B) — deep-chain LZ77 beats libwebp's config choice
   on exactly that content class — while on 256² libwebp m0 is both faster
   AND 3× smaller than their eff=6.
4. Speed itself is real but modest on real content: ~1.4–4× faster than
   libwebp m4 at eff=4 (not 30-80×), at the size costs above.

## Measured (selected; full grid in the TSV)

| image | crab eff4 | libwebp m4 | zenwebp m2 |
|---|---|---|---|
| photo CID22 512² | 50 ms / 219.0 KB | 104 ms / 174.9 KB | 108 ms / 173.3 KB |
| screenshot 314×365 | 3.7 ms / 30.1 KB | 28 ms / 18.2 KB | 14 ms / 18.2 KB |
| line art frymire | 76 ms / 308.6 KB | 109 ms / 173.3 KB | 110 ms / 173.7 KB |
| alpha dice 800×600 | 29 ms / 207.6 KB | 123 ms / 159.4 KB | 134 ms / 167.1 KB |
| gradient 800×600 | 4.2 ms / 6.6 KB | 89 ms / 5.6 KB | 76 ms / 5.5 KB |

## What zenwebp should take from this

1. **We have no competitive fast lossless tier.** zenwebp m0 is 2.6–11×
   slower than libwebp m0 (71 vs 28 ms photo 512²; 60 vs 7.8 ms on the
   800×600 gradient; 74 vs 9.2 ms dice) — our m0 produces smaller files
   (−16% on photo) but there is no "thumbnail-grade, milliseconds" operating
   point. crab eff0 does photo 512² in 18 ms. A true fast path (single crunch
   config, cheap fixed predictor decision, skip meta-huffman clustering)
   could serve the transient/caching use case their project targets.
2. **Method-ladder anomaly (follow-up):** on `gradients_png` zenwebp m6
   (24.7 KB) is +15% larger than m2 (21.6 KB); photo m4/m6 (175.7 KB) larger
   than m2 (173.3 KB). libwebp shows the same shape on gradients_png (m6
   25.3 KB > m4 24.0 KB) so it is partly content/heuristic weirdness, but our
   spread is wider. Higher method should not cost bytes.
3. **The comparison surfaced a real decoder bug** (fixed in efddb6c3): the
   VP8L pixel-loop bit-window refills were under-budgeted (literal RBA needs
   30 bits after RED, dist path needs 33), so valid libwebp `-m 0 -lossless`
   deep-tree streams got a spurious `BitStreamError`. See LOG.md "VP8L
   Decoder Bit-Window Refill Under-Budget" and
   `tests/vp8l_deep_tree_decode.rs`.
4. Their repo vendors imazen crates: `jxl-encoder`/`jxl-encoder-simd`
   (credited in their NOTICE, with imazen.io/pricing pointer) plus `zenjpeg`
   and `zen-jp2` directories (zenjpeg is NOT in their NOTICE — worth a look
   if attribution matters).

## Techniques actually present in their encoder (for the record)

- Effort-scaled chain depth (16/32/64/128/256 for eff 0-1/2-3/4/5/6+)
- Single-pixel hash (`0x1e35a7bd` multiply), chain built once over all pixels
- 1-step lazy matching that *recomputes* the next match (2× match-finder cost)
- Color cache with size-heuristic bits (8/9/10 by pixel count) and a full
  re-tokenize without cache when hits are zero
- Always subtract-green; predictor transform = single-entry TOP for the whole
  image (encoded as a 1-symbol predictor sub-image)
- Meta-huffman bit written 0 (single group); distance alphabet 40 with the
  standard plane-code mapping

None of these are ahead of what zenwebp's VP8L pipeline already does at
m0–m6; the only structural idea worth borrowing is the *existence of a
cheap tier*, not their implementation of it.
