# Sharp-YUV dig: zenyuv's sharp vs libwebp's SharpYUV (2026-07-16)

Requested as part of #38 follow-up ("dig into yuv"). The byteparity
permutation sweep showed sharp_yuv at 0/96 — this dig characterizes WHY and
what it costs.

## The two implementations

- **libwebp SharpYUV** (`sharpyuv/*.c`, ~600 lines): fixed-point s16
  iterative refinement, 4 iterations, forward-matrix gradient steps, gamma
  via `kGammaToLinearTabS16`-family tables, `SharpYuvUpdateY` luma
  compensation. The reference implementation.
- **zenyuv sharp** (`zenyuv-0.1.3 src/sharp.rs`, ~1100 lines): a deliberate
  REDESIGN, not a port — Newton steps with the correct inverse-matrix
  Jacobian, f32 math, 2 iterations default, its own gamma LUTs, optional Y
  refinement pass. Its docs claim "2 iterations gives better quality than 4
  iterations of the traditional forward-matrix gradient approach".

Byte parity is impossible by construction; the meaningful comparison is
delivered quality.

## Measurement

`dev/sharpyuv_compare.rs` (committed): 15-image A/B corpus (same recipe as
`tuned_candidates_2026-07-16.md`), q ∈ {50, 75, 90}, m4. Three encodes per
cell: zen sharp (`with_sharp_yuv(true)`), libwebp sharp
(`sharp_yuv(true)`), zen standard. All decoded and zensim-scored against the
source. Raw data: `sharpyuv_compare_2026-07-16.tsv`.

| q | zen-sharp vs lib-sharp | zen-sharp vs zen-std | lib-sharp vs zen-std |
|---|---|---|---|
| 50 | −2.2% bytes, **−0.72 zsim** | −0.1% bytes, +0.24 zsim | **+0.95 zsim** |
| 75 | −3.2% bytes, **−0.99 zsim** | −0.0% bytes, +0.23 zsim | **+1.22 zsim** |
| 90 | −4.8% bytes, **−1.46 zsim** | −0.2% bytes, +0.32 zsim | **+1.78 zsim** |

## Findings

1. **libwebp's SharpYUV substantially outperforms zenyuv's sharp in
   delivered quality**: it buys 3-6× the zsim improvement over standard
   conversion (+0.95..+1.78 vs +0.23..+0.32). libwebp's files are 2-5%
   larger, but even discounting the byte spend at typical RD slopes
   (~0.1-0.2 zsim per % bytes) a genuine ~+0.6-1.0 zsim algorithmic
   advantage remains at mid-high quality.
2. zenyuv's "2 Newton iterations beat 4 gradient iterations" claim does
   not hold up end-to-end on this corpus — either the iteration-quality
   claim was measured in a narrower setting, or the gamma tables /
   Y-refinement details dominate.
3. The sharp_yuv parity axis (0/96) is therefore not just a byte-format
   gap; the algorithms genuinely differ, libwebp's favorably.

## Next chunk (queued)

Port libwebp's SharpYUV exactly (fixed-point, self-contained — gamma tables
+ filter/update loops) into zenwebp as the parity sharp converter, which
closes the sharp_yuv axis byte-for-byte AND becomes a tuned-adoption
candidate for quality-focused encodes (the current zenyuv sharp would remain
as the fast option). Keeping the port inside zenwebp avoids touching the
zenyuv crate.
