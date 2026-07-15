# `max_i4_header_bits` — tuned default adopts libwebp's formula (2026-07-14)

## Decision

The tuned default (`CostModel::ZenwebpDefault`) now derives its per-MB I4-header
cap from `partition_limit` using **libwebp's formula**, exactly like the
`StrictLibwebpParity` path:

```
max_i4_header_bits = 256 * 16 * 16 * (100 - partition_limit)² / 100²
                   = 65536   at the default partition_limit = 0
```

Previously the tuned default hardcoded `256*16*16/4 = 16384` (the
`partition_limit = 50` point). Source: `src/encoder/vp8/mode_selection.rs`,
the `pick_best_intra4` header-bit cap.

## Why 16384 existed (git archaeology, done before the sweep)

`16384` was **not** a measured optimum — it was a band-aid. Introduced in
`faa3846` explicitly "to prevent excessive I4 usage" back when zenwebp's I4
*coefficient costs were inaccurate*; reinforced by `71aeea4` ("full I4
suppression needed at 26MP") and `d994e54`. With inaccurate costs, I4 looked
cheaper than it was, so it over-selected and bloated files; clamping the header
budget was the cheap fix.

That premise no longer holds. This session fixed the I4 cost accuracy:
- zigzag scan order in the residual cost (`3b90e27`)
- the dropped I4 flatness penalty, 140/block (`4d41a33`)

So the reason to suppress I4 is gone — the question is purely empirical: with
accurate costs, does lifting the clamp help or re-open the over-selection bug?

## Measurement

**Corpus (per user request):** 3 CID22 + 12 representative imazen-26 images,
one per content class (photos, nature, food, people, textures, museum, EPA
report, plots, web screenshots, AI clipart / illustrations / products).
Downscaled to ≤1024 max-dim (Lanczos3) per sweep-source discipline.

**Grid:** tuned default (SNS=50, filter=60, segs=4), methods {4, 6} (the tiers
that consult `max_header_bits`), q ∈ {25, 50, 75, 90}. 120 cells per variant.

**A/B:** `clamp(16384)` (old tuned) vs `full(65536)` (libwebp formula), via a
`OnceLock`-cached `ZEN_MAXI4HB_FULL` toggle (temporary, not committed). Raw
data: `maxi4_header_bits_2026-07-14.tsv`. zsim = zensim latest profile,
distorted vs source, higher = better.

| m | clamp size | full size | Δsize | clamp zsim | full zsim | Δzsim |
|---|-----------|-----------|-------|-----------|-----------|-------|
| 4 | 3,842,428 | 3,836,046 | **−0.17%** | 71.383 | 71.395 | **+0.012** |
| 6 | 3,658,212 | 3,652,890 | **−0.15%** | 71.257 | 71.272 | **+0.015** |
| **all** | 7,500,640 | 7,488,936 | **−0.16%** | 71.320 | 71.333 | **+0.014** |

74/120 cells changed.

## Conclusion

Lifting the clamp is a small but clean win on the tuned default: **−0.16% size
AND +0.014 zsim** (slightly *better* quality, not a size/quality trade). The old
over-selection failure mode does **not** re-open — files got *smaller*, not
larger, which is the direct signature of I4 winning only where it genuinely
pays now that its costs are accurate. The two paths are unified on libwebp's
formula.

Gates after the change: 323 lib tests, the 14-cell zensim regression matrix,
and v2 pixel-perfect all green; `StrictLibwebpParity` remains 14/14
byte-identical to libwebp (segs1 + segs4, m0–m6) — the parity path was already
on this formula, so it is untouched.
