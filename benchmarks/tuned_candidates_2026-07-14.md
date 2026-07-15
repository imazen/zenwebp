# Tuned-default adoption of parity-gated fixes (2026-07-14)

Issue #38 landed byte-exact `StrictLibwebpParity` by adding a set of fixes gated
to the parity path. This asks the follow-up question: which of those behaviors is
*also* better for the tuned default (`CostModel::ZenwebpDefault`) and should be
adopted? Three candidates were measured on a 15-image corpus (3 CID22 + 12
representative imazen-26, one per content class), downscaled ≤1024 (Lanczos3).

Raw data: `tuned_candidates_2026-07-14.tsv` (baseline / fastalpha / uvdiff_rd
variants, m0–m6, q∈{25,50,75,90}, size + zsim + min-of-3 encode ms).

## 1. `max_i4_header_bits` → **ADOPTED** (separate commit)

Unify the tuned default onto libwebp's `partition_limit`-derived formula (65536
at the default) instead of the historical 16384 band-aid. −0.16% size, +0.014
zsim at m4/m6. Full write-up: `maxi4_header_bits_2026-07-14.md`.

## 2. FastMBAnalyze alpha at m0/m1 → **REJECTED**

libwebp uses `FastMBAnalyze` (which returns `best_alpha = 0`) at method ≤ 1;
the parity path matches this, but the tuned default runs the fuller
`analyze_best_intra16_mode` for segmentation. Candidate: should the tuned default
also take the fast `alpha=0` path at m0/m1?

| m | Δsize | Δzsim | Δtime |
|---|-------|-------|-------|
| 0 | **+4.26%** | +0.917 | −7.4% |
| 1 | **+4.25%** | +0.929 | −9.0% |
| all | **+4.25%** | +0.923 | −8.1% |

Rejected. The fast path trades **+4.25% bytes** for +0.92 zsim and −8% time —
a size regression on a draft tier whose entire purpose is minimum size (tuned m0
is documented to trade time for size, running ~7–10% under libwebp m0 bytes).
The fuller analysis is delivering exactly that size advantage; the fast path's
quality/speed gain is a different RD operating point, not a Pareto win, and the
wrong direction for this tier. The tuned default keeps the fuller analysis; only
parity takes the fast path.

## 3. UV-diffusion in the RD scoring loop → **ADOPTED for m3-6 only**

libwebp's `ReconstructUV` runs chroma-DC error diffusion (`CorrectDCValues`) for
every candidate UV mode at quality ≤ 98. zenwebp's *emission* path already
diffuses (both cost models), but the tuned RD *scoring* saw the undiffused
reconstruction — so it scored UV modes against different pixels than it emitted.

| m | Δsize | Δzsim | Δtime |
|---|-------|-------|-------|
| 3 | −0.22% | +0.113 | +1.5% |
| 4 | −0.22% | +0.145 | −0.2% |
| 5 | −0.23% | +0.075 | −0.2% |
| 6 | −0.23% | +0.095 | +0.7% |
| all | **−0.22%** | **+0.107** | +0.5% |

Adopted at m3-6: a clean Pareto win (smaller *and* better) at negligible time
cost, consistent across all four RD tiers. Scoring on the emitted reconstruction
is simply more correct.

**Not** applied to the tuned **m0** arm, which also routes through `pick_best_uv`
(mode_selection.rs ~2311, "m0 keeps zenwebp's RD pick — measured better").
Diffusing m0's RD pick dips the synthetic `noise` image below its zsim floor at
q10 (`tests/zensim_regression_matrix.rs`: 41.27 < 42) — a real regression on
smooth-noise content, caught by the floor. The gate is
`parity || method >= 3`, so tuned m0 is untouched and parity (which only reaches
this fn at m3-6) stays 14/14 byte-identical.

## 4. I4 flatness penalty in the running total → **REJECTED (wash)**

libwebp folds a `FLATNESS_PENALTY` (~140/block) into `rd_i4.R` before the
I4-vs-I16 comparison; the parity path matches this, but the tuned default omits
it (I4 looks ~140/block cheaper). This pulls I4/I16 selection opposite to the
`max_i4_header_bits` lift (#1), so it was measured as a pair.

| m | Δsize | Δzsim | Δtime |
|---|-------|-------|-------|
| 3 | −0.02% | +0.014 | +1.8% |
| 4 | −0.01% | −0.001 | +0.1% |
| 5 | −0.02% | +0.006 | +0.9% |
| 6 | −0.01% | −0.008 | +0.9% |
| all | **−0.02%** | **+0.003** | +0.9% |

Rejected as a wash: byte-neutral (−0.02%) and quality-neutral (+0.003 zsim,
mixed sign by method) at +0.9% time. It also **confirms the `max_i4` decision
is safe** — the penalty pulls I4 selection the opposite direction yet moves
almost nothing, so the lifted clamp is not causing harmful I4 over-selection.
Kept parity-only.

## 5. Mid-row `level_costs` refresh → **REJECTED (previously measured)**

Rebuilding the level-cost tables from the refreshed image-adapted probabilities
mid-row (libwebp's `VP8CalculateLevelCosts` after `FinalizeTokenProbas`) is
parity-only. Already measured as a tuned-default *regression* (1.0101x → 1.0114x
on default compression, documented at `vp8/mod.rs:1564`) — not re-run. Kept
parity-only.

## Other parity gates (not candidates)

`FinalizeTokenProbas` default-proba coding (tuned already keeps the strictly-
smaller image-adapted update), container even-padding-inside-chunk (a 1-byte
layout detail), and the m5 `StoreMaxDelta` non-trellis nonzero test (tuned uses
the more-correct trellis test) are byte-exactness matching quirks where the
tuned default is already equal or richer — nothing to adopt.

## Gates after adoption

323 lib tests, the 14-cell zensim regression matrix, and v2 pixel-perfect all
green. `StrictLibwebpParity` remains 14/14 byte-identical to libwebp (segs1 +
segs4, m0–m6).
