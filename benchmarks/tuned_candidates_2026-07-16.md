# Tuned-default adoption of parity-derived candidates, round 2 (2026-07-16)

Follow-up to `tuned_candidates_2026-07-14.md`: the #38 grid-completion work
(4004/4004 byte-exact `StrictLibwebpParity`, see
`byteparity_scope_2026-07-14.md`) surfaced four new root causes. This measures
which of them also improve the tuned default (`CostModel::ZenwebpDefault`).

## Method

Harness: `dev/tuned_ab_sweep.rs` (committed, `[[example]]` wired) — encodes a
corpus at the tuned default across (method × q), scoring size, zensim
(default profile), and min-of-3 encode ms; one TSV row per cell, baseline and
candidate builds append with distinct `--variant` labels.

Corpus (15 images, deterministic recipe since the 2026-07-14 set was lost in
the /tmp wipe): the 3 CID22-512 validation images (382297, 1025469, 1418519)
as-is, plus the FIRST sorted PNG of each of 12 content classes from
`/mnt/v/output/imazen-26-png-v3/` (photos-general, nature, food, people,
textures, met-museum, epa-report, plots, web-screenshots, ai-clipart,
ai-illustrations, ai-products), downscaled to ≤1024 long edge with ImageMagick
`-filter Lanczos`. m ∈ {4,5,6} (m4 = no-trellis control), q ∈ {25,50,75,90}.

Raw data: `tuned_trellis_skip_ab_2026-07-16.tsv`.

## 1. m5/m6 skip from FINAL trellis levels → **ADOPTED**

The tuned trellis path decided the per-MB skip with a separate simple-quant
test (`check_all_coeffs_zero`) while the emission recorded trellis-quantized
levels. The trellis (neutral-bias level0 + sharpen + RD) keeps borderline
coefficients the simple bias drops, so on those MBs the encoder reconstructed
WITH a kept coefficient — and predicted subsequent MBs from it — while
signaling a skip the decoder honors with prediction only. That is an
encoder-side reference mismatch: the encoder's prediction sources drift from
the decoder's actual pixels.

Candidate: derive the skip from the recorded levels
(`stored_coeffs.is_all_zero`), exactly like the parity arm (#38, `a9fc2da`).

| m | Δsize | Δzsim | Δtime |
|---|-------|-------|-------|
| 4 (control) | **0.000%** | **0.0000** | +0.6% |
| 5 | +0.011% | **+0.104** | +0.2% |
| 6 | +0.015% | **+0.097** | +0.2% |

Per-q (m5+m6): q25 +0.002% / **+0.156** · q50 +0.024% / +0.112 ·
q75 +0.008% / +0.080 · q90 +0.019% / +0.054. 77 of 180 cells changed; the
zsim deltas are consistently positive (largest per-cell +0.62), the size cost
never exceeds +0.024% at any q.

Adopted: +0.10 zsim for ≤0.015% bytes is far above the RD curve (a tenth of a
zensim point normally costs a percent-class byte increase), the gain is
largest exactly where quality matters most (low q), and it removes a real
correctness wart. The m4 control at exactly zero confirms the change touches
only the trellis tiers. `check_all_coeffs_zero` deleted; the tuned and parity
trellis arms now differ only in StoreMaxDelta gating (tuned: final-mode blocky
test on non-skipped MBs; parity: I16-candidate test before the skip decision).

## Non-candidates from the same #38 batch (not measured, reasoning recorded)

- **m0-m2 skip-proba StatLoop count** — parity-only bookkeeping to reproduce
  libwebp's stats-pass counting. The tuned default already enables the skip
  flag from its own full-frame rounded estimate, which is self-consistent and
  arguably better-calibrated than libwebp's subset count; nothing to adopt.
- **Segment-quant via libm pow** — the fast-pow approximation only matters
  when byte-matching libwebp's truncation boundary; for the tuned default a
  ±1 quant index at a rare boundary is direction-free noise. Not worth the
  (tiny) libm cost on a path that runs 4×/encode either way; keep fast path.
- **I4 tie-break order** — on an exact RD tie both candidates are equally
  good by definition; libwebp's visit order is not better, just *its*.
  The tuned SSE-presort keeps its early-exit speed advantage.

## 2. Alpha pipeline (filters + full VP8L + raw fallback) → **ADOPTED** (2026-07-16, later)

The tuned default's ALPH payloads used a literal-only VP8L fallback (the
`implicit_dimensions` branch) with no prediction filters and no raw
fallback. The libwebp pipeline ported for parity — filter trials
(none/h/v/gradient via `GetFilterMap`), the FULL VP8L coder on the
alpha-in-green plane, per-trial raw fallback — re-encodes the SAME quantized
plane losslessly, so decoded pixels are bit-identical and the change is a
pure size win:

| probe (64×64/33×17 planes) | before | after |
|---|---|---|
| gradient alpha, aq100 | 54 B | **26 B** |
| checker alpha, aq100 | 109 B | **32 B** |
| gradient, aq90 (quantized) | 54 B | **26 B** |
| gradient 33×17, aq100 | 61 B | **32 B** |

2-3.5× smaller ALPH chunks at a small encode-time cost (filter trials ×
full VP8L on a tiny plane). The tuned default keeps its historical uniform
level-quantizer mapping (`1 + aq·255/100` levels) so decoded alpha VALUES
at aq<100 are unchanged; only the lossless representation shrank. Parity
uses libwebp's `QuantizeLevels` k-means + its level mapping.

## 3. SharpYUV port for `.sharp_yuv(true)` → **ADOPTED** (2026-07-16, later)

The libwebp SharpYUV port built for the parity sharp_yuv axis (96/96)
replaces zenyuv's Newton-refinement converter on the plain
`.sharp_yuv(true)` path: **+1.0..+1.8 zsim** over standard conversion vs
zenyuv's +0.18..+0.32, at the technique-inherent +2-5% bytes, and the
scalar port is 1.5× faster than libwebp's own SSE2 build. Opt-in flag
only — tuned DEFAULT bytes unchanged; `sharp_yuv_config(custom)` still
selects zenyuv. Full analysis, A/B table, and speed data:
`sharpyuv_port_2026-07-16.md` (+ `sharpyuv_port_ab_2026-07-16.tsv`).

## 4. Transparent-area cleanup (lossy `exact=false`) → **ADOPTED** (2026-07-16, later)

`WebPCleanupTransparentArea` (YUV flavor) was the documented-but-missing
behavior behind the public `exact` flag: libwebp's default (`exact=0`)
smoothens invisible luma in mixed-alpha 8×8 blocks and flattens fully
transparent blocks before encoding. Now implemented for RGBA/BGRA/La8 on
BOTH cost models (the flag already promised it; libwebp does it
unconditionally at default): checker-alpha probe VP8 layer 1554 → 1058 B
(−32%), visible pixels bit-identical, `exact(true)` opts out. The
La8↔RGBA cross-format equivalence suite pins the layouts to identical
behavior.

## 5. VP8L: huffman tie-break + rb_zero cross-color skip → **ADOPTED** (2026-07-16, later)

Both are libwebp-exact behaviors adopted on both models (see CHANGELOG):
equal-cost Huffman trees now match libwebp's assignment, and cross-color
is skipped when R/B are constant-zero (pure-gray content in subtract-green
modes saves ~5 stored trees + a transform image). A/B corpus lossless
ratio: 1.0011× (m4), 0.9999× (m6) — within the historical band. The
hash-chain iteration accounting was NOT adopted for tuned (parity-gated
via `Vp8lConfig::parity`): zenwebp's stall-budget + always-on row-above
seed measured +8.5..75% bytes BETTER than the flat cap on smooth
gradients at m0 (see `hash_chain.rs` comments).

## 6. segs1 uv_alpha dq_uv → **REJECTED** (2026-07-16, later)

The last queued candidate from the extremes-axis work: libwebp applies
uv_alpha-derived UV quant deltas at every segment count; zenwebp's tuned
single-segment path skips them. Measured (`tuned_ab_sweep --segments 1
--sns 50`, m2/4/6 × q25-90, 180 paired cells, all changed;
`segs1_dquv_ab_2026-07-16.tsv`):

| axis | Δsize | Δzsim |
|---|---|---|
| m2 / m4 / m6 | +2.76% / +2.77% / +2.88% | +0.34 / +0.36 / +0.38 |
| q25 / q50 / q75 | +1.3% / +1.2% / +1.5% | +0.53 / +0.32 / +0.18 |
| q90 | **+7.26%** | +0.42 |

Rejected: ~+0.35 zsim for ~+2.8% bytes sits ON the tuned RD curve (not
above it like the adopted candidates), the q90 leg is clearly below it,
and at m2+ the delta is driven by libwebp's UNSET `uv_alpha = 0` default
(its analysis doesn't run there) rather than content signal — libwebp is
spending bytes on a hardcoded "assume chroma is bad" guess. No shipped
preset uses segments=1, so the affected path is explicit-override only.
Verdict recorded at the parity gate in `vp8/mod.rs`.
