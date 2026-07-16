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
