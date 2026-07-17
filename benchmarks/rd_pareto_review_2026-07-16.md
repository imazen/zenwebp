# RD Pareto review: what stepped onto the frontier, and generalize-vs-pick (2026-07-16)

Question: after the #38 parity program and the two tuned-candidate adoption
rounds, (a) where does the tuned default sit on the rate-distortion plane vs
libwebp, (b) do the adopted candidates generalize per-image, and (c) do any
REJECTED candidates hide content pockets where they win — i.e. is a per-image
"pick" mechanism warranted for any of these knobs?

Method: `dev/tuned_ab_sweep.rs` grew an `--encoder libwebp` arm (webpx,
default preset, same method/quality/knob grid) so one TSV carries zen
variants and a libwebp anchor. Comparison metric: **matched-quality size
ratio** — for each zen point, log-interpolate libwebp's size at the same
zensim score on the same (image, method) curve; geomean over the overlapping
range. `< 1` = zen needs fewer bytes at equal quality (above libwebp's
curve). Corpora: the 15-image A/B set (3 CID22-512 + 12 imazen-26 classes,
≤1024 Lanczos), a 13-image line-art expansion (9 plots subclasses + patents
scans + NPS brochures), and a 2-image scans pocket probe. q ∈
{10,25,35,50,65,75,85,90(,95)} — low-q covered per the web-focus discipline.

Raw data: `rd_anchor_2026-07-16.tsv` (15-image, m0/2/4/6 × 9q × both
encoders), `lineart_fastalpha_2026-07-16.tsv`, `lineart_rdtier_2026-07-16.tsv`
(incl. `no_csf` / `no_maskblend` ablation variants),
`scans_sns_2026-07-16.tsv` (sns + flatness ablations).

## (a) Encoder-level: where the tuned default sits vs libwebp

Matched-quality size ratio (zen/libwebp), 15-image corpus:

| method | overall | q10-35 | q50-75 | q85-95 |
|---|---|---|---|---|
| m0 | **0.910** | 0.909 | 0.912 | 0.906 |
| m2 | **1.010** | 1.017 | 1.008 | 1.008 |
| m4 | **0.997** | 0.998 | 0.990 | 1.005 |
| m6 | **0.988** | 0.989 | 0.978 | 0.997 |

- **m0 is ~9% above libwebp's curve** across the whole quality range (the
  documented time-for-size trade of the draft tier), on every image
  (0.805–0.995), including line-art.
- **m6 is ~1.2% above**, strongest at mid-q (0.978); worst image 1.006.
- **m4 is at parity** (ahead at mid-q, ~0.5% behind at q85-95).
- **m2 is the one tier BELOW the curve** (−1.0%), and its loss concentrates
  exactly where the web focus cares: q10-35 on dense-detail photographic
  content (nature 1.058, textures 1.044, ai-clipart 1.037). m2 is also the
  tier where zen is now FASTER than libwebp (0.97x wall after the speed
  program) — there is speed budget to spend on RD here.

## (b) Adopted candidates hold per-image (generalization was right)

Per-image tails from the adoption TSVs:

- **uvdiff_rd** (m3-6): size negative on all 15 images (−0.04%…−1.88%);
  mean zsim positive on 12/15, the 3 near-zero means come with real size
  wins. Individual cells swing ±(mode-flip noise); no content pocket of harm.
- **trellis_skip** (m5/m6): quality positive on 13/15 images, worst cell
  −0.14 zsim, size ≤ +0.07% — uniform.
- **flatness penalty** (the rejected wash): per-image confirms pure noise
  (±0.04%, ±0.05 zsim, domination cells ~random 2-8/28). No pocket.

## (c) Rejected candidates: pick-rule hypotheses tested and REFUTED

- **fastalpha at m0/m1**: the 07-14 corpus showed one image
  (`7000-lilith-plots`) where the fast path strictly DOMINATED (6/8 cells
  smaller AND better). Tested on 13 line-art images: geomean **1.016**
  (a matched-quality LOSS), domination only on 2 aliased-pattern images
  (10/16, 7/16 cells) and nowhere else. The pocket is a narrow
  "aliased repeating line patterns" artifact, not a content class. **No
  pick rule; the rejection generalizes.** (zen m0/m1 baseline also beats
  libwebp on line-art: libwebp needs +1.1% at matched zsim.)
- **segs1 dq_uv**: per-image confirms a uniform operating-point shift
  (every image lands in the bigger+better quadrant, ~zero domination
  cells). Nothing to pick; only reachable via explicit segments=1 anyway.

## (d) New content map: the scans pocket, and psy generalizes

Line-art RD tiers (13 images, m4+m6 combined, vs libwebp): geomean 1.001 —
**parity**. But not uniform:

- plots/charts/brochures: 0.954–1.017, centered on parity.
- **patent scans: 1.033** (both images, both methods, 2.5–4.5% per cell) —
  the one measured content class where the tuned default is behind.

Single-knob ablations on the scans pocket, all NULL:

| variant | scans ratio vs libwebp |
|---|---|
| tuned default | 1.033 |
| enhanced CSF off | 1.033 |
| SATD masking blend off | 1.034 |
| SNS 0 (both encoders) | 1.038 |
| I4 flatness penalty ON | 1.033 |

The gap is structural (mode/RD decisions on near-binary strokes+flat
content), not any of the tested tuned extensions. Needs a mode-decision-level
diff (per-MB I4/I16/skip comparison against libwebp on a scan image) — the
harness is in place; recorded as future work.

Counter-finding worth keeping: **the SATD masking blend HELPS line-art**
(removing it moves the line-art geomean 1.001 → 1.005, grids 0.987 → 1.012).
The psy extensions generalize beyond the photographic content they were
tuned on; the earlier worry that they'd hurt graphics is refuted.

## Verdicts: generalize vs pick

1. **Everything adopted stays generalized.** Per-image tails are clean; no
   content-conditional gating needed.
2. **No per-image knob pick is warranted for any tested candidate.** The
   two hypothesized pockets (fastalpha-on-plots, segs1-dq_uv-anywhere)
   dissolve under a wider sample. The measured content variance of the
   tuned-vs-libwebp ratio at m4/m6 is ±1% — smaller than what a
   misclassifying picker would risk. This matches the earlier picker
   post-mortem: the axes where a picker could help are not these knobs.
3. **The Pareto work that remains is TIER-level, not per-image:**
   - **m2 (−1.0%, worst at low q on dense detail)** is the concrete target.
     It is faster than libwebp with a ~3-9% wall surplus; candidate levers:
     RD-score the I16-vs-I4 cut (or the top-2 SSE candidates) instead of
     pure SSE at m2, or import the m3 UV-RD pick. Any candidate must be
     A/B'd with the established harness (size + zsim + ms, this corpus).
   - **scans-class gap (+3.3% at RD tiers)**: needs the per-MB decision
     diff first; no knob-level fix exists among the tested ones.
4. **m0's +9% and m6's +1.2%** are the standing wins to protect: any future
   tuning A/B should keep this anchor sweep as its regression gate
   (`--encoder libwebp` arm makes it one command per side).
