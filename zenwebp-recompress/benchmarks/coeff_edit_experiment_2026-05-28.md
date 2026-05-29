# CoeffEdit experiment — VP8 coefficient-domain recompression vs Reencode

**Date:** 2026-05-28
**Crate commit:** see `git log` for `zenwebp-recompress/src/vp8x/` landing
**Question:** Can editing a WebP's VP8 DCT coefficients in place (no IDCT/FDCT
spatial round-trip) hit a smaller size at a target quality *better* than just
re-encoding the decoded pixels (`Reencode`)? The hypothesis — "coefficient
editing avoids generation loss" — is the premise behind treating CoeffEdit as
the project's headline strategy.

**Answer: No, for VP8/WebP.** The coefficient transcoder is correct and
pixel-exact, but every size-reducing coefficient edit is RD-dominated by
`Reencode` at matched output size. The reason is VP8 spatial intra-prediction
drift. CoeffEdit's no-generation-loss advantage is only realisable at the
verbatim (zero-size-change) point, which `LosslessRemux` already provides.
**The router does not select CoeffEdit.**

---

## What was built (and is kept)

A self-contained VP8 keyframe coefficient transcoder, `src/vp8x/`:

- `bool.rs` — matched boolean arithmetic decoder/encoder (round-trip tested).
- `tables.rs` — VP8 constants (token trees, prob tables, quant tables, zigzag).
- `parse.rs` — `parse_vp8_keyframe` → header + per-MB modes + quantized
  coefficient **levels** (no dequant, no IDCT).
- `emit.rs` — `emit_vp8_keyframe` re-emits the token stream.
- `edit.rs` — the edits: `drop_high_freq_ac` (zero AC scan positions ≥ keep)
  and `requantize` (coarsen the level grid by `f_dc`/`f_ac`, header untouched).

**Correctness is established, not assumed.** The verbatim transcode
(parse → re-emit unchanged) and the no-op requantize (`f = 1.0`) are
**bit-for-bit pixel-exact** against libwebp's decoder (MAD 0) on real corpus
images and synthetic gradients up to 512×384. This was the M2 gate; it passes.

---

## Method

Clean references: the five lossless WebPs in `tests/images/gallery2/*_webp_ll.webp`
(decoded to RGBA — true, uncompressed ground truth). For each:

1. Encode to an "already-compressed source" at libwebp q75, method 4.
2. **CoeffEdit RD curve:** transcode the source's VP8 chunk at a grid of
   `keep_ac` (AC-drop) and `(f_dc, f_ac)` (requantize), rewrap the container,
   decode, score zensim **Profile A vs the clean reference**, record bytes.
3. **Reencode RD curve:** `run_reencode_at_q` at a grid of q, decode, score
   vs the same clean reference, record bytes.
4. Compare quality at **matched output size**.

Scoring is zensim Profile A (`expert::score_rgba`). Raw logs archived at
`/mnt/v/zen/zenwebp-recompress/coeff-edit-2026-05-28/` (`rd_compare.log`
AC-drop, `rd_requant2.log` requantize).

Reproduce with the `expert` API:
`expert::run_coeff_edit_keep(src, keep_ac)` /
`expert::run_coeff_edit_requant(src, f_dc, f_ac)` vs
`expert::run_reencode` at chosen q, each decoded and scored against the clean
reference with `expert::score_rgba`.

---

## Result 1 — AC-drop is RD-dominated by Reencode

zensim-A vs clean reference, at matched output size (source = q75):

| ref | size (B) | CoeffEdit AC-drop | Reencode | winner |
|-----|----------|-------------------|----------|--------|
| 1 (400×301) | ~12300 | keep10 → **51.1** | q75 (12260) → **69.0** | Reencode +18 |
| 2 (386×395) | ~11100 | keep14 (11162) → **72.4** | q75 (11066) → **78.3** | Reencode +6, smaller |
| 3 (800×600) | ~54100 | keep12 (54082) → **76.0** | q75 (53746) → **74.9** | CoeffEdit +1.1 (wash) |
| 4 (421×163) | ~16850 | keep14 (16842) → **60.9** | q75 (16870) → **80.5** | Reencode +19.6 |
| 5 (300×300) | ~53000 | keep14 (53268) → **65.5** | q75 (52658) → **68.5** | Reencode +3, smaller |

Reencode wins decisively on 4/5; ref 3 (smooth photo) is a ~1-point wash.

## Result 2 — Requantization is worse still

Coarsening the level grid (header/dequant untouched, `f_dc=1.0` to protect DC):

| ref | requant `f_ac=1.5` | Reencode at ~same size | gap |
|-----|--------------------|------------------------|-----|
| 1 | 12502 B → **23.1** | q70 (12040) → **64.6** | −41 |
| 2 | 10900 B → **29.4** | q70 (10864) → **77.9** | −48 |
| 3 | 53950 B → **39.4** | q75 (53746) → **74.9** | −35 |
| 4 | 16430 B → **31.3** | q60 (16134) → **77.0** | −46 |
| 5 | 51652 B → **12.7** | q70 (51812) → **64.4** | −52 |

Requantize is catastrophically dominated — even a *mild* `f_ac=1.25` produces
**MAD ≈ 15.6** against the source decode for only a ~0.5 % byte saving. (The
Y2-block DC carrier was handled correctly — protecting it changed the result by
< 0.2 zensim, so it is not the cause.)

---

## Why — VP8 spatial intra-prediction drift

VP8 keyframes predict every block from its neighbours' **reconstructed**
pixels, then add the block's residual (the DCT coefficients). The encoder chose
each block's residual *relative to a specific predicted base*. Perturb any
coefficient and:

1. that block's reconstruction changes;
2. the next block predicts from the changed pixels, so its prediction base
   shifts even though its own residual is untouched;
3. the error **compounds across the whole frame** along the prediction chain.

A localised, mild coefficient change therefore produces a large, image-wide
pixel error. `Reencode` sidesteps this entirely: it re-derives residuals from
the clean decoded pixels *after* prediction and re-runs RD optimisation
(trellis, mode selection at method 4), so it spends bits where they matter at
the new rate. Coefficient editing has neither the clean prediction base nor any
RD re-optimisation.

**This is codec-specific.** Baseline JPEG has no inter-block spatial
prediction (DC is differentially coded but losslessly reversible), so JPEG
coefficient requantization *is* drift-free and competitive — which is why
jpegtran/mozjpeg-style coefficient transcoding works there. VP8's intra
prediction is exactly the feature that defeats the same idea for WebP.

---

## Result 3 — drift compensation does not rescue it either

The natural objection: Reencode re-FDCTs *everything* (generation loss even on
coefficients that didn't need changing), whereas a coefficient edit preserves
untouched levels *exactly* (zero generation loss) — measured here as a 3–10.7
zensim-A advantage of the source over a same-quality (q75→q75) re-encode. If
the drift from editing a few coefficients could be *cancelled*, that advantage
might survive.

So a closed-loop **DC drift compensator** was built (`src/vp8x/compensate.rs`):
decode the unedited source as the target, then after the edit iterate
emit→decode→measure-per-MB-mean-error→nudge the DC level (Y2 DC for luma,
UV DC for chroma) using the analytic level→pixel DC gain, to a fixed point.

It does not help:

- **Unstable.** VP8's prediction chain couples every block, so a simultaneous
  (Jacobi) correction forms a traveling wave that overshoots toward the
  bottom-right. At relax ≥ 0.5 it diverges into a limit cycle (luma mean-abs
  drift 22 → 54). Only heavy under-relaxation (≤ 0.25) is stable.
- **Insufficient.** Even stable, it removes only ~10 % of the drift (22.25 →
  19.9 luma mean-abs on a synthetic): the bulk of the error is the *intended*
  high-frequency loss plus *non-DC* structural drift, which DC nudging can't
  touch. It also adds DC magnitude → larger files.
- **RD still dominated.** zensim-A vs clean at matched size (source q75):

  | ref | uncomp keep10 | DC-compensated keep10 | Reencode (matched) |
  |-----|---------------|------------------------|--------------------|
  | 1 | z=51.1@12304 | z=12.9@12896 (worse + bigger) | q75 z=69.0@12260 |
  | 3 | z=66.8@53866 | z=63.0@54152 | q70 z=73.1@53360 |
  | 4 | z=39.9@15788 | z=42.8@16054 (+2.9) | q75 z=80.5@16870 |

  Compensation is inconsistent (tiny help on ref 4, catastrophic on ref 1) and
  never closes the gap to Reencode.

The structural reason, now empirically grounded: *complete, stable* drift
compensation requires sequential (Gauss-Seidel) reconstruction in decode order
with full-frequency residual re-derivation — which is exactly what re-encoding
does. Partial (DC-only) compensation is both unstable on the prediction chain
and far too weak. There is no coefficient-domain shortcut that beats Reencode
for VP8.

## Decision

- **Keep** the transcoder + both edits + the DC compensator as validated
  infrastructure, reachable via `expert::run_coeff_edit_keep` /
  `expert::run_coeff_edit_requant` / `expert::run_coeff_edit` (research +
  reproduction). The compensator lives in `vp8x::compensate` (crate-internal).
- **Do not** let the router select CoeffEdit (`router::filter_candidates`
  excludes it, alongside the separately-falsified `DeblockReencode`).
- CoeffEdit's only loss-free operating point is verbatim, already covered by
  `LosslessRemux`.
- For WebP recompression, the productive strategies remain `Reencode`,
  `LosslessReencode`, and `LosslessRemux`; the `better_handled_by_jxl` signal
  on the result type points callers at JXL when even those waste bytes.
