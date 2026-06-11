# Variant Generation: zenwebp's adoption of the zenjpeg patterns

Written 2026-06-11. Codec-neutral patterns:
`zenjpeg/docs/VARIANT_GENERATION.md`. Code: `src/sweep.rs` (`__expert`),
`examples/sweep_validate.rs`, results in
`benchmarks/sweep_validate_webp_2026-06-11.tsv`.

## Pattern-by-pattern

1. **Discrimination**: mode is the axis — `SweepVariant{Lossy,Lossless}`
   over the already-separate `LossyConfig`/`LosslessConfig` types; a
   lossless cell cannot spell an SNS strength. Excluded with reasons (in
   module docs): `target_*` closed loops, `preset` macro-knob,
   `alpha_quality`/`exact` (class-conditional, pattern 10),
   `near_lossless` (metric-class), `multi_pass_stats` outside m4
   (gate-shadowed).
2. **Dominance/trial/metric audit**: the **entire curated lossless space
   is trial-class** — VP8L method and "quality" are effort dials; decoded
   pixels are identical across every lossless cell (harness-gated as
   exact roundtrip), so `min(bytes)` over lossless cells is exact. An
   exact lossless trial helper is the natural follow-up (encode the 5
   strata, ship the smallest — no shared state needed at these sizes).
   Lossy: everything pixel-affecting (method, sns, filter, segments,
   sharp-yuv, cost model) is metric-class; no further dominance cases
   found beyond what the encoder already does unconditionally.
3. **Resolution**: holds trivially — `SweepVariant::build()` constructs
   the literal config; no hidden resolution layer exists (presets are
   excluded), so there is no second implementation to drift.
4. **Fingerprints**: every field hashed (resolved internal params, not
   the label). No exclusions to prove wrong. 0 aliases in the curated
   space (no anchor clamps/plateaus in webp's quality chain).
5. **Planner**: rd_core / modes_full, main-effects-first,
   lossy-before-lossless per deviation class, one-value-at-a-time budget
   ladder with floors (methods ≥2 lossy / ≥1 lossless).
6. **Validation** (first run, 41 dev≤1 cells × 7 images, 18.5 s): ALL
   HARD CHECKS PASSED, 0 warnings — every step live (diff rates
   14–100 %), every cell decoded (pattern 14), every lossless cell
   roundtripped exactly, ordering intact. Corpus crosses the 16-px
   macroblock topology via a 509×381 crop (pattern 15). Notable data:
   m6 −3.2 % bytes, sns0 +4.9 %, seg1 +2.9 %, mpass live at m4 (−0.3 %),
   `vp8l-m6` only 14 % byte-diff vs m4 (weak but live).
7. **Id grammar**: `vp8-m4_def[-syuv][-sns<v>][-flt<v>][-seg<n>]_q<q>` /
   `vp8l-m<m>[-ql<v>]`; `variant_from_cell_id` + label registry
   (def/parity/mpass/smooth/plim50); grammar-totality roundtrip test.
8. **Executor wiring**: zenmetrics plan-cell bridge — next commit.

## Known limits / open items

- `vp8l-m6` near-ties m4 on this corpus (14 % diff) — keep, but a
  larger-image corpus leg would characterize where m6 pays.
- Alpha axes (`alpha_quality`, `exact`) need an RGBA corpus + alpha-aware
  metric before they can be swept honestly (pattern 10).
- Exact lossless trial helper (pattern 2 finding) not yet shipped.
- `near_lossless` sweeps belong in metric-scored fleet runs, not the
  curated trial-class axes.
