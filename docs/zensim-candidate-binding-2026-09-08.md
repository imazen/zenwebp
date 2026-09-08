# WebP complete-candidate binding and accounting registration — 2026-09-08

Registered before implementation or candidate runs. Base: zenwebp `018753cf`.
This follows the September 8 JXL, AVIF and JPEG binding/neutral-control work;
none of those experiments qualified a model or demonstrated a spatial RD win.

## Public behavior repairs

The existing `target-zensim` loop owns encoding and segment decisions. Keep its
registry zensim 0.2 scorer and the separately calibrated recompress Profile A
workspace unchanged. Repair these observable targeting semantics:

- Every targeted encode, including a one-pass budget, is decoded and measured.
  The strict undershoot error applies to one pass too; no extra encode is added.
- `passes_used` counts completed full encodes, including an early no-move stop.
- `targets_met` means the emitted candidate is in the configured ship band
  `[target - max_undershoot_ship.unwrap_or(0), target + max_overshoot.unwrap_or(infinity)]`.
  Best effort may return `Ok` with this flag false; `max_undershoot` independently
  controls errors. No-target metrics keep their documented NaN/one/true values.
- Candidate selection respects the strict failure floor first, then prefers a
  candidate in the ship band, then fewer bytes
  among accepted candidates, then the existing best-effort fallback.
- Accept all finite negative targets without rescaling: the public validation
  range becomes `f32::MIN..=100`. NaN, infinities and targets above 100 fail.
  Existing legacy seed tables remain legacy heuristics, with endpoint saturation.

Observe new regression tests fail against the old implementation before repairs.
Validate legacy RGB/RGBA targeting, strict errors, real pass accounting and API
documentation locally. No crate publication is part of this work.

## Opt-in candidate experiment

Add hidden `__zensim-research` feature for the current phase-3 trace driver only.
Use private optional dependency aliases for the exact zensim complete BakeScorer
and zenpredict Model runtime (Git revisions recorded in Cargo.lock). These types
do not cross WebP public signatures or change the zenanalyze Offer contract.

An explicit bake file, formula revision 1, finite seed quality in [0,100], and
`scalar|neutral|active` mode are mandatory. Initial scope is tightly packed opaque
sRGB8, dimensions at least 8, with unsupported spatial feature IDs and corruption
gates rejected. Do not infer an approximate profile from a file name. Do not use
legacy calibrated anchors to seed a different score function. Negative scores
retain their original scale. Preserve the existing legacy RGBA path.

Use the complete BakeScorer scalar and cached finite-difference attribution
surface. Query absolute integrated density over each actual clipped 16x16 block
from an 8-pixel binned map. Keep an explicit macroblock-map representation;
do not expand it into a full-resolution raster or infer its layout from width.
Aggregate those integrals divided by actual pixel counts over the encoder's
current k-means segment assignment, reusing the existing override policy.

Scalar skips maps and segment changes. Neutral evaluates maps but supplies zero
segment overrides. Active applies the existing bounded overrides. For a forced
phase-3 engagement probe, neutral executes the same attempted correction even
when overrides are zero; this deliberate no-op encode is counted. After that,
the ordinary global-q trajectory resumes. Fix multi-pass statistics off for all
research passes so that neutral comparisons are not confounded by an unrelated
encoder setting. Never report a computed but unused map as consumed.

Extend the existing trace driver with a fresh output directory and strict opaque
PNG input. Save every encoded WebP, exact decoded pixels, block integrals, segment
assignments, requested overrides and actual quantizer indices. Independently
decode and complete-model score selected bytes; record bake/source/binary hashes,
commands, time and full encode/decode/scalar/map counts.

The initial bounded screen uses canonical imazen-26 training origin 2010 (the
same source as the JPEG screen), fixed method 4 and explicit seed quality.
Compare scalar, neutral, active and an exact active repeat, including a forced
fine-gap engagement case and partial right-edge blocks. Success means honest
accounting, model binding, neutral identity and actuator engagement. It is not
model qualification, target-range coverage, a fitted heuristic or an RD claim.
Independent SSIMULACRA2 and Butteraugli measurements accompany changed outputs.
Realistic train-fitted 1/2/3-shot targeting and held-out matched RD remain separate
required work after this binding screen.
