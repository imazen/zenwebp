# WebP complete-candidate binding — 2026-09-08

The exact packed model now drives scalar measurements and native segment
corrections through the existing WebP targeting loop. Neutral controls preserve
the fixed-quality bitstream; active controls change the actual segment quantizer
and decoded pixels. This is **binding and engagement evidence**, not model
qualification or a matched-quality compression win.

Registration: [candidate binding](../docs/zensim-candidate-binding-2026-09-08.md).
Machine-readable results: [JSON](zensim_candidate_binding_2026-09-08.json).
Full artifacts and scripts: `/mnt/v/output/zensim/webp-candidate-binding-2026-09-08/`.
Windows delivery directory: `zensim-validation-2026-09-08/webp-candidate-binding`
under the configured work share.

## What changed and why

`target-zensim` previously returned NaN and optimistic success for one-pass
budgets, bypassing strict undershoot enforcement. A quality-endpoint stop could
report eight passes after executing one. Best-effort misses/overshoots reported
success, and selection could discard an accepted undershoot in favor of an
out-of-band overshoot. New tests were observed failing against those defects
before repairs. Selection now honors the strict failure floor before the ship
band; a separate regression also caught and repaired the interaction between a
tight failure floor and a wider ship band.

Every targeted encode is now measured. `passes_used` is actual full encodes;
`targets_met` means membership in the ship band. Strict errors remain independent.
Finite negative targets are valid on their original scale. The RGBA test fixture
was always a large overshoot (~89 for target 80); its former assertion relied on
the optimistic flag. The updated test retains its quality floor and verifies the
actual band flag and alpha-bearing decode.

The hidden `__zensim-research` feature uses exact private runtime aliases:
zensim `6c7c14a300ace3b0e1631b754faf8bd99aebbcdf` and zenpredict
`05de3cbc44b19d8769077d99710b0146c761d6df`. Cargo's required SIMD dependencies
advance from archmage/magetypes 0.9.27 to 0.9.29. Registry zensim 0.2.7 and the
separate recompress Profile A calibration retain their owners and scale.

Complete `BakeScorer` attribution uses the cached source and current decoded
candidate, with explicit formula revision 1. Absolute integrals over clipped
16x16 macroblocks come from the retained 8-pixel binned map. Existing segment
policy aggregates those integrals over the encoder's actual assignments, using
actual pixel areas. There is no full-resolution candidate raster expansion.
Unsupported map terms/gates fail explicitly. Opaque sRGB8 is the experiment's
scope; the legacy RGB/RGBA scorer remains separate.

## Controlled screen

Canonical imazen-26 **training** origin 2010, original 205x256 variant; method 4,
4 segments, WebP 4:2:0, explicit initial q=80. Model D is the 1420-byte bake with
SHA-256 `cd1098b450ef6941b6925b24bcbd129715b6f07c4fe84838a92e13ab364ddea6`.
The release driver SHA-256 is
`0639c763ebef16b933267e2042815247dddb7e79dcd87f9b1f8248724c05ad6e`.

The one-pass baseline scored 73.673485. The engagement request was baseline minus
5, or 68.673485, with zero ship tolerances and a three-full-encode budget. The
preregistered fine-gap override 1000 forces one segment correction; this is not a
test of default controller convergence. Multi-pass statistics stay off for every
research pass. Neutral deliberately spends one counted encode with zero overrides.

| Selected output | Bytes | D score | SSIMULACRA2 ↑ | Butteraugli pnorm3 ↓ |
|---|---:|---:|---:|---:|
| Scalar | 16,040 | 73.673485 | 76.767407 | 1.069968 |
| Neutral | 16,040 | 73.673485 | 76.767407 | 1.069968 |
| Active | 15,728 | 72.955315 | 76.141553 | 1.090189 |

The active correction changes segment 3's quantizer **11 → 14**, matching its
requested +3 override; the other quantizers remain `[27,21,16]`. The assignment
has 208 macroblocks, including the partial right-edge blocks. Segment aggregation
counts sum to all 52,480 source pixels. The zero-override correction preserves
bytes, pixels, maps and segment assignments exactly. Every active repeat matches
bytes, pixels, maps and segment assignments exactly; the map changes after the
active pixels change.

All three selected outputs miss the screen's zero-width acceptance band. The
scalar trajectory does visit score 68.213150, but its strict lower ship bound
retains the initial feasible output. Neutral and scalar spend their budgets on
different trajectories; their selected equality alone is insufficient evidence
of neutrality. The fixed-q pass comparison above is the actual neutral control.

The 312-byte saving (1.945%) accompanies lower quality according to **all three
judges**. One source and a forced engagement probe do not establish matched RD,
target-range coverage, useful default spatial steering or a qualified model.

## Accounting and verification

- 13 full encodes, 13 ordinary independent decode/comparisons, 9 attribution
  evaluations (included in those comparisons), 2 non-neutral maps consumed
  across active and its repeat. No extra inner reconstruction/metric loop.
- 5 additional terminal decodes and complete-model scalar comparisons, each
  agreeing within 1e-5. The selected WebP bytes and PNG pixel hashes are checked.
- Both independent judges cover all 13 emitted probes: 26 comparisons total.
  All scores are finite; exact reference/distorted pair identities, coverage and
  uniqueness are checked. The PNG judge inputs reproduce the scored packed RGB8
  pixels exactly.
- Ten invalid-input controls reject missing/nonfinite/out-of-range seeds,
  unknown modes, profile aliases, alpha input, zero budgets, wrong formula
  revision, unsupported map terms and NaN targets before primary result output.
- Per-pass encode/decode/measurement and whole-loop/terminal times are retained
  in TSV/JSON. `run-heavy` reports each process peak RSS rounded to 0.01 GiB;
  that resolution and this single fixed-order screen do not support speed or
  memory comparisons.

Local checks: 354 library tests, 15 targeting tests and 33 validation tests pass
with the candidate feature (one pre-existing ignored library test). The legacy
targeting path also passed its library/targeting/validation checks. Candidate
trace and legacy-target library Clippy pass with warnings denied; release build,
scoped formatting and API snapshot check pass. A clipped-block regression compares
pixel and integral segment aggregation, including exact area counts.

The committed API snapshots were stale and generated on another architecture.
Regenerating base `018753cf` on this host proves that this change's only snapshot
delta is the excluded `__zensim-research` feature header; all other regenerated
lines already belong to the base. The preserved comparison is `API_DELTA.json`.

Required next work remains train-fitted, range-aware 1/2/3-shot targeting and
held-out matched RD using a qualified frozen model. No terminal holdout was used.
