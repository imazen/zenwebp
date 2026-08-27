# zenwebp target-zensim: instrument census + Zq-head wave (2026-08-27)

REGISTERED BEFORE ANY RUN. GOAL criterion 4 requires, per encoder, a
registered k2/k3 census on the REAL 27-cell instrument (9 corpus9 refs ×
t∈{70,80,88}) + a one-shot Zq head (consts form per the recorded
`feedback_no_zenpredict_in_codecs` resolution, zensim plan doc 2026-08-27).
zenwebp today: `ZensimTarget` loop (bucket-anchor first-pass q + one-pair
secant, max_passes) + a SYNTHETIC-image convergence CI (`967db71`) — the
instrument census has never run.

## Phase A — CONTROL census (no arm, the baseline measurement)
- Instrument: the same 9 refs the jxl waves use (3 coherence photos, 3
  CID22-512 validation, 3 gb82 576² crops) × t{70,80,88}; k2 = max_passes 2
  and k3 = max_passes 3 arms; JUDGE = decoded pixels scored by the v47-A
  bake through the canonical extractor route — never the loop's own
  internal score (independent-judge rule).
- Harness: `examples/zensim_census.rs` (zenwebp-owned, loop-ownership
  directive) emitting the same TSV schema as the jxl driver
  (image/class/target/seed_q/achieved/abs_err/bytes/passes).
- Report: median |achieved−target| overall + per class + per t; ±2-hit
  count; bytes. NO gate — phase A is the baseline the arm must beat.

## Phase B — Zq head arm (fit + census A/B)
- Fit: per-codec copy of `fit_zq_seed.py` (family rule) on the 07-01
  canonical `zenwebp_lossy` 9-pt q→zensim curves (PAVA-isotonic q*(t)
  labels, same 8-feature pool + robust-L1 + greedy LOO-origin-p90). Unit
  bridge: NONE — zenwebp's knob IS encoder quality q.
- Arm B = head-seeded first pass (env-gated hook, fallback = bucket
  anchors, G-J3 shape); arm A = the bucket anchors (the shipped baseline —
  a STRONGER control than jxl's staircase was; expected margin smaller).
- **Gates (family bar): PASS iff median decoded |err| improves ≥15% vs the
  bucket baseline AND ±2 hits do not regress.** FAIL = numbers committed,
  buckets stay — an acceptable outcome (the census itself closes the
  criterion's census requirement either way).

## Endgame
Census TSVs + harness committed here; verdict appended; zensim plan +
memory updated; ship decision (if PASS) user-gated as always.

## AMENDMENT (2026-08-27, before any run) — model-vintage discovery
`zenwebp`'s target-zensim stack depends on PUBLISHED zensim `0.2` (the
`PreviewV0_2` model), NOT the campaign-standard v47/B path-dep family: its
targets, bucket anchors, and tolerances are all calibrated on that scale.
Consequence for phase A: a v47-A judge alone would measure CROSS-MODEL
disagreement, not loop error. Phase A therefore judges decoded pixels with
BOTH models — `err_pub02` (self-consistent loop error, the loop's own
contract) and `err_v47a` (the fleet-standard offset, reported not gated).
Phase B's bar applies to `err_pub02` (like-for-like vs the bucket baseline).
A model-family upgrade for zenwebp's loop (path-dep zensim + re-anchored
buckets) is a SEPARATE registered wave if the user wants it — it invalidates
the shipped anchor tables and is not smuggled into this census.
