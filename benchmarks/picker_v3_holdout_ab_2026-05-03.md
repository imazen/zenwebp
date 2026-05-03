# Picker v0.2 (ZNPR v3) held-out A/B — 2026-05-03

Held-out gate before shipping the v0.2 ZNPR v3 bake. Pass criterion
(per the v3 picker brief): total bytes ≤ bucket-table at every matched
target_zensim.

**Result: REGRESSION at zq=75 and zq=80.** Picker is +0.013% across
the full grid (basically flat) but the per-target breakdown shows it
trades a 0.4-0.6% win at zq≥85 for a 1.0-1.4% loss at zq≤80. The brief
explicitly forbids shipping when the picker loses against the bucket
table; we did not promote the v0.2 .bin into the production runtime
path.

## Setup

- Held-out corpus: `~/work/zentrain-corpus/mlp-validate/cid22-val`
  (41 images, fully separate from the v0.1 sweep training corpus).
- Targets: zq ∈ {75, 80, 85, 90}.
- Tool: `dev/picker_ab_eval.rs`, run twice — once with
  `--features "target-zensim analyzer picker"` (v0.2 ZNPR v3 bake),
  once with `--features "target-zensim analyzer"` (bucket-table fallback,
  exact same convergence loop, exact same Auto preset).
- Encoder: zenwebp 0.4.5 + the runtime changes from this branch
  (Predictor::predict_with_specs, Option<u8> scalar heads,
  bucket-default fallback for Default sentinels).
- Picker bake: `benchmarks/zenwebp_picker_v0.2.bin` (35.4 KB i8,
  ZNPR v3, schema_hash 0x2599fdceee7e9aee, n_inputs=78, n_outputs=24).
- Trainer: `train_hybrid.py --activation leakyrelu --seed 51966
  --hidden 128,128`. Argmin acc 45.8% on held-out (val) split,
  mean overhead 2.24%.

## Per-target totals

| target_zq | picker bytes | bucket bytes | Δ bytes | Δ % |
|-----------|--------------|--------------|---------|-----|
| 75        | 1,225,566    | 1,213,896    | +11,670 | +0.961% |
| 80        | 1,707,644    | 1,684,446    | +23,198 | +1.377% |
| 85        | 2,430,250    | 2,439,218    | −8,968  | −0.368% |
| 90        | 3,910,428    | 3,935,128    | −24,700 | −0.628% |
| **Total** | **9,273,888**| **9,272,688**| **+1,200** | **+0.013%** |

Achieved-zensim averages match (82.23 picker / 82.14 bucket); both
hit 100% reach. The size difference is genuine, not noise from
different operating points.

## Diagnosis

Two contributing factors, in order of likely impact:

1. **Cell mis-pick at low zq.** The picker assigns more zq=75/80 rows
   to cells with 4-segment AQ (worse for already-low-bitrate, where
   the segment overhead doesn't pay back). The bucket-table's
   per-content-type defaults route low-q work through 1-segment cells.
   Argmin acc on the v0.1 sweep was 45.8%, mean overhead 2.24% —
   the student MLP picks the right cell less than half the time, and
   when it misses at low quality the cell delta is large because every
   byte matters more.

2. **Scalar-head bias at low zq.** SNS / filter_strength regression
   targets concentrate near 0 at low quality (the encoder doesn't
   need much filtering when the bitrate is already aggressive). The
   student RMSE on these heads is 33.5 / 22.7, which is teacher-floor
   (i.e. the student matches the teacher's noise) but the absolute
   error overwhelms the picker's value at zq=75 specifically.

## Concrete next steps (not done in this run)

- Re-train against an objective that weights low-zq error proportionally
  to its production cost. Today's MSE loss on log-bytes treats a 1%
  byte miss at zq=75 the same as a 1% miss at zq=90, but the absolute
  bytes-at-stake at zq=90 is 3.2× larger so the loss cares more about
  the high-zq tail.
- Add `output_specs` sparse overrides for the (cell, scalar) pairs
  that consistently miss at low zq, pinning them to the bucket-table
  values. The v3 format supports this — `SPARSE_OVERRIDES` is wired
  end-to-end in `dev/inject_v3_specs.py` — but tuning the overrides
  needs a separate held-out sweep.
- Investigate whether the v0.1 sweep's effective_max_zensim gap (the
  DATA_STARVED_SIZE / UNCAPPED_ZQ_GRID safety_report violations) is
  poisoning the low-zq labels via unreachable-cell contamination.

## Artifacts

All bake artifacts are archived to block storage at
`/mnt/v/output/zenwebp/picker_v0.2_2026-05-03/` rather than committed
to git (the .bin is 36 KB which trips the >30 KB-no-explicit-confirm
rule, and the trainer JSON is 1 MB). The repo's `.gitignore` now
covers `benchmarks/zenwebp_picker_v0.*.bin` and the matching
`.manifest.json` / `_bake_*.log` siblings.

- Picker .bin (not promoted to production):
  `/mnt/v/output/zenwebp/picker_v0.2_2026-05-03/zenwebp_picker_v0.2.bin`
  (36,132 bytes ZNPR v3 i8, schema_hash 0x2599fdceee7e9aee)
- Trainer JSON:
  `/mnt/v/output/zenwebp/picker_v0.2_2026-05-03/zenwebp_hybrid_v3.json`
- Train log + bake log: same directory.
- Per-row TSVs: `/tmp/picker_v3_ab.tsv`, `/tmp/bucket_ab.tsv`
  (164 rows each — ephemeral).
- Trainer-side updates landed under
  `~/work/zen/zenanalyze/zentrain/examples/zenwebp_picker_config.py`
  (v0.1 sweep paths, OUTPUT_SPECS, retired-feature handling) and
  `~/work/zen/zenanalyze/tools/bake_picker.py` (Infinity literals
  in feature_bounds → finite f32-max sentinels).
- Bake-time v3-spec injection lives at `dev/inject_v3_specs.py`.

## Verdict

Do not promote `zenwebp_picker_v0.2.bin` to the runtime. The shipped
`zenwebp_picker_v0.1.bin` stays active as the production picker until
a v0.3 bake clears the held-out gate at every target the brief
specifies.
