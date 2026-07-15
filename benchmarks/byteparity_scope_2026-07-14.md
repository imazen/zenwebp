# StrictLibwebpParity byte-identity — actual scope (2026-07-14)

## CURRENT STATE (2026-07-15): 3407/4004 = 85.1% byte-identical

Four parity-gated fixes this session took the grid **24% → 85.1%**: base-quant
`52cf96f2`, segmentation-collapse `41923466`, trailing-slots `7acdd775`,
**skip-proba `91c96168`**. The skip-proba one was NOT a StatLoop rearchitecture
(I wrongly called it deep and stopped — it was a one-line gate): instrumented
libwebp always writes `use_skip_proba = 0` (there's an unconditional
`assert(use_skip_proba == 0)` at `VP8EncTokenLoop` entry; the flag is never
enabled in the shipping encoder), so parity just forces
`macroblock_no_skip_coeff = None`. Closed the whole low-q cluster (+256 cells).

**Remaining ~15% is luma I4/I16 mode-RD**, in two overlapping clusters:
- **m6, all q (~207 cells).** The m5→m6 delta is trellis-during-I4-mode-selection
  (RD_OPT_TRELLIS_ALL): q40 m5 has `b4_same` 832/832 (all I4 sub-modes match) but
  q40 m6 has 377/768 + `y_same` 90.8% + i4-count off by ~4. **Traced (q40 m6,
  382297):** it's a per-MB coefficient-RATE cascade, NOT a lambda/DP bug. The
  I4-trellis lambdas match (`(7·q²)>>3`), block-score lambda matches
  (`lambda_mode=18`), D/SD/H/modes match — only R diverges, driven by an nz-CONTEXT
  difference (mb(11,8) sub-block 0: zen `ctx=1 t1+l0`, lib `ctx=2 t1+l1`). The R
  drift starts BEFORE the first mode flip (mb(10,8), not a flip, already has zen
  I16 R≈56210 vs lib 56470), so it cascades MB-by-MB. **Root traced to mb(4,0)**
  (an all-MB I16-D/R diff found it as the first diverging MB in raster order).
  At mb(4,0) sub-block 0, zen and libwebp have the SAME incoming context (`ctx=0
  t0+l0`) and matching lambdas — yet zen's I4 sub-mode selection picks HU while
  libwebp picks TM. Both winning sub-modes are **nz=0** (empty block), so the
  coefficient rate should be mode-independent, yet zen's R=89 vs libwebp's R=229.
  So the divergence is the **per-candidate-mode I4 *trellis rate* at m6** — the
  deepest layer: lambda (`lambda_i4=(3·q²)>>7=56`, matches), context, trellis DP
  (byte-identical at m5), and D are ALL ruled out. Next: per-mode trellis-rate
  dump at mb(4,0)#0 (same-mode zen-vs-lib) to pin the rate-accounting difference;
  it's mode-independent-looking (nz=0 R differs), so likely one systematic fix
  closes the cluster. Instrumented libwebp `LIBI16`/`LIBI4blk`/`LAMDBG`/`A16`
  hooks + zen `MB_DEBUG` are in place in the scratchpad.
- **High-q q80–95, m3–m5.** Milder luma mode flips (`y_same` ~97%) + `n_proba_updates`
  off by a few. The n_proba is DOWNSTREAM of modes (when modes match, n_proba
  matches — verified at q40 m5), so the mode-RD is the root.

Both are per-MB RD-score matching (the genuinely harder tail), but NOT assumed
deep — trace each flip against instrumented libwebp before concluding.

## TL;DR

The #38 parity work makes `CostModel::StrictLibwebpParity` **byte-identical to
libwebp at the specific operating point it was traced against** (q75, CID22
382297, two configs), across all 7 methods. It is **NOT** byte-identical in
general. The broad grid below started at **972/4004 (24%)** byte-identical. #38's
north-star (byte-identical at *all* matching settings) is not met — this doc
records exactly how far it reaches so the next session doesn't over-trust the
`methodcmp` 14/14.

**Update (2026-07-15): trailing-segment-slot fix → 3151/4004 (78.7%).** The VP8
segment header always carries 4 quant+filter slots; libwebp leaves the slots
beyond the configured count (`[config..4]`) at base/seg0 (a 2-segment encode has
`dqm[2]==dqm[3]==dqm[0]`), zen held seg1's values. Fix (parity-gated,
`vp8/mod.rs`): point `[config..4]` at seg0 using the pre-simplify count (so
`SimplifySegments`' correct `[num_final..config]` replication isn't clobbered).
sns30/segs2 **30.4% → 78.8%**; all four configs now converged at **78–79%** — the
residual is now a COMMON cluster (low-q skip/quant, high-q, m6 mode-RD),
config-independent. Commit `7acdd775`.

**Update (2026-07-15): segmentation-collapse fix → 2666/4004 (66.6%).** Biggest
single jump. libwebp writes `segmentation_enabled = (num_segments > 1)` *after*
`SimplifySegments` merges equivalent segments; at sns=0 the SNS quantizer spread
is 0, so all segments are uniform → libwebp collapses to 1 and turns segmentation
OFF. zen set `segments_enabled` unconditionally, emitting a full 4-segment header
where libwebp emits none — so the entire sns0/segs>1 config diverged. Fix
(parity-gated, `vp8/mod.rs`): `segments_enabled = num_segments > 1`. Per-config:
sns0/segs4 **0% → 78.3%** (now identical to sns0/segs1), sns50/segs4 **48.6% →
79.3%**, sns30/segs2 **0% → 30.4%**, sns0/segs1 78.3% (unchanged). q75 14/14 held;
tuned byte-unchanged. Commit `41923466`. Tuned-adoption candidate (strict
byte-saving when it fires) pending a sweep.

**Update (2026-07-15): base-quant round→truncate fix → 1270/4004 (32%).**
First generalization step: `setup_encoding` computed the segs1 base quant with
`quality_to_quant_index` (which **rounds** `127*(1-c)`), but libwebp truncates
(`VP8SetSegmentParams`). They diverge by +1 at q10/30/50/80 (frac ≥ 0.5), which
q75 (26.20, rounds==truncates) could never expose. Parity now uses
`quality_to_quant_index_trunc`. Effect on the clean **sns0/segs1** config:
**48.6% → 78.3%** identical (q30 0→6/7, q50 0→7/7; q10 and q80 still carry a
*second* divergence). Tuned default byte-unchanged, q75 still 14/14. The
remaining open axes below are unchanged (segs4 mechanism, q5/q10/q80 second
divergence, high-q q90/q95).

## Why the 14/14 was misleading

`methodcmp` (the gate that reported 14/14) fixes **q=75** and **one image**
(382297) and tests exactly two configs: (SNS=0, filter=0, segs1) and (SNS=50,
filter=60, segs4). Those are the settings every differential trace this session
used (`TARGX`/`TARGY` at q75 on 382297). So the fixes are tuned to make *that
point* byte-exact; nothing forced the other points to converge.

## The broad sweep

`dev`-style bin `byteparity_sweep` (in the scratchpad harness): 13 images
(3 CID22 512² + 10 synthetic incl. 1×1, 2×2, 3×3, 17×17, 33×17 odd-chroma,
edge-partial MBs) × q ∈ {5,10,20,30,40,50,60,70,80,90,95} × 4 configs
{(0,0,1),(50,60,4),(0,0,4),(30,20,2)} × m0–6 = **4004 cells**. The 3032-line
raw fail list (196 KB) is reproducible via the `byteparity_sweep` bin and is not
committed (>30 KB); the breakdown tables below are the durable record.

**972 / 4004 = 24.3% byte-identical.**

### By config (of 1001 each) — the dominant axis
| config (sns/flt/segs) | % identical |
|---|---|
| (0, 0, 1)   | 48.6% |
| (50, 60, 4) | 48.6% |
| (0, 0, 4)   | **0.0%** |
| (30, 20, 2) | **0.0%** |

The two `methodcmp` configs sit at ~49%; the two it never tested are **0%**.
`(SNS=0, segs4)` and `(SNS=30, segs2)` never produce a byte-identical file at any
q/method/image — a concrete segmentation/SNS-interaction divergence to trace.

### By quality (of 364 each)
q5 31% · q10 12% · q20 32% · q30 13% · q40 33% · q50 12% · q60 32% · q70 34% ·
q80 13% · q90 29% · q95 25%. Clear odd/even oscillation (odd steps ~12%, even
~32%) — a quantizer-index parity effect. **No q exceeds 34%** (q75 itself, the
traced point, is not in this grid — it is the outlier at 100% for the two tested
configs).

### By method (of 572 each)
m0 29% · m1 29% · m2 30% · m3 22% · m4 22% · m5 21% · m6 17%. The RD tiers (m3–6)
are worse — more places to diverge.

### By image (of 308 each)
382297 24% (the traced image, best of the reals) · 1025469 8% · 1418519 11% ·
tiny synth 16% · larger synth 27–40%. Other CID22 images are far worse than the
one that was traced — direct evidence the fixes are 382297-specific.

### First divergence
2462 of 3032 diffs first differ in the header (byte < 8 → the RIFF/VP8 size
field, i.e. total size differs); 570 match a prefix then diverge in content.

## What this means

- The parity fixes are correct **where traced** and buy real value there (a
  user encoding at ~q75 with those configs gets byte-identical output). But the
  mode "byte-identical to libwebp" claim only holds at that point.
- Generalizing is open work along three axes, most tractable first:
  1. **Config:** `(SNS=0, segs4)` and `(SNS=30, segs2)` are 0% — a specific
     segment-quantizer / SNS-derivation divergence. Trace like the q75 work but
     at those configs.
  2. **Quality:** the odd/even q oscillation points at quantizer-table index
     rounding (`VP8SetSegmentParams` / `q` → `qi` mapping). One fix likely lifts
     many cells.
  3. **Content:** once config+q converge on 382297, re-sweep the other CID22 +
     synthetics.
- **Do not re-report 14/14 as "parity complete."** Use this grid's 24% as the
  headline number and drive it up.

## Default config (sns50/filter60/segs4) — diagnosis (2026-07-15)

Per the scope decision, the Default preset was investigated first.

### Two facts established with a real-libwebp-C oracle

The vendored libwebp source (`scratchpad/lwsrc/`, `trace_driver`) was rebuilt
with instrumentation and used as ground truth:

1. **`webpx` == real libwebp C, byte-for-byte.** At q30 m4 sns50 segs4, zen ==
   webpx == `trace_driver` output (all 27944 bytes identical). So the
   `webpx`-based byteparity baseline *is* canonical libwebp — the grid numbers
   are trustworthy.
2. **The base-quant fix (`52cf96f2`) closed far more than the 4004-grid +8%
   implied for this config.** The Default preset on 382297 is now **47/77 (61%)**
   byte-identical, not ~49%.

### METHODOLOGY WARNING — rebuild every harness binary after a lib change

The first diagnosis pass (committed then corrected here) chased a **phantom**:
`segfielddiff` was built *before* the base-quant fix, so its zen encode still
rounded the q30 base quant (53 vs the fixed 52), producing a fake `seg_lf`
divergence at q30 **m4**. The 3-way oracle proved q30 m4 is fully
byte-identical. The harness (`webp-ll-compare`) builds zenwebp into its *own*
target, so rebuilding the lib in the zenwebp repo does **not** refresh a harness
binary — you must rebuild the binary. Always rebuild ALL harness bins
(methodcmp, byteparity_sweep, segfielddiff, …) after any lib edit before trusting
their output. (Also: `bitexact_diff`'s field diffs print *before* their method's
summary line — attribute each diff to the method whose summary follows it.)

### Actual current divergence map (382297, post base-quant fix)

```
q\m  0 1 2 3 4 5 6      . = byte-identical
  5  X X . X X X X      m2 is 100% identical across all q
 10  X X . . . X X      m0/m1 identical except low-q (skip_prob)
 20  . X . . . X X      m3/m4 identical mid-q
 30  . . . . . X X      divergences cluster at LOW-q and HIGH-q,
 40  . . . . . . .        consistent with parity traced at q75
 50  . . . . . . .
 60  . . . X . . .
 70  . . . . X . X
 80  . . . X X X X
 90  . . . X X X X
 95  . . . X X X X
```

### Real remaining causes (distinct, q-regime-clustered)

- **m0/m1 low-q `use_skip`/`skip_prob`** (q5/q10): zen sets `use_skip=1`
  (`skip_prob=248`) where libwebp uses `use_skip=0`. NOT a shallow threshold fix:
  the decision is `use_skip_proba = (skip_proba < 250)` on both sides and the
  formula is the same modulo rounding (zen rounds `255*non_skip/total`; libwebp's
  `CalcSkipProba` truncates — worth aligning, but the wrong direction to explain
  248<250 alone), so the divergence is the **`nb_skip` count**. libwebp counts
  `nb_skip` inside `StatLoop` via `VP8Decimate` per MB (`frame_enc.c:625`),
  entangling this with the multi-pass `StatLoop` architecture — i.e. issue #25
  (SKIP_PROBA) **and** #27 (full multi-pass StatLoop), both major open items, not
  a one-line gate.
- **m5 `seg_lf`** (low-mid q): the trellis-path StoreMaxDelta (`6b4fa0c` fixed it
  at q75; still diverges off-q75). NOTE the earlier "seg_lf/max_edge" narrative
  applied to m5-trellis, **not** the non-trellis m4 path — m4 is identical.
- **m6 mode-RD** (`y_same` 90–99%) + `n_proba_updates` — I4/mode RD divergence.
- **High-q (q80+) across m3–m6** — proba/RD divergence at fine quant.

**Next step:** the m0/m1 low-q `skip_prob` threshold is the most tractable;
trace with the now-instrumented `trace_driver` (env dumps `LWSMD`/`LWSFS`/`LWADJ`
added for filter work; add a skip-proba dump). The `seg_lf`/`max_edge` machinery
(`adjust_filter_strength`, `store_max_delta` in `vp8/mod.rs`) was verified
CORRECT on the non-trellis path (zen and libwebp both compute seg2 max_edge=2,
final=20 at q30 m4) — the m5 divergence is trellis-specific.

## Method note

`methodcmp` should be extended to sweep q and multiple images (or replaced by
`byteparity_sweep`) so a single-point 14/14 can't again be mistaken for general
byte-exactness. The zensim/pixel gates are unaffected — this is purely about the
*byte-identical* claim.
