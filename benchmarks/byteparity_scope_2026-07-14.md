# StrictLibwebpParity byte-identity — actual scope (2026-07-14)

## TL;DR

The #38 parity work makes `CostModel::StrictLibwebpParity` **byte-identical to
libwebp at the specific operating point it was traced against** (q75, CID22
382297, two configs), across all 7 methods. It is **NOT** byte-identical in
general. The broad grid below started at **972/4004 (24%)** byte-identical. #38's
north-star (byte-identical at *all* matching settings) is not met — this doc
records exactly how far it reaches so the next session doesn't over-trust the
`methodcmp` 14/14.

**Update (same day): base-quant round→truncate fix lands → 1270/4004 (32%).**
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

## Default config (sns50/filter60/segs4) — diagnosis (2026-07-14)

Per the scope decision, the Default preset was investigated first. Field-level
diff (via `dev/bitexact_diff.rs`, retargeted to sweep q at this config) shows the
**dominant divergence is `seg_lf` — the per-segment loop-filter levels**. At
m2–m5 across many q, one segment's filter comes out lower in zen (e.g. q30 m4
seg2: zen=20 lib=50; q20 m4/m5 seg3: 18 vs 36; q5 m3/m4 seg1: 29 vs 55). All
other fields (`seg_q`, modes: `y_same`/`uv_same`/`b4_same` = 100%) match.

**This is contained, not a cascade.** The VP8 loop filter doesn't affect
keyframe intra prediction (neighbours are read pre-filter), so these cells have
`1st-diff@36-38` (the filter header field) with identical coefficient content —
getting `seg_lf` right makes them byte-identical, no downstream effect.

`seg_lf` is produced by `adjust_filter_strength` (`vp8/mod.rs`) from
`max_edge_per_segment`, accumulated per-MB by `store_max_delta` over "blocky"
I16 MBs (all Y1 AC zero, Y2 nonzero), gated on `D > min_disto`. At q30 m4 the
divergence is `max_edge[seg2]`: zen=2, libwebp≈5 — zen under-counts the edge.

**Ruled out (instrumented + source-compared against vendored libwebp):**
- **Y2 coefficient order** — libwebp `QuantizeBlock_C` writes `out[n]` in zigzag
  order, so `y_dc_levels[1,2,4]` = zigzag 1,2,4, matching zen's `y2_zigzag[1,2,4]`.
- **`D > min_disto` gate** — zen's `min_disto = 20*ydc` matches libwebp's
  `20*m->y1.q[0]`; the gate structure matches.
- **Flat-source D-doubling** — libwebp doubles `rd->D` for flat blocks before the
  gate; making zen's gate use the doubled `d_final` was a **no-op** on the 4004-
  cell grid (these blocky MBs aren't flat-source), so it wasn't shipped.

**Narrowed to:** the Y2 coefficient *values* `store_max_delta` reads
(`stored_coeffs.y2_zigzag`, `mode_selection.rs` call site ~L1784) differ per-MB
from libwebp's `rd->y_dc_levels` — a stored-coefficient / quant-pass mismatch,
the same class as the m5 blocky-nz fix (`6b4fa0c`). **Next step:** build
instrumented libwebp (scratchpad `libwebp-dbg/`) to dump per-MB `(segment,
blocky, D, Y2[1,2,4], max_edge)` and diff against zen — the q75-style
methodology. Separately, m6 diverges on I4/mode RD (`y_same` 90–99%) and m0/m1
on `use_skip`/`skip_prob` (the #25 SKIP_PROBA path); both are distinct from
`seg_lf`.

## Method note

`methodcmp` should be extended to sweep q and multiple images (or replaced by
`byteparity_sweep`) so a single-point 14/14 can't again be mistaken for general
byte-exactness. The zensim/pixel gates are unaffected — this is purely about the
*byte-identical* claim.
