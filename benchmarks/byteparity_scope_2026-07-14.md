# StrictLibwebpParity byte-identity — actual scope (2026-07-14)

## TL;DR

The #38 parity work makes `CostModel::StrictLibwebpParity` **byte-identical to
libwebp at the specific operating point it was traced against** (q75, CID22
382297, two configs), across all 7 methods. It is **NOT** byte-identical in
general. A broad grid is only **972/4004 (24%)** byte-identical. #38's
north-star (byte-identical at *all* matching settings) is not met — this doc
records exactly how far it reaches so the next session doesn't over-trust the
`methodcmp` 14/14.

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

## Method note

`methodcmp` should be extended to sweep q and multiple images (or replaced by
`byteparity_sweep`) so a single-point 14/14 can't again be mistaken for general
byte-exactness. The zensim/pixel gates are unaffected — this is purely about the
*byte-identical* claim.
