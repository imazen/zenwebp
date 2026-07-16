# Finish #38 — byte-exact `StrictLibwebpParity` across the FULL grid

> **Durable playbook.** Re-entrant: each run lands ONE verified, parity-gated fix
> and re-measures. Read this + `byteparity_scope_2026-07-14.md` (the living state)
> before touching anything.

Make `CostModel::StrictLibwebpParity` produce **byte-identical** output to libwebp
at matching `(quality, method, sns, filter, segments)`. Decoded pixels are already
bit-exact and gated; this is about matching libwebp's exact **bytes**.

## STATE (2026-07-15)

**3829/4004 = 95.6%** on the committed grid. 175 cells remain:

| axis | count | note |
|---|---|---|
| **m0-m2** | **94** (20/38/36) | **54% of the remainder — the next target** |
| m5 / m6 | 39 / 35 | trellis residue |
| m3 / m4 | 4 / 3 | 99.3% identical — effectively closed |
| sns0/flt0/segs1 · segs4 | 18 · 18 | ~98% closed |
| sns30/flt20/segs2 · sns50/flt60/segs4 | 70 · 69 | was 201 combined, now 139 |

**The axis flipped.** The RD methods (m3-m6) used to dominate; now m0-m2 do. Those
are `RD_OPT_NONE`, which take a **completely different libwebp path** — plain
`VP8EncLoop` → `RefineUsingDistortion` + `RecordResiduals` → `VP8RecordCoeffs`,
not the token loop. Trace that path, not the RD one.

## TOOLS — all committed, all `--features __expert`

| tool | what it answers |
|---|---|
| `dev/byteparity_sweep.rs` | **THE SCORE.** `X/4004` + every diverging cell with first-diff offset |
| `dev/mbpixdiff.rs` | **Run FIRST.** Decodes both bitstreams, diffs per-MB → the first *emitted* divergence |
| `dev/bitexact_diff.rs` | Frame-header fields + per-MB mode agreement for one cell |
| `tests/libwebp_byte_parity.rs` | **The gate.** In CI via a `--features __expert` step |
| `~/work/zen/libwebp--zen38trace` | Instrumented libwebp (`zen38_driver`, `build_zen38.sh`) |

```bash
cargo run --release --features __expert --example byteparity_sweep
cargo run --release --features __expert --example mbpixdiff     -- [img] [q] [m] [sns] [flt] [segs]
cargo run --release --features __expert --example bitexact_diff -- [img] [q] [m] [sns] [flt] [segs]
~/work/zen/libwebp--zen38trace/zen38_driver <rgb> <w> <h> <method> [segs] [sns] [flt] [q]
```

`~/work/webp-porting/libwebp` is a **READ-ONLY** reference tree — never modify it
(it is a different repo; CLAUDE.md forbids touching it). `libwebp--zen38trace` is
the writable copy. RGB input: `convert img.png -depth 8 RGB:~/tmp/img.rgb`.

**Fidelity caveat:** the reference tree is NOT pristine upstream — it carries
pre-existing uncommitted debug `fprintf`s in `src/dec/vp8_dec.c` from an earlier
session (20 insertions, 0 deletions, print-only). They are decoder-side and change
no behaviour, so encoder traces are faithful and the copy inherits them harmlessly.
Do not "clean them up" — they are not ours. If you ever need a guaranteed-pristine
reference, re-vendor rather than reverting someone else's working tree. Note the
SCORE never depends on this tree at all: `byteparity_sweep` compares against
`webpx`, which links real libwebp C.

## THE LOOP

1. **SCORE.** Rebuild, run `byteparity_sweep`. This byte comparison IS the score
   (`webpx` links real libwebp C).
2. **PICK** the biggest cluster sharing a plausible root.
3. **LOCATE** with `mbpixdiff` → the first EMITTED divergence.
   * **Small `1st-diff@` offset (< ~200) ⇒ a HEADER field, not content** → go
     straight to `bitexact_diff`. It found the seg_lf root in ONE run.
   * `NO per-MB pixel difference` ⇒ header/probability-only → `bitexact_diff`.
   * Otherwise trace ONLY the MB it names.
4. **TRACE by dumping BOTH sides at runtime.** Reading C and reasoning is a
   *hypothesis*, not a finding (see PITFALLS).
5. **FIX, parity-gated ALWAYS:**
   `self.cost_model == crate::encoder::api::CostModel::StrictLibwebpParity`
   (add `&& self.method >= N` where libwebp's behaviour is method-scoped). The
   tuned default MUST stay byte-unchanged.
6. **VERIFY every gate:** score UP · `cargo test --release` green ·
   `tests/libwebp_byte_parity.rs` green · tuned unchanged · `cargo clippy
   --all-targets -- -D warnings` clean (**and** with `--features __expert`).
7. **LAND:** `jj describe --stdin < file`, `jj bookmark set main -r @`,
   `jj git push --bookmark main`, verify `git merge-base --is-ancestor @ origin/main`.
8. **UPDATE** `byteparity_scope_2026-07-14.md` + `CHANGELOG.md` in the same change.

## PITFALLS — every one of these cost real time on 2026-07-15

**Trace the EMISSION, not a probe.** zen calls `pick_best_intra*` ~4x per MB and
`MB_DEBUG` mixes probe calls with the emission. A probe reads *exactly* like a real
divergence. A long trace of mb(3,0) at q40/m6 turned out to be a probe — zen emits
the same mode libwebp does there. **Always `mbpixdiff` first.**

**libwebp often has TWO of something — check which one actually runs.** This bit
twice in one day:
* Two coefficient recorders that **disagree**: `VP8RecordCoeffTokens`
  (`token_enc.c`, token loop, m3-m6) vs `VP8RecordCoeffs` (`cost_enc.c`,
  `VP8EncLoop`, m0-m2). Applying one rule everywhere regressed m0-m2 (20/38/36 →
  36/69/54).
* Two magnitude-recording variants in `VP8RecordCoeffs` itself: the
  `USE_LEVEL_CODE_TABLE` path ships; the explicit tree under
  `#if !defined(...)` is **dead code upstream**.
* `VP8EncLoop` vs `VP8EncTokenLoop`, selected by `use_tokens = rd_opt >= RD_OPT_BASIC`.

**Never count table rows by eye.** Miscounting `VP8LevelCodes` produced a confident,
wrong conclusion and a no-op "fix" to a dead path. Parse programmatically or print
the table at runtime.

**Confirm a fix took effect before measuring it.** The above edited a function that
was never called (`record_residual_stats` is dead; the live one is
`record_coeff_tokens`). Re-dump and see the value change.

**Don't publish a root cause you haven't dumped.** "zen never runs StatLoop" was
committed as the root cause and was FALSE — zen's refresh works, and StatLoop's
`FinalizeTokenProbas` never even runs at m3 (it early-outs on `size_p0 == 0`). It
had to be retracted on main.

**Plausible fixes regress.** Enabling zen's two-pass under parity: 89.4% → **57.4%**.
Measure before committing; revert and record the negative result *in code* so nobody
re-tries it.

**A gate CI doesn't run is not a gate.** `tests/libwebp_byte_parity.rs` needs
`__expert`; CI only ran default/no_std/imgref/mode_debug, so a parity regression
would have shipped silently. There is now an explicit CI step — keep it.

**Run CI's EXACT clippy.** Scoping clippy to one example missed 12 findings that CI
denies. Wiring a new `dev/` example makes clippy start checking it.

**`cargo test` captures stderr.** Pass `-- --nocapture` or your debug prints vanish
(this looked like "the code never runs").

**`/tmp` is BANNED — use `~/tmp`.** A mid-session wipe destroyed the whole harness,
the instrumented libwebp, and the pre-fix baseline log, forcing a rebuild onto a
*different* synthetic grid (the score moved 87.1% → 89.4% for purely tooling
reasons). Anything reusable goes in `dev/` + an `[[example]]` entry, committed.

**Write commit bodies to a file** (`jj describe --stdin < file`). An unquoted `!`
in an inline shell string triggers history expansion and silently eats text —
`c9abe85`'s body lost a phrase that way.

**Rebuild every binary after a lib change.** A stale bin fakes divergences.

## ALREADY REFUTED — do not re-check

`VP8_LEVEL_FIXED_COSTS` (byte-identical to `VP8LevelFixedCosts`) · the level-cost
precomputation (== `VP8CalculateLevelCosts`) · `get_residual_cost` (==
`GetResidualCost_SSE2`) · `ExpandMatrix` / `zthresh` / `QFIX` / `BIAS` · the default
coefficient-probability table · `VP8LevelCodes` · zen's mid-row proba refresh (it
fires, and `max_count` already matches libwebp's `(mb_w*mb_h)>>3` floored at 96).

**StatLoop is NOT a prerequisite** for the remaining cells: its
`FinalizeTokenProbas` never runs at m3 (early-out on `size_p0 == 0`); all 8
finalizes come from the token loop. #27 stands on its own merits only.

## ALREADY SOLVED

| fix | commit | gain |
|---|---|---|
| I16-AC-trellis nz-context seed | `f996eef` | +81 |
| Cat5/Cat6 stat-node, per encode-loop path | `44ae3a0` | +189 |
| StoreMaxDelta from the I16 CANDIDATE, not the final mode | `c9abe85` | +62 |

Earlier: base-quant truncate `52cf96f2` · segmentation-collapse `41923466` ·
trailing-slots `7acdd775` · skip-proba forced off `91c96168`.

## STRATEGY FOR THE REMAINING 175

**1. m0-m2 (94 cells — start here). ROOT ALREADY FOUND: `use_skip` / `skip_prob`.**

Measured at q5/m1/sns50/flt60/segs4, where every mode decision already matches
(y 100%, uv 100%, seg 100%) and the ONLY divergence is two header fields:

```
use_skip:   zen=0  lib=1
skip_prob:  zen=0  lib=243
```

`mod.rs:1469` forces `macroblock_no_skip_coeff = None` for **every** method under
parity, justified by libwebp's `assert(proba->use_skip_proba == 0)`. **That assert
is at `frame_enc.c:816` — inside `VP8EncTokenLoop`, which only runs when
`use_tokens` is set (`rd_opt >= RD_OPT_BASIC`, i.e. m3-m6).** m0-m2 run plain
`VP8EncLoop`, which *reads* the flag (`dont_use_skip = !proba.use_skip_proba`) and
lets `FinalizeSkipProba` enable it. The code comment claims verification "across
q5..q75, m0..m6" — that only ever exercised sns0/flt0/segs1, where libwebp emits 0
too. **Treat that comment as disproven.**

**ATTEMPTED AND REVERTED (measured):** gating the force to `method >= 3` and
computing `skip_proba` libwebp's way for m0-m2 scored **3778/4004 (−51 net)** —
but the split says the direction is right and only the *count* is wrong:

| config | before | after |
|---|---|---|
| sns30/flt20/segs2 | 70 | **39** |
| sns50/flt60/segs4 | 69 | **39** |
| sns0/flt0/segs1 | 18 | **74** |
| sns0/flt0/segs4 | 18 | **74** |
| m0 | 20 | **62** |

Enabling the flag fixed −61 in the SNS/segs configs and broke +112 in the sns0
ones, i.e. **zen's `skip_mb` != libwebp's `nb_skip`**. Why:

* `nb_skip` is counted ONLY in `StatLoop`'s `OneStatPass`
  (`if (VP8Decimate(...)) ++enc->proba.nb_skip;`) — a **separate pass**, not the
  emission pass zen counts in. `ResetStats` zeroes it.
* `FinalizeSkipProba` then divides by the **FULL** frame:
  `nb_mbs = enc->mb_w * enc->mb_h`, while `nb_skip` may have been accumulated over
  only the `fast_probe` subset (m0: `nb_mbs>>2`, m3: `nb_mbs>>1`). At m0 that
  inflates `skip_proba` toward 255, pushing it past the 250 threshold so
  `use_skip` lands 0 — which is exactly why m0 broke worst (20 → 62).
* libwebp's `CalcSkipProba` = `(total - nb) * 255 / total` — **truncated, no
  clamp**. zen's tuned formula rounds (`+ total/2`) and clamps to 1..254.

**Next step:** replicate `nb_skip` from a StatLoop-shaped count (subset-limited per
`fast_probe`, denominator = full frame) before re-enabling the flag. zen already
has the subset machinery for probabilities (`fast_probe_stat_limit`,
`fast_probe_snapshot`) — mirror it for the skip count. Verify against
`bitexact_diff` on BOTH a sns0 cell and a sns50 cell before measuring the grid;
either alone will mislead you.

Other m0-m2 facts: libwebp runs `RefineUsingDistortion` (not `PickBestIntra16`) and
`RecordResiduals` → `VP8RecordCoeffs` (table recorder, node 10 for Cat5/6);
`StoreMaxDelta` never fires (it lives at the end of `PickBestIntra16`), which is why
`store_max_edge_active()` gates on `method >= 3`. Known upstream quirk documented in
`dev/bitexact_diff.rs`: at m0-m2 with one effective segment StatLoop bails early
(`size_p0 == 0`) and libwebp ships DEFAULT probas, costing it 25-35% size — zen must
reproduce that under parity but must NOT adopt it for the tuned default.

**2. m5/m6 trellis residue (74 cells).** m5 = `RD_OPT_TRELLIS`, m6 =
`RD_OPT_TRELLIS_ALL` (trellis during I4 mode selection). The I16-AC-trellis context
fix (`f996eef`) closed the big part; what remains is likely the I4 trellis path.

**3. The 4+3 m3/m4 stragglers.** Lowest leverage, but a clean root may be visible
now that everything else on those methods matches.

## DONE

`byteparity_sweep` at 4004/4004 — or a measured max with each residual documented
as a *named* libwebp-internal requiring a specific rearchitecture. Then: all gates
green, doc + CHANGELOG current, `.workongoing` removed.

**Do not advertise unqualified "bit-exact libwebp encode" below 100%.** What is
true and gated today: byte-exact at q75 across m0-m6 on two configs, plus
tiny/odd dimensions, plus q90 across both recorder paths. Decode bit-exactness IS
complete and gated (`tests/v2_pixel_perfect.rs` asserts `max_diff == 0` vs real
libwebp).

## GUARDRAILS

Heavy work through `~/work/zen/scripts/run-heavy`; ONE heavy job at a time. Commit
after each verified change; push early; fix CI first. Only VERIFIED claims. Never
relax a test/floor/threshold. Every fix parity-gated so the tuned default and the
zensim floors stay green by construction.
