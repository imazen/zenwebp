# Finish #38 — byte-exact `StrictLibwebpParity` across the FULL grid

> **Durable playbook.** Re-entrant: each run lands ONE verified, parity-gated fix
> and re-measures. Read this + `byteparity_scope_2026-07-14.md` (the living state)
> before touching anything.

Make `CostModel::StrictLibwebpParity` produce **byte-identical** output to libwebp
at matching `(quality, method, sns, filter, segments)`. Decoded pixels are already
bit-exact and gated; this is about matching libwebp's exact **bytes**.

## STATE (2026-07-16, final): ✅ **DONE — every sweep axis at 100%, no follow-ons left**

Base grid 4004/4004, and EVERY phase-2 axis: filter_sharpness 1-7
(1960/1960), segments-3 + sns/filter extremes incl. sns>0/segs1
(1120/1120), quality edges q0/q1/q99/q100 (1456/1456), pinned
partition_limit (252/252), **sharp_yuv 96/96** (closed by the exact
SharpYUV port, `src/encoder/sharpyuv.rs` +
`benchmarks/sharpyuv_port_2026-07-16.md`), and **alpha 192/192**
(opaque + gradient + checker; six roots — Huffman tie-break, rb_zero
cross-color skip, WebPCleanupTransparentArea + re-pad, weighted chroma,
hash-chain accounting under `Vp8lConfig::parity`; see the alpha row in
`byteparity_scope_2026-07-14.md`). CI anchors:
`tests/libwebp_byte_parity.rs` incl. `sharp_yuv_matches_libwebp` and
`transparent_rgba_matches_libwebp`.

Out-of-scope axes (no matched knob): target_size/PSNR search, pass>1,
autofilter, filter_type, preprocessing bits, multi-partition,
low_memory, emulate_jpeg_size.

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
| m0-m2 skip-proba: StatLoop-shaped count + size_p0 bailout | `46e2a2c` | +93 |
| m5/m6 skip from FINAL trellis levels, not simple re-quant | `a9fc2da` | +67 |
| segment-quant via libm pow (fast-pow flipped trunc boundary) | `9a6a289` | +5 |
| I4 tie-break in LIBWEBP's enum order (LD/RD/VR permuted) | 2026-07-16 | +10 → **4004/4004** |

Earlier: base-quant truncate `52cf96f2` · segmentation-collapse `41923466` ·
trailing-slots `7acdd775` · skip-proba forced off `91c96168`.

## STRATEGY — none needed; grid complete. Solved-root index:

**(solved) I4 tie-break order** — the final 10 cells, one root: zen's B-mode
constants are spec-order (LD=4,RD=5,VR=6), libwebp's enum is permuted
(RD=4,VR=5,LD=6), and exact RD ties resolve to the first-visited mode.
`LIBWEBP_I4_ORDER` in `mode_selection.rs` now drives the parity iteration in
all three evaluator paths. Ties concentrate at high q.

**(solved) synth_33x17 q90 cluster** — was 5 cells, one root: the fast
`pow`/`cbrt` approximations flipped `compute_segment_quant`'s truncated quant
index at an integer boundary (seg1 12 vs 11). Parity now routes the whole
`QualityToCompression → pow(c_base, expn) → (int)(127*(1-c))` chain through
`libm::pow`, mirroring libwebp literally. `bitexact_diff` found it in one
run; `synth:WxH:SEED` specs now feed the sweep's synthetics to both trace
tools.

**(solved) m5/m6 trellis-skip** — was 74 cells. zen decided the per-MB skip
with `check_all_coeffs_zero` (simple re-quant of the raw DCT); libwebp skips
on `rd->nz == 0` — the FINAL trellis nz. The trellis keeps borderline
coefficients the simple bias drops (neutral-bias level0 + sharpen + RD), so
zen skipped MBs libwebp coded. Parity arm now records actual levels and
derives skip from `stored_coeffs.is_all_zero`; StoreMaxDelta fires before the
skip test. Full trace + tooling (`REFRESHDBG2`/`LEVFINAL`/`TRELDBG`) in
`byteparity_scope_2026-07-14.md`.

**(solved) m0-m2 skip-proba** — was 94 cells, the whole m0-m2 axis. Full
mechanism + SKIPDBG evidence in `byteparity_scope_2026-07-14.md` ("SOLVED:
m0-m2 use_skip/skip_prob"): StatLoop-subset `nb_skip` count (m0 `fast_probe`
freeze via `fast_probe_skip_count`), full-frame truncated unclamped
`CalcSkipProba`, `< 250` threshold, and the `size_p0 == 0` bailout gating the
finalize off at single-effective-segment configs (`!segments_enabled`).

## DONE — ✅ criteria met 2026-07-16

`byteparity_sweep` at **4004/4004**; all gates green; docs + CHANGELOG
current; `.workongoing` removed at session end.

**Advertise the claim with its scope:** byte-exact across the committed grid
(13 images × q5-95 × 4 configs × m0-m6), gated in CI by
`tests/libwebp_byte_parity.rs` (q75 pin, tiny/odd dims, q90 recorder paths,
plus regression anchors for the four 2026-07-16 roots). Decode bit-exactness
was already complete and gated (`tests/v2_pixel_perfect.rs`, `max_diff == 0`
vs real libwebp). Unswept encode axes remain (sharpness, partitions, alpha,
target_size) — extend the sweep before extending the claim.

## GUARDRAILS

Heavy work through `~/work/zen/scripts/run-heavy`; ONE heavy job at a time. Commit
after each verified change; push early; fix CI first. Only VERIFIED claims. Never
relax a test/floor/threshold. Every fix parity-gated so the tuned default and the
zensim floors stay green by construction.
