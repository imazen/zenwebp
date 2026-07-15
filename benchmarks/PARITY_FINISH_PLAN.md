# Finish #38 — byte-exact `StrictLibwebpParity` across the FULL grid

> Durable mission doc. The `/finish-parity` slash command
> (`.claude/commands/finish-parity.md`, local/gitignored) is a thin launcher for
> this. Re-entrant + loop-safe: each run lands ONE verified, parity-gated fix and
> re-measures. Run repeatedly or under `/loop` until the score stops improving.

Make `CostModel::StrictLibwebpParity` produce **byte-identical** output to libwebp
(`cwebp`) at matching `(quality, method, sns, filter, segments)` — **not just q75**.
Decoded pixels are already bit-exact; this is matching libwebp's exact *bytes*.
An optional argument names a cluster to target; default is the highest-leverage one.

## STEP 0 — orient (EVERY run)

1. **Read the living state first:** `benchmarks/byteparity_scope_2026-07-14.md` — the
   current %, divergence map, ruled-out causes, and per-cluster file/function
   pointers. Then skim `CLAUDE.md`'s "byte-exact" section + `git log --oneline -15`.
2. Claim the repo: write `.workongoing` (CLAUDE.md protocol), `jj git fetch`,
   `jj new main@origin -m "finish-parity: <cluster>"`.
3. **Rebuild the harness if missing** (scratchpad is per-session/ephemeral — see
   HARNESS). If present, **rebuild every bin anyway** — a stale bin fakes divergences
   (a stale `segfielddiff` once reported a phantom seg_lf diff at a byte-identical
   cell after the base-quant fix; it cost a whole trace).

## THE LOOP (one pass = one landed fix)

1. **SCORE (ground truth).** Rebuild + run `byteparity_sweep` → `X/4004` byte-identical
   + the diverging `(image, q, config, method)` cells with first-diff offsets. This
   byte comparison **is** the score. `webpx` is verified == real libwebp C, so canonical.
2. **PICK a cluster** (`$ARGUMENTS` or the largest sharing a root cause — divergences
   cluster by **low-q**, **high-q**, and by **method**: m5 trellis, m6 mode-RD, m0/m1
   skip-proba). Prefer a shared root that closes many cells.
3. **CONFIRM it's real** with the **3-way oracle**: `zen(StrictLibwebpParity)` vs
   `webpx` vs vendored **`lwsrc/trace_driver`** output. All three match ⇒ the cell is
   already identical and your bin was stale — rebuild and re-pick.
4. **TRACE to root.** Attribute the first diff with `segfielddiff` (frame-header
   fields — each diff line prints *before* the method-summary it belongs to). Then
   instrument BOTH sides per-MB: zen (`#[cfg(feature="std")]` + env-gated `eprintln!`)
   and `lwsrc` (`getenv`+`fprintf`; `LWSMD`/`LWSFS`/`LWADJ` dumps exist for the filter
   path). Read libwebp's ACTUAL source in `lwsrc/src/enc/` — never guess its logic.
5. **FIX — parity-gated, ALWAYS.** Gate on
   `self.cost_model == crate::encoder::api::CostModel::StrictLibwebpParity`
   (or `|| self.method >= N` where libwebp's behavior is method-scoped). Tuned default
   (`ZenwebpDefault`) MUST stay byte-unchanged. Match libwebp's arithmetic exactly
   (round vs truncate: `(x) as i32` truncates, `round(x)` rounds — libwebp usually
   `(int)`-truncates).
6. **VERIFY every gate before committing:**
   - `byteparity_sweep` % went **UP** (measured, not speculative — a grid no-op gets
     DROPPED, don't ship speculative complexity).
   - `methodcmp` still **14/14** (q75, both configs, m0–m6).
   - `cargo test --release` green — esp. the **zensim regression matrix** (synthetic
     floors) + **v2_pixel_perfect**. NEVER relax a floor or the 14/14; parity-gating
     keeps them green by construction.
   - Tuned default byte-unchanged (its byteparity/zensim numbers must not move).
7. **LAND it.** `jj describe` (measured byteparity delta + exact root cause +
   parity-gated), `jj bookmark set main -r @`, `jj git push --bookmark main`, verify
   `git merge-base --is-ancestor @ origin/main`. Watch CI (`gh run list`); if red,
   `gh run view <id> --log-failed`, fix forward now (common: `cargo fmt`, clippy
   `unusual_byte_groupings`).
8. **UPDATE the record** in the SAME change: new % + what closed + what remains in
   `benchmarks/byteparity_scope_2026-07-14.md`; a `CHANGELOG.md` entry. If a
   parity-gated behavior is *also* a measured win for the tuned default on real
   content (CID22 + imazen-26), note it as a tuned-adoption candidate — adopt only
   after a separate sweep (how max_i4/UVDIFF were adopted).
9. **REPEAT** on the next cluster.

## DONE

`byteparity_sweep` at **100%** (or the measured achievable max, each residual
documented as a *named* libwebp-internal you can't match without a specific
rearchitecture). Then: all gates green, doc + CHANGELOG reflect final state, remove
`.workongoing`, report the final `X/4004`.

## HONEST-STOP (legitimate, not lazy)

- A cell needing a **named rearchitecture** you can't land in one chunk (e.g. the
  multi-pass `StatLoop` port for skip-proba, #27) → decompose, land the smallest
  demoable sub-chunk, document the next. Switch clusters to keep the score climbing;
  document the blocked one precisely.
- Do **not** stop for context length / "clean session" (banned). Stop only on a
  measured-and-verified blocker or at 100%.

## KNOWN REMAINING WORK (re-verify; don't trust blindly)

- **base-quant round→truncate** landed (`52cf96f2`) — closed q10/30/50/80.
- **skip-proba (m0/m1 low-q)**: the `nb_skip` count, entangled with libwebp's
  `StatLoop`/`VP8Decimate` (#25 + #27). Align the skip-proba round→truncate first
  (cheap, correct); the count itself likely needs the StatLoop port.
- **m5 seg_lf**: trellis-path `StoreMaxDelta` (non-trellis path verified CORRECT —
  both compute seg2 `max_edge`=2 at q30 m4). `6b4fa0c` fixed q75 only.
- **m6 mode-RD** (`y_same` < 100%) + `n_proba_updates`.
- **high-q (q80+) across m3–m6**: proba / RD at fine quant.
- Non-default configs `sns0/segs4` + `sns30/segs2` are 0% — a segmentation-at-low-sns
  divergence; big lever if scope widens beyond the default config.

## HARNESS (rebuild if the scratchpad is gone — ephemeral & per-session)

Cargo workspace `webp-ll-compare` in a **persistent** sibling dir (NOT `/tmp`), deps:
`zenwebp` (path, `features=["__expert","mode_debug"]`), `webpx`, `zensim` (path),
`image`, `png`. Bins:
- **byteparity_sweep** — `zen(StrictLibwebpParity)` vs `webpx` across 13 images
  (3 CID22 512² + tiny/odd-dim synthetics) × q{5,10,20,30,40,50,60,70,80,90,95} ×
  4 configs {(0,0,1),(50,60,4),(0,0,4),(30,20,2)} × m0–6; print `X/4004` + every
  non-identical cell with first-diff offset. **The score.**
- **methodcmp** — `zen` vs `webpx`, q75, CID22 382297, configs (sns0,flt0,segs1) +
  (sns50,flt60,segs4), m0–6 = 14 cells. **The q75 regression gate.**
- **segfielddiff** — copy of repo `dev/bitexact_diff.rs`, config loop retargeted;
  decodes frame-header fields + per-MB mode agreement (diff lines print BEFORE the
  method summary).
- **threeway** — zen vs webpx vs saved `trace_driver` output; the oracle.
- **trace_driver** — vendored libwebp (`lwsrc/`, `build_trace.sh`, args
  `<rgb> <w> <h> <method> [segs] [sns] [flt] [q]`). If gone: vendor `libwebp-sys`
  source, add `getenv`/`fprintf` dumps in `src/enc/`. RGB via
  `convert img.png -depth 8 RGB:img.rgb`. Consider committing byteparity_sweep +
  methodcmp to the repo's `dev/` for durability.

## GUARDRAILS (from ~/.claude/CLAUDE.md — non-negotiable)

Heavy build/run through `~/work/zen/scripts/run-heavy`; ONE heavy job at a time.
Commit after each verified change; push early; fix CI first. Only VERIFIED claims.
Never relax a test/floor/threshold. Every fix parity-gated so the tuned default +
synthetic gates stay green by construction.
