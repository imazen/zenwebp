# zenwebp recovery register — 2026-05-08

## Verdict table

| branch | commit | date | item | what it adds | verdict | files |
|---|---|---|---|---|---|---|
| main | `fcceda5` | 2026-05-05 | picker v0.3 held-out A/B sweep | extended A/B (q30–q90, 13 target points); parity within ±1 zensim to libwebp | kept | `benchmarks/`, examples + training scripts |
| fix/security-audit-2026-05-06 | `60fd977` | 2026-05-06 | ICCP chunk bounds | clamps ICCP chunk range to buffer length | **kept (security)** | mux ICCP path |
| chore/dev-fmt | (recent) | 2026-05-06 | fmt cleanup | rustfmt | kept | various |
| feat/expert-internal-params (worktree `zenwebp--expert`) | `d34723a` | — | expert internal params stub | matches zenavif approach; gated behind `__expert` cargo feature | **STAGED — current path forward** | `tests/validation.rs`, encoder API |
| release/0.4.5 | `a022c37` | 2026-05-02 | **YANKED release** | `__expert` feature + `InternalParams` + validation gating + theory docs per field | **YANKED** | — |
| spike/zenpicker-knobs | (recent) | 2026-04-29 | zenpicker knobs integration spike | exploratory | research | — |
| spike/chroma-spatial | `4a1d837` | 2026-04-26 | chroma spatial discriminator spike | photo-vs-drawing classifier features | research | — |
| spike/chroma-entropy | `25d9cfe` | 2026-04-26 | chroma entropy spike | similar | research | — |
| spike/chroma-palette-count | `cc82cad` | 2026-04-27 | palette-count spike | similar | research | — |
| investigate/issue-50-q-monotonicity | `1219881` | 2026-04-27 | issue 50 root cause | non-monotonic q70→q80 quality regression traced to **stale ledger PNGs from pre-2026-03-05 buggy encoder** (alpha invert, UV quant mismatch, signed-value sign). Encoder bugs fixed; fresh re-encode is q-monotonic | **closed — fixed in main** | issue thread |
| investigate/issue-50-followup-high-q-gap | `b77b427` | 2026-04-28 | issue 50 followup repro | repro tool — now obsolete after stale-ledger fix | closed | — |
| work-target-zensim-rgba | (recent) | 2026-04-27 | zensim-side RGBA support investigation | feasibility for RGBA inputs to zensim | research | — |
| pr48-cleanup | (recent) | 2026-04-27 | PR 48 cleanup work | merge prep | superseded | — |
| salvage/zenwebp-picker-prior-agent-2026-04-30 | (in zenanalyze actually) | 2026-04-29 | salvage of prior agent's picker work | preserved on a branch | unverified — relevance? | — |

## Locked agent worktrees (`.claude/worktrees/`)

Each one represents a previous session's work. Need a sweep to check each for unmerged commits ≥ 2026-04-25; for now noted as inventory:

- `agent-a01bb99b2d014cdc9` (worktree-agent-...)
- `agent-a089591fd9f7df921` (`investigate/issue-50-followup-high-q-gap`) — issue 50 closed
- `agent-a18a54643c2593719` (`fix/issue-32-fast-mb-analyze`) — fix branch
- `agent-a26e05459e6f21352` (worktree-agent-...)
- `agent-a29f8c6180cb1b300` (detached)
- `agent-a4683163dba7bbdd6` (`work-target-zensim-rgba`) — RGBA investigation
- `agent-a474f77ac110c5f4b` (`fix/25-full-skip-proba`)
- `agent-a5c8a54d1f6e34b2e` (`fix/35-cleanup-bundle`)

## Re-release path (Phase 2, parallel to zenavif)

Same shape as zenavif. The 0.4.5 yank reason: bundled bake. Path forward:

1. **Merge `feat/expert-internal-params` (worktree `zenwebp--expert`) → main**.
2. **Remove any `include_bytes!`** for the picker bake; drop the bundled `.bin`.
3. **Add `with_baked_model(bytes)` API**.
4. **Errors at runtime** rather than compile-time when bake missing.
5. **Integration tests**: bake-supplied vs no-bake-default-encode.
6. **CHANGELOG**: yank reason + caller-supplied API explanation.
7. **Tag as 0.5.0** (minor bump for API; 0.4.5 yanked).
8. **DO NOT publish** until ZNPR v3 + zenpredict 0.2.0 are out.

## Cherry-picks for main (anti-bloat)

1. `fix/security-audit-2026-05-06` → main (security-critical: ICCP bounds).
2. `feat/expert-internal-params (zenwebp--expert)` → main (replacement for the yanked bundled-bake design).
3. `chore/dev-fmt` → main (small cleanup; trivial).

## Drop / archive

- All `spike/chroma-*` branches: superseded by zenanalyze's own chroma features (UniformitySmooth/FlatColorSmooth were even retired in zenanalyze 2026-05-06 cleanup) — archive as `recovered-archive/spike-chroma-*`.
- `investigate/issue-50-*`: closed; archive after preserving findings note.
- `release/0.4.5`: archive (yanked).
- Locked `.claude/worktrees/agent-*`: unlock + git worktree remove if no unmerged commits ≥ 2026-04-25.

## Notable docs / artifacts

- The picker v0.3 held-out A/B writeup (find under `benchmarks/`) — preserve.
- Issue 50 root-cause writeup — preserve as `docs/recovered/issue-50-stale-ledger-rca.md`.
- The `__expert` feature + `InternalParams` theory docs from `release/0.4.5` (per-field rationale) — preserve in `docs/expert-knobs.md`.
