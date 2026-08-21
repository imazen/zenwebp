# zenwebp bug retrospective — PR/issue inventory, root-cause practices, and the guards that answer them

*2026-08-21. Written after inventorying all 31 PRs and 44 issues to date, the
resolved-bug log (`LOG.md`), and the three external bug reports of 2026-08-20
(PRs #73/#74/#75). Every claim below is traceable to a PR, issue, commit, or a
measurement run during this audit. This document is the durable record; the
policy bullets at the end are enforced by tests/CI landed alongside it.*

## Why this exists

Three fix PRs arrived from production users in one day (2026-08-20):

- **#73** — every lossy WebP decoded on wasm32+simd128 came out solid green,
  shipped since v0.4.2.
- **#74** — a 70-byte malformed container spins parsing forever at 100% CPU on
  32-bit targets, shipped since 2026-02-03.
- **#75** — lossy RGBA at m6 was ~40x slower than m4 (alpha-plane VP8L
  escalated to quality 100), a cliff bad enough to pin nginx workers.

Meanwhile **#72** (VP8L encoder silently emits a valid-but-wrong stream, 84% of
pixels corrupt, at default settings) is fixed on `main` but still shipping in
crates.io 0.4.5. None of these were found by our own tests, CI, benches, or
fuzzing. Each one belongs to a *category* with prior members — the practices
that allowed them are recurring, not one-off.

## Bug taxonomy, with the practice that allowed each class

### A. Hand-port transcription errors

Wrong operand semantics, swapped enums, inverted signs, misplaced guards,
wrong tables — the port compiles and mostly works.

Instances: #73 (`i8x16_shuffle` operand semantics — every index selected the
`zero` operand, all samples discarded); its NEON sibling (`c321c2b`, same
class); the VP8L refill guard misplaced during porting (`efddb6c` — upstream
had it right, the port moved it); TokenType I16DC/I16AC swapped; #21 (mode-cost
tables in wrong enum order — libwebp's `B_*` enum permutes LD/RD/VR vs spec
order); #24 (wrong Y2 quantizer table); the 2026-03-05 trio (`get_alpha()`
inverted, `write_optional_signed_value` sign inverted); trellis sign-bit
double-count; SNS c_base double-apply.

**Root practice: porting C by hand with the differential harness built
*after* the port, not during it.** The #38 byte-parity grid (now 4004/4004
cells, CI-gated) was built months after the encoder; every class-A bug above
predates it, and the classes it covers have stopped recurring. The wasm kernel
in #73 had no reference-comparison test at all.

**Guard:** a differential test against an independent reference (libwebp via
`webpx`, or the scalar implementation for SIMD kernels) must land in the same
change as any ported kernel. Now covered by: byte-parity grid (`tests/
libwebp_byte_parity.rs`, CI), `tests/simd_tier_parity.rs` (6 platforms),
`fused_row_2uv_matches_scalar_reference` (#73, arch-generic — one test body
gates v3/NEON/SIMD128), transform differential tests.

### B. Encoder-state vs bitstream divergence

The encoder applies X internally but writes Y (or nothing) to the stream; its
own reconstruction diverges from every decoder's.

Instances: UV quant deltas applied internally but never written to the frame
header (2026-03-05); multi-pass probability signaling compared against the
wrong baseline — encoder used pass-0 probs, decoder got defaults, 99.9% of
pixels wrong; **#72** — phase-5 histogram rebuild allocated a slot per *active*
cluster including remap-stranded ones, so trees were written that the entropy
image's max-symbol count says don't exist; decoders parse the extra trees as
pixel data.

**Root practices:** (1) self-referential validation — zen-encode → zen-decode
roundtrips can't see mirrored or "all decoders agree" bugs (#72's garbage was
reproduced *identically* by old zenwebp, new zenwebp, and libwebp — a
self-consistency roundtrip proves nothing; only comparison against the
ORIGINAL pixels catches it). (2) Roundtrip tests pinned at default settings
only — #72 lived at m4/m5/m6 × q25 on specific content, a cell no test
visited.

**Guard:** exact-oracle roundtrips against source pixels across the settings
grid, plus fuzzing over content × settings. Landed with this audit:
`encode_lossless_roundtrip` / `encode_lossy_roundtrip` fuzz targets (structured
content generator over the method×quality grid, decoded pixels must equal the
source exactly; the alpha plane gives lossy the same exact oracle at
`alpha_quality=100`). Existing: `tests/lossless_stranded_cluster.rs` (the #72
gate, verified-to-fail), `tests/lossless_roundtrip.rs`.

### C. Untested execution tier / target

Code compiled for a tier but never *executed* anywhere before shipping.

Instances: #73 (wasm CI was `cargo check` only — and without `+simd128` the
SIMD128 kernels were not even *compiled*; the kernel had zero tests); #74
("invisible because CI never executed a 32-bit target" — the i686 job exists
but nothing covered the malformed-container path, and fuzzing is x86_64-only);
#5 (VP8L broken on non-x86, fragmented coverage); #45 (alpha decode regression).

**Root practice: treating "it compiles for the target" as coverage.** A tier
that CI never runs is exactly as trustworthy as an untested code path — the
green checkmark just hides it better.

**Guard:** every tier the dispatcher can select must execute its tests in CI.
Landed: the wasmtime CI job (`+simd128`, from #73) — 292 lib tests now execute
on wasm32; #74's pure-arithmetic gates fail *cleanly* on any 32-bit target
(they run in the i686 job and the wasm job) instead of timing out. Existing:
`simd_tier_parity` with `testable_dispatch` across 6 platforms + AVX-512.

### D. Robustness/DoS on malformed input

Instances: #6 (panic on exhausted bitstream), #68 (decompression bomb vs fuzz
timeout), #4 (assert panics, unbounded allocs, missing limits), #55 (limits
not routed through lossless transforms; demuxer bounds), #74 (infinite loop —
32-bit only), #63 (panic-on-OOM buffers; `AllocPreference` infrastructure has
since landed on main).

**Root practices:** the parsing layer predates its adversarial harness; fuzz
targets covered only the main decode path (`WebPDecoder`), so the demuxer and
metadata entry points — public API — were never fuzzed (#74's seed "would
otherwise have been inert"); casts narrowed before range checks.

**Guard:** every public parsing entry point appears in a fuzz target and in
`tests/fuzz_regression.rs` (#74 added `run_demux`; this audit added the encode
runners); range-check in u64 before narrowing, now swept across all of
`SliceReader` including the `io::Seek` `End`/`Current` arms #74 missed
(`io_seek_end_current_do_not_truncate`).

### E. Wrong operating point shipped as default (perf cliffs, quality cliffs)

Instances: #75 (m6 alpha q100 escalation — libwebp-faithful, wrong as a tuned
default, ~40x); #17 (sharp_yuv −17..−26 zensim regression, later re-adopted
properly via the exact port); #50 (non-monotonic quality vs q); #32 (m0 4x
outliers on tiny low-color images); #22 (I4 penalty 14x off).

**Root practice: adopting a knob, port, or default without sweeping the
settings × content × size grid** — the workspace sweep discipline exists
precisely for this, and the cells it mandates (method × format × quality,
including tiny images) are where every one of these lived. #75 specifically:
the alpha pipeline adoption (`376b7b6`) was byte/quality-swept but not
wall-time-swept per method × format.

**Guard:** operating points are *named, unit-pinned decisions*, not inline
expressions — `alpha_vp8l_quality()` landed with this audit after observing
that **both of #75's own tests stay green with the fix reverted** (alpha is
bit-exact either way; the cliff was one `&& parity` revert from returning).
Plus: wall-time ratio sweeps belong in the adoption benchmark for any change
that touches per-method work (see policy below).

### F. Gates that don't gate

Instances: #46 (`v2_pixel_perfect` silently skipped VP8L and VP8X+ALPH files —
the byte-exact gate covered a fraction of what its name claimed); the fuzz
workflow's regression step wrapped the harness in `|| echo`, converting test
failure into a green step (fixed in this audit); #72's first regression sweep
"passed while proving nothing" (content never stranded — the trigger had to be
*found* by instrumenting the encoder); #75's guard tests (above); three tests
in `lossless_stranded_cluster.rs` documented explicitly as "general nets, not
#72 gates" because their content doesn't strand — that honesty is the model.

**Root practice: writing the test after the fix and trusting that it passing
means it gates.** A regression test that was never seen to *fail* against the
broken code is a hope, not a gate.

**Guard: negative-control verification** — reintroduce the defect, watch the
gate fail, revert. #73 and #74 both did this and said so in their PR bodies;
this audit re-verified both independently (kernel defect → wasmtime abort;
seek narrowing → 32-bit gate failure) and applied the same standard to the new
operating-point pin. This is now repo policy (below).

### F addendum — found DURING this audit's final gate

The full-suite gate flaked once:
`decoder::yuv::tests_simd::test_yuv_to_rgb_matches_scalar` panicked "SSE4.1
required for SIMD YUV" on an AVX2 machine. Root cause: the yuv_exact tier
tests call `archmage::testing::for_each_token_permutation`, which disables
tokens **process-wide**, while libtest runs sibling tests on parallel threads
— any concurrently-running test whose production path summon-expects a token
dies. (`testable_dispatch` reaches the lib tests through dev-dependency
feature unification, which is what makes tokens disableable there at all.)
Global mutable state inside a parallel test suite is class F with a random
victim. Fixed with `src/test_token_lock.rs`: every lib test that toggles
tokens and every lib test that summon-expects them takes one lock
(`tests/simd_tier_parity.rs` was already safe — its CI job runs
`--test-threads=1` for exactly this reason). Verified 0/10 failures at 32
threads post-fix.

Related, noted for later: the `tests_simd` tests guard themselves with
`if X64V3Token::summon().is_none() { return; }` — a silent runtime skip
(banned pattern) that would quietly pass on a pre-SSE4.1 machine. Low
practical exposure (x86-64 CI runners all have AVX2), but it should become a
visible, caller-controlled skip when that module is next touched.

### G. Process/meta findings from this audit

- **Fork PRs show a green-looking check while running zero CI.** All three
  2026-08-20 PRs displayed only "GitGuardian: SUCCESS"; the repo's 20-job CI
  sat in `action_required` (first-contributor approval) and nobody approved
  it. A maintainer glancing at the checks column could merge fully unverified
  code. Practice: approve workflow runs for plausible fork PRs immediately,
  and never treat the rollup as green without confirming the *repo's own*
  jobs ran.
- **Main CI was red (Format + Clippy) from 2026-08-01 to this audit** — three
  weeks of "broken window" normalizing failure, which also poisons every fork
  PR's checks. The 2026-08-01 pushes went out without `cargo fmt`/`clippy`
  having been run.
- **Local claims in PR bodies are claims.** "332 passed locally" was accurate
  for all three PRs — but verifying it required a full local re-run because no
  CI had executed. Trust after verification, per the workspace's
  "done requires verification on the remote" rule.
- Three working-tree files were found zero-filled (NUL) with sizes preserved —
  filesystem-level corruption from an earlier crash, restored from git. jj's
  stat-based snapshotting missed it; a full-content re-hash
  (`git hash-object --stdin-paths` vs the index) confirmed the rest of the
  tree clean. Worth repeating after any unclean shutdown.

## The scoreboard

| Class | Members (sample) | Guard | Status |
|---|---|---|---|
| A. Port transcription | #73, #21, #24, `efddb6c`, 2026-03-05 trio | byte-parity grid; SIMD-vs-scalar kernel tests; differential-with-port rule | in CI; rule below |
| B. State/bitstream divergence | #72, UV-deltas, multi-pass probs | encode fuzz w/ exact source oracle; stranded-cluster gate | **landed in this audit** + existing |
| C. Untested tier | #73, #74, #5 | wasmtime CI job; 32-bit-clean gates; tier-parity suite | landed via #73/#74 |
| D. Malformed input | #74, #6, #68, #55, #63 | fuzz all public entry points; u64-before-narrow sweep | landed via #74 + this audit |
| E. Wrong default operating point | #75, #17, #50, #32 | named+pinned operating points; sweep discipline | pin landed; sweep = policy |
| F. Gates that don't gate | #46, fuzz `\|\| echo`, #75's tests | negative-control rule; no silent skips | fixed + policy |
| G. Process | fork-PR zero-CI, red main | approve fork CI; fix-CI-immediately | fixed; policy |

## Policy (enforced from 2026-08-21)

1. **A regression test lands only after it has been watched to FAIL against
   the defect it gates** (negative control). Say so in the commit/PR body. A
   test whose content or settings can't reach the trigger must be labeled a
   general net, not a gate.
2. **Every ported kernel lands with its differential test in the same
   change** — vs libwebp for bitstream/pixel behavior, vs the scalar reference
   for SIMD tiers (arch-generic, so one test body covers every tier the build
   selects).
3. **A dispatch tier or target that CI does not *execute* is treated as
   untested.** `cargo check` is not coverage. New tiers ship with a CI job
   that runs them (wasmtime for wasm32; `cross` for i686; the tier-parity
   suite for ISA tiers).
4. **Encoder validation compares against the ORIGINAL pixels** (exact for
   lossless and for alpha at `alpha_quality=100`), never only
   self-consistency, and sweeps method × quality × content, not just
   defaults.
5. **Every public parsing entry point appears in a fuzz target** and replays
   in `tests/fuzz_regression.rs` on stable.
6. **Container/size arithmetic is checked in u64 before narrowing to usize.**
   When fixing one instance of a class, sweep the whole surface in the same
   change (the `End`/`Current` arms were missed one function below the fix).
7. **Default operating points are named functions with pinned unit tests**,
   and any adoption that changes per-method work is wall-time-swept across
   method × format before it becomes the default.
8. **No `|| true`-shaped steps in workflows; no silent test skips.** A step
   that can't run must fail or be visibly gated by the caller.
9. **Fork PRs: approve the workflow run before reviewing further**, and treat
   the checks column as unverified until the repo's own jobs report.

## What landed with this audit (2026-08-21)

- PRs #73, #74, #75 verified end-to-end locally (full 53-binary release
  suite; wasm32+simd128 lib suite under wasmtime; byte-parity 8/8; negative
  controls for #73's kernel and #74's seek) and landed with contributor
  authorship preserved.
- `fix(ci)`: main returned to green (fmt drift + 3 clippy doc errors).
- `fix`: io::Seek `End`/`Current` u64 sweep + `alpha_vp8l_quality()`
  operating-point pin.
- `feat(fuzz)`: `encode_lossless_roundtrip` + `encode_lossy_roundtrip`
  targets, shared-core replay in `fuzz_regression`, nightly matrix, and the
  `|| echo` gate fix.
- This document; CLAUDE.md points here.

### Review + coverage pass (later same day)

A 4-agent review (mux/container, decoder+limits, encoder-config plumbing,
VP8L emit invariants) plus a coverage/test-quality pass. Every fix was
verified against source with a repro or negative control before landing —
the agents' findings were treated as leads, not truth (several proposed
"obvious fixes" had hidden hazards; see #77 and #78). Landed:

- **VP8L max-dimension (16384) wrap → 0 pixels** — `(1 + header) & mask`
  instead of `(field) + 1`; a legal max-size lossless image *failed to
  decode*. A **correctness** bug (wrong/absent pixels), the highest-severity
  find. Gate `tests/vp8l_max_dimensions.rs`.
- **`DecodeConfig::limits` were dead** — applied after `read_data` ran the
  gates against defaults. Gate `tests/decode_limits_enforced.rs`.
- **Encoder `quality > 100` panicked the library** from public constructors.
  Gate `tests/encode_quality_no_panic.rs`.
- **Mux untrusted-input:** 38-byte ANIM OOB panic (repro'd), ANMF quadratic
  amplification, `take_slice`/`peek_slice`/streaming 32-bit wraps.
- **VP8L #72 emit-time invariant** (`debug_assert`) — catches the class at
  emit, including non-stranding content the roundtrip gate cannot.
- **~1900 lines dead v1 diagnostics removed**; weak unit tests strengthened
  to assert pixels/properties (were `is_ok()`-only or assertion-free);
  `StreamingDecoder` + `demux_container` coverage added.
- Config consistency: enum setters clamp; `LossyConfig` Debug completed.

Backlog of design-level / measurement-needed findings: **#78**. Palette
cache-bits determinism: **#77**. Coverage started at 82.5% src lines; the
0% files were dead code (removed) or the now-tested `StreamingDecoder`.

## Still open after this audit

- **Publish.** crates.io 0.4.5 (2026-05-02) still ships #72 (silent lossless
  corruption at defaults) and #73's green-screen wasm decode. The release
  needs the user's go-ahead (README review, version choice — the Unreleased
  section includes queued breaking changes).
- #27 (multi-pass StatLoop), #35 (cleanup tracker), #39/#40 (search/ML), #60
  (whereat tags), #71 (fine-tile predictor gap) — tracked, not bugs shipping
  wrong pixels.
- A scheduled zenbench perf-regression workflow (category E's long-term
  guard) — the paired-stats harness exists; wiring baselines into CI is the
  remaining work.
