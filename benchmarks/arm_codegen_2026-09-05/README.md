# ARM kernel and codegen audit

The wider codec audit is incomplete: existing edits prevented syncing zenjpeg,
zenpng, zenjxl, and zenjxl-decoder without owner approval. This report covers
zenwebp's existing 17 kernel comparisons and an archmage/magetypes codegen check.
No codec implementation or dispatch policy changed.

Measured on Apple M4 Pro, Mac16,11, 24 GiB RAM, macOS Darwin 25.5.0,
Rust 1.98.0 (`88d9e12ae`), LLVM 22.1.8, aarch64-apple-darwin.
Run date: 2026-09-05 America/Denver (2026-09-06 UTC).

Source revisions:

- zenwebp remote main: `f7f1451dbc9037c9c7666d44d20b457a6a711716`.
- zenwebp's locked archmage/magetypes: **0.9.27**, registry source.
- Separately inspected archmage remote main:
  `ebc204640c64f407cee9f6521b5b2e1385d0dbd6`, version **0.9.29**.
  Syncing that checkout does not update zenwebp's locked dependencies.

## Findings

| Kernel / input | NEON mean | Scalar mean | Interpretation |
|---|---:|---:|---|
| Inverse subtract-green, 1,048,576 pixels | 489.53 us | 489.73 us | No measured tier benefit; assembly is scalar byte work |
| Dequantize, 16 coefficients | 23.6 ns | 21.3 ns | Scalar faster in this benchmark; investigate caller/codegen before changing dispatch |
| Add i16 residue, 4x4 in a 256-byte block | 40.9 ns | 38.8 ns | Scalar faster; ARM path widens to i32 |
| Encoder subtract-green, 1,048,576 pixels | 89.74 us | 196.76 us | Explicit magetypes loop emits NEON shuffle/subtract |
| Exact YUV420 to RGB, 1280x720 | 736.5 us | 3817.6 us | NEON path provides a substantial benefit |
| Exact YUV420 to RGBA, 1280x720 | 832.9 us | 4157.4 us | NEON path provides a substantial benefit |
| Spectral distortion, 4x4 | 21.9 ns | 30.3 ns | NEON wins despite avoidable-looking load/check overhead |

Full measurements, confidence intervals, variance flags, and process resource
statistics are in [kernels.log](kernels.log). This is one interleaved run on a
shared Mac. Several groups have substantial variance; RGB conversion has a
drift flag. Small differences are candidates, not established production
regressions. These are kernel inputs, not a content/size/quality sweep and not
whole-codec profiles. No constants or dispatch decisions should be calibrated
from this run. Other ARM microarchitectures were not measured.

### Lossless decoder add-green: scalar code behind the NEON dispatch

[`add_green_portable`](../../src/decoder/lossless_transform_simd.rs)
at line 682 takes a backend token but does scalar byte updates on 16-byte
chunks. The release benchmark's dispatched NEON body contains `ldrb`/`ldurb`,
integer `add`, and `strb`/`sturb`; it does not vectorize that main loop.
See [decoder-add-green.asm](decoder-add-green.asm).

The encoder's analogous generic `u8x16` implementation at
[`transforms.rs`](../../src/encoder/vp8l/transforms.rs):138 produces
`ldr q`, `tbl.16b`, `sub.16b`, `str q`, with no helper calls in the main loop.
Its scalar array construction becomes a shuffle, without a stack round-trip.
There is still a range check per chunk. See
[encoder-subtract-green.asm](encoder-subtract-green.asm).

Next experiment: express decoder add-green with the same generic byte-vector
pattern and wrapping addition; compare against its existing scalar oracle over
all byte values, chunk tails, and row/stride cases before measuring a change.
No speedup for that proposed implementation has been measured.

### Small integer kernels and row loading

[`prediction.rs`](../../src/common/prediction.rs):285 widens 16 i16
residuals to i32 before calling the ARM add-residue implementation. The i16
IDCT similarly widens, transforms, then narrows in
[`transform.rs`](../../src/common/transform.rs):103. The IDCT path is
called by encoder mode selection; its isolated i16 cost was not measured here.

The single-block DCT and IDCT `neon` dispatch functions at lines 360 and 400
already call scalar implementations. Their benchmark labels therefore do not
compare explicit NEON transforms with scalar transforms. Two-block DCT and
`ftransform2` still call explicit NEON implementations. Recheck those actual
call paths instead of interpreting a near-equal tier result as coverage of all
transform implementations.

[`tdisto_4x4_fused_inner`](../../src/common/simd_neon.rs):219 builds row
vectors with individually indexed bytes. Its stride-32 specialization retains
32 input-length conditional branches before loading the first pixel, plus
scalar byte loads and widening. The retained function occupies 323 assembly
instruction lines including cold panic paths, not 323 executed instructions
per call. See [tdisto-4x4.asm](tdisto-4x4.asm).
Next experiment: validate each row as a fixed four-byte slice once, then
construct/load the paired row vector. Preserve arbitrary stride support.
NEON currently wins this benchmark; this is further optimization potential,
not evidence that the entire kernel is poor.

### Macro/backend checks

Latest archmage's `arcane_impl_sibling` generates an always-inline wrapper and
an inline target-feature sibling. The NEON magetypes backend uses inline
storage conversions and arcane-wrapped arithmetic. The existing
`magetypes/examples/idiomatic_patterns_all.rs` was compiled and executed on
ARM at `ebc2046`. All its assertions passed. In its extracted generic f32x8 dot
calculation, assembly uses paired 128-bit loads and `fmla.4s`, then vector
reductions, with no arithmetic helper calls in that sequence. The source
example uses small known inputs, so this is an inlining/codegen check, not a
general performance bound. See [magetypes-dot.asm](magetypes-dot.asm) and
[archmage-patterns.log](archmage-patterns.log).

`magetypes/benches/generic_vs_concrete.rs` and
`magetypes/examples/polyfill_demo.rs` both have empty non-x86 mains. Neither
provides an ARM measurement. Extend the former with native ARM cases to test
non-inlined versus inlined generic helpers and larger working sets.

Latest magetypes also has native signed i8-to-i16 widening implementations in
`magetypes/src/simd/impls/arm_neon.rs`:4360. Codec comments that justify scalar
cross-color transforms solely by absent widening primitives need reviewing
against the dependency version actually selected. This audit did not implement
or measure a replacement cross-color transform.

## Reproduction and validation

Exact commands are in [commands.txt](commands.txt). No `target-cpu=native` or
custom target-feature flags were used. zenwebp release uses its checked-in
`lto = true`, `codegen-units = 1`. Builds were serialized with four jobs;
the recorded benchmark ran with `nice -n 19`. The Linux `run-heavy` script
requires `/proc`, systemd and ionice and was not usable on this Mac. macOS
`/usr/bin/time -l` recorded resources; there was no cgroup memory cap.

The recorded kernel run exited 0 in 219.86 seconds. The full benchmark log
reports peak RSS 30,195,712 bytes. The release benchmark build exited 0 in
24.18 seconds, with peak RSS 425,951,232 bytes. These are process observations,
not decoder memory budgets.

- zenwebp scoped formatting check passed.
- zenwebp release library tests: 318 passed, 0 failed, 1 pre-existing ignored.
- zenwebp ARM clippy with `_dev` and `-D warnings` failed on 16 existing lint
  diagnostics: three unused macros, one unused import, one unused mutable
  binding, one unused variable, eight private-interface diagnostics, and two
  unused functions. See [clippy.log](clippy.log). No lint was suppressed.
- archmage generation, registry validation, token validation, and soundness
  checks passed and left tracked files unchanged; xtask emitted warnings.
- archmage scoped formatting check passed; ARM idiomatic-pattern execution
  passed. This does not replace its full test suite.

Two preliminary benchmark launches were interrupted after discovering that
`--help` starts the benchmark. Their measurements are excluded. Only the
subsequent serialized, niced complete run is reported here.

## Proposed documentation corrections

Owner review is required by the shared instructions before changing existing
documentation. The evidence above supports reviewing these claims together:

- `src/decoder/lossless_transform_simd.rs`: the add-green comment saying the
  compiler autovectorizes this well does not describe this ARM build.
- Encoder/decoder cross-color comments about missing widening primitives
  need version-qualified wording.
- `benchmarks/zenwebp_arm_profile_2026-05-30.tsv` says no scalar hot kernel is
  missing a NEON path and names `src/encoder/simd/*`, which is not the current
  layout. Retain its historical scope, but annotate the newer findings.

No existing documentation claims have been rewritten in this audit.
