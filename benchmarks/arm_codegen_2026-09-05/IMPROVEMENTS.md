# ARM kernel improvements

Two zenwebp kernels now beat their scalar references on the measured Apple M4
Pro. These results cover add-green and spectral distortion, not end-to-end
codec throughput or the whole ARM implementation. The initial
[audit](README.md) describes the pre-change source and measurements; its
pending experiments and sync blocker are historical. The owner subsequently
approved preserving existing work and syncing the other codec checkouts.

## Add-green

The old portable decoder body accepted a SIMD token but emitted scalar byte
loads and stores. `c1cf4a1b` expressed the operation with `u8x16`;
`093193ce` uses a `u8x32` main loop, a `u8x16` tail, and a scalar partial tail.
It adds green to red and blue with wrapping byte arithmetic, preserving green,
alpha, incomplete pixels, and caller-selected row boundaries.

| Image dimensions | New NEON mean | Scalar mean | Scalar / NEON |
|---|---:|---:|---:|
| 64x64 | 405.3 ns | 3.34 us | 8.25x |
| 256x256 | 9.37 us | 55.67 us | 5.94x |
| 1024x1024 | 105.57 us | 618.02 us | 5.85x |
| 4096x4096 | 1.65 ms | 9.55 ms | 5.79x |

[Paired measurements](zenwebp-green32-sizes.log) use 30 interleaved rounds per
size. The unchanged scalar path is timed in the same run. The
[16-byte experiment](zenwebp-green16-sizes.log) also wins against scalar; it
was a separate run, so it does not establish a paired 16-versus-32 codec delta.
The separate archmage seven-way benchmark provides that row-kernel comparison.

[Assembly](decoder-add-green32.asm) shows paired vector loads/stores and two
`tbl.16b`/`add.16b` operations per 32 bytes, followed by a vector tail. The
optimized function inlines into the benchmark runner, so the excerpt includes
its enclosing symbol and original addresses. There are no calls or bounds
checks inside this vector loop. This uses zenwebp's locked magetypes 0.9.27;
no dependency update was required.

## Spectral distortion

`e510152f` validates each input row as a four-byte array before combining it
with the other block's row. Transform arithmetic and weights are unchanged.

| Block | Previous NEON mean | New NEON mean | New-run scalar mean |
|---|---:|---:|---:|
| 4x4 | 20.5 ns | 9.9 ns | 27.7 ns |
| 8x8 | 83.7 ns | 43.4 ns | 130.9 ns |
| 16x16 | 352.1 ns | 168.1 ns | 535.2 ns |

[Before](tdisto-before.log) and [after](tdisto-after.log) are separate runs,
each pairing NEON with scalar. Their scalar controls remain close, while the
new NEON implementation takes about half the previous time. Shared-host
variance is substantial (some CVs exceed 20%); exact cross-run percentages
should not be treated as stable CPU characteristics.

The stride-32 specialization changes from 32 to 14 input-length branches
before the first pixel load. [New assembly](tdisto-fixed-rows.asm) uses
four-byte vector loads and widening instead of individually loaded bytes.
The retained function shrinks from 323 to 151 instruction lines, including
cold panic paths; these are code-size counts, not executed-instruction counts.
Remaining bounds checks are visible and are not claimed to be eliminated.

## Correctness and tooling

- [Native release library tests](final-native-tests.log): 319 passed,
  zero failed, one existing ignored test unchanged.
- [Native SIMD parity tests](parity-tests.log): 16 passed, zero failed.
  New cases exercise all 256 deterministic block seeds, offsets 0 through 15,
  strides 4, 5, 7, 16, 31, and 64, poisoned row padding, and a minimal final row.
- [WASM SIMD release library tests](wasm-tests.log): 305 passed, zero failed,
  one existing ignored test unchanged, executed with wasmtime on wasm32-wasip1
  and SIMD128 enabled. This verifies the changed portable add-green body;
  the spectral-distortion edit is ARM-only.
- Add-green tests cover every channel/green byte pair, offsets, partial tails,
  padding, and strided row slices against an independent byte oracle.
- Scoped formatting passes. The final ARM library clippy check still fails
  on 15 existing diagnostics; none were suppressed. The formerly unused
  `u8x32` import is now used, reducing the original count of 16.
- Archmage's new ARM benchmark passes clippy with warnings denied. Its
  generation, registry, token, and soundness checks passed. A full archmage
  cross-platform test suite was not run.

No unsafe code, public API, pixel expectations, or dispatch policy changed.
No x86 performance claim or WASM speedup is made. Other ARM CPUs and
end-to-end codec throughput remain unmeasured.

## Reproduction and provenance

Host/compiler match the initial report: M4 Pro, 24 GiB RAM, macOS Darwin
25.5.0, rustc 1.98.0 (88d9e12ae), LLVM 22.1.8. Run date is 2026-09-05 local /
2026-09-06 UTC. No native CPU or custom ARM target-feature flags.

Source commits: `aadce837` (byte oracle), `c1cf4a1b` (16-byte implementation),
`bd677255` (size sweep), `093193ce` (32-byte implementation and WASM oracle),
`e510152f` (fixed-row spectral distortion and strided oracle).
Logs sometimes identify the preceding commit because the measured change was
still in the jj working copy. The green32 run contains the code committed as
`093193ce`; tdisto-after contains the code committed as `e510152f`.

Use `just arm-kernel-audit-macos inverse_subtract_green` and
`just arm-kernel-audit-macos tdisto`. Full logs are saved under `~/tmp`.
Timing excludes input construction and allocation/drop. Inputs are fixed,
deterministic byte data; these kernels have no quality knob. No coefficients,
thresholds, or codec-wide defaults are calibrated from this experiment.

Commands for correctness:

```sh
cargo test --locked --release --lib
cargo test --locked --release --features _dev --test simd_dispatch_arch_parity
cargo test --locked --lib --release --target wasm32-wasip1 --no-default-features
```

All heavy invocations were serialized with `nice -n 19` and four
build/Rayon/OpenMP threads. macOS cannot run the Linux cgroup-based run-heavy
wrapper. `/usr/bin/time -l` recorded cargo build/run peak RSS: green32
424,706,048 bytes (28.79 s), tdisto-before 96,829,440 bytes (13.10 s),
tdisto-after 426,180,608 bytes (18.62 s). These include any build work and are
not codec memory usage. Full benchmark logs preserve the resource counters.
