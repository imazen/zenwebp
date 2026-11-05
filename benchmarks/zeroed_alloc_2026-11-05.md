# Zeroed allocation unification

`decoder::alloc_util::alloc_zeroed` now uses one
[`bytemuck::allocation::try_zeroed_vec`](https://docs.rs/bytemuck/1.25.2/bytemuck/allocation/fn.try_zeroed_vec.html)
success path. The caller's allocation preference is evaluated only on failure.
Fallible sites return `AllocFailed`; infallible sites retain capacity-overflow
panics and the allocation-error handler. The public API is unchanged.

Bytemuck checks the layout and calls `alloc_zeroed` for nonempty, non-ZST
buffers ([source](https://docs.rs/bytemuck/1.25.2/src/bytemuck/allocation.rs.html#92-145)).
It needs `Zeroable` and `extern_crate_alloc`, both already available for this
byte helper. It does not need `std`.

## Measurement

[Raw results](zeroed_alloc_2026-11-05.txt), from
[benches/zeroed_alloc.rs](../benches/zeroed_alloc.rs), on 2026-11-05 UTC:

- CPU: AMD Ryzen 9 7900X; host `r7900x`; Linux 7.0.0-38-generic.
- Rust 1.99.1, LLVM 23.1.1; release optimization, thin LTO, no native CPU flag.
- Locked bytemuck 1.25.2 and zenbench 0.1.8.
- Production helper: commit `ea007729340176acca55101598b5bb25951c3831`.
- Benchmark source: commit `fddedbc791df058f7019c3a01553767d8fbeccc2`.
  The raw runner's git header names base `2e58d1e5`: the test host received
  the committed files by rsync before the refactor landed. These content
  hashes identify what was built:

```text
dd60e4b37002486cdd9500d04e4d7b1a847d4569b5dfcdbc2f0b70ec307a2d48  src/decoder/alloc_util.rs
2e13b4081c04b25c959fa9140014649bb27782353bc45b1e3a0daf2bba76aae4  benches/zeroed_alloc.rs
74140bca422cb862fc44ac0caaa118b9d21f84ee9e43237278f17fe83ad39798  benchmark executable
```

Run from the checkout under a disk-backed `TMPDIR`:

```sh
run-heavy --mem 12G --jobs 8 -- just zeroed-alloc-perf --no-busy-gate
```

The explicit scanner opt-out was necessary because zenbench 0.1.8 repeatedly
classified idle agent command lines and threads as competing benchmarks.
Earlier scanner-blocked attempts were interrupted and are excluded. The
exclusive benchmark file lock remained enabled. Host inspection showed no
other cargo/rustc job; the idle review process measured 0.50% CPU over two
one-second pidstat samples. This is one host-supervised run, not qualification
on a dedicated machine. The completed lint/build/measurement scope reported:

```text
run-heavy: done rc=0 49s | peak-RSS 0.28GiB | min-avail 20404MiB | peak-load 0.93
```

All arms include allocation and deallocation. The fill workload exposes the
zeroed slice to `black_box`, overwrites every byte with `0xAB`, exposes it
again, then drops it. Repeated allocations may reuse allocator storage.
Allocation-only timings defer first-touch costs; they are not codec timings.

Selected reported means:

| Bytes / workload | `vec!` | Reserve + resize | Unified fallible | Unified infallible |
|---|---:|---:|---:|---:|
| 0 / allocate | 0.3 ns | 2.2 ns | 1.3 ns | 1.3 ns |
| 16,384 / allocate | 110.3 ns | 113.2 ns | 111.9 ns | 112.3 ns |
| 262,144 / allocate | 1.55 µs | 1.55 µs | 1.56 µs | 1.55 µs |
| 4,194,304 / allocate | 28.18 µs | 28.28 µs | 28.28 µs | 28.24 µs |
| 67,108,864 / allocate | 5.65 µs | 21.16 ms | 5.71 µs | 5.71 µs |
| 67,108,864 / fill | 21.48 ms | 23.53 ms | 21.42 ms | 21.34 ms |

The largest allocation-only group has five rounds; its iteration estimator
picked 176 calls per round before accounting for the slower reserve/fill arm.
The largest fill group has 23 rounds, and the other groups have 30. Keep those
sample counts beside the numbers. Empty-buffer differences also include the
runtime preference barrier used only by the unified arms.

These measurements support removing the separate `vec!` success path from
this byte helper. They do not show a general speedup for smaller buffers or
measure end-to-end decoding, parallel page faults, another allocator, ARM,
Windows, or Zen 5.

## Validation

The baseline passed formatting, strict library Clippy and its 14 allocation
tests. The changed source passed `just zeroed-alloc-check`: 17 allocation
tests, the golden codec integration test with independent libwebp readback,
strict library/benchmark Clippy, no-default-features check and formatting.
The complete default library run passed 363 tests with one pre-existing
ignored test; doctests passed 33 with three pre-existing ignored examples.
No established expectation or ignore was changed.

The new controls check empty/page-boundary buffers in all three policies and
both site defaults, writable/growable storage, and both infallible
capacity-overflow cases. The impossible-size error test checks layout refusal;
it does not inject valid-layout allocator failure. Actual allocator-refusal
handling was reviewed against the source, not exercised by exhausting RAM.

## Related helper audit

This audit classified conditional fresh allocations in the active Rust trees;
it was not an exhaustive inventory of every allocation. Only zenwebp was
changed. The remaining candidates, observed 2026-11-05 UTC, are:

| Repository and helper | Conversion constraint |
|---|---|
| zenjpeg `foundation/alloc.rs`: `try_alloc_zeroed_bytes_pref`, `try_alloc_zeroed_pref`, `try_alloc_zeroed_f32_pref`, `try_alloc_dct_blocks_pref` | Concrete zero-only candidates. Keep overflow categories, byte counts and context; enable bytemuck's allocation feature. The existing reserve/resize comments incorrectly describe the `SpecFromElem` allocation specialization. |
| zenjxl `alloc_util::alloc_zeroed` | Direct byte candidate; needs a direct bytemuck dependency. |
| zenbitmaps `alloc_util::alloc_zeroed` | Direct byte candidate; dependency must work with empty default features. |
| zenpng `alloc_zeroed`, `alloc_zeroed_typed` | Already use bytemuck when fallible; the remaining `vec!` branch can adopt error-only policy handling. |
| zengif `alloc_util::alloc_zeroed` | Specialize the zero wrapper and retain `Stats` limit checks and rollback. Generic canvas background fills are not necessarily zero. |
| zenraw `alloc_util::alloc_filled` | Split known-zero u8/f32 callers from arbitrary fills. |
| zenavif `alloc_util::alloc_filled` | Split known-zero callers; some RGBA fills require alpha 255. Enable the allocation feature explicitly. |
| zentiff `alloc_util::alloc_filled` | Verified conversion callers use u8/u16/f32; `Default + Clone` alone is insufficient to promise zero bytes. |
| HEIC `alloc_util::alloc_filled` | Its uncompressed-image caller uses zero u8; bytemuck must be reachable without the separate zencodec feature. |
| jxl-encoder `budget::{vec_f32_zeroed_fallible, vec_i64_zeroed_fallible, try_alloc_zeroed_permanent}` | `OutOfMemory(TryReserveError)` cannot directly receive bytemuck's `()`. Preserve the error contract and budget accounting before converting. |
| HEIC/SVT feature-selected fill macros | Public generic contracts include nonzero sentinels; use a separate zero-only path. HEIC's parent `fallible-alloc` flag also does not forward to heic-core's allocator flag. |

zenpixels' separate fallible buffer constructors are a secondary opportunity.
SVG/PDF/JP2 byte helpers and zentiff's separate byte helper were unused
templates in the inspected source. Capacity-only helpers must remain empty,
growth helpers must preserve existing data, and zenjxl-decoder's aligned
`AVec` helpers must retain 64-byte alignment.

For generic conversions, `Zeroable` does not imply that zero equals
`Default`. A ZST `Vec` has capacity `usize::MAX`, so the byte helper's exact
capacity assertion must not be copied into a generic ZST test. Numeric
comparison to `0.0` also cannot distinguish positive and negative zero.
