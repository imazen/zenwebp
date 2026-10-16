# WebP kernel tiers on macOS ARM64, 2026-10-16

Source before this measurement: `62b4ead5` (M4 Pro, macOS ARM64). Command:
`~/work/claudehints/scripts/run-heavy --mem 12G --jobs 4 -- just arm-kernel-audit-macos`.
The recipe runs `cargo bench --features _dev --bench kernel_tiers -- --cpu-policy=unrestricted --format=llm`.
The full local log is `~/tmp/zenwebp-arm-kernels-20261016T230651Z.log`.
The scheduler was unrestricted; these are exploratory kernel measurements, not whole-codec speedups.

| Kernel | NEON mean | Forced scalar mean | Finding |
| --- | ---: | ---: | --- |
| `yuv420_to_rgb_exact/rgb` | 565.03 µs | 2.95 ms | NEON faster |
| `apply_subtract_green/1MP` | 93.84 µs | 210.36 µs | NEON faster |
| `quantize_dequantize_block` | 33.6 ns | 48.1 ns | NEON faster |
| `dct4x4` | 33.3 ns | 32.6 ns | Small scalar advantage; high variability |
| `dequantize_block` | 18.3 ns | 16.8 ns | Existing measured dispatch overhead; see source note in `src/encoder/quantize.rs` |
| `is_flat_source_16/flat` | 7.5 ns | 98.5 ns | NEON faster |
| `is_flat_source_16/early-exit` | 6.6 ns | 5.2 ns | Scalar exits earlier for this input |

The dequantization gap also appears in the committed 2026-09-05 ARM audit. Its source note records a whole-encode cost estimate and why per-block dispatch restructuring was rejected. The flatness results depend strongly on whether the input exits early. No dispatch change is supported by these measurements.

`run-heavy`: peak RSS 0.41 GiB, minimum available memory 5224 MiB, peak load 3.43.
