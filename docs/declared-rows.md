# Declared WebP row encoding

The zencodec adapter accepts `begin(width, height, descriptor)` and returns
`OutputTiming::AT_FINISH`. It validates the declaration before changing state.
A rejected declaration permits retry. Once admitted, one-shot encode and a second
begin refuse. Rows must retain the declared width and descriptor, cannot exceed
the height, and finish requires the complete canvas. The older direct
`push_rows` path remains available and validates consistency across strips.

This adapter accumulates the full raw raster. It does not provide low-memory
streaming or compressed output before finish. Dimension checks and retained raw
byte admission precede accumulation; native workspace estimation remains a
separate check in final encoding. No whole-operation proven bound is claimed.

Only visible row bytes enter the accumulator. The pixel buffer contract requires
a stride divisible by pixel size; non-pixel-aligned RGB stride is rejected by
PixelSlice before this adapter runs. The regression compares packed and two
whole-pixel-padded layouts, declared and legacy row paths,
lossless and lossy, and validates lossless samples with libwebp via webpx.

Qualification: four new controls, all 51 existing codec tests and strict Clippy
pass on ARM64. The baseline reproduces begin refusal and padded-row divergence.
Original assertions and native codec algorithms are unchanged. See the
[full receipt and limits](../benchmarks/declared_rows_2026-10-29.md).
