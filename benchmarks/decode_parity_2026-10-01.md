# Wide-corpus decoder parity vs libwebp — 2026-10-01

**Result: 43,414 / 43,414 comparisons byte-identical (tolerance 0) across
13,601 files**, after three fixes found by this sweep (below). Reference:
libwebp **1.6.0** (vendored in `libwebp-sys` 0.14.4, driven through `webpx`
0.4.0). Host: r5900xt (x86_64, AVX2 dispatch, no `target-cpu=native`).

Tool: `dev/decode_parity_sweep.rs`
(`cargo run --release --features __expert --example decode_parity_sweep`).

## What is compared

Every file is decoded by both decoders in each mode that applies, and the
output buffers must match byte for byte:

| mode | applies to | zenwebp entry | libwebp entry |
|---|---|---|---|
| `rgba` | stills | `oneshot::decode_rgba` | `WebPDecodeRGBA` |
| `rgb` | stills | `oneshot::decode_rgb` | `WebPDecodeRGB` |
| `rgba_nofancy` | lossy stills | `DecodeConfig::upsampling(Simple)` | `WebPDecode`, `no_fancy_upsampling=1` |
| `yuv` | lossy stills | `oneshot::decode_yuv420` | `WebPDecodeYUV` |
| `anim` | animations | `mux::AnimationDecoder::decode_all` (every composited canvas) | `WebPAnimDecoder` |

Files rejected by both decoders count as agreement; one-sided rejection,
panics, or dimension mismatches count as failures (none occurred).

## Corpus

- **272 existing `.webp` files**: codec-corpus `webp-conformance/valid` (225),
  `image-rs` (9), `imageflow` (4), this repo's `tests/images`, the zencodec
  animation corpus, zenmetrics fixtures, libwebp-sys test files.
- **13,329 generated files** from 702 PNG sources (codec-corpus CID22,
  clic2025, gb82, gb82-sc, kadid10k, qoi-benchmark, pngsuite, png/apng
  conformance, image-rs, imageflow, webp-conformance PNGs). Each source yields
  the full image, random-position crops at 1×1, 3×5, 17×13, 33×31, 127×65,
  250×199 (each ~50 % of sources), and a synthetic-alpha crop (gradient /
  checker / noise / disc). Each variant is encoded **twice by libwebp and twice
  by zenwebp** with a seeded random draw over: quality 0–100, method 0–6,
  segments 1–4, SNS 0–100, filter strength 0–100, sharpness 0–7, filter type,
  partitions 0–3, sharp-YUV, alpha quality / filter / compression,
  preprocessing (libwebp), `StrictLibwebpParity` (zenwebp), and lossless with
  quality, method, near-lossless and `exact` (30 % of draws). Plus one
  animation per encoder per source (2–5 frames, lossy or lossless, synthetic
  alpha, and — zenwebp side — explicit sub-frames with random blend/dispose).
  4,104 generated files are lossless, 2,381 lossy+alpha, 1,221 animated.

Seeds derive from the source path, so the run is reproducible.

## Results by class (after fixes)

| source | anim | rgb | rgba | rgba_nofancy | yuv |
|---|---|---|---|---|---|
| existing | 13/13 | 259/259 | 259/259 | 237/237 | 237/237 |
| encoded by libwebp | 582/582 | 6083/6083 | 6083/6083 | 4272/4272 | 4272/4272 |
| encoded by zenwebp | 639/639 | 6026/6026 | 6026/6026 | 4213/4213 | 4213/4213 |

Still-image fancy RGB/RGBA and YUV were already exact **before** the fixes.

## Divergences found and fixed

Before the fixes: 6,424 of 43,414 comparisons failed.

1. **`UpsamplingMethod::Simple` / `DecodeConfig::no_fancy_upsampling()` was a
   no-op** (5,967 failures). `lossy_upsampling` was stored and never read;
   "nofancy" output was fancy output. Now `DecoderContext` carries the method
   and `Simple` takes the full-frame path with point sampling
   (`yuv420_to_rgb_sampled`, libwebp's `EmitSampledRGB`). The zencodec
   streaming decoder rejects `Simple` so the caller falls back to the full
   decode. Commit `bacf13b` had regenerated `tests/reference/gallery1_nofancy`
   from zenwebp's own (fancy) output and loosened lossy reftests to
   mean-diff < 10, which hid this; the original dwebp `-nofancy` references
   (from `481096d`) are restored and match exactly.
2. **Animation compositing diverged from `WebPAnimDecoder`** (451 failures,
   up to 255 levels): no keyframe logic (keyframes were blended against the
   cleared canvas, losing up to ~50 levels on translucent pixels and zeroing
   RGB under alpha 0), pixels inside a background-disposed previous rect were
   blended instead of copied raw, the previous rect was not cleared when the
   current frame had no alpha (stale pixels), and the blend used a rounded
   divide-by-255 instead of libwebp's `>> 8` (off-by-one). `composite_frame`
   now mirrors `WebPAnimDecoderGetNext` step for step. The two
   `random_lossless` reference frames that commit `34223fa` (upstream
   "Faster alpha blending") re-blessed from the crate's own output are
   restored to the webpmux-derived originals, which match libwebp.
3. **Transparency created by compositing was dropped** (6 failures): an
   animation whose frames carry no alpha (VP8X alpha flag clear) can still
   composite to transparent pixels — a first frame smaller than the canvas, or
   a background dispose the next frame does not repaint. `has_alpha()`
   reported `false`, frames came back as RGB, and the holes turned opaque
   black. `has_alpha()` is now `true` for such animations (detected from the
   ANMF headers at parse time); opaque-only animations still decode to RGB.

Regression tests: `tests/libwebp_decode_parity.rs` (7 tests, each watched to
fail on the pre-fix code) and exact lossy comparison in `tests/decode.rs`.

## Round 2: every decoder option and output format, plus wasm

Same corpus (13,599 files after de-duplicating same-named sources), now 39
modes for lossy stills, 29 for lossless, 1 for animations: **397,593
comparisons**. libwebp side is `WebPDecode` via raw `libwebp-sys` FFI in the
matching `WEBP_CSP_MODE` with `no_fancy_upsampling` / `dithering_strength`.

**Exact on every file** (tolerance 0):

| zenwebp entry point | libwebp equivalent |
|---|---|
| `oneshot::decode_{rgba,rgb,bgra,bgr,argb}` | `MODE_{RGBA,RGB,BGRA,BGR,ARGB}` |
| `decode_{rgba,rgb,bgra,argb,bgr}_into` at stride w+3 (padding verified untouched) | same modes |
| `StreamingDecoder` fed in 977-byte chunks | `MODE_RGBA` |
| `WebPDecoder::read_image` (fancy and `Simple`) | `MODE_RGBA` (+`no_fancy`) |
| zencodec `decode()`, `push_decoder` RGBA / BGRA / `Simple` / dither 100 | `MODE_RGBA` (+options) |
| `DecodeConfig` dithering 50 / 100 (RGBA, RGB, and combined with `Simple`) | `dithering_strength` 50 / 100 |
| `decode_yuv420` | `WebPDecodeYUV` |

**Differ by convention, not by decoded pixels** — confirmed by diagnostic
modes that re-derive libwebp's output from zenwebp's exact RGBA:

| output | zenwebp | libwebp | diagnostic that matches 100 % |
|---|---|---|---|
| `decode_{rgba,bgra,argb}_premultiplied` (1,514 alpha files) | `garb`: `C*A/255` rounded | `floor(C*A/255)` (`(x*a*32897)>>23`) | zen RGBA + libwebp formula |
| `decode_rgb565` | `garb`: `(c*31+128)>>8` rounding, little-endian u16 (documented) | bit truncation (`c & 0xf8`), high byte first (`WEBP_SWAP_16BIT_CSP=0`) | zen RGBA + libwebp packing |
| `decode_rgba4444` | `garb`: `(c*15+128)>>8` rounding, little-endian u16 (documented) | bit truncation (`c & 0xf0`), high byte first | zen RGBA + libwebp packing |

### wasm32-wasip1

`dev/decode_parity_dump.rs` digests zenwebp's output for every (file, mode)
on any target and compares against the libwebp digests the native sweep
wrote (`--save-gen` / `--hash-out`); `dev/decode_parity_wasm.sh` shards it
across wasmtime processes (`just decode-parity-wasm`). Note
`~/.cargo/config.toml` adds `+simd128` for wasm32-wasip1 on this box, so the
scalar build must pass `-C target-feature=-simd128` explicitly.

- **simd128, before the fix: 3,339 lossy files wrong in every mode,
  including raw YUV** — the loop filter. `do_filter6` (macroblock-edge
  filter, non-HEV pixels) used `sat(p0 - q0)` instead of libwebp's base
  delta `p1 - q1 + 3*(q0 - p0)`, with the opposite sign; and the base delta
  saturated `3*(q0 - p0)` before adding `p1 - q1` (libwebp accumulates
  `q0 - p0` three times with saturation) in all three wasm filters. Every
  failing file had the loop filter on; lossless was unaffected.
- **simd128, after: all 397,593 digests identical to x86_64** (so exact vs
  libwebp in every mode where x86 is).
- **scalar wasm: all 397,593 digests identical to x86_64.**

### aarch64-unknown-linux-gnu (NEON), qemu-user

`decode_parity_dump` cross-built (`CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc`)
and run under `qemu-aarch64 -L /usr/aarch64-linux-gnu` in 14 shards: **all
397,593 digests identical to x86_64**. qemu-user emulates NEON
instruction-for-instruction, so this exercises the NEON kernels' arithmetic,
not Apple-silicon timing.

Pinned by `loop_filter::wasm_tests::wasm_filters_match_scalar_spec` (20,000
randomized cases across the six luma kernels vs the scalar spec filters),
which runs in CI's wasmtime `--lib` job and was watched to fail on the old
kernels (kernel 3, case 3).

## Not covered

- Truncated / corrupted inputs (no invalid corpus; error-agreement is untested).
- Cropping / scaling / `bypass_filtering` / flip: zenwebp exposes none of them.
- Animation decoding with `Simple` upsampling or dithering: libwebp's
  `WebPAnimDecoder` has no such options, so there is no reference.
- macOS (Apple silicon) not yet run; aarch64 coverage so far is Linux under
  qemu-user (below).

## Reproduce

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/imazen/codec-corpus ~/tmp/codec-corpus-sparse
(cd ~/tmp/codec-corpus-sparse && git sparse-checkout set webp-conformance image-rs imageflow CID22 \
  pngsuite png-conformance apng-conformance gb82 gb82-sc kadid10k qoi-benchmark clic2025)
C=~/tmp/codec-corpus-sparse
cargo run --release --features __expert --example decode_parity_sweep -- \
  --files $C --files tests/images --gen-src $C --encodes-per-variant 2 \
  --out ~/tmp/decode-parity/run.tsv --mismatch-dir ~/tmp/decode-parity/mismatch
```

Or `just decode-parity` then `just decode-parity-wasm`. Runtime: ~2 min
for the sweep on 28 threads, ~2 min per wasm flavour on 14 shards. The full per-file TSV (4 MB) is not committed.
`--detail FILE` prints a per-frame/per-channel breakdown for an animated
mismatch; with `DP_REF_PREFIX=<dir/name>` it also diffs `<name>-N.png`.
