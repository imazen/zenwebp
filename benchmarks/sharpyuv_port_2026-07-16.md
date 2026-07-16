# SharpYUV port: byte-exact parity, the "why", and tuned adoption (2026-07-16)

Continuation of `sharpyuv_dig_2026-07-16.md` (#38 "port fully optimized
sharpyuv to rust, and figure out the why, and why zenyuv isn't an exact
match"). `src/encoder/sharpyuv.rs` is an exact Rust port of libwebp's
SharpYUV library (`sharpyuv/sharpyuv{,_dsp,_gamma,_csp}.c`) at the WebP
encoder's operating point: `rgb_bit_depth=8`, `yuv_bit_depth=8`,
`kSharpYuvMatrixWebp`, sRGB transfer.

## Verification (byte-exactness)

- `dev/sharpyuv_selftest.rs` vs `dump_sharpyuv` (instrumented libwebp tree
  `~/work/zen/libwebp--zen38trace`, calls `SharpYuvConvert` directly):
  **IDENTICAL on 17 shapes** — 1×1, 2×2, 3×3, 5×2, 2×5, 16×16, 17×17,
  31×47, 64×64, 127×129, 256×255, 512×512, 640×480, 1023×7, 7×1023,
  999×501, 63×999.
- vs `dump_encoder_yuv` (replicates webpx's exact flow: `use_argb=1` import
  → `WebPPictureSharpARGBToYUVA`): **IDENTICAL** on CID22 382297 (real
  512×512 photo).
- Whole-file: `dev/byteparity_sweep.rs` sharp_yuv axis **0/96 → 96/96
  byte-identical** (3 CID22 images × 2 configs × 4 q × 4 m), base grid
  unchanged 4004/4004, all other axes still 100%. CI anchor:
  `tests/libwebp_byte_parity.rs::sharp_yuv_matches_libwebp`.

Two port bugs found on the way, both by plane diffing (not by eyeballing):
1. **Exit threshold uses PADDED dims**: libwebp computes
   `diff_y_threshold = 3.0·w·h` with its already-rounded-to-even locals.
   Using `width·height` changes the iteration count on odd-dimension
   images (caught: single ±1 V sample on 1023×7).
2. **The conversion interface stride is in PIXELS** (matching
   `convert_image_yuv_fast`); passing it to the port as bytes sheared the
   planes 3× vertically. Caught with the `ZYUVDUMP` env dump (mode_debug)
   of the encoder-input planes vs `dump_encoder_yuv` — the whole-file
   diff alone pointed at segmentation, which was a red herring.

## Why libwebp's SharpYUV is so much better than plain conversion

Standard 4:2:0 conversion computes Y per pixel and box-averages chroma per
2×2 block, each in gamma space, independently. SharpYUV instead solves a
small optimization problem: **choose the Y plane and chroma deltas so that
what the DECODER reconstructs matches the source in linear light.**

Mechanically (all fixed-point):
1. Work in a 10-bit W/(R−W,G−W,B−W) space (`GetPrecisionShift(8)`=2 extra
   bits; W = BT.709-weighted gray `(13933r+46871g+4732b+32768)>>16`).
2. Targets are computed in LINEAR light: per-pixel target luma
   (`UpdateW`: sRGB→linear per channel, gray, linear→sRGB) and per-2×2
   target chroma (`UpdateChroma`/`ScaleDown`: each channel's 4 samples
   averaged in linear space). The sRGB↔linear tables are 16-bit
   fixed-point (1026/514 entries, linear-interpolated).
3. Up to 4 gradient iterations: reconstruct full-res RGB from the current
   (Y, chroma) using the decoder's OWN 9-3-3-1 fancy-upsampling filter
   (`SharpYuvFilterRow`), re-derive its luma/chroma (`UpdateW`/
   `UpdateChroma`), and move Y and chroma by the difference
   (`SharpYuvUpdateY`/`UpdateRGB`). Exit when Σ|Y error| < 3·w·h (padded)
   or grows.
4. Final W/RGB → YUV through `kWebpMatrix` in 16.16 fixed point.

Three ingredients do the work:
- **Linear-light averaging** kills the classic gamma-space darkening of
  saturated chroma edges (red/blue text fringes, thin colored lines).
- **Optimizing against the decoder's actual upsampling filter** means
  chroma ringing that the 9-3-3-1 filter would produce is pre-compensated;
  the box-average has no model of the decoder at all.
- **The luma plane participates**: Y absorbs the residual the
  half-resolution chroma cannot express (that's why sharp output Y differs
  from standard Y — and why the encoded files are 2-5% larger: Y carries
  real added detail).

## Why zenyuv's sharp converter isn't (and can't be) an exact match

zenyuv `sharp.rs` (0.1.3) is a deliberate redesign, not a port:

| | libwebp SharpYUV (and our port) | zenyuv sharp |
|---|---|---|
| Arithmetic | 10-bit fixed-point W/RGB-deltas, u16/i16 | f32 SoA |
| Objective space | linear light (16-bit fixed tables) | gamma-encoded RGB |
| Reconstruction model | decoder's 9-3-3-1 fancy-upsample across blocks | per-2×2-block constant chroma (box model), blocks independent |
| Solver | forward-gradient, whole-plane, ≤4 iterations, global Σ\|ΔY\| exit | per-block Newton step with inverse-matrix Jacobian, 2 iterations |
| Luma treatment | Y plane iteratively co-optimized | Y fixed from SIMD kernel; optional separate `refine_y` pass |
| Exit criterion | Σ\|ΔY\| < 3·w·h or error grows | fixed 2 iterations (threshold field unused in SIMD path) |

Byte-parity is impossible by construction (f32 vs fixed point, different
objective, different reconstruction model, different solver). The deeper
issue is QUALITY: zenyuv's per-block box reconstruction model never sees
the decoder's cross-block 9-3-3-1 filter, and its objective is gamma-space
L2, so it fixes only a fraction of what SharpYUV fixes. Measured (below):
zenyuv sharp buys +0.18..+0.32 zsim over standard; the port buys
+1.0..+1.8.

## Quality A/B (15-image corpus, m4, q ∈ {25,50,75,90})

Harness: `dev/sharpyuv_compare.rs`; corpus = 3 CID22-512 validation images
+ 12 imazen-26 content classes at ≤1024 (`~/tmp/abcorpus`, recipe in
`tuned_candidates_2026-07-16.md`). "port" = the port's planes through the
tuned zen encoder via Yuv420 input; zenyuv/lib rows through their own
sharp flags. Raw data: `sharpyuv_port_ab_2026-07-16.tsv`.

Δzsim vs zen standard conversion (mean over 15 images), bytes ratio vs
standard:

| q | zenyuv sharp | port | libwebp sharp | bytes: zenyuv / port / lib |
|---|---|---|---|---|
| 25 | +0.175 | **+0.997** | +0.835 | 1.003 / 1.021 / 1.014 |
| 50 | +0.238 | **+1.189** | +0.953 | 0.999 / 1.030 / 1.022 |
| 75 | +0.232 | **+1.304** | +1.222 | 1.000 / 1.036 / 1.034 |
| 90 | +0.318 | **+1.776** | +1.781 | 0.998 / 1.050 / 1.050 |

- port vs zenyuv, same tuned encoder: **+1.08 zsim mean, 47/60 cells win**.
- port (zen encoder) vs libwebp sharp (libwebp encoder): +0.12 zsim mean at
  1.004× bytes — i.e. same converter, zen's encoder keeps its usual edge.

## Speed

`dev/sharpyuv_selftest.rs --bench` vs the instrumented libwebp
(`dump_sharpyuv` bench mode), same synthetic content (seed 99), min-of-N:

| | 512×512 | 1024×1024 |
|---|---|---|
| port (scalar Rust, autovectorized) | 5.93 ms (44.2 Mpix/s) | 24.10 ms (43.5 Mpix/s) |
| libwebp (SSE2 build) | 9.31 ms (28.2 Mpix/s) | 36.73 ms (28.5 Mpix/s) |

**The scalar port is ~1.5× faster than libwebp's own SSE2-dispatched
build.** libwebp's SIMD covers only `UpdateY`/`UpdateRGB`/`FilterRow`; its
gamma chain (`UpdateW`/`UpdateChroma`/`ScaleDown` — the bulk of the work)
is scalar C. Our tight u16 loops autovectorize (LLVM) and the gamma
interpolation is branch-free, so no explicit SIMD was needed to beat the
"fully optimized" original. (No further SIMD work planned unless a profile
shows conversion mattering in a real workload; it is an opt-in path.)

Cost in context: at 1 MP the conversion adds ~24 ms — comparable to an m4
encode of the same image (m4 512×512 ≈ 11.6 ms ⇒ ~46 ms/MP). cwebp's
`-sharp_yuv` costs more on the same machine (37 ms/MP conversion alone).

## Tuned adoption → ADOPTED for `.sharp_yuv(true)`

The plain `.sharp_yuv(true)` / `with_sharp_yuv(true)` path now runs the
port (was: zenyuv's converter). Rationale: the flag is an explicit opt-in
whose meaning — matching cwebp's `-sharp_yuv` — is "spend encode time for
the best 4:2:0 chroma"; delivering +0.2 zsim where the same flag in
libwebp delivers +1.2 was a trap for users switching from cwebp. The
+2-5% byte cost is inherent to the technique (Y carries the chroma
compensation) and applies to libwebp's flag equally.

- The tuned DEFAULT (sharp off) is byte-unchanged; no preset enables
  sharp_yuv.
- `sharp_yuv_config(custom)` still routes to zenyuv's Newton converter
  (those knobs parameterize it); only the default-config path switched.
- Below libwebp's `kMinDimensionIterativeConversion` (4 px), sharp falls
  back to standard conversion on both cost models, like libwebp.

## Repro

```bash
# byte-exactness vs libwebp (needs the instrumented tree built once):
cd ~/work/zen/libwebp--zen38trace && ./build_zen38.sh   # + dump tools, see file headers
./dump_sharpyuv 129 65 77 > /tmp/p.bin
cargo run --release --features __expert --example sharpyuv_selftest -- /tmp/p.bin 129 65 77

# full parity gate:
cargo run --release --features __expert --example byteparity_sweep

# quality A/B:
cargo run --release --features __expert --example sharpyuv_compare -- <corpus> 25,50,75,90 4

# speed:
cargo run --release --features __expert --example sharpyuv_selftest -- --bench 1024 1024
```
