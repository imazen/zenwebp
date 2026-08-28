# `chunks_exact(N)` → `as_chunks::<N>()` migration, slice 1 — 2026-08-27 (#76)

Measured migration of the VP8L decoder's inverse-transform loops
(`src/decoder/lossless_transform.rs`: predictors 5/7/10/11/12/13, scalar
color transform, scalar subtract-green, color-indexing ≥17 colors and the
packed-index expansion; `src/decoder/lossless_transform_simd.rs`: the scalar
tails of the SSE2 subtract-green / color-transform kernels and the
`row_tf_data` block iterators). 25 of the 26 `chunks_exact` sites in
`lossless_transform.rs` and 6 of 12 in `lossless_transform_simd.rs` were
converted; the runtime-width row iterators (`chunks_exact_mut(width * 4)`)
cannot use a const `N` and the 4 test-scaffold sites were left alone.

Host: Apple M4 Pro (aarch64, NEON tier), release, `lto = true`,
`codegen-units = 1`, no `target-cpu=native`. Harness: a scratch decode-only
binary (`~/tmp/zenwebp-76-bench`, not committed) decoding each input 20x and
reporting the median; run A/B/A/B (before, after, before, after) so drift
shows up as a before-vs-before difference.

## Output invariance

Every input's pixel checksum is identical before and after (the harness
folds every decoded byte); `tests/libwebp_lossless_golden.rs`,
`decoder_vs_libwebp.rs`, `lossless_roundtrip.rs`, `decode.rs`,
`fuzz_regression.rs` and the `lossless_transform_simd` scalar-vs-SIMD
equivalence unit tests all pass on the migrated code.

## `cargo asm` (`--features _dev`), panic-path counts

The kernels are `#[inline(always)]`, so the symbols below are the dispatch
entries they inline into. "panics" counts `panic_bounds_check` /
`slice_*_index` / `core::panicking` references.

| symbol | before lines / panics | after lines / panics |
|---|---|---|

| `subtract_green` | 317 / 0 | 315 / 0 |
| `dispatch_predictor_neon` | 1255 / 5 | 1252 / 5 |
| `add_body_scalar` | 401 / 5 | 401 / 5 |
| `add_body_neon` | 504 / 7 | 504 / 7 |
| `avg_body_scalar` | 829 / 10 | 829 / 10 |
| `predictor_avg_body::<NeonToken>` | 979 / 13 | 979 / 13 |

**No bounds check was removed.** LLVM already proved the `chunks_exact(4)`
bodies in range; the remaining panic paths are the `range`/`old[..]` slicing
and `last_chunk().unwrap()` at the kernel entries, which `as_chunks` does
not touch.

## Wall-clock (median of 20 decodes, ms)

| input | before #1 | after #1 | before #2 | after #2 |
|---|---|---|---|---|
| synth2048_m4 | 59.536 | 59.620 | 59.781 | 59.623 |
| palette1024_m4 | 3.820 | 3.763 | 3.885 | 3.780 |
| 1_webp_ll.webp | 1.297 | 1.269 | 1.344 | 1.290 |
| 2_webp_ll.webp | 1.022 | 1.016 | 1.024 | 1.010 |
| 3_webp_ll.webp | 3.963 | 4.008 | 3.982 | 3.966 |
| 4_webp_ll.webp | 0.538 | 0.558 | 0.548 | 0.555 |
| 5_webp_ll.webp | 1.108 | 1.122 | 1.077 | 1.121 |

Flat: every after-vs-before delta is smaller than the before-vs-before
drift. (`synth2048_m4` is a 2048² gradient+grain image encoded at m4 —
predictor + cross-color + subtract-green heavy; `palette1024_m4` is a
200-color 1024² image — color-indexing heavy; `*_ll.webp` are the
`tests/images/gallery2` lossless stills.)

## Conclusion

For the decoder's inverse-transform loops the lint fix is **hygiene, not an
optimization**: identical bytes, identical bounds-check count, identical
wall-clock. The migration is kept (it is the cleaner spelling and the lint
will stop firing there), but it carries no performance claim. #76's
"real optimization candidate" hypothesis remains open for the remaining 58
sites (`codec.rs` swizzles, `vp8l` encoder, `yuv.rs`), which should be
measured the same way before being converted; the package-level
`chunks_exact_to_as_chunks = "allow"` stays until all sites are migrated.


---

# Slice 2 — 2026-08-28: the remaining sites, migration complete, allow dropped

Scope: every `chunks_exact(N)` / `chunks_exact_mut(N)` with a constant `N`
that clippy 1.98's `chunks_exact_to_as_chunks` reports, across the lib
(24 sites), tests, `dev/`, `examples/` and the fuzz targets (~100 sites).
Only runtime-width iterators (`chunks_exact_mut(width * 4)`, `(stride)`,
`(bpp)`, `(buffer_width / 2)`, `(dst_w)`) remain; the lint does not cover
them and `as_chunks` needs a const `N`. The package-level
`[lints.clippy] chunks_exact_to_as_chunks = "allow"` is removed.

Lib sites by path: VP8L decoder color-cache insert on copies and the
entropy-image unpack (`lossless.rs`), animation canvas clear + composite
(`extended.rs`, `decoder/api.rs`), `iwht4x4` (`common/transform.rs`), the
alpha-fallback lossless writer `encode_frame_lossless` and its
`count_run` / `write_run` helpers (now `Peekable<slice::Iter<[u8; 4]>>`),
`Bgr8` / `L8` → `Rgb8` expansion in `convert_to_contiguous`,
`codec.rs::as_f32_slice`, the `LibwebpExact` alpha-plane extraction
(`yuv.rs`), the unused `fill_rgba_row_simple_scalar` (paired via
`as_chunks_mut::<BPP>().0.as_chunks_mut::<2>()` — no `{BPP * 2}` const
expression needed), the `target_zensim` luma plane, the `analyzer`
RGBA→RGB helper, and the big-endian VP8L argb packing (not compiled here;
converted by symmetry with the LE path).

Host as slice 1 (Apple M4 Pro, aarch64, NEON tier, release `lto = true`,
`codegen-units = 1`, no `target-cpu=native`), with another build agent
active on the box (visible as the occasional +10 ms outlier below).

## Output invariance

- `dev/output_hash.rs` (`--features __expert`, 6 synthetics + 2 imazen-26
  screenshots, both cost models, lossy RGB/RGBA + sharp_yuv + lossless):
  every section hash and `COMBINED: b1309ba43b9b5e43` identical before and
  after.
- 16-case harness (`~/tmp/zenwebp-76-bench`, scratch — decode lossless /
  lossy RGB / lossy RGBA / lossy+ALPH / two animations / two gallery
  stills; encode lossless from Rgb8 / Bgr8 / Bgra8 / L8 and lossy from
  Rgba8-with-alpha / Bgr8 / L8): all 16 output checksums identical.
- Full local run of the CI matrix: default, `--no-default-features`,
  `--features imgref`, `mode_debug`, `__expert` byte-parity gate, `cms`,
  `std,pixel-types,imgref,avx512,cms`, `std`-only lib, wasm32-wasip1 under
  wasmtime with `+simd128` (301 passed), `simd_tier_parity` with
  `testable_dispatch`; `cargo clippy --all-targets -D warnings` on both
  aarch64-apple-darwin and x86_64-apple-darwin (covers the x86-gated test
  sites); `cargo fmt --check`.

## `cargo asm` (`--features _dev --everything`, before vs after)

| symbol | before lines / panics | after lines / panics |
|---|---|---|
| `common::transform::iwht4x4` | 118 / 0 | 118 / 0 |
| `WebPDecoder::read_frame` | 1271 / 7 | 1261 / 7 |
| `LosslessDecoder::decode_image_stream` | 1911 / 8 | 1911 / 8 |
| `encoder::api::encode_frame_lossless` | 937 / 5 | 975 / 5 |
| `decoder::extended::composite_frame` | 693 / 6 | 691 / 6 |
| `codec::as_f32_slice` | 98 / 1 | 99 / 1 |
| whole crate | 3,507,303 / 1490 | 3,501,780 / 1485 |

"panics" = `panic_bounds_check` / `slice_*_index_fail` /
`core::panicking::panic` references. `decode_backward_reference` and
`convert_to_contiguous` are inlined into their callers (no standalone
symbol either side). **No bounds check was removed by any migrated site**;
the −5 crate-wide and −217 symbols are the dropped `ChunksExact` iterator
monomorphizations.

## Wall-clock (median of 20 decodes / 5 encodes, ms; A/B/A/B × 5)

| case | before (5 runs, median ms) | after (5 runs, median ms) |
|---|---|---|
| `dec_lossless_synth2048_m4` | 65.57 / 66.33 / 66.19 / 68.09 / 66.12 | 76.47 / 66.45 / 66.38 / 66.61 / 77.14 |
| `dec_lossless_palette1024_m4` | 5.66 / 5.69 / 5.70 / 6.28 / 5.71 | 5.74 / 5.74 / 5.70 / 5.68 / 6.17 |
| `dec_lossy_rgb_2048` | 42.86 / 43.04 / 43.12 / 44.60 / 43.09 | 43.04 / 43.37 / 42.95 / 43.28 / 44.55 |
| `dec_lossy_rgba_2048` | 45.36 / 45.15 / 45.35 / 47.35 / 45.63 | 45.48 / 45.62 / 45.33 / 45.69 / 47.14 |
| `dec_lossy_alpha_rgba_2048` | 45.79 / 43.58 / 43.60 / 45.01 / 43.61 | 43.55 / 43.78 / 43.35 / 43.64 / 44.25 |
| `dec_anim_random_lossless.webp` | 0.18 / 0.17 / 0.16 / 0.18 / 0.17 | 0.19 / 0.16 / 0.17 / 0.18 / 0.21 |
| `dec_anim_random_lossy.webp` | 0.86 / 0.79 / 0.83 / 0.81 / 0.79 | 0.81 / 0.78 / 0.79 / 0.80 / 0.79 |
| `dec_1_webp_ll.webp` | 1.57 / 1.45 / 1.51 / 1.55 / 1.42 | 1.59 / 1.50 / 1.50 / 1.53 / 1.63 |
| `dec_3_webp_ll.webp` | 5.25 / 4.75 / 4.75 / 4.99 / 4.75 | 4.77 / 4.76 / 4.73 / 4.73 / 4.83 |
| `enc_ll_m2_rgb8` | 249.87 / 246.34 / 247.63 / 251.27 / 245.44 | 247.97 / 245.81 / 247.61 / 244.60 / 244.72 |
| `enc_ll_m2_bgr8` | 247.05 / 254.35 / 246.34 / 251.85 / 245.57 | 248.28 / 243.19 / 246.04 / 246.84 / 242.68 |
| `enc_ll_m2_bgra8` | 583.20 / 582.40 / 602.00 / 588.38 / 584.54 | 613.64 / 598.98 / 601.03 / 603.66 / 599.87 |
| `enc_ll_m2_l8` | 272.26 / 267.50 / 274.35 / 273.32 / 286.23 | 272.35 / 277.91 / 270.31 / 274.70 / 270.93 |
| `enc_lossy_m2_rgba8_alpha` | 60.67 / 61.20 / 60.91 / 61.28 / 62.76 | 60.03 / 60.07 / 59.90 / 59.93 / 60.28 |
| `enc_lossy_m2_bgr8` | 28.33 / 28.72 / 28.28 / 28.33 / 28.89 | 28.19 / 28.44 / 28.41 / 27.96 / 28.27 |
| `enc_lossy_m2_l8` | 18.34 / 19.31 / 18.73 / 18.58 / 19.28 | 18.38 / 18.65 / 18.55 / 18.28 / 18.71 |

Flat within before-vs-before drift on 15 of 16 cases. The exception is
`enc_ll_m2_bgra8` (1024² lossless m2 from `Bgra8`, i.e. with alpha):
min-of-run 582–586 ms before vs 597–602 ms after, +2.7%, reproducible.
Per-file bisection (rebuild the harness with exactly one migrated file
reverted, run `bisect` / `after` / `before` interleaved):

| reverted file | `enc_ll_m2_bgra8` min ms |
|---|---|
| (none — after) | 599–601 |
| `src/encoder/api.rs` | 603 |
| `src/decoder/lossless.rs` | **578–582 (= before)** |
| `src/decoder/api.rs` | 587 |
| `src/decoder/extended.rs` | 588–591 |
| `src/codec.rs`, `yuv.rs`, `transform.rs`, `lossless_transform_simd.rs` | 596–606 |
| (before) | 584–587 |

The shift is caused by `src/decoder/lossless.rs` — the VP8L *decoder*,
which the lossless *encoder* never calls (`grep` of `src/encoder/` for
`decoder::` / `decode_rgba` / `oneshot::` finds only tests). So it is a
code-layout effect of the LTO/`codegen-units = 1` build on the encoder's
hot loops, not the migrated loop itself; `decode_image_stream`, the
function that contains the migrated site, is byte-for-byte the same size
with the same panic paths, and every *decode* case is flat. Recorded here
rather than chased: a 2.7% one-config layout shift is below what this
harness can attribute further without instruction-level profiling, which
this host (macOS, no valgrind) cannot do.

## Conclusion

#76's hypothesis — that `as_chunks` would drop bounds checks in the pixel
loops — is **falsified across the whole crate** (slice 1: the inverse
transforms; slice 2: everything else). LLVM already proved the
`chunks_exact(N)` bodies in range. The migration is kept as the cleaner
spelling and to retire the package-level allow; it carries no performance
claim.

## zenbench A/B (the issue's named gate)

`--save-baseline=pre76_*` from a `jj workspace` at `main@origin`
(47c562b1), then `--baseline=pre76_*` on the migrated tree; same host, same
concurrent build agent. Cross-process baseline deltas on this box are
dominated by that agent (the **libwebp control** in every group moves by the
same amount as zenwebp — e.g. `decode_ph_576_flowers` lib +25.0% / zen
+24.1%, `decode_sc_2k_wiki` lib −27.5% / zen −27.6%, `lossless_decode_photo_512`
lib −51.5% / zen −50.1%), so the attributable statistic is the zen/lib ratio
from each run's interleaved rounds, which is immune to that drift:

`decode_lossless_compare` — zen/libwebp ratio from the in-process interleaved rounds, before → after:

| group | zen / lib before (ms) | ratio | zen / lib after (ms) | ratio | Δratio |
|---|---|---|---|---|---|
| `lossless_decode_photo_512` | 5.7 / 5.0 | 1.140 | 2.8 / 2.4 | 1.167 | +2.3% |
| `lossless_decode_codec_wiki` | 24.9 / 10.4 | 2.394 | 24.4 / 10.0 | 2.440 | +1.9% |
| `lossless_decode_terminal` | 4.2 / 2.6 | 1.615 | 2.5 / 1.6 | 1.562 | -3.3% |

`decode_compare` (lossy) — zen/libwebp ratio from the in-process interleaved rounds, before → after:

| group | zen / lib before (ms) | ratio | zen / lib after (ms) | ratio | Δratio |
|---|---|---|---|---|---|
| `decode_sc_4k_wiki` | 24.2 / 17.3 | 1.399 | 24.8 / 17.6 | 1.409 | +0.7% |
| `decode_sc_3k_imac` | 20.9 / 16.7 | 1.251 | 20.3 / 16.0 | 1.269 | +1.4% |
| `decode_sc_2k_wiki` | 11.3 / 7.0 | 1.614 | 8.2 / 5.1 | 1.608 | -0.4% |
| `decode_sc_2k_ui` | 10.1 / 5.7 | 1.772 | 9.1 / 4.9 | 1.857 | +4.8% |
| `decode_sc_1k_term` | 8.9 / 6.4 | 1.391 | 7.5 / 5.4 | 1.389 | -0.1% |
| `decode_ph_2k_sq` | 17.4 / 13.9 | 1.252 | 16.8 / 13.4 | 1.254 | +0.2% |
| `decode_ph_2k_43` | 53.3 / 50.5 | 1.055 | 55.8 / 53.0 | 1.053 | -0.2% |
| `decode_ph_2k_32` | 16.7 / 12.8 | 1.305 | 17.4 / 13.3 | 1.308 | +0.3% |
| `decode_ph_2k_uw` | 19.2 / 16.4 | 1.171 | 19.6 / 16.9 | 1.160 | -0.9% |
| `decode_ph_2k_pt` | 11.9 / 8.7 | 1.368 | 12.9 / 9.5 | 1.358 | -0.7% |
| `decode_ph_576_baby` | 2.8 / 2.0 | 1.400 | 2.7 / 1.9 | 1.421 | +1.5% |
| `decode_ph_576_city` | 5.0 / 4.2 | 1.190 | 5.3 / 4.4 | 1.205 | +1.2% |
| `decode_ph_576_flowers` | 3.5 / 3.0 | 1.167 | 4.3 / 3.7 | 1.162 | -0.4% |
| `decode_ph_512_cid` | 2.0 / 1.3 | 1.538 | 1.5 / 1.0 | 1.500 | -2.5% |

Every Δratio is inside the ±5% the run-to-run ratio jitter shows for the
unchanged libwebp control; the largest (`decode_sc_2k_ui` +4.8%) is a
5.7 ms case whose lib side moved −13.5% between runs.

`encode_vs_libwebp` (lossy m0/m2/m4/m6 + q50/75/90, `CID22/792079.png`,
criterion-compat timing): `--baseline=pre76_enc` reports **0 regressions,
0 improvements, 22 unchanged** — every zenwebp and libwebp cell within
−1.7%..+0.4% of its baseline (`~/tmp/zenwebp-76-zb-enc-after.log`).
