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

