# zenwebp benchmarks

How the numbers in the [README](../README.md#performance) and
[`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) are produced, and how to
reproduce them. This directory also holds committed result files (sweep TSVs,
ablation JSON, profiles) from past runs — those are dated artifacts, not the
live harness.

The live benchmark harnesses are the Rust files in [`../benches/`](../benches/),
built on [zenbench](https://github.com/imazen/zenbench) (interleaved A/B
execution with paired statistics — purpose-built to cancel the thermal/turbo
bias that back-to-back runs bake in). New comparison benches go here, not into
criterion.

## Ground rules (these make the comparison fair)

- **No `-C target-cpu=native`.** Build with the default target so the binary uses
  runtime SIMD dispatch (SSE2/SSE4.1/AVX2 / NEON / SIMD128) — that is what ships
  to users. `native` bakes in ISA extensions at compile time and reports speeds
  no user gets. (Some bench-file header comments still say to set `native`; do
  not — `docs/PERFORMANCE.md` numbers were measured without it.)
- **I/O is outside the timed region.** Each harness decodes the corpus PNG and
  encodes the WebP test bytes into RAM *before* the measured loop; the timed
  closure only decodes/encodes in-memory `&[u8]`/`Vec<u8>`. Output is consumed
  with `black_box` so it can't be optimized away. No file open/read/write is ever
  inside a timed closure.
- **Single-threaded, both sides.** zenwebp encode and decode are single-threaded.
  libwebp is driven through its simple API, whose `use_threads` defaults to OFF —
  so this is 1-thread vs 1-thread. (libwebp's 2-thread decode pipeline was
  measured and is a net negative on this corpus; see `docs/PERFORMANCE.md`.)
- **Apples-to-apples inputs.** Both contenders decode the *same* WebP bytes (or
  encode the *same* pixels) at the *same* dimensions, pixel format, and
  quality/method target. Lossy decode benches use Q75 m4; state the setting when
  you add a cell.
- **No faked names, no synthetic-gradient overfit.** Corpus images come from the
  [`codec-corpus`](https://crates.io/crates/codec-corpus) crate (CLIC2025 photos,
  screenshots, CID22, gb82). See `~/work/claudehints/topics/benchmarking.md`.

## Competitors and pinned versions

All pinned in the repo `Cargo.lock`; the dev-dependency floors are in
[`../Cargo.toml`](../Cargo.toml):

| Contender | Crate | Version | Backend |
|-----------|-------|---------|---------|
| libwebp (C reference) | [`webp`](https://crates.io/crates/webp) | 0.3.1 | libwebp via `libwebp-sys` |
| libwebp (C, low-level) | [`libwebp-sys`](https://crates.io/crates/libwebp-sys) | 0.14 | libwebp |
| libwebp (C, w/ threading probe) | [`webpx`](https://crates.io/crates/webpx) | 0.4.0 | libwebp |
| pure-Rust upstream | [`image-webp`](https://crates.io/crates/image-webp) | 0.2 | pure Rust |

zenwebp is built from this repo at the commit under test (record the full SHA in
any results file you commit — `git rev-parse HEAD`).

## Reproduce

```sh
git clone https://github.com/imazen/zenwebp && cd zenwebp
git checkout <full-commit-sha>          # the commit the numbers came from

# Decode throughput, zenwebp vs libwebp (14-image corpus, lossy Q75 m4):
cargo bench --bench decode_compare

# Lossless decode throughput:
cargo bench --bench decode_lossless_compare

# Encode speed vs libwebp:
cargo bench --bench encode_vs_libwebp
cargo bench --bench encode_benchmark
```

Do **not** prefix with `RUSTFLAGS="-C target-cpu=native"`. The corpus is fetched
on first run by the `codec-corpus` crate; benches print and skip any image the
corpus can't supply.

Available harnesses (`[[bench]]` in `Cargo.toml`): `decode_compare`,
`decode_lossless_compare`, `decode_benchmark`, `decode_zenbench`,
`encode_benchmark`, `encode_vs_libwebp`, `predictor_coalesce` (needs
`--features _benchmarks` on nightly).

## Charts

zenbench renders sorted-throughput bar charts in the terminal directly; for a
self-contained report use `--format=html`, or the `charts` feature for standalone
SVGs. For "speed vs size" codec comparisons, plot an RD/Pareto scatter (x = bytes
or bpp, y = SSIMULACRA2 / butteraugli / zensim) from the exported per-cell
samples — one line per codec swept across quality. Avoid pie/3D/dual-axis charts.
See the zen* README conventions (§7) for chart-choice guidance.

## Profiling (instruction-level, not wall-clock)

Wall-clock ratios live in `docs/PERFORMANCE.md`; instruction-count breakdowns vs
libwebp (callgrind) and the optimization history live in
[`../CLAUDE.md`](../CLAUDE.md). Profile under callgrind *without* `native`
(valgrind can't model AVX-512). The `callgrind_*` examples under `../examples/`
are minimal drivers for this.
