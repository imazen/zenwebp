# Decoder dispatch and Wasm size, 2026-11-23

Baseline codec: `b123f328`. Implementation: `87ce925f`. The explicit gallery
benchmark mode is in `0317ada9`; the original baseline harness used equivalent
untimed `chunks_exact` oracle checks before clippy requested `as_chunks`.

This changes private decoder code only. The public API, dependency constraints,
pixel expectations, encoder settings and nested recompress workspace are unchanged.

## Changes and attribution

Four generic VP8L predictor bodies must inline into their x86 feature regions.
The release decoder negative control retained **five `__simd_inner_*` definitions
and seven calls**. Forced inlining is specific to x86, preserving shared portable
predictor bodies on Wasm and ARM.

Wasm loop-filter edge kernels and four column gather/scatter helpers are shared
out of line. Contiguous UV rows use two checked eight-byte halves, replacing
sixteen separately checked byte lanes. All arithmetic and filtering order are
unchanged. `attune(wasm128, inline(never))` specifies the edge policy because the
legacy `rite` macro replaces a written inline attribute with an inline hint.

The unused free `pipeline::process_mb_row` and its architecture variants are
eliminated from the linked consumer. The similarly named **live** function belongs
to `DecoderContext`, whose type is declared in `context.rs` and whose decode loop
is implemented in `vp8v2/mod.rs`. Removing annotations from the unused pipeline
would not shrink this artifact. The live row body can absorb the smaller Wasm
filter dispatcher; compare the whole code section rather than one symbol.

The optional allocation audit found an existing borrowed tight-RGB path in
`decoder::yuv::to_tight_rgb`. No extra VP8L bounds or YUV allocation change was
included.

## Reproduction

All size and throughput artifacts were built on **i134, CPU 2**, with Rust
1.99.0 (`b940084d7`, LLVM 23.1.1), default features, release LTO and one codegen
unit. The shared remote runner serialized execution with the other lanes, used
`nice -n 19 taskset -c 2`, and kept scratch below `~/tmp`. Builds used the resource
wrapper with a 12 GiB memory cap and one job. Native `RUSTFLAGS` was empty;
`target-cpu=native` was never used. The lockfile SHA-256 on both revisions was
`43345e6eb0f7caba169d7823b9980b656fa17ef13c06764a0480a6339fce8ce7`.

The original corpus-dependent benches can acquire external datasets. This run
explicitly selects committed photographic fixtures instead, with no skipped
files: five VP8 files in `tests/images/gallery1` and five VP8L files matching
`tests/images/gallery2/*_ll.webp`. Each fixture is checked against independent
libwebp pixels before timing. Timed work includes decoding and output allocation.
Input reading, oracle checks and byte preparation are outside timing.

Commands executed for both revisions:

```sh
export RUSTFLAGS='' ZENWEBP_BENCH_FIXTURES=gallery
cargo bench --locked --bench predictor_coalesce --features _benchmarks -- \
  --format=llm --cpu-policy=unrestricted --rounds=30
cargo bench --locked --bench decode_zenbench -- \
  --format=llm --cpu-policy=unrestricted --rounds=30
cargo build --locked --release --example decode_parity_dump
nm -S -C --defined-only target/release/examples/decode_parity_dump
objdump -d -C target/release/examples/decode_parity_dump
size -A target/release/examples/decode_parity_dump
RUSTFLAGS='-C target-feature=+simd128' cargo build --locked --release --lib \
  --target wasm32-unknown-unknown
RUSTFLAGS='-C target-feature=+simd128' cargo build --locked --release \
  --example decode_parity_dump --target wasm32-wasip1
wasm-tools objdump target/wasm32-wasip1/release/examples/decode_parity_dump.wasm
wasm-tools strip --all INPUT.wasm -o stripped.wasm
wc -c stripped.wasm
gzip -n -9 -c stripped.wasm | wc -c
```

The unknown-unknown library build is a compilation gate. Linked Wasm size uses
the existing WASI `decode_parity_dump` consumer, since an rlib is not a linked
module. Function sizes are body bytes from the code section, associated through
the function-name subsection; the code-section payload is the denominator.

Rsync preserves timestamps. The affected source files were touched before each
final build to invalidate Cargo's timestamp cache. An initial after-run reused
old example/library artifacts; that run was marked invalid and excluded. An
edge-only candidate increased raw code by 765 bytes despite saving gzip bytes;
the column sharing and UV load/store changes address that result.

## Measurements

| Artifact / count | Before | After | Delta |
|---|---:|---:|---:|
| Native SIMD shim definitions | 5 | 0 | −5 |
| Native SIMD shim call sites | 7 | 0 | −7 |
| Native ELF `.text`, bytes | 774,018 | 775,602 | +1,584 (+0.20%) |
| Native ELF file, bytes | 1,126,680 | 1,126,480 | −200 |
| Wasm code-section payload, bytes | 498,778 | 465,640 | −33,138 (−6.64%) |
| Wasm without custom sections, bytes | 553,825 | 521,018 | −32,807 (−5.92%) |
| Stripped Wasm gzip, bytes | 197,610 | 186,771 | −10,839 (−5.49%) |

The measured savings are smaller than the initial 35–48 KB estimate for the
larger application consumers. No extrapolation to those applications is made.
The final Wasm predictor dispatcher remains 3,799 bytes, matching baseline;
forcing portable predictor inlining on Wasm had unnecessarily duplicated it.
The live macroblock-row body is 31,597 → 32,428 bytes because it absorbs the
smaller filter dispatcher. Normal vertical UV edge/inner kernels are now
1,849 / 1,575 bytes, down from 6,031 / 5,721 in the column-only candidate.

All seven linked native loop-filter function instruction sequences match the
baseline after normalizing relocated addresses and padding. The SIMD row-filter
body remains 15,134 native bytes. The ELF text increase comes with the predictor
specializations, while five tiny vector-operation boundaries disappear.

Native ELF SHA-256:

- Before: `7ed84e953347a3ea0db93cac60b9a3618f7003d58fed7190c3bab0021e0d8f34`
- After: `6b51184e100284a8a82ab89409268dacf3fc92fae3f28f55ff52a763ae767568`

Named Wasm SHA-256:

- Before: `a7adb9db8e66b591a2b3cf4a03bbb15d2d716024680603566ca71f50e55fe735`
- After: `aa9d2112bf598a2db6e7ada71346063c89cdad0eff6959755cd26a6314c137cf`

The resource wrapper reported peak RSS of 0.87 GiB for the initial native
benchmark build and 0.59 GiB for each before/final measurement sequence.
These are build/run receipts, not a decoder-memory comparison.

A fixed **three-block ABBA** sequence ran retained before/after benchmark
executables on i134 CPU 2: six fresh processes per revision per bench, each
with 30 rounds per fixture. The statistic below is the median of six process
medians, including every raw sample. No process or sample was discarded.
The wrapper reported 106 seconds and 0.02 GiB peak RSS for this sequence.

| Fixture | Dimensions | Before, Mpixel/s | After, Mpixel/s | Throughput delta |
|---|---:|---:|---:|---:|
| VP8 1 | 550×368 | 109.91 | 109.97 | +0.06% |
| VP8 2 | 550×404 | 71.61 | 71.65 | +0.06% |
| VP8 3 | 1280×720 | 83.77 | 83.81 | +0.05% |
| VP8 4 | 1024×772 | 83.35 | 83.18 | −0.20% |
| VP8 5 | 1024×752 | 134.58 | 134.71 | +0.10% |
| VP8L 1 | 400×301 | 123.62 | 130.77 | +5.78% |
| VP8L 2 | 386×395 | 182.78 | 191.14 | +4.57% |
| VP8L 3 | 800×600 | 155.93 | 165.40 | +6.08% |
| VP8L 4 | 421×163 | 160.07 | 176.62 | +10.34% |
| VP8L 5 | 300×300 | 74.91 | 79.90 | +6.66% |

VP8 is unchanged within process variation; VP8L throughput improves on all five
fixtures. The −0.20% VP8-4 point is disclosed rather than rounded to zero. A
paired bootstrap of the six log time ratios (20,000 resamples, seed 20261123)
gives VP8-4 a 95% time-change interval of −2.20% to +0.24%; none of the five VP8
intervals establishes a slowdown. The largest VP8 upper bound is +0.62%.
This establishes no resolved native regression at this sampling precision,
not literal zero timing difference on every possible input.

Zenbench records `cpu_policy=unrestricted` because the shared runner enforces
thread affinity rather than a delegated cpuset. Every group's recorded thread
set was `[2]`. These are disclosed exploratory measurements, not a promoted
zenbench confirmation baseline. Wasm throughput was not benchmarked.

The repeat command, within the same shared runner/resource wrapper, was:

```sh
export ZENWEBP_BENCH_FIXTURES=gallery
for bench in decode_zenbench predictor_coalesce; do
  for block in 1 2 3; do
    for phase in before after after before; do
      "$retained/$phase/$bench" --format=llm \
        --cpu-policy=unrestricted --rounds=30
    done
  done
done
```

## Correctness and scope of validation

On i265 with Rust 1.99.1, native `cargo test --release --all-targets` passed **836 tests**, with zero
failures and 11 existing ignored tests; no skip messages. The complete gate also
executed its harness-free benchmark targets. It required
`ZENWEBP_PERF_FIXTURES=$PWD/tests`, `ZENWEBP_BENCH_FIXTURES=gallery` and
`ZENBENCH_CPU_POLICY=unrestricted` on the asymmetric local host. The successful
full native gate preceded the final Wasm-only load/store refinements; x86
predictor code was unchanged by those refinements.

Strict `cargo clippy --all-targets -- -D warnings` and
`cargo fmt --check -p zenwebp` passed. `cargo test --doc` passed 33 tests, with
three existing ignored examples. Native golden codec hashes and independent
libwebp gates retain their exact assertions. No golden files were re-blessed.

Final source passed Wasmtime library tests with `--release --lib
--target wasm32-wasip1 --no-default-features`: **320 passed / 3 existing ignored**
with `RUSTFLAGS='-C target-feature=+simd128'`, and **313 passed / 3 existing
ignored** with `-simd128`. The SIMD run includes the loop-filter differential
test; both configurations include the golden codec hashes.

The bounded native decode-parity sweep covered 17 committed still/animation
files and matched **402/402** independent libwebp comparisons across its output
formats and options. The final i134 Wasm consumer also matched **402/402** against the same
oracle digests under Wasmtime. Existing target-specific dead-code warnings on Wasm are outside
the native strict-lint gate and were not suppressed.
