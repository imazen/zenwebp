# PR105 focused performance evidence

Measured on Apple M4 Pro (aarch64 macOS), 2026-10-18.

No regression met zenbench’s configured significance criterion in any of the
three interleaved passes. Median time changes ranged from -2.49% to +3.37%.
Short decode samples have substantial host variation; small effects remain
unresolved. These cases do not establish performance on other CPUs, metadata
loads, alpha-heavy inputs, effort settings or large images.

| Operation | Main ms | PR ms | Time change |
|---|---:|---:|---:|
| native_anim_encode_lossless | 126.839 | 127.050 | +0.17% |
| shared_anim_encode_lossless | 127.328 | 127.912 | +0.46% |
| native_anim_decode_lossless | 0.294 | 0.287 | -2.37% |
| shared_anim_decode_lossless | 0.325 | 0.320 | -1.48% |
| native_anim_encode_lossy | 14.183 | 13.912 | -1.91% |
| shared_anim_encode_lossy | 14.263 | 13.908 | -2.49% |
| native_anim_decode_lossy | 1.483 | 1.533 | +3.37% |
| shared_anim_decode_lossy | 1.668 | 1.683 | +0.90% |
| shared_still_decode_lossy | 2.720 | 2.725 | +0.20% |

Each encode operation processes three 256x256 photographic crops at lossy q75/m4
or lossless m4. Decode operations consume a complete three-frame 64x63 lossless
animation, four-frame 99x87 lossy animation, or 550x368 lossy still. Input hashes,
exact source revisions and resource settings are in [provenance](provenance_2026-10-18.json).
Fixture origins are recorded in [tests/CREDITS.md](../../tests/CREDITS.md).

Both source revisions link into one process and share the PR’s pinned dependencies.
The main control’s package version gets a suffix solely to disambiguate Cargo identities.
This isolates source changes. Separate-process runs using each revision’s own graph
showed wide host drift, so they do not support attributing small differences to the
dependency changes. Their full logs and preserved binaries remain local.

The table uses medians of three per-pass means. [Per-pass comparisons](passes_2026-10-18.csv)
retain the paired confidence intervals and significance decisions. zenbench 0.1.8’s
aggregate result retains its first-pass analysis record while replacing summary
statistics; those cached analyses were not treated as inference for the three-pass table.

```text
run-heavy: done rc=0 233s | peak-RSS 0.04GiB | min-avail 18616MiB | peak-load 3.25
```

Set `RUN_HEAVY` to the resource wrapper used for your machine. Reproduce from
the repository root, then execute outside ancestor Cargo overrides:

```sh
python3 tools/animation-perf/prepare.py a3fec3fec04cad9322af0a9e6bed023e72f1929c
repo=$(pwd)
cd "$HOME/tmp"
export TMPDIR="$HOME/tmp" CARGO_INCREMENTAL=0 RUSTC_WRAPPER=
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_TARGET_DIR="$HOME/tmp/webp-pr105-perf-target"
export ZENWEBP_PERF_FIXTURES="$repo/tests"
"$RUN_HEAVY" --mem 12G --jobs 4 -- cargo build --locked --release --manifest-path "$repo/tools/animation-perf/Cargo.toml"
"$RUN_HEAVY" --mem 12G --jobs 4 -- "$CARGO_TARGET_DIR/release/zenwebp-animation-perf" --cpu-policy=unrestricted --format=json --median-of-passes=3
```

Keep complete stdout/stderr logs when reproducing. The original private logs are
`webp-perf-main-measured.log`, `webp-perf-pr-measured.log` and
`webp-perf-interleaved-measured.log`; public compact result records are linked above.
