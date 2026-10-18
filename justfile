# zenwebp justfile

# Default test image path (Kodak test suite)
test_image := env_var_or_default("TEST_IMAGE", "~/work/codec-corpus/kodak/1.png")
test_webp := env_var_or_default("TEST_WEBP", "/tmp/test.webp")

# Build and test natively
test:
    cargo test --lib

# PR105 integration: both timing contracts plus retained whole-file behavior.
animation-contract-check:
    cargo test --locked --test animation_timing --test animation_decode_contract --test sansio --test streaming_decoder

animation-contract-lint:
    cargo clippy --locked --lib --tests -- -D warnings

# Run clippy
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Format code + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the self-contained apidoc/ package, so it is
# never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt
    cargo test --manifest-path apidoc/Cargo.toml

# Build for WASM target with SIMD128
wasm-build:
    RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasip1 --features _wasm_profiling --bin profile_wasm

# Run WASM benchmark with wasmtime (requires wasmtime installed)
# Usage: just wasm-bench path/to/image.webp [iterations]
wasm-bench webp_path iterations="100":
    wasmtime run --dir=. --dir=/home --dir=/tmp target/wasm32-wasip1/release/profile_wasm.wasm {{webp_path}} {{iterations}}

# Run native benchmark for comparison
# Usage: just native-bench path/to/image.webp [iterations]
native-bench webp_path iterations="100":
    cargo run --release --bin profile_wasm --features _wasm_profiling -- {{webp_path}} {{iterations}}

# Compare WASM vs native performance
# Usage: just bench-compare path/to/image.webp [iterations]
bench-compare webp_path iterations="100":
    @echo "=== Native Performance ==="
    cargo run --release --bin profile_wasm --features _wasm_profiling -- {{webp_path}} {{iterations}}
    @echo ""
    @echo "=== WASM Performance ==="
    RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasip1 --features _wasm_profiling --bin profile_wasm
    wasmtime run --dir=. --dir=/home --dir=/tmp target/wasm32-wasip1/release/profile_wasm.wasm {{webp_path}} {{iterations}}

# Run profiling benchmarks
profile-decode webp_path="" iterations="100":
    cargo run --release --bin profile_decode --features _profiling -- {{webp_path}} {{iterations}}

profile-encode png_path="" iterations="10":
    cargo run --release --bin profile_encode --features _profiling -- {{png_path}} {{iterations}}

# Run WASM lib tests via wasmtime
wasm-test:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime" RUSTFLAGS="-C target-feature=+simd128" cargo test --target wasm32-wasip1 --features simd --lib

# Check that WASM target compiles (without running)
wasm-check:
    RUSTFLAGS="-C target-feature=+simd128" cargo check --target wasm32-wasip1 --features simd

# Build without simd feature (scalar fallback)
build-scalar:
    cargo build --release --no-default-features --features std

# Build for no_std
build-no-std:
    cargo build --no-default-features

# Run I4 diagnostic harness (compares zenwebp vs libwebp bitstream internals)
diag:
    cargo test --release --features _corpus_tests --test i4_diagnostic_harness -- --nocapture

# Cross-test i686 (32-bit x86) via QEMU
test-i686:
    cross test --features "std,simd,pixel-types" --target i686-unknown-linux-gnu

# Cross-test armv7 (32-bit ARM) via QEMU
test-armv7:
    cross test --features "std,simd,pixel-types" --target armv7-unknown-linux-gnueabihf

# Cross-test all non-native targets
test-cross: test-i686 test-armv7
    @echo "All cross tests passed!"

# Check all locally-resolvable feature permutations
feature-check:
    cargo test --no-default-features
    cargo test --features "std,simd,pixel-types"
    cargo test --features imgref
    cargo test --features mode_debug
    cargo check --features avx512

# Run all quality checks
check: fmt clippy test
    @echo "All checks passed!"

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml
# Reproduce the ARM kernel audit on macOS, retaining full output in ~/tmp.
arm-kernel-audit-macos group="":
    #!/usr/bin/env bash
    set -euo pipefail
    test "$(uname -s)" = Darwin
    mkdir -p "$HOME/tmp"
    audit_log="$HOME/tmp/zenwebp-arm-kernels-$(date -u +%Y%m%dT%H%M%SZ).log"
    group_args=()
    if [[ -n '{{group}}' ]]; then group_args+=("--group={{group}}"); fi
    TMPDIR="$HOME/tmp" CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 \
      nice -n 19 /usr/bin/time -l cargo bench --features _dev --bench kernel_tiers -- "${group_args[@]}" --cpu-policy=unrestricted --format=llm \
      2>&1 | tee "$audit_log"

# Wide-corpus decoder parity vs libwebp (benchmarks/decode_parity_*.md).
# corpus = a codec-corpus checkout; writes ~/tmp/decode-parity/{run.tsv,gen,mismatch}.
decode-parity corpus="~/tmp/codec-corpus-sparse":
    #!/usr/bin/env bash
    set -euo pipefail
    out="$HOME/tmp/decode-parity"; mkdir -p "$out"; rm -rf "$out/gen" "$out/mismatch"
    cargo build --release --features __expert --example decode_parity_sweep
    nice -n 19 ./target/release/examples/decode_parity_sweep --files {{corpus}} --files tests/images \
      --gen-src {{corpus}} --encodes-per-variant 2 --out "$out/run.tsv" --mismatch-dir "$out/mismatch" \
      --save-gen "$out/gen" --hash-out "$out/gen/lib.hashes" 2>&1 | tee "$out/run.log"

# Re-check the `decode-parity` corpus on wasm32-wasip1 (scalar + simd128) under wasmtime.
decode-parity-wasm:
    #!/usr/bin/env bash
    set -euo pipefail
    out="$HOME/tmp/decode-parity"
    RUSTFLAGS="-C target-feature=-simd128" CARGO_TARGET_DIR=target/wasm-scalar \
      cargo build --release --target wasm32-wasip1 --example decode_parity_dump
    RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_DIR=target/wasm-simd128 \
      cargo build --release --target wasm32-wasip1 --example decode_parity_dump
    for v in scalar simd128; do
      dev/decode_parity_wasm.sh target/wasm-$v/wasm32-wasip1/release/examples/decode_parity_dump.wasm \
        "$out/gen" "$out/wasm-$v" 14 | tee "$out/wasm-$v.summary"
    done

# Structural metadata tests, default-off contract, and compiler checks.
metadata-check:
    cargo test --features inventory --test inventory --test inventory_both_directions --test inventory_feature --test fuzz_regression
    cargo clippy --features inventory --lib --tests -- -D warnings
    cargo check --no-default-features --features inventory
    cargo test --test inventory_feature

inventory-oracle:
    INVENTORY_ORACLE_EXIFTOOL="$(command -v exiftool)" cargo test --features inventory --test inventory -- --ignored --nocapture

metadata-portability:
    CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_RUNNER=env cargo test --features inventory --target i686-unknown-linux-gnu --test inventory --test inventory_feature --test fuzz_regression
    cargo check --manifest-path fuzz/Cargo.toml --bin inventory
    just api-doc

metadata-api-check:
    #!/usr/bin/env bash
    set -euo pipefail
    baseline=$(cargo read zenwebp==0.4.4 --path-only)
    RUSTC_BOOTSTRAP=1 cargo rustdoc --lib --features inventory -- -Zunstable-options --output-format json
    RUSTC_BOOTSTRAP=1 cargo rustdoc --manifest-path "$baseline/Cargo.toml" --target-dir "$PWD/target/published-api/0.4.4" --lib -- -Zunstable-options --output-format json
    cargo semver-checks check-release --current-rustdoc target/doc/zenwebp.json --baseline-rustdoc target/published-api/0.4.4/doc/zenwebp.json --release-type patch

inventory-mutation:
    cargo test --features inventory --test inventory_both_directions -- --ignored --nocapture

metadata-input-limit-check:
    cargo test --features inventory --test inventory job_ -- --nocapture
    cargo clippy --features inventory --lib --test inventory -- -D warnings

# Populate and validate the same corpus mounted into cross by CI.
metadata-corpus-check:
    cargo test --features inventory --test inventory corpus_webp_conformance_valid -- --exact

animation-timing-check:
    cargo test -p zenwebp --test animation_timing
    cargo clippy -p zenwebp --lib --test animation_timing -- -D warnings

animation-timing-oracle output:
    cargo run -p zenwebp --example animation_timing_oracle -- "{{output}}"
    python3 tests/animation_timing_oracle.py "{{output}}"

animation-integration-check: animation-timing-check
    cargo test -p zenwebp --lib codec::
    cargo test -p zenwebp --features inventory --test inventory

# The nested workspace resolves patches independently of the parent package.
recompress-contract-check:
    cargo test --locked --manifest-path zenwebp-recompress/Cargo.toml --workspace
    cargo test --locked --manifest-path zenwebp-recompress/Cargo.toml --all-features
    cargo clippy --locked --manifest-path zenwebp-recompress/Cargo.toml --all-targets --all-features -- -D warnings
    cargo fmt --manifest-path zenwebp-recompress/Cargo.toml -- --check

# Each standalone workspace must resolve the same unpublished interface source.
sync-wholefile-locks:
    cargo update -p zencodec
    cargo update --manifest-path fuzz/Cargo.toml -p zencodec
    cargo update --manifest-path zenwebp-recompress/Cargo.toml -p zencodec

wholefile-lock-check:
    cargo check --locked --manifest-path fuzz/Cargo.toml --bins
    just recompress-contract-check
    just wholefile-lock-sources

wholefile-lock-sources:
    #!/usr/bin/env python3
    from pathlib import Path
    import tomllib
    root = tomllib.loads(Path("Cargo.toml").read_text())
    revision = root["patch"]["crates-io"]["zencodec"]["rev"]
    shared = {"zencodec", "zencodec-io", "zencodec-testkit", "zenmeta"}
    for path in ["Cargo.lock", "fuzz/Cargo.lock", "zenwebp-recompress/Cargo.lock"]:
        packages = tomllib.loads(Path(path).read_text())["package"]
        assert sum(p["name"] == "zencodec" for p in packages) == 1, path
        for package in packages:
            if package["name"] in shared:
                assert f"rev={revision}#{revision}" in package.get("source", ""), (path, package)
        print(f"{path}: one zencodec source at {revision}")

# New shared contracts, with inventory both enabled and absent.
shared-contract-check:
    cargo test -p zenwebp --features inventory --test shared_contracts
    cargo test -p zenwebp --test shared_contracts
    cargo clippy -p zenwebp --features inventory --test shared_contracts -- -D warnings

# Whole-file incremental fixture parity and resource admission.
wholefile-admission-check:
    cargo test --locked --test sansio known_length_limit_is_independent_of_the_offered_offset -- --nocapture

sansio-check:
    cargo test --locked --test sansio -- --nocapture

# Complete post-landing whole-file decoder validation.
wholefile-gate:
    cargo test --locked
    cargo test --locked --features inventory --test inventory --test inventory_feature --test fuzz_regression --test shared_contracts
    cargo clippy --locked --all-targets -- -D warnings
    cargo fmt -p zenwebp -- --check
    cargo test --locked --doc
    cargo doc --locked --no-deps
    just wholefile-api-check

# Supply rustdoc JSON directly because Cargo's generated dependency wrapper
# does not inherit the root patches for unpublished dependencies.
wholefile-api-check:
    #!/usr/bin/env bash
    set -euo pipefail
    target_dir=$(python3 -c 'import os,pathlib; print(pathlib.Path(os.environ.get("CARGO_TARGET_DIR", "target")).resolve())')
    baseline=$(cargo read zenwebp --path-only)
    features() {
      python3 -c 'import sys,tomllib; f=tomllib.load(open(sys.argv[1],"rb"))["features"]; print(",".join(k for k in f if k not in ("unstable","nightly","bench","no_std") and not k.startswith(("_","unstable_","unstable-"))))' "$1"
    }
    RUSTC_BOOTSTRAP=1 cargo rustdoc --locked --lib --features "$(features Cargo.toml)" -- -Zunstable-options --output-format json
    RUSTC_BOOTSTRAP=1 cargo rustdoc --manifest-path "$baseline/Cargo.toml" --locked --lib --features "$(features "$baseline/Cargo.toml")" --target-dir "$target_dir/published-api" -- -Zunstable-options --output-format json
    cargo semver-checks check-release --current-rustdoc "$target_dir/doc/zenwebp.json" --baseline-rustdoc "$target_dir/published-api/doc/zenwebp.json"
