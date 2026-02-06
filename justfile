# zenwebp justfile

# Default test image path (Kodak test suite)
test_image := env_var_or_default("TEST_IMAGE", "~/work/codec-corpus/kodak/1.png")
test_webp := env_var_or_default("TEST_WEBP", "/tmp/test.webp")

# Build and test natively
test:
    cargo test --lib

# Run clippy
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Format code
fmt:
    cargo fmt

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

# Run all quality checks
check: fmt clippy test
    @echo "All checks passed!"
