# Encoder speed-parity program, session 1 (2026-07-16)

Directive: hold exactness constant (every output byte, both cost models),
close the wall-clock gap vs libwebp. Six chunks landed (`ec577bba`,
`4868ce95`, `7caf05a4`, `74ca25f2`, `af19d414`+`90bb08bf`, `fdb928ab`);
every change gated by
`dev/output_hash.rs` (grid hash over 21 images × both cost models ×
m0-6 × q{5,50,75,90} + alpha/sharp/lossless sections — byte-identical
before/after each slice), the full 52-binary test suite, clippy ×3, fmt.

## Result (792079.png 512×512 q75, zenbench interleaved, quiet box)

zen/libwebp wall ratios, session start → end:

| method | default preset | diagnostic (sns0/flt0/segs1) |
|---|---|---|
| m0 | 2.41x → **1.88x** | 1.92x |
| m2 | 1.24x → **1.00x (parity)** | **0.91x — 9% faster than libwebp** |
| m4 | 1.47x → **1.25x** | **1.13x** |
| m6 | 1.57x → **1.53x** | 1.50x |

Instructions (callgrind, default preset): m0 90.6M → 75.1M (lib 41.1M),
m4 213.3M → **183.5M** (lib 170.7M = **1.075x**), m6 408.2M → 392.5M
(lib 250.8M). m4 wall (~1.25x) now exceeds its instruction ratio by ~17%
— the I1-footprint/branch wall is the binding constraint there.

Context for the m0 gap: zen's m0 deliberately does more (hint analysis +
RD UV picks) and produces −7..−10% bytes vs libwebp m0 — that tier trades
time for size by design; the wall gap shrinks only by making the same
decisions cheaper. The old CLAUDE.md table (1.76x/1.30x/1.36x/1.41x,
2026-07-14) predates the #38-D adoptions that bought bytes with time; the
2.41x baseline here is the same code the adoptions shipped.

## What landed

1. **Fused final pass** (`ec577bba`): the final coding pass quantized /
   dequantized / IDCT'd with per-coefficient scalar loops + staging
   arrays, then the token path re-quantized the same DCT input a second
   time. Now: the fused SIMD primitives the RD path already used, and the
   reconstruction pass carries its zigzag levels
   (`LumaBlockResult::{y1_zigzag,y2_zigzag}`, `ChromaBlockResult::zigzag`)
   straight into the recorder — `quantize_mb_coeffs` is pure assembly.
2. **Border hoisting + one-region cost walkers** (`4868ce95`): bordered
   prediction workspaces built once per MB and shared across mode loops
   (`predict_luma_16x16_into` / `predict_chroma_8x8_into`);
   `get_cost_luma16`/`get_cost_uv` walk all blocks inside one arcane
   region instead of dispatching per residual-cost call.
3. **I16-RD batching + code-footprint work** (`7caf05a4`): one-region
   fused quantize+dequantize of all 16 I16-AC blocks; whole-16×16
   residual+DCT in one region; the `#[rite]` residual-cost kernel de-duplicated
   to one shared `#[arcane]` copy (it was inlined 4+ times ≈ I1 pressure:
   cachegrind 872k I1 misses vs libwebp 121k); m6-only trellis arms moved
   out-of-line; trellis micro (de-Optioned cost pointers, hoisted biases
   + psy gate).
4. **Trellis table hoist** (`74ca25f2`).
5. **I4 winner carry** (`af19d414` + i686 fix `90bb08bf`): the RD pick's
   fully-reconstructed I4 winner (recon + zigzag levels) feeds the final
   pass directly on the non-trellis tiers; SIMD-eval paths only (the
   scalar fallback disables it — see the process note below).
6. **UV winner carry** (`fdb928ab`): same for chroma, with the diffusion
   contract encoded in the gate (carriable ⟺ RD diffusion behaviour ==
   final-pass behaviour; tuned m0-m2 and parity-m5 excluded); the final
   pass replays only the diffusion error STORE.

## Where the remaining gap lives (measured, for the next session)

- **m6 (1.5x): `trellis_quantize_block` is ~190M of 395M** vs libwebp's
  ~104M inlined equivalent. Call counts are fine (zen 130k I4-candidate
  trellis calls vs libwebp 250k `ReconstructIntra4` — the presort +
  max_modes + early-bail already halve the candidates); the per-call cost
  is ~2x. Root causes: **i32 coefficient arrays vs libwebp's `int16_t`**
  (double data traffic through the DP, double-width output/input clears)
  and codegen density. The structural fix is an i16 coefficient
  migration through the coeff pipeline — large but mechanical.
- **Winner recompute**: I4 and UV winners now carry (chunks 5-6). Still
  open: the I16 winner (non-trellis levels are context-free ⇒ carryable
  at m2-m4; small on photo content, bigger on I16-heavy screenshots),
  and m5/m6 luma recomputes stay by design (trellis context: m5 RD
  quantizes simple while the final trellises; m6 tuned seeds all-false
  in RD vs real ctx in the final pass).
- **I1 cache (wall vs instruction gap)**: zen's per-MB loop walks ~54KB
  of code (872k I1 misses vs 121k). De-duplication of rite kernels helped
  little; the executed footprint itself is the issue — i16 migration and
  less monomorphized inlining are the levers that shrink it.
- **memcpy ~3.2M at m4**: mostly `LumaBlockResult` (~2.5KB) returned by
  value per MB + remaining prediction staging; an out-param refactor of
  `transform_luma_block` would remove the big one.

## Process note (added after the i686 CI break)

The `output_hash` byte gate only covers the platform it runs on. The
speed work's cfg'd arch arms (x86_64 / wasm32 / scalar-fallback) broke
i686 twice: a missing `quantize_dequantize_ac_only_simd` definition
(`5c37ee33`) and the I4 winner carry staying enabled with empty levels on
the non-SIMD arms (`90bb08bf` — corrupted every I4 MB on 32-bit). Before
pushing changes that touch arch-gated code paths, run the FULL suite on
`--target i686-unknown-linux-gnu` locally; it exercises the scalar tiers
the x86-64 gate never sees.

## Repro

```bash
# byte-invariance gate (run before + after, diff):
cargo run --release --features __expert --example output_hash

# wall: cargo bench --bench encode_vs_libwebp -- method_default
# instructions:
cargo build --release --features _profiling --example callgrind_encode
valgrind --tool=callgrind target/release/examples/callgrind_encode \
    ~/tmp/792079_512x512.rgb 512 512 75 4 default
```
