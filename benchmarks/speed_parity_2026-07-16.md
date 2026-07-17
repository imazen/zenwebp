# Encoder speed-parity program, sessions 1-2 (2026-07-16)

Directive: hold exactness constant (every output byte, both cost models),
close the wall-clock gap vs libwebp. Session 1 landed six chunks
(`ec577bba`, `4868ce95`, `7caf05a4`, `74ca25f2`, `af19d414`+`90bb08bf`,
`fdb928ab`); session 2 landed the i16 port + m4 code-weight chunks
(`dd6236ec`, `dd352c55`, `7c33e0ae`, `f62983ab`, `479681c7`). Every
change gated by `dev/output_hash.rs` (grid hash over 21 images × both
cost models × m0-6 × q{5,50,75,90} + alpha/sharp/lossless sections —
byte-identical before/after each slice, COMBINED 958f376a6c8b118f), the
full 52-binary x86-64 suite, the full i686 suite, wasm32 build, clippy
×3, fmt.

## Result (792079.png 512×512 q75, zenbench interleaved)

zen/libwebp wall ratios, program start → session-2 end (box under
load-avg ~5-7 for the session-2 numbers; ratios are interleaved A/B):

| method | default preset | diagnostic (sns0/flt0/segs1) |
|---|---|---|
| m0 | 2.41x → **1.79x** | 1.84x |
| m2 | 1.24x → **0.97x — beats libwebp** | **0.91x** |
| m4 | 1.47x → **1.16x** | **1.015x — wall parity** |
| m6 | 1.57x → **1.40x** | 1.39x |

Instructions (callgrind, default preset, session-2 end): m0 90.6M →
**72.5M** (lib 41.1M), m2 **58.2M**, m4 213.3M → **171.2M** (lib 170.7M
= **1.003x — instruction parity**), m6 408.2M → **374.0M** (lib 250.8M
= 1.49x, all in the trellis DP — see below).

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

- **m6 (1.4x): `trellis_quantize_block` is ~191M of 374M** (51%).
  Measured precisely in session 2: libwebp's trellis lives INLINED in
  `ReconstructIntra4` (75.6M self over 125,100 calls ≈ 560/call
  including quantize glue); zen's is 1468/call over 130,330 calls —
  ~2.6x per call. The i16 in/out conversion + caller de-shimming moved
  m6 only 395→374M: the DP body itself is the cost. Static code size is
  near-identical (602 vs 569 instructions) and the algorithm is a
  line-faithful port; the excess is per-line codegen density (bounds
  checks, `Ord::min` sequences ≈ 320/call in cmp.rs, table addressing).
  A bounds-proof pass (`& 1`/`.min(15)` masks) was tried and REVERTED:
  +3M — the mask ops cost more than the checks they eliminate. Next
  lever if resumed: assembly-diff the DP loop body against libwebp's
  and chase specific codegen (e.g. the `&level_cost_t[band][ctx]`
  addressing chain, 8.2M).
- **Winner recompute: DONE.** I4 + UV (session 1) + I16 (session 2,
  `479681c7`) all carry. m5/m6 luma recomputes stay by design (trellis
  context: m5 RD quantizes simple while the final trellises; m6 tuned
  seeds all-false in RD vs real ctx in the final pass).
- **I1 cache**: ~848k I1 misses at m4 (cachegrind) vs libwebp 121k;
  biggest per-fn miss counts are the per-MB orchestrators evicted
  between MBs (choose_macroblock_info 144k, pick_best_uv 126k,
  encode_macroblock 119k). Wall parity at m4 was reached anyway — real
  hardware prefetch tolerates this pattern better than cachegrind's
  model suggests. Remaining lever: shrink the walked span per MB
  (outline runtime-dead fallback arms).
- **memcpy ~3.2M at m4**: mostly `LumaBlockResult` (~2.5KB) returned by
  value per MB + remaining prediction staging; an out-param refactor of
  `transform_luma_block` would remove the big one.

## i16 port state — COMPLETE (session 2)

Stage A (`dd6236ec`): quantized LEVELS i16 end-to-end (Residual/cost
kernels — the SSE2 pack prelude gone — recorder, trellis OUT, every
carrier struct).

Stage B (`7c33e0ae`): the block arrays + kernel family. DCT carriers
(luma_blocks/u_blocks/v_blocks/LumaBlockResult::coeffs/y1_quant/
y1_dequant/uv_quant/uv_dequant) are [i16;N]; `ftransform_from_u8_4x4*`
return [i16;16] (the sign-extend tails deleted); the fused
`quantize_dequantize_*` family is i16 in/out on every tier (the SSE2/
NEON/WASM kernels were already 16-bit inside — the input packs and
output sign-extend unpacks are deleted, −12 instructions per SSE2
call); `idct_add_residue_i16` + `idct4x4_i16` + `add_residue_i16`
encoder variants (decoder keeps its i32 `_inner` paths);
`sse4x4_with_residual` takes i16 residuals; chroma DC diffusion runs on
i16 blocks. Dead code deleted alongside: the four unused legacy record
paths (~690 lines), `Plane`, the DCT token constants, the i32
`add_residue`/`idct_add_residue_inplace` dispatch families.

Stage B trellis (`f62983ab`): `trellis_quantize_block` is i16 in/out
like libwebp's `int16_t in[16]/out[16]` (DP stays i32 in locals); every
caller-side widen shim deleted; the trellis arms reuse the dequantized
values trellis writes back instead of re-deriving them.

m4 code-weight audit (`dd352c55` bounds-strip + `479681c7`):
analysis-prediction bounds checks stripped via fixed windows
(pred_chroma8_tm 200→4 panic sites); `Segment::{y1,y2,uv}_matrix`
de-Optioned (11 hot-loop unwrap paths gone); I16 winner carry ends the
final-pass 16×16 recompute at m3/m4. `encode_image` retains ~31
bounds-check sites but they are outside the per-MB loop (header/emit/
finalize — not hot).

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
