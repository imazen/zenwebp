# zenwebp-libwebp-shim — design notes

## What this is

A prototype Rust crate that produces a `libwebp.{so,dll,dylib}` exposing the
libwebp 1.6 C ABI subset that **libwebp-net actually calls** — backed by
zenwebp under the hood, with zero changes required on the .NET side.

This crate is exploratory. Not published, not stable. Use it to test the
idea; don't ship it as-is.

## Why the subset, not the whole libwebp ABI

The libwebp-net P/Invoke surface declares ~80 functions, but a focused audit
of the managed wrapper code (`src/Imazen.WebP/*.cs`) found that **only 33
distinct C entry points are actually invoked** from production code, plus
**2 callbacks** and **6 ref-passed structs**. The other ~48 declarations are
dead surface that the wrapper never reaches. We can ignore them.

Required surface, grouped:

| Category               | Functions                                                                                      | Status |
|------------------------|------------------------------------------------------------------------------------------------|--------|
| Version                | `WebPGetEncoderVersion`, `WebPGetDecoderVersion`                                               | WIRED  |
| Probe                  | `WebPGetInfo`, `WebPGetFeaturesInternal`                                                       | WIRED  |
| Simple lossy encode    | `WebPEncode{RGB,BGR,RGBA,BGRA}`                                                                | WIRED  |
| Simple lossless encode | `WebPEncodeLossless{RGB,BGR,RGBA,BGRA}`                                                        | WIRED  |
| Decode into buffer     | `WebPDecode{RGB,BGR,RGBA,BGRA}Into`                                                            | WIRED  |
| Memory                 | `WebPMalloc`, `WebPFree`                                                                       | WIRED* |
| Advanced encode        | `WebPConfigInitInternal`, `WebPConfigLosslessPreset`, `WebPValidateConfig`,                    | STUB   |
|                        | `WebPPictureInitInternal`, `WebPPictureImport{RGB,BGR,RGBA,BGRA}`, `WebPEncode`,               |        |
|                        | `WebPPictureFree`                                                                              |        |
| Animation encode       | `WebPAnimEncoderOptionsInitInternal`, `WebPAnimEncoderNewInternal`, `WebPAnimEncoderAdd`,      | STUB   |
|                        | `WebPAnimEncoderAssemble`, `WebPAnimEncoderGetError`, `WebPAnimEncoderDelete`                  |        |
| Animation decode       | `WebPAnimDecoderOptionsInitInternal`, `WebPAnimDecoderNewInternal`, `WebPAnimDecoderGetInfo`,  | STUB   |
|                        | `WebPAnimDecoderGetNext`, `WebPAnimDecoderHasMoreFrames`, `WebPAnimDecoderReset`,              |        |
|                        | `WebPAnimDecoderDelete`                                                                        |        |

*WebPFree currently relies on Box::from_raw, which leaks the slice length.
Real implementation needs to either track allocations in a side table or use
`libc::malloc`/`libc::free` so the lifetime crosses the FFI boundary cleanly.

## Three strategies considered

### Strategy A — Drop-in C ABI shim (THIS CRATE)

Ship `libwebp.{so,dll,dylib}` from a Rust crate. libwebp-net's `Extern/*.cs`
P/Invokes are unchanged; only the native binary in
`Imazen.WebP.NativeRuntime.<rid>/runtimes/<rid>/native/` swaps out. Could be
selected at install time via a new NuGet package
`Imazen.WebP.NativeRuntime.<rid>.Zen`.

- **Effort**: medium. Simple encode/decode is already done in this prototype
  (~250 lines, half a day). WebPPicture pipeline (~600 lines) and
  Animation (~400 lines) are the remaining work; estimate 1–2 weeks for full
  parity.
- **Risks**: WebPPicture/WebPConfig have ~50 fields each whose byte layout must
  match libwebp 1.6 *exactly* — any drift causes silent corruption on the .NET
  side. Mitigated by writing a struct-layout assertion test that compiles
  against libwebp headers via bindgen and `static_assertions::assert_eq_size!`.
- **Preserves**: 100% of the existing Imazen.WebP managed API surface,
  including Bitmap APIs, raw buffer APIs, animation APIs, advanced encoder
  config — no managed changes at all.

### Strategy B — Parallel managed-side backend

Add a new `Imazen.WebP.Extern.Zen` namespace with idiomatic-Rust FFI
signatures (no WebPPicture/WebPConfig dance — just `(bytes, width, height,
stride, quality) -> bytes`). Add a `WebPBackend.Active` static knob to switch
at runtime.

- **Effort**: small for simple path, large for animation/advanced encode (need
  parallel WebPEncoderConfig API or feature-flag the Bitmap-based advanced
  pipeline as libwebp-only).
- **Risks**: doubles the managed surface area; advanced encode users who rely
  on `WebPEncoderConfig.SetMethod(...)` etc. either get a stripped backend or
  the knobs get silently ignored on the zenwebp side.
- **Preserves**: all simple APIs; partial coverage for advanced/animation.

### Strategy C — Managed-layer abstraction

Introduce `IWebPEncoderBackend` / `IWebPDecoderBackend` interfaces; existing
libwebp code moves behind one implementation, zenwebp behind another.
Selection via DI or static config.

- **Effort**: large. Refactor every existing entry point. Touches every
  managed file under `src/Imazen.WebP/`.
- **Risks**: API churn for downstream consumers; tests need to run twice
  (once per backend); the existing public types (`WebPEncoderConfig`,
  `WebPPixelFormat`) leak libwebp concepts that don't map cleanly to a
  generic interface.
- **Preserves**: all APIs but at the cost of changing the *internals*
  significantly.

## Recommendation: Strategy A

Lowest .NET-side disruption, cleanest packaging story (just another NuGet
package), strictly additive: users opt in by installing the
`Imazen.WebP.NativeRuntime.<rid>.Zen` package instead of (or alongside) the
upstream `*.NativeRuntime.<rid>` package. Native loader's existing search
order picks up whichever `libwebp.{ext}` is closer in the resolution path.

Risk concentration is fully in the Rust crate's struct layouts; **make those
testable** as the first non-stub commit:

```rust
#[test]
fn webp_config_layout_matches_libwebp() {
    assert_eq!(size_of::<WebPConfig>(), 196); // from libwebp/encode.h
    assert_eq!(offset_of!(WebPConfig, lossless), 0);
    assert_eq!(offset_of!(WebPConfig, quality), 4);
    // ... 30+ field-offset assertions
}
```

Layout drift bugs caught at compile time > debugging mysterious .NET crashes
in production.

## Open questions for the user

1. **Feature scope** — is the simple encode/decode path enough to demonstrate
   value, or does the prototype need the WebPPicture pipeline before it's
   shippable? The .NET wrappers default to the advanced pipeline for
   `WebPEncoderConfig`-based encoding; simple encode is only used for the
   bare `quality` overloads.
2. **Animation** — same question. Animation support is six WebPAnim* functions
   each side; non-trivial but bounded.
3. **Bit-exactness expectation** — zenwebp produces spec-conformant WebP within
   0.02–5% of libwebp's file size, not byte-identical bytes. Any downstream
   consumer relying on byte-for-byte reproducibility (file hashes, golden
   tests) will see drift. Worth a CHANGELOG note on the NuGet package.
4. **Packaging name** — `Imazen.WebP.NativeRuntime.linux-x64.Zen`?
   `Imazen.WebP.NativeRuntime.linux-x64-zen`? `Imazen.ZenWebP.NativeRuntime.*`?
5. **Licensing** — zenwebp is AGPL-3.0-or-later OR commercial. libwebp is
   BSD-3-Clause. Mixing means the shim+zenwebp is AGPL, and users of the
   `*.Zen` NuGet pull AGPL transitive obligations unless they have a commercial
   license. Worth flagging prominently.

## Layout of this prototype

```
zenwebp--libwebp-shim/
├── Cargo.toml              # cdylib + rlib, name="webp" -> libwebp.{so,dll,dylib}
├── src/lib.rs              # 33 extern "C" functions
├── DESIGN.md               # this file
├── .workongoing            # claim marker per ~/.claude/CLAUDE.md
└── .gitignore
```

`cargo build --release` produces `target/release/libwebp.so` (Linux),
`webp.dll` (Windows), `libwebp.dylib` (macOS). The cdylib name matches what
libwebp-net's `[DllImport("libwebp", ...)]` looks for.

## Next steps if we proceed

1. **Layout-assert tests for WebPConfig / WebPPicture** (block all other
   advanced-encode work behind this).
2. **Wire WebPConfigInitInternal + WebPValidateConfig** — these are pure
   field-default writes, no algorithm.
3. **Wire WebPEncode + the Picture import functions** — this is the bulk of
   the work. Storage strategy: a `Mutex<HashMap<*mut WebPPicture, Box<PicState>>>`
   side-table keyed by the picture pointer, where `PicState` holds the
   imported pixels, format, and any other state libwebp's internal `memory_argb_`
   field would track.
4. **Wire animation** — `AnimationEncoder` in zenwebp is feature-complete; the
   shim mostly translates timestamp deltas and packs frames.
5. **CI matrix**: add the seven RIDs libwebp-net ships
   (win-x64/x86/arm64, linux-x64/arm64, osx-x64/arm64). Cross-compile via
   `cargo zigbuild` or `cross`. Output goes into the existing
   `Imazen.WebP.NativeRuntime.<rid>` packaging layout.
6. **Validation**: run libwebp-net's existing test suite against the shim
   `libwebp.so` and triage diffs. The PageHeap heap-validation job (already in
   `dotnet.yml`) is a good first canary — any UAF / OOB in the FFI shim
   surfaces there.
