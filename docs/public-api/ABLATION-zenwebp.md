# zenwebp Public-API Ablation Report

**Date:** 2026-06-11  
**Snapshot commit:** `8502a70a7a3e` (refactor(api-doc): exclude underscore-prefixed features from all-features snapshot)  
**Snapshot:** `docs/public-api/zenwebp.txt` — 1,583 default / 1,831 all-except-`_*` items  
**Mode:** COMMIT (report only — no source changes)

---

## Grep Protocol

External-consumer scan performed with:

```
ugrep -r <ITEM> /home/lilith/work \
  --include="*.rs" --include="*.toml" \
  --exclude-dir=zenwebp --exclude-dir=target --exclude-dir=.jj
```

Confirmed consumers (as of this scan):

| Repo | Dependency on zenwebp |
|---|---|
| `zenpipe/zencodecs` | `features = ["zencodec"]` — uses `zenwebp::zencodec::*` |
| `imageflow` | `features = ["zencodec"]` — uses `zenwebp::zencodec::*` |
| `zenmetrics` | `features = ["__expert"]` (sweep feature) — uses `InternalParams`, `SharpYuvSetting` |
| `coefficient` | `features = ["__expert"]` — uses zencodec-trait path |
| `zenwebp-recompress` (nested) | uses `detect::*`, `decoder::decode_yuv420`, `decoder::YuvPlanes`, `mux::WebPMux`, `oneshot::*`, `encoder::*` |
| `jxl-encoder`, `zenjpeg`, `zenavif`, `zenjxl` | reference `InternalParams` / `AblationToggles` / `SharpYuvSetting` naming patterns in their own analogous expert surfaces — NOT consuming zenwebp's |
| `zenanalyze` | comment-level reference to zenwebp's alpha classifier concept — NOT importing |
| `coefficient` | `ImageContentType` referenced in comments, but using a local re-implementation — NOT importing zenwebp's |

---

## Summary

| Scope | Count | Flagged | Flagged % |
|---|---|---|---|
| Default surface (1,583 items) | 1,583 | 3 | 0.2% |
| Extra all-features items (248 extra) | 248 | 3 | 1.2% |
| Combined | 1,831 | 6 | 0.3% |

Total flagged: **6 items across 4 locations**. Well below the 10% aggregation threshold. Conservative posture — KEEP applied wherever any uncertainty exists.

Action class breakdown: A = `#[doc(hidden)]`/`#[deprecated]` (no API break); B = `pub(crate)`/remove (queued breaking, next 0.x minor).

---

## Module-by-Module Findings

### `zenwebp` crate root

| Item | External hits | Flag | Action | Notes |
|---|---|---|---|---|
| `pub use encoder::ClassifierDiag` | 0 | FLAG | A | Re-exported with no `#[cfg]` gate; the type is available at `zenwebp::ClassifierDiag` in default features. The struct exposes `edge_density`, `uniformity`, `is_bimodal` — useful diagnostics but zero external consumers. `ImageContentType` (sibling export) also has 0 external hits (coefficient has a local copy). Could be `#[doc(hidden)]` without API break. |
| `pub use encoder::ImageContentType` | 0 | FLAG | A | Same gate as `ClassifierDiag`; shipped without hiding. Consumers can still use `classify_image_type` without needing the type name. `#[doc(hidden)]` is the conservative action; removal is B. |
| `pub use zenyuv::SharpYuvConfig` | 0 zenwebp-specific hits (jxl-encoder / zenjpeg use it directly from `zenyuv`) | KEEP | — | Required by `LossyConfig::with_sharp_yuv_config()` (default-feature public API). The re-export is necessary — callers constructing a custom `SharpYuvConfig` need the type. Not a mistake. |

### `zenwebp::encoder` (pub mod, sub-modules doc-hidden)

| Item | External hits | Flag | Action | Notes |
|---|---|---|---|---|
| `pub mod validation` (no `#[doc(hidden)]`) | 0 external hits directly via `zenwebp::encoder::validation::*` | FLAG | A | `ValidationError` is correctly re-exported at crate root (`zenwebp::ValidationError`); the sub-path `zenwebp::encoder::validation::ValidationError` is a redundant second route into the same type. `TARGET_ZENSIM_RANGE` (all-features only, `target-zensim` gate) has 0 external hits. The validation module should carry `#[doc(hidden)]` to match the other internal sub-modules (`analysis`, `cost`, `quantize`, `tables`, `vp8`, `vp8l`, `zensim_target`, `picker`). Not a breaking change (hidden items can still be used). |

### `zenwebp::encoder` — `#[cfg(feature = "__expert")]` surface

| Item | External hits | Flag | Action | Notes |
|---|---|---|---|---|
| `InternalParams` (default-features hidden; expert-gated) | **2 confirmed consumers**: `zenmetrics` (partition_limit, multi_pass_stats, smooth_segment_map, sharp_yuv fields); `coefficient` (feature-gated via `zenwebp` feature) | KEEP | — | Intentional expert surface. Correctly gated. Both consumers build with `__expert`. No action. |
| `SharpYuvSetting` | **1 confirmed consumer**: `zenmetrics` (Off/On variants for sweep knobs) | KEEP | — | Used by zenmetrics sweep. Required alongside `InternalParams::sharp_yuv`. |
| `AblationToggles` + `set_ablation_toggles` | 0 external hits | FLAG | B | Correctly gated behind `ablation` feature (which implies `target-zensim`). No external consumer anywhere in the workspace. The `ablation` feature is explicitly documented as dev-only for `dev/zensim_*.rs` measurement binaries. The type IS useful internally but has never escaped. Proposal: move the global-static `set_ablation_toggles` to `pub(crate)` on the Rust side, keeping only the test-binary `dev/` paths building it in. Would not break any external build. |

### `zenwebp::detect` module

| Item | External hits | Flag | Action | Notes |
|---|---|---|---|---|
| `BitstreamType::Lossy { quantizer_index, filter_level, has_segment_quant, quality_estimate, sharpness_level }` | Consumed by `zenwebp-recompress` (recompress-only; `source.rs` extracts all five fields) | KEEP | — | Recompress is a known external-ish consumer. All five fields serve a documented purpose: `quantizer_index` drives deblock strength, `filter_level` gates the deblock strategy, `quality_estimate` seeds the calibration. |
| `WebPProbe::icc_profile` (returns `Option<Vec<u8>>`) | `zenwebp-recompress` — header probe, not ICC | KEEP | — | Struct fields are intentional. Detection module is a deliberate public API surface for header inspection without full decode. |

### `zenwebp::decoder`

| Item | External hits | Flag | Action | Notes |
|---|---|---|---|---|
| `YuvPlanes` (struct with 7 public fields) | `zenwebp-recompress` (via `decode_yuv420`); 0 hits in any other external repo | KEEP | — | YuvPlanes is the return type of the pub `decode_yuv420` API, used by recompress for chroma-aware compensation. Removing it would break recompress. |
| `decode_yuv420` (free fn in `decoder` + `oneshot` modules) | `zenwebp-recompress` | KEEP | — | Intentional semi-internal path for recompress. |

### `zenwebp::encoder::EncodeStats::transform_bits`

| Item | External hits | Flag | Action | Notes |
|---|---|---|---|---|
| `transform_bits: u8` | 0 external hits | KEEP (conservative) | — | `EncodeStats` is `#[non_exhaustive]` (not struct-constructible by callers). `transform_bits` is a lossless VP8L diagnostic field alongside 7 sibling lossless fields. Since the struct is `#[non_exhaustive]`, no field can be removed without a semver break anyway — and it's not one of the flagged items above. Noted for completeness. |

### `zenwebp::heuristics` module

| Item | External hits | Flag | Action | Notes |
|---|---|---|---|---|
| `EncodeEstimate`, `DecodeEstimate`, `estimate_encode`, `estimate_decode`, `estimate_animation_encode`, `estimate_animation_decode` | 0 external hits | KEEP (conservative) | — | The heuristics module is a common pattern across zen codecs (zenjpeg has its own). Even with 0 current external hits, resource estimation is a natural companion to a codec's public API and carries no implementation risk. Not flagged. |

### `zenwebp::mux` module

No unexpected items. All mux types (`WebPMux`, `WebPDemuxer`, `AnimationEncoder`, `DemuxFrame`, `MuxFrame`, `BlendMethod`, `DisposeMethod`, `MuxError`, `MuxResult`) are intentional container-layer API. `zenwebp-recompress` consumes `mux::WebPMux`. No external hits outside the repo, but the mux API is a coherent and intentional surface. No flags.

### `zenwebp::zencodec` (all-features only, `zencodec` feature)

All 9 public structs (`WebpDecoder`, `WebpEncoder`, `WebpDecodeJob`, `WebpEncodeJob`, `WebpDecoderConfig`, `WebpEncoderConfig`, `WebpStreamingDecoder`, `WebpAnimationFrameDecoder`, `WebpAnimationFrameEncoder`) are confirmed consumed by `zenpipe/zencodecs` and `imageflow`. No flags.

### `zenwebp::encoder::validation::TARGET_ZENSIM_RANGE` (all-features only)

`pub const TARGET_ZENSIM_RANGE: RangeInclusive<f32>` — gated on `target-zensim` feature, published at `zenwebp::encoder::validation::TARGET_ZENSIM_RANGE`. Zero external hits. Used internally by validation logic. Covered by the `validation` module `#[doc(hidden)]` flag above — no separate action needed.

---

## Flagged Items Summary

| # | Item path | Default or All | Action | Class |
|---|---|---|---|---|
| 1 | `zenwebp::ClassifierDiag` | default | `#[doc(hidden)]` on the re-export in `lib.rs` (line 71 of `encoder/mod.rs`) | A |
| 2 | `zenwebp::ImageContentType` | default | `#[doc(hidden)]` on the same re-export block | A |
| 3 | `zenwebp::encoder::validation` (the module) | default | Add `#[doc(hidden)]` to match other internal sub-modules | A |
| 4 | `zenwebp::encoder::validation::TARGET_ZENSIM_RANGE` | all-features | Covered by #3 | A |
| 5 | `zenwebp::AblationToggles` | all-features (ablation feature) | Consider `pub(crate)` — no external consumers, explicitly dev-only | B (proposal) |
| 6 | `zenwebp::set_ablation_toggles` | all-features (ablation feature) | Same as #5 | B (proposal) |

---

## `picker` → `__expert` Architecture Note

The `picker` feature implies `__expert` because the picker runtime uses `InternalParams::partition_limit`, `multi_pass_stats`, and `smooth_segment_map` to populate its predicted tuning axes. This is intentional: picker-generated configs must be fed via `LossyConfig::with_internal_params()`, which is `#[cfg(feature = "__expert")]`.

The practical consequence is that any downstream that enables `picker` (for its MLP-driven `pick_tuning()` → `TuningPick` path) automatically gets the full `__expert` surface: `InternalParams`, `SharpYuvSetting`, and their methods. Whether that surface is "narrower than it needs to be" for typical picker consumers is a design observation, not a mistake — the picker itself sets all five expert axes when its schema advances to v0.2+. The current transitivity is architecturally intentional per the Cargo.toml comment on line for `picker`.

The `encoder::picker` module is `#[doc(hidden)]`; `PickError`, `TuningPick`, and `pick_tuning` are pub-within-hidden — zero external hits, zero concern.

---

## Top-3 Digest

1. **`zenwebp::encoder::validation` module not `#[doc(hidden)]`** — All sibling internal sub-modules (`analysis`, `cost`, `quantize`, `tables`, `vp8`, `vp8l`, `zensim_target`, `picker`) are `#[doc(hidden)]`. `validation` is the sole exception. `ValidationError` is correctly available at crate root; the sub-path is redundant. Add `#[doc(hidden)]` to match. (Class A)

2. **`ClassifierDiag` + `ImageContentType` re-exported at crate root without hiding** — The two classifier types appear in `zenwebp::encoder::mod.rs` as unconditional pub re-exports without `#[doc(hidden)]`. Zero external consumers. They're useful diagnostics but shouldn't be first-class crate root items without intentional promotion. `#[doc(hidden)]` on the re-exports is the conservative action. (Class A)

3. **`AblationToggles` + `set_ablation_toggles` have no external consumers** — Correctly gated behind the `ablation` feature (documented as dev-only), but the feature itself is not underscore-prefixed, so it appears in the all-features snapshot. The global-static mutating `set_ablation_toggles` is the higher-risk item: it mutates a `static AtomicU64` that affects all concurrent encoders in the same process. No production consumer should call it. Queued as a B proposal: `pub(crate)` in the next 0.x minor that touches this area. (Class B, proposal-level)

---

## Files Consulted

- `/home/lilith/work/zen/zenwebp/docs/public-api/zenwebp.txt` — API snapshot
- `/home/lilith/work/zen/zenwebp/Cargo.toml` — feature definitions
- `/home/lilith/work/zen/zenwebp/src/lib.rs` — crate-root re-exports
- `/home/lilith/work/zen/zenwebp/src/encoder/mod.rs` — encoder re-exports and `#[doc(hidden)]` pattern
- `/home/lilith/work/zen/zenwebp/src/encoder/config.rs` — `InternalParams`, `SharpYuvSetting`, `LossyConfig`
- `/home/lilith/work/zen/zenwebp/src/encoder/zensim_target.rs` — `AblationToggles`, `set_ablation_toggles`
- `/home/lilith/work/zen/zenwebp/src/encoder/api.rs` — `EncodeStats`
- `/home/lilith/work/zen/zenwebp/src/encoder/analysis/classifier.rs` — `ClassifierDiag`, `ImageContentType`
- `/home/lilith/work/zen/zenwebp/src/encoder/validation.rs` — `TARGET_ZENSIM_RANGE`
- `/home/lilith/work/zen/zenwebp/src/detect.rs` — `WebPProbe`, `BitstreamType`
- `/home/lilith/work/zen/zenmetrics/crates/zen-metrics-cli/src/sweep/encode.rs` — confirmed `__expert` consumer
- `/home/lilith/work/zen/zenpipe/zencodecs/src/lib.rs`, `config.rs` — confirmed `zencodec` consumer
- `/home/lilith/work/imageflow/imageflow_core/src/codecs/zen_encoder.rs` — confirmed `zencodec` consumer
