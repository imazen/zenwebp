# zenwebp-recompress

Smart recompression of already-encoded WebP files to a target
[zensim](https://crates.io/crates/zensim) Profile A score with **minimal generation
loss** and **no size regression**.

Given any WebP — lossy or lossless, libwebp, zenwebp, sharp, cwebp, ImageMagick —
and a target perceptual quality, this crate picks the cheapest strategy that
shrinks the file at-or-above target quality, or refuses to recompress when
recompression would lose. The chosen path is exposed so callers can know whether
the recompression they got was a coefficient-domain edit, a deblocked
pixel-domain re-encode, a lossless container re-mux, or whether they should look
elsewhere (e.g. transcode to JXL).

## Status

**Pre-alpha (0.1.x).** Designed to mirror
[zenjpeg-recompress](https://github.com/imazen/zenjpeg-recompress)'s approach
applied to WebP. The frozen public API is stable; calibration tables are still
landing.

## Usage

```rust,no_run
use zenwebp_recompress::{recompress, Budget, RecompressOptions, RecompressResult};

let webp_bytes = std::fs::read("photo.webp")?;
let opts = RecompressOptions {
    target_zensim_a: 82.0,
    budget: Budget::OneShot,
    ..Default::default()
};

match recompress(&webp_bytes, &opts)? {
    RecompressResult::Recompressed { bytes, strategy, projected_zensim_a, .. } => {
        println!("Recompressed via {strategy:?}, projected zensim-A = {projected_zensim_a:.2}");
        std::fs::write("photo.opt.webp", bytes)?;
    }
    RecompressResult::LosslessOnly { bytes, reason, .. } => {
        println!("Recompression would lose ({reason:?}); shipping a lossless re-mux");
        std::fs::write("photo.opt.webp", bytes)?;
    }
    RecompressResult::NoOp { reason } => {
        println!("Source is already at target ({reason:?})");
    }
    // The enum is `#[non_exhaustive]` — match the wildcard so future
    // additions don't break callers.
    _ => {}
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## How it picks

Strategies, each at a different point on the (generation-loss × bitrate
× CPU) surface:

| Strategy             | Status | Decodes? | When it wins                                   |
|----------------------|--------|----------|-----------------------------------------------|
| `NoOp`               | shipped | no      | source already at target                       |
| `LosslessRemux`      | shipped | no      | metadata strip; no recompression beats source  |
| `Reencode`           | shipped | yes     | source has headroom above target → re-encode lower |
| `LosslessReencode`   | shipped | yes     | graphics/screen, or a suboptimal lossless source |
| `DeblockReencode`    | **de-selected** | — | measured net-negative (VP8 deblocks in-loop) — [details](./benchmarks/deblock_experiment_2026-05-28.md) |
| `CoeffEdit`          | **deferred** | — | needs a VP8 token-stream API in zenwebp; not yet implemented |

The router keys a calibration table on the **decode-based effective
quality** (`(eff_q, target_zensim_a, content_class, strategy)`) — header
quality detection is unreliable for segmented WebP, so a re-compression
self-consistency estimate is used instead (see
[docs/QUALITY_DETECTION.md](./docs/QUALITY_DETECTION.md)). It picks the
strategy with the smallest projected output size whose projected zensim-A
is at-or-above target. A **ground-truth size guard** re-checks the actual
output and falls back to `LosslessRemux` if it didn't shrink — so the
result never grows the file, regardless of projection error.

## Design

See [DESIGN.md](./DESIGN.md) for the full taxonomy, calibration design, and
decision-router blueprint.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. See [LICENSE-AGPL3](./LICENSE-AGPL3)
and [LICENSE-COMMERCIAL](./LICENSE-COMMERCIAL).
