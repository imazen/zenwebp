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

Six strategies, each at a different point on the (generation-loss × bitrate
× CPU) surface:

| Strategy             | Decodes? | Re-encodes? | Generation loss | When it wins                                   |
|----------------------|----------|-------------|-----------------|-----------------------------------------------|
| `NoOp`               | no       | no          | none            | source already at target                       |
| `LosslessRemux`      | no       | no          | none            | container metadata strip, no recompression wins |
| `CoeffEdit`          | partial  | partial     | none in IDCT    | per-segment Q tighten on VP8 only              |
| `Reencode`           | yes      | yes         | one round       | medium-quality source, target close to source  |
| `DeblockReencode`    | yes      | yes (deblock)| one round + deblock | source has visible artifacts, target much lower bpp |
| `LosslessReencode`   | yes      | yes (VP8L)  | none after decode | graphics/screen content, lossless wins        |

The router queries a calibration table indexed by
`(encoder_family, source_q, content_class, target_zensim_a, strategy)` and
picks the strategy with the smallest projected output size whose projected
zensim-A is at-or-above target. When **no** strategy beats source size at
target, it ships a `LosslessRemux` instead.

## Design

See [DESIGN.md](./DESIGN.md) for the full taxonomy, calibration design, and
decision-router blueprint.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. See [LICENSE-AGPL3](./LICENSE-AGPL3)
and [LICENSE-COMMERCIAL](./LICENSE-COMMERCIAL).
