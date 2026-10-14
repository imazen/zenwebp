# Upstream notices

`zenwebp` began as a fork of [`image-webp`](https://github.com/image-rs/image-webp),
the pure-Rust WebP crate from the [image-rs](https://github.com/image-rs) project.

## What carries an upstream notice

The decoder, the lossless pipeline, the loop filter, the pixel transforms, and
the shared VP8 tables still follow the upstream implementation closely; the full
texts of the upstream licenses live beside this note:

- [`LICENSE-MIT`](./LICENSE-MIT) — MIT license of `image-webp`
- [`LICENSE-APACHE`](./LICENSE-APACHE) — Apache-2.0 license of `image-webp`

Thank you to the image-rs maintainers for the well-structured, battle-tested
codebase this project started from.
