# Changelog

All notable changes to zenwebp are documented here. (Started 2026-06-10;
earlier history lives in git log and LOG.md.)

## [Unreleased]

### Added
- `cms` feature: moxcms-backed ICC synthesis (`zenpixels-convert/cms-moxcms`,
  weak passthrough — takes effect with `zencodec`) covering PQ/HLG and any
  CICP moxcms can express. Failing to synthesize a needed ICC is now an
  encode **error** (`EncodeError::IccSynthesisUnavailable`), not a silent
  skip: WebP has no CICP carrier, so an embedded ICC is the only way the
  color survives. CI tests `--features zencodec,cms`; tests
  `cicp_pq_without_cms_is_an_encode_error` / `cicp_pq_with_cms_synthesizes_icc`.
- zencodec 0.1.21 color-emit integration: the encode path reconciles ICC vs
  CICP via `resolve_color_emit` under the caller's `ColorEmitPolicy`. WebP has
  no CICP carrier, so a CICP-only source synthesizes an embedded ICC (via
  zenpixels-convert `synthesize_icc_for_cicp`, transfer-aware) instead of
  silently emitting an untagged sRGB-assumed file. Metadata retention now
  flows through `with_metadata_policy` / `Metadata::filtered`. Deps bumped to
  published zencodec 0.1.21 / zenpixels 0.2.11 / zenpixels-convert 0.2.12
  (8bc51dbe).

### Known issues
- dev-dependency `webpx = "0.1.4"` is yanked on crates.io. Builds resolve via
  the committed `Cargo.lock`; a fresh `cargo update` fails until the webpx dep
  is migrated to a current release (0.3.4 at time of writing).
