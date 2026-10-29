# Declared row validation, 2026-10-29

Base: d5893295. Tested source snapshot: e07d7cd045b06775316de08cab2cec0acdfb472f.
ARM64 Neoverse-N1, isolated export outside workspace patches, unchanged committed
Cargo.lock. Existing core pin feb09f9f and pixels 283655c5; this is not a broader
consumer repin qualification. Native algorithms and existing assertions unchanged.

Command: `run-heavy --mem 3G --jobs 2 -- timeout 900 just declared-rows-check`
with `CARGO_TARGET_DIR` outside the export and `CARGO_PROFILE_DEV_DEBUG=0`.
All stdout/stderr retained under `~/tmp/zenwebp-declared-rows-20261029/`.

- Initial baseline build: rc101, 404s, peak RSS 0.55GiB, failed on the new
  regression's incorrect config import. Second import correction still addressed
  a private module: rc101, 1s, 0.23GiB. Correct public path is `zenwebp::zencodec`.
- Baseline with corrected tests: rc101, 2s, 0.18GiB. Declared begin refused.
- Baseline with separate padding control: rc101, 1s, 0.17GiB; three failures
  (begin, declared flow, padded legacy rows), one conformance pass. Conformance
  permits configuration-specific begin refusal; passing it alone does not prove
  a row route is usable.
- Initial repair: rc101, 12s, 0.45GiB; three pass, one test fixture rejected by
  PixelSlice's StrideNotPixelAligned validation. Replaced this newly introduced
  invalid fixture with a second valid pixel-aligned stride and removed the
  unnecessary repacking branch. Existing assertions were not changed.
- Final repair: rc0, 95s, peak RSS 0.66GiB, minimum available 14264MiB,
  peak load 1.66. Four new controls, 51 existing codec tests, and strict library
  plus regression-target Clippy pass. Existing library filter leaves 275 tests
  outside this focused run; no full-crate or cross-platform qualification claim.

The new controls cover packed and two padded RGB layouts, lossy/lossless,
declared/legacy rows, retry after rejected begin, duplicate begin, one-shot after
begin, width mismatch, missing/excess rows and identical packed output bytes.
Lossless readback through libwebp (webpx) matches exact source samples.

Source SHA256: `2bd38cba852dbcae0e2011d809c0af3b3a06deb6366eae20cd1b852093fdcf5a`.
Test SHA256: `429f1e31352b5d98268e52f6d4fb721e1a1a6f9b11634756cb3c54e3a8409f44`.
Lock SHA256: `be21ce005cb548314c04251b2c89d48cfef9618b66491382cee527fe3a1e75c0`.

Wrapper resources measure validation, not encoder peak memory or speed.
Allocation/polling costs have not been benchmarked. Row accumulation retains a
whole raster; capacity growth and conversion/native workspace are not a proven
whole-operation bound. Checked plan/color/deferred-metadata routes remain outside
this qualification.
