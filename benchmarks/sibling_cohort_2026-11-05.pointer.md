# WebP sibling-cohort qualification (2026-11-05)

The update retains incoming main `7f7ab908` and converges root, fuzz and
recompress on one shared core/pixel identity. No codec algorithm, existing
assertion, score profile or calibration table changes.

| Dependency | Revision |
|---|---|
| zencodec family | `1816b4d082234c04c3805411b437b8396040f522` |
| zenpixels / convert | `fa836abf72afe9826e866dc25712b7c800323962` |
| analyzer / API | `38dc80f73887dafae6bf5ebfe9a03be5ef78ebe0` |
| PNG | `02a7f0df5f72dd79b0afa58d66ced7cdd8395e31` |
| dev-only resampler | `e3975fb9d6d6b7baa96038a0eb8e27febb37c012` |

Recompress keeps zensim `bab88fde6aae8aaf54f42767bc1272d90340c095` and Profile A.
The candidate research pins in the root remain unchanged as well.

## Gates

| Configuration | Runtime pass | Existing runtime ignores | Doc pass | Existing doc ignores |
|---|---:|---:|---:|---:|
| Root default | 831 | 11 | 33 | 3 |
| Root all 18 stable features | 951 | 13 | 35 | 4 |
| Recompress default workspace | 40 | 0 | 1 | 0 |
| Recompress all features | 40 | 0 | 1 | 0 |

Every executed test passes. Root and recompress strict all-target lint pass;
seven fuzz targets compile. `wholefile-lock-sources` checks all three lockfiles,
requiring one core and one pixel source with the selected exact revisions. It
uses Python 3.12.12 here (the pre-existing tomllib script needs Python 3.11+).
Formatting passes. No older encoder qualification is rerun or restamped.

The stable feature recipe excludes only the nightly `_benchmarks` feature and
the default alias. It prints the actual 18 flags and fails if selection is empty.
The initial harness mistakenly suppressed that stdout through `just --quiet`;
its 831-test run is retained as default coverage, not full-feature coverage.

## Compatibility boundary

Checked metadata retention remains unsupported, as directed for this dependency
convergence phase. The inherited checked setter rejects empty, ICC and CICP
records (3/3) with MetadataRetention. The diagnostic
`just sibling-metadata-probe` reports admission only and never encodes or
certifies retention. The missing contract is also in CLAUDE.md Known Bugs.

The 13 legacy-setter functions/helpers have narrow deprecation allowances
(12 new, one already present). Removing only the new allowance lines reproduces
all six original source/test files byte for byte: every assertion is retained.
The reviewed implementation is `a6f84b56`. Implementing checked retention or
other new codec behavior is outside this phase.

Profile A guards pass in both recompress configurations. SHA-256 remains:

- `src/calibration/calib_tables.rs`:
  `35f0305c1e6f0d7575e4c5e663463ac8cb10ef8d0eb8cfe82f810f0fb80900fa`.
- `src/measure.rs`:
  `00d830a55724f3d069723eb3fddbbc29fc3da07321ccf5de623ae3239af1735d`.

## Resource and artifact receipts

The root 18-feature test/doc scope completed before a duplicate allowance lint
finding: rc101, 258s, peak RSS 1.74 GiB, minimum available 6,627 MiB, peak load
4.45. Removing only our duplicate preserves the existing allowance. The next
scope passes strict root lint, seven fuzz compiles, both recompress suites and
recompress lint/fmt, then stops on Python 3.10's missing tomllib: rc1, 87s, peak
RSS 0.76 GiB, minimum available 7,577 MiB, peak load 4.10. The unchanged lock
checks then pass with installed Python 3.12: rc0, 0s, peak RSS 0.01 GiB, minimum
available 8,505 MiB, peak load 2.23. These are component gate receipts, not a
claim of a single successful combined invocation or a performance comparison.
All failed logs remain available.

Evidence on dev: `/mnt/v/output/zensim-siblings-20261105/zenwebp/`.
The initial boundary snapshot is at that root; final qualification is under
`qualified-legacy/`. Mirror:
`/mnt/tower/output/zensim-siblings-20261105/zenwebp/`.

Checked trait contract:
[EncodeJob metadata retention](https://github.com/imazen/zencodec/blob/1816b4d082234c04c3805411b437b8396040f522/src/traits/encoding.rs#L426).
