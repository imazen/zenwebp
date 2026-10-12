# Metadata integration

Structural inventory reuses [PR106](https://github.com/imazen/zenwebp/pull/106)
at `cd09b5bd`. Enable `inventory` explicitly; normal metadata transport and color
handling remain available without it. Without the feature, DecodeJob::inventory
returns typed unsupported. Public signatures use zencodec contracts only.

The inventory models the existing decoder, demuxer and probe differences and
identifies their readers in part details. Compressed VP8/VP8L/ALPH image-data
slack remains opaque; locating it requires pixel entropy decoding.

## Validation

Native and i686 inventory/replay/feature-boundary tests pass. The native no_std
inventory check and strict clippy pass. ExifTool matched 264 chunks across 228
files. The fuzz target compiles and the stable replay exercises existing crash
seeds through inventory. The separate two-direction mutation test remains an
explicit ignored-by-default run, not counted as executed here.

## Published API

The forced patch comparison against crates.io 0.4.4 reports 222 checks passed,
one failed and 31 skipped. The failure is the second EncodeError::LimitExceeded
tuple field, already present on main. The existing 0.5.0 version and queued
breaks predate this extraction. Associated-error migrations need manual review
in addition to the compiler check; an additive inventory API does not certify
all other main-branch changes as compatible.

Input-size rejection currently preserves PR106's decoder-equivalent error;
changing its existing test to require the shared typed InputSize cause is
pending owner approval. Part errors already retain InventoryError and classify
part-cap failures as InventoryParts.
