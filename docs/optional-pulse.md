# Optional per-call control

The shared incremental decoder accepts `Option<&dyn zencodec::Pulse>`.
Pass `None` when no extra per-call control is needed. The wrapper forwards
absence unchanged; neither `None` nor an explicit no-op disables the retained
job's cancellation. Native owned controls and their reporting units are unchanged.

## Qualification (2026-10-29)

Source `6cba688f`, using core `feb09f9f` and pixels `283655c5`, passed:

```sh
cargo test --locked --test pulse_calls --test pulse_polls --test sansio
cargo clippy --locked --lib --test pulse_calls --test pulse_polls --test sansio -- -D warnings
```

The 20 tests include retained timeout before sink effects, both optional call
forms, supplied control polling and incremental pixel parity. This is selected
Linux contract coverage; broader feature coverage and platform CI are not
qualified by these commands.

The three-codec supervising run reported:
`rc1 373s; peak-RSS 0.81GiB; min-avail 13902MiB; peak-load 2.45`.
Its WebP test and Clippy steps both returned zero. The aggregate failure belongs
to bitmap Clippy and remains recorded in the
[zencodec qualification receipt](https://github.com/imazen/zencodec/blob/encoding-route/benchmarks/optional_pulse_2026-10-29.json).
This receipt is not a throughput or per-codec memory measurement.

The committed root lock is the tested portable graph, resolved outside parent
workspace patches. Native PNG tests use the paired PNG source; retaining old
PNG 0.1.4 introduced a second core and failed its layout assertion before tests.
The old assertion was not changed. Required PNG/zenflate/rustix changes were
reviewed explicitly; the root contains one core, pixel, converter and Pulse
package. Nested workspaces have separate locks and need their own qualification.

## Standalone graph follow-up

On `fb4388e2`, the standalone fuzz and recompress workspaces resolved and passed
portable graph audits. Their only changed lock packages were core, pixels and
converter. `cargo check --locked --bins` passed in `fuzz/` and
`cargo check --locked --lib` passed in `zenwebp-recompress/`. These are compile
checks; they do not rerun the recompression quality or runtime calibration gates.
Both tested locks are committed here.

The follow-up wrapper recorded `rc1 62s; peak-RSS 0.62GiB;
min-avail 14490MiB; peak-load 0.99`. Both WebP workspaces passed; unrelated
JPEG/bitmap fuzz roots failed resolution and remain recorded as failures.
