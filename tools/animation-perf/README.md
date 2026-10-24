# PR105 animation comparison

This standalone tool interleaves main and PR source in one zenbench process.
Both use the PR's locked dependency graph and release settings. The exported
main control changes only its package-version suffix to let both libraries
link together. This isolates source cost; it does not measure dependency
upgrades. The root `animation_contract_perf` bench can measure each revision
with its own graph, but separate processes are more sensitive to host drift.

Prepare from the repository root (the control must not already exist):

```sh
python3 tools/animation-perf/prepare.py a3fec3fec04cad9322af0a9e6bed023e72f1929c
```

Build from outside ancestor Cargo overrides, then run serially under the
workspace resource wrapper. Set `ZENWEBP_PERF_FIXTURES` to this repository's
absolute `tests` directory for every run. Use the same directory for both
arms. Run with `--cpu-policy=unrestricted --format=json --median-of-passes=3`.
The exact invocation and measured limits are in [the result record](../../benchmarks/pr105/README.md).

Encode cases use three 256x256 photographic crops, lossy q75/m4 and lossless
m4. Decode cases use the existing three-frame 64x63 lossless animation,
four-frame 99x87 lossy animation and 550x368 lossy still. Native and shared
animation paths are separate cases. Setup and file I/O are outside timed calls.
The fixture provenance remains in [tests/CREDITS.md](../../tests/CREDITS.md).

`main/` and `baseline.json` are generated local controls and are not published.
Do not replace an existing control or discard result files when rerunning.
