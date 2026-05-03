#!/usr/bin/env python3
"""
Inject ZNPR v3 `output_specs` + `sparse_overrides` into a hybrid-heads
trainer JSON, ready to feed `~/work/zen/zenanalyze/tools/bake_picker.py`.

Reads:
  - The trainer JSON written by `train_hybrid.py` (via codec config's
    OUT_JSON path).
  - The codec config's OUTPUT_SPECS dict (keyed by output-block name —
    `bytes_log` and each scalar axis name) and SPARSE_OVERRIDES list.
  - The trainer-emitted `hybrid_heads_manifest.output_layout` to
    decide where each block lives in the n_outputs vector.

Writes:
  - <input>.v3.json with `output_specs` (length n_outputs) and
    `sparse_overrides` (validated index range) appended at the top
    level. Other keys preserved verbatim.

train_hybrid.py does not currently know about OUTPUT_SPECS — the v3
shape lives entirely on the bake side, and threading the schema
through the trainer would be a wider zenanalyze change. This shim
keeps the v3 work codec-local until that lands upstream.

Usage:
    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples \\
        python3 dev/inject_v3_specs.py \\
            --codec-config zenwebp_picker_config \\
            --in benchmarks/zenwebp_hybrid_v3.json \\
            --out benchmarks/zenwebp_hybrid_v3_specs.json
"""

import argparse
import importlib
import json
import sys
from pathlib import Path


def _expand_specs(output_layout: dict, output_specs_by_name: dict, n_outputs: int) -> list[dict]:
    """Walk `output_layout` (head → [start, end] indices) and emit one
    OutputSpec dict per index in [0, n_outputs). Each block uses the
    spec from `output_specs_by_name` keyed by its head name; absent
    head keys raise loudly so misalignment surfaces at bake time."""
    out: list[dict | None] = [None] * n_outputs
    for head_name, span in output_layout.items():
        if head_name not in output_specs_by_name:
            raise SystemExit(
                f"OUTPUT_SPECS is missing a spec for head {head_name!r}. "
                f"Add it to the codec config or strip the head from the "
                f"trainer output."
            )
        spec = output_specs_by_name[head_name]
        start, end = int(span[0]), int(span[1])
        if start < 0 or end > n_outputs or end <= start:
            raise SystemExit(
                f"output_layout[{head_name!r}] = [{start}, {end}] is not a "
                f"valid sub-range of [0, {n_outputs})"
            )
        for i in range(start, end):
            out[i] = dict(spec)
    if any(s is None for s in out):
        missing = [i for i, s in enumerate(out) if s is None]
        raise SystemExit(
            f"output_layout doesn't cover every index in [0, {n_outputs}); "
            f"missing: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    return out  # type: ignore[return-value]


def _validate_sparse_overrides(overrides: list[dict], n_outputs: int) -> list[dict]:
    cleaned: list[dict] = []
    for entry in overrides:
        idx = int(entry["idx"])
        if idx < 0 or idx >= n_outputs:
            raise SystemExit(
                f"sparse_overrides idx {idx} out of range (n_outputs={n_outputs})"
            )
        value = entry.get("value")
        if value is None:
            cleaned.append({"idx": idx, "value": None})
        else:
            cleaned.append({"idx": idx, "value": float(value)})
    return cleaned


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--codec-config",
        required=True,
        help="Codec config Python module exporting OUTPUT_SPECS + SPARSE_OVERRIDES",
    )
    ap.add_argument("--in", dest="input", required=True, type=Path, help="trainer JSON in")
    ap.add_argument("--out", dest="output", required=True, type=Path, help="v3-augmented JSON out")
    args = ap.parse_args()

    cfg = importlib.import_module(args.codec_config)
    output_specs_by_name = getattr(cfg, "OUTPUT_SPECS", None)
    if not isinstance(output_specs_by_name, dict):
        raise SystemExit(
            f"codec config {args.codec_config!r} does not export OUTPUT_SPECS"
        )
    sparse_overrides_in = getattr(cfg, "SPARSE_OVERRIDES", []) or []
    # Optional codec-supplied schema_version_tag — must match the
    # runtime's compile-time SCHEMA_HASH derivation. zenwebp's runtime
    # uses "zenwebp.picker.v0.1"; without this, bake_picker.py defaults
    # to "zentrain.v1.generic" and the resulting hash won't validate
    # against the runtime's compile-time const.
    schema_version_tag = getattr(cfg, "SCHEMA_VERSION_TAG", None)
    bake_name = getattr(cfg, "BAKE_NAME", None)

    model = json.loads(args.input.read_text())
    n_outputs = int(model["n_outputs"])
    hh = model.get("hybrid_heads_manifest") or {}
    output_layout = hh.get("output_layout")
    if not isinstance(output_layout, dict):
        raise SystemExit(
            "trainer JSON missing hybrid_heads_manifest.output_layout — "
            "can't decide which dim gets which spec."
        )

    expanded = _expand_specs(output_layout, output_specs_by_name, n_outputs)
    validated = _validate_sparse_overrides(sparse_overrides_in, n_outputs)

    model["output_specs"] = expanded
    model["sparse_overrides"] = validated
    if schema_version_tag is not None:
        model["schema_version_tag"] = str(schema_version_tag)
    if bake_name is not None:
        model["bake_name"] = str(bake_name)
    args.output.write_text(json.dumps(model, indent=2))
    sys.stderr.write(
        f"wrote {args.output}: {len(expanded)} output_specs, "
        f"{len(validated)} sparse_overrides"
        + (f", schema_version_tag={schema_version_tag!r}" if schema_version_tag else "")
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
