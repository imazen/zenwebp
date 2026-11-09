#!/usr/bin/env python3
"""Run the exact cross-built allocation control on the native Linux host.

The cross container can refuse subprocess creation before the probe runs.
This executes the same 32-bit test binary outside that container. Missing or
ambiguous artifacts are errors; the allocation assertions are unchanged.
"""
import json
from pathlib import Path
import subprocess
import sys

receipt, target_dir = map(Path, sys.argv[1:])
artifacts = []
for line in receipt.read_text().splitlines():
    if not line.startswith("{"):
        continue
    item = json.loads(line)
    if (item.get("reason") == "compiler-artifact"
            and item.get("target", {}).get("name") == "zenwebp"
            and item.get("profile", {}).get("test")
            and item.get("executable")):
        artifacts.append(item["executable"])
if len(artifacts) != 1:
    raise RuntimeError(f"expected one zenwebp test executable, found {artifacts!r}")
binary = target_dir / "i686-unknown-linux-gnu" / "debug" / "deps" / Path(artifacts[0]).name
with binary.open("rb") as stream:
    header = stream.read(20)
if header[:6] != b"\x7fELF\x01\x01" or int.from_bytes(header[18:20], "little") != 3:
    raise RuntimeError(f"not a 32-bit little-endian x86 ELF: {binary}")
test = "decoder::alloc_util::tests::valid_layout_allocation_failure"
listing = subprocess.run(
    [str(binary.resolve()), "--list", "--exact", test],
    check=True, text=True, capture_output=True,
)
if f"{test}: test" not in listing.stdout.splitlines():
    raise RuntimeError(f"allocation control is absent: {listing.stdout}")
subprocess.run([str(binary.resolve()), "--exact", test, "--nocapture"], check=True)
