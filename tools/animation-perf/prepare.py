#!/usr/bin/env python3
"""Create an untouched source control except for disambiguating its package ID."""
import io, json, re, subprocess, tarfile
from pathlib import Path
import argparse
args=argparse.ArgumentParser()
args.add_argument("revision")
a=args.parse_args()
root=Path(__file__).resolve().parent
repo=root.parent.parent
sha=subprocess.check_output(["git","-C",str(repo),"rev-parse",a.revision],text=True).strip()
dest=root/"main"
if dest.exists(): raise SystemExit("main control already exists; preserve or move it before preparing another")
dest.mkdir()
data=subprocess.check_output(["git","-C",str(repo),"archive",sha])
with tarfile.open(fileobj=io.BytesIO(data)) as archive: archive.extractall(dest,filter="data")
p=dest/"Cargo.toml"
s=p.read_text()
s,n=re.subn(r'(?m)^version = "([^"]+)"',r'version = "\1-main-perf"',s,count=1)
assert n==1
p.write_text(s)
(root/"baseline.json").write_text(json.dumps({"source_revision":sha,"identity_only_change":"package version suffix -main-perf","dependency_graph":"shared PR pins, to isolate source cost"},indent=2)+"\n")
if not (root/"Cargo.lock").exists(): (root/"Cargo.lock").write_bytes((repo/"Cargo.lock").read_bytes())
print(sha)
