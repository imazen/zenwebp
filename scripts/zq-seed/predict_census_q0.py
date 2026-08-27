#!/usr/bin/env python3
"""Per-(corpus9 ref, target) Zq-head q0 table for the census B arm.
Reads the fit line (benchmarks/zq_seed_fit_2026-08-27.tsv JSON tail) + the
print_features TSV; mirrors fit_zq_seed.py's design() basis EXACTLY."""
import json, math, sys, csv

names = coefs = None
for line in open("benchmarks/zq_seed_fit_2026-08-27.tsv"):
    k, _, v = line.strip().partition("\t")
    if k == "# features":
        names = v.split(",")
    elif k == "coef":
        coefs = [float(x) for x in v.split(",")]
assert names and coefs, "chosen/coef rows not found"
LOG1P = {"dct_compressibility_y","dct_compressibility_uv","laplacian_variance",
         "laplacian_variance_peak","variance","variance_spread","distinct_color_bins"}

feats = {}
dims = {}
import subprocess
for row in csv.DictReader(open(sys.argv[1]), delimiter="\t"):
    path = row["path"]
    fv = []
    for n in names:
        v = float(row[n])
        fv.append(math.log1p(max(v, 0.0)) if n in LOG1P else v)
    feats[path] = fv
    # pixel count from the png header via file? use python-free: read IHDR
    with open(path, "rb") as f:
        f.seek(16)
        import struct
        w, h = struct.unpack(">II", f.read(8))
    dims[path] = w * h

print("path\tt\tq0")
for path, fv in feats.items():
    for t in (70.0, 80.0, 88.0):
        tn = (t-65)/25; h80 = max(t-80,0)/10
        base = [1.0, tn] + [max(t-k,0)/10 for k in (50,60,70,80,85)] + [(math.log(dims[path])-13)/3]
        x = base + fv + [v*tn for v in fv] + [v*h80 for v in fv]
        q0 = sum(a*b for a, b in zip(coefs, x))
        q0 = min(100.0, max(1.0, q0))
        print(f"{path}\t{t:.0f}\t{q0:.2f}")
