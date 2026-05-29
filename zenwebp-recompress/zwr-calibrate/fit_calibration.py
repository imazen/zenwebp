#!/usr/bin/env python3
"""Fit zenwebp-recompress calibration tables from a zwr-calibrate raw sweep.

Input: one or more CSVs from `zwr-calibrate --refs ... --reencode-qs ...`
(rows with strategy in {source, reencode, remux, vp8l}; content class parsed
from the input_path `/<class>/` segment; size from width/height).

Output: Rust `const` tables (per content class) for src/calibration/data.rs,
a size-dependence report, and held-out validation error.

Per the source-informing sweep discipline: per-class fit, size-dependence
fitted+reported (ratio intercept vs slope), low-q weighted same as high-q.

Usage: fit_calibration.py --train T.csv [T2.csv ...] [--val V.csv] \
           [--anchors 30,40,50,60,70,80,90] [--targets 20:95:5] --out tables.rs
"""
import argparse, csv, re, math, sys
from collections import defaultdict

CLASS_RE = re.compile(r"/(photo|screen|lineart|mixed)/")
# TRUE synthetic source quality is in the label (`_synth_q{q}`). The CSV
# `source_q` column is the unreliable detect estimate — never key on it.
TRUEQ_RE = re.compile(r"_synth_q(\d+)")
CLASSES = ["photo", "screen", "lineart", "mixed"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="*", default=[])
    p.add_argument("--anchors", default="30,40,50,60,70,80,90")
    p.add_argument("--targets", default="20:95:5")
    p.add_argument("--out", default=None)
    # The router keys the calibration on the decode-based estimate, which is
    # biased ~-5 vs true q. Key the table on `est` so the bias is baked in and
    # runtime indexing is self-consistent. `true` is for diagnostics only.
    p.add_argument("--key", choices=["est", "true"], default="est")
    return p.parse_args()


def parse_targets(s):
    a, b, st = (float(x) for x in s.split(":"))
    out, x = [], a
    while x <= b + 1e-6:
        out.append(int(round(x)))
        x += st
    return out


def size_bucket(maxdim):
    if maxdim <= 96:
        return "tiny"
    if maxdim <= 384:
        return "small"
    if maxdim <= 1280:
        return "medium"
    return "large"


def load(paths, key="est"):
    """rows: list of dicts with class, sq (the chosen key: est or true q),
    sq_true, sq_est, reencode_q, strat, ratio, cum, maxdim, pixels."""
    rows = []
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["result_kind"] != "ok":
                    continue
                m = CLASS_RE.search(r["input_path"])
                tm = TRUEQ_RE.search(r["input_path"])
                if not m or not tm:
                    continue
                cls = m.group(1)
                try:
                    w, h = int(r["width"]), int(r["height"])
                    sq_true = float(tm.group(1))  # TRUE encode q
                    eq = r.get("est_quality", "")
                    sq_est = float(eq) if eq not in ("", "NaN", "nan") else float("nan")
                    cum = float(r["measured_zensim_a_vs_reference"])
                    ratio = float(r["size_ratio"])
                except (ValueError, KeyError):
                    continue
                if w == 0 or h == 0 or math.isnan(cum):
                    continue
                # The calibration key: the decode-based estimate (matches the
                # router) unless --key=true. Fall back to true if est missing.
                sq = sq_est if (key == "est" and not math.isnan(sq_est)) else sq_true
                rq = r["reencode_q"]
                rq = float(rq) if rq not in ("", "NaN", "nan") else None
                rows.append(dict(cls=cls, sq=sq, sq_true=sq_true, sq_est=sq_est,
                                 rq=rq, strat=r["strategy"], ratio=ratio, cum=cum,
                                 maxdim=max(w, h), pixels=w * h))
    return rows


def nearest_anchor(sq, anchors):
    return min(anchors, key=lambda a: abs(a - sq))


def fit_source_cum(rows):
    """Per class: linear fit source_cum = a*q + b from strategy==source rows."""
    out = {}
    for cls in CLASSES:
        pts = [(r["sq"], r["cum"]) for r in rows if r["cls"] == cls and r["strat"] == "source"]
        if len(pts) < 2:
            out[cls] = (0.32, 57.0, len(pts))
            continue
        n = len(pts)
        sx = sum(q for q, _ in pts); sy = sum(c for _, c in pts)
        sxx = sum(q * q for q, _ in pts); sxy = sum(q * c for q, c in pts)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-9:
            out[cls] = (0.0, sy / n, n); continue
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n
        out[cls] = (a, b, n)
    return out


def fit_grid(rows, anchors, targets):
    """Per (class, anchor, target): mean ratio + mean cum over refs/sizes.
    Returns tables[cls] = (ratio[anchor][target], cum[...], counts)."""
    acc = defaultdict(lambda: ([], []))  # (cls,anchor,target) -> (ratios, cums)
    for r in rows:
        if r["strat"] != "reencode" or r["rq"] is None:
            continue
        anc = nearest_anchor(r["sq"], anchors)
        tq = min(targets, key=lambda t: abs(t - r["rq"]))
        if abs(tq - r["rq"]) > 0.6:
            continue
        ratios, cums = acc[(r["cls"], anc, tq)]
        ratios.append(r["ratio"]); cums.append(r["cum"])
    tables = {}
    for cls in CLASSES:
        ratio = [[float("nan")] * len(targets) for _ in anchors]
        cum = [[float("nan")] * len(targets) for _ in anchors]
        cnt = [[0] * len(targets) for _ in anchors]
        for ai, anc in enumerate(anchors):
            for ti, tq in enumerate(targets):
                rs, cs = acc.get((cls, anc, tq), ([], []))
                if rs:
                    ratio[ai][ti] = sum(rs) / len(rs)
                    cum[ai][ti] = sum(cs) / len(cs)
                    cnt[ai][ti] = len(rs)
        tables[cls] = (ratio, cum, cnt)
    return tables


def enforce_monotone_cum(cum):
    """Cumulative should be non-decreasing in reencode-q within a row."""
    for row in cum:
        run = float("-inf")
        for i, v in enumerate(row):
            if v == v:  # not nan
                run = max(run, v); row[i] = run


def fill_nans(grid):
    """Forward/backward fill NaNs in each row so the table is dense."""
    for row in grid:
        last = None
        for i, v in enumerate(row):
            if v == v: last = v
            elif last is not None: row[i] = last
        nxt = None
        for i in range(len(row) - 1, -1, -1):
            if row[i] == row[i]: nxt = row[i]
            elif nxt is not None: row[i] = nxt
        for i, v in enumerate(row):
            if row[i] != row[i]: row[i] = 0.0


def size_dependence(rows, anchors):
    """Report ratio vs size: for a mid anchor/target, ratio per size bucket."""
    print("\n== size-dependence of size_ratio (reencode, per class) ==", file=sys.stderr)
    print("   (ratio at source≈70, reencode≈70, by size bucket)", file=sys.stderr)
    for cls in CLASSES:
        by_bucket = defaultdict(list)
        for r in rows:
            if r["cls"] != cls or r["strat"] != "reencode" or r["rq"] is None:
                continue
            if abs(r["sq"] - 70) <= 6 and abs(r["rq"] - 70) <= 3:
                by_bucket[size_bucket(r["maxdim"])].append(r["ratio"])
        parts = []
        for b in ["tiny", "small", "medium", "large"]:
            v = by_bucket.get(b, [])
            parts.append(f"{b}={sum(v)/len(v):.3f}(n{len(v)})" if v else f"{b}=-")
        print(f"   {cls:8s} {'  '.join(parts)}", file=sys.stderr)


def validate(val_rows, scum, tables, anchors, targets):
    """Predict cumulative for val reencode rows via the fitted tables;
    report MAE per class."""
    if not val_rows:
        return
    print("\n== held-out validation (cumulative-zensim MAE) ==", file=sys.stderr)
    for cls in CLASSES:
        errs = []
        ratio, cum, _ = tables[cls]
        for r in val_rows:
            if r["cls"] != cls or r["strat"] != "reencode" or r["rq"] is None:
                continue
            ai = anchors.index(nearest_anchor(r["sq"], anchors))
            ti = min(range(len(targets)), key=lambda i: abs(targets[i] - r["rq"]))
            pred = cum[ai][ti]
            if pred == pred:
                errs.append(abs(pred - r["cum"]))
        if errs:
            errs.sort()
            print(f"   {cls:8s} MAE={sum(errs)/len(errs):5.2f}  "
                  f"p50={errs[len(errs)//2]:5.2f}  p90={errs[int(len(errs)*0.9)]:5.2f}  n={len(errs)}",
                  file=sys.stderr)


def emit_rust(scum, tables, anchors, targets, out):
    def fmt_row(row):
        return "[" + ", ".join(f"{v:.3f}" for v in row) + "]"
    lines = []
    lines.append("// AUTO-GENERATED by zwr-calibrate/fit_calibration.py — do not hand-edit.")
    lines.append(f"// Anchors (source eff-q): {anchors}")
    lines.append(f"// Targets (reencode q): {targets}")
    lines.append("")
    anchors_lit = ", ".join(f"{float(a):.1f}" for a in anchors)
    targets_lit = ", ".join(str(int(t)) for t in targets)
    lines.append(f"pub const SOURCE_Q_ANCHORS: &[f32] = &[{anchors_lit}];")
    lines.append(f"pub const TARGET_Q_GRID: &[u8] = &[{targets_lit}];")
    lines.append("")
    for cls in CLASSES:
        ratio, cum, _ = tables[cls]
        C = cls.upper()
        lines.append(f"pub const {C}_RATIO: [[f32; {len(targets)}]; {len(anchors)}] = [")
        for row in ratio: lines.append(f"    {fmt_row(row)},")
        lines.append("];")
        lines.append(f"pub const {C}_CUM: [[f32; {len(targets)}]; {len(anchors)}] = [")
        for row in cum: lines.append(f"    {fmt_row(row)},")
        lines.append("];")
        a, b, n = scum[cls]
        lines.append(f"// {cls} source_cum = {a:.4f}*q + {b:.3f}  (n={n})")
        lines.append(f"pub const {C}_SOURCE_CUM: (f32, f32) = ({a:.4f}, {b:.3f});")
        lines.append("")
    text = "\n".join(lines)
    if out:
        open(out, "w").write(text)
        print(f"\nwrote {out}", file=sys.stderr)
    else:
        print(text)


def main():
    args = parse_args()
    anchors = [int(x) for x in args.anchors.split(",")]
    targets = parse_targets(args.targets)
    train = load(args.train, args.key)
    val = load(args.val, args.key) if args.val else []
    print(f"loaded {len(train)} train rows, {len(val)} val rows (key={args.key})", file=sys.stderr)
    for cls in CLASSES:
        n = sum(1 for r in train if r["cls"] == cls)
        print(f"   {cls:8s} {n} train rows", file=sys.stderr)
    scum = fit_source_cum(train)
    tables = fit_grid(train, anchors, targets)
    for cls in CLASSES:
        ratio, cum, _ = tables[cls]
        fill_nans(ratio); fill_nans(cum); enforce_monotone_cum(cum)
    size_dependence(train, anchors)
    validate(val, scum, tables, anchors, targets)
    emit_rust(scum, tables, anchors, targets, args.out)


if __name__ == "__main__":
    main()
