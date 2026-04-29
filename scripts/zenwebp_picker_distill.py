#!/usr/bin/env python3
"""
Distill HistGradientBoosting teachers into a small shared MLP for the
zenwebp picker spike.

This is the Phase 3 step of the spike. Reads the two TSVs produced by
`dev/zenwebp_picker_sweep.rs`, fits one HistGB regressor per cell
predicting log-bytes from the simple feature vector, then trains a
single MLP on the soft targets.

Schema (must match `src/encoder/picker/spec.rs::FEAT_COLS` order):

  raw zenanalyze: 14 columns, named `feat_*`
  + log_pixels, log_pixels^2
  + target_zensim_norm, ^2, * log_pixels
  + target_zensim_norm * each feat_*
  + icc_bytes (placeholder, always 0)

Output: JSON consumable by `~/work/zen/zenanalyze/tools/bake_picker.py`.

Usage:
  python3 scripts/zenwebp_picker_distill.py \\
      --pareto   /mnt/v/output/zenwebp/picker-sweep/pareto_<TS>.tsv \\
      --features /mnt/v/output/zenwebp/picker-sweep/features_<TS>.tsv \\
      --out      benchmarks/zenwebp_picker_distill_<DATE>.json
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Must match `spec.rs::FEAT_COLS` exactly.
FEAT_COLS = [
    "feat_screen_content",
    "feat_text_likelihood",
    "feat_natural_likelihood",
    "feat_flat_color_block_ratio",
    "feat_distinct_color_bins",
    "feat_variance",
    "feat_edge_density",
    "feat_uniformity",
    "feat_high_freq_energy_ratio",
    "feat_palette_fits_in_256",
    "feat_indexed_palette_width",
    "feat_line_art_score",
    "feat_skin_tone_fraction",
    "feat_edge_slope_stdev",
]
N_FEATS = len(FEAT_COLS)
SEED = 0xC0DE
HOLDOUT_FRAC = 0.20


def load_pareto(path):
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                rows.append({
                    "image_path": r["image_path"],
                    "corpus": r["corpus"],
                    "size_class": r["size_class"],
                    "width": int(r["width"]),
                    "height": int(r["height"]),
                    "cell_idx": int(r["cell_idx"]),
                    "cell_name": r["cell_name"],
                    "target_zensim": float(r["target_zensim"]),
                    "bytes": int(r["bytes"]),
                    "achieved_zensim": float(r["achieved_zensim"]),
                    "passes_used": int(r["passes_used"]),
                    "targets_met": int(r["targets_met"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_features(path):
    feats = {}
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            feats[r["image_path"]] = np.array(
                [float(r[c]) for c in FEAT_COLS], dtype=np.float32
            )
    return feats


def n_configs_from_pareto(pareto):
    return max(r["cell_idx"] for r in pareto) + 1


SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}


def build_dataset(pareto, feats):
    """
    For each (image, target_zensim) decision row, build:
      X_simple:     [feats..., size_onehot(4), log_pixels, target_z_norm]
                                                                       (N_FEATS+4+2 = 20)
      X_engineered: [feats..., size_onehot(4),
                     log_pixels, log_pixels^2,
                     target_z_norm, target_z_norm^2,
                     target_z_norm*log_pixels,
                     target_z_norm*feats[i] for i,
                     icc_bytes(=0)]                                    (N_FEATS+4+5+N_FEATS+1 = 38)
      Y: log(bytes) for each cell, NaN if cell ran out of attempts.

    Layout matches bake_picker.py's derive_extra_axes() expectation:
      n_inputs = N_FEATS + 4 (size_oh) + 5 (poly) + N_FEATS (cross) + 1 (icc).
    """
    n_configs = n_configs_from_pareto(pareto)
    by_key = defaultdict(dict)  # (image_path, target) -> cell_idx -> row
    for r in pareto:
        key = (r["image_path"], r["target_zensim"])
        by_key[key][r["cell_idx"]] = r

    Xs_rows, Xe_rows, Y_rows, meta = [], [], [], []
    for (image_path, target), cells in by_key.items():
        if image_path not in feats:
            continue
        any_row = next(iter(cells.values()))
        w, h = any_row["width"], any_row["height"]
        size_class = any_row["size_class"]
        f = feats[image_path]
        log_px = math.log(max(1, w * h))
        target_norm = target / 100.0
        size_oh = np.zeros(len(SIZE_CLASSES), dtype=np.float32)
        if size_class in SIZE_INDEX:
            size_oh[SIZE_INDEX[size_class]] = 1.0

        xs = np.concatenate([
            f, size_oh,
            np.array([log_px, target_norm], dtype=np.float32),
        ])
        xe = np.concatenate([
            f, size_oh,
            np.array([
                log_px, log_px * log_px,
                target_norm, target_norm * target_norm,
                target_norm * log_px,
            ], dtype=np.float32),
            target_norm * f,
            np.array([0.0], dtype=np.float32),  # icc_bytes placeholder
        ])
        y = np.full(n_configs, np.nan, dtype=np.float32)
        for cell_idx, row in cells.items():
            # "Feasible" for this target if the cell hit (or overshot).
            # We still record the bytes even on undershoot — the picker
            # uses the achieved score later; we don't want to discard
            # signal.
            if row["bytes"] > 0:
                y[cell_idx] = math.log(row["bytes"])
        if not np.any(~np.isnan(y)):
            continue
        Xs_rows.append(xs)
        Xe_rows.append(xe)
        Y_rows.append(y)
        meta.append((image_path, target, w, h))
    return np.stack(Xs_rows), np.stack(Xe_rows), np.stack(Y_rows), meta, n_configs


def evaluate_argmin(Y_pred_log, Y_actual_log, name):
    n = Y_pred_log.shape[0]
    overheads, correct, unreach = [], 0, 0
    for i in range(n):
        actual = Y_actual_log[i]
        pred = Y_pred_log[i]
        m = ~np.isnan(actual)
        if not np.any(m):
            unreach += 1
            continue
        ab = np.where(m, np.exp(actual), np.inf)
        pb = np.where(m, np.exp(np.clip(pred, -30, 30)), np.inf)
        actual_best = int(np.argmin(ab))
        pred_best = int(np.argmin(pb))
        if pred_best == actual_best:
            correct += 1
        ov = (ab[pred_best] - ab[actual_best]) / ab[actual_best]
        overheads.append(ov)
    if not overheads:
        return None
    overheads = np.array(overheads)
    return {
        "name": name,
        "n": int(len(overheads)),
        "unreachable": unreach,
        "argmin_accuracy": correct / len(overheads),
        "overhead_mean_pct": float(100 * np.mean(overheads)),
        "overhead_p50_pct": float(100 * np.percentile(overheads, 50)),
        "overhead_p75_pct": float(100 * np.percentile(overheads, 75)),
        "overhead_p90_pct": float(100 * np.percentile(overheads, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config-names", default=None,
                    help="optional path to write a cell_idx -> name JSON")
    args = ap.parse_args()

    sys.stderr.write(f"loading pareto: {args.pareto}\n")
    pareto = load_pareto(args.pareto)
    feats = load_features(args.features)
    sys.stderr.write(f"  pareto rows: {len(pareto)}, feature rows: {len(feats)}\n")

    Xs, Xe, Y, meta, n_configs = build_dataset(pareto, feats)
    sys.stderr.write(
        f"  decision rows: {len(Xs)} × {n_configs} cells "
        f"(Xs:{Xs.shape[1]}, Xe:{Xe.shape[1]})\n"
    )

    # Per-cell name table for the manifest.
    cell_names = {}
    for r in pareto:
        cell_names.setdefault(r["cell_idx"], r["cell_name"])

    rng = np.random.default_rng(SEED)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * HOLDOUT_FRAC))
    val_set = set(images[:n_val])
    train_idx = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    val_idx = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(f"  train rows: {len(train_idx)}, val rows: {len(val_idx)}\n")

    Xs_tr, Xs_va = Xs[train_idx], Xs[val_idx]
    Xe_tr, Xe_va = Xe[train_idx], Xe[val_idx]
    Y_tr, Y_va = Y[train_idx], Y[val_idx]

    # ------ Step 1: HistGB teacher per-cell ------
    sys.stderr.write(f"\nfitting {n_configs} HistGB teachers...\n")
    teachers = []
    cfg_means = np.nanmean(Y_tr, axis=0)
    for c in range(n_configs):
        mask = ~np.isnan(Y_tr[:, c])
        if mask.sum() < 20:
            teachers.append(None)
            continue
        gbm = HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, learning_rate=0.05,
            l2_regularization=0.5, random_state=SEED,
        )
        gbm.fit(Xs_tr[mask], Y_tr[mask, c])
        teachers.append(gbm)

    # ------ Step 2: dense soft targets ------
    sys.stderr.write("generating soft targets...\n")
    soft_tr = np.zeros((len(train_idx), n_configs), dtype=np.float32)
    soft_va = np.zeros((len(val_idx), n_configs), dtype=np.float32)
    for c in range(n_configs):
        if teachers[c] is None:
            soft_tr[:, c] = cfg_means[c] if not np.isnan(cfg_means[c]) else 0.0
            soft_va[:, c] = cfg_means[c] if not np.isnan(cfg_means[c]) else 0.0
        else:
            soft_tr[:, c] = teachers[c].predict(Xs_tr)
            soft_va[:, c] = teachers[c].predict(Xs_va)

    t_metrics = evaluate_argmin(soft_va, Y_va, "HistGB teacher")
    sys.stderr.write(f"  teacher: {t_metrics}\n")

    # ------ Step 3: MLP student ------
    sys.stderr.write("\ntraining MLP student...\n")
    scaler = StandardScaler()
    Xe_tr_s = scaler.fit_transform(Xe_tr)
    Xe_va_s = scaler.transform(Xe_va)
    student = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=2e-3,
        batch_size=128,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        tol=1e-6,
        random_state=SEED,
        verbose=False,
    )
    student.fit(Xe_tr_s, soft_tr)
    sys.stderr.write(f"  fit, loss={student.loss_:.4f}, n_iter={student.n_iter_}\n")

    Y_va_pred = student.predict(Xe_va_s)
    s_metrics = evaluate_argmin(Y_va_pred, Y_va, "MLP student")
    sys.stderr.write(f"  student: {s_metrics}\n")

    # Bucket-table baseline: for each row, evaluate where the bucket
    # table would have sent us. (Photo->cell with sns=80 f=30 seg=4
    # = sns_idx=3, f=0, seg=1 -> idx = 3*4+0*2+1 = 13 in our flat
    # layout. Drawing/Default -> sns=50 f=60 seg=4 -> sns_idx=2, f=1,
    # seg=1 -> idx = 2*4+1*2+1 = 11. Icon -> sns=0 f=0 seg=4 (we
    # don't have f=0 in the grid; nearest is f=30 seg=4 -> idx
    # = 0*4+0*2+1 = 1.) For headline numbers we compare picker vs
    # picking THE Drawing default — that's what the existing
    # auto-preset path picks for ~80% of CID22. Caller can re-derive
    # this from the manifest later.
    BUCKET_DRAWING_IDX = 11  # sns=50, f=60, seg=4
    BUCKET_PHOTO_IDX = 13    # sns=80, f=30, seg=4
    BUCKET_ICON_IDX = 1      # sns=0,  f=30, seg=4 (closest to Icon)
    # Use the picker's own argmin-best as oracle, bucket-default as baseline.
    overheads_drawing = []
    overheads_oracle_diff = []
    for i in range(len(val_idx)):
        actual = Y_va[i]
        m = ~np.isnan(actual)
        if not np.any(m):
            continue
        ab = np.where(m, np.exp(actual), np.inf)
        oracle_idx = int(np.argmin(ab))
        if not m[BUCKET_DRAWING_IDX]:
            continue
        ov = (ab[BUCKET_DRAWING_IDX] - ab[oracle_idx]) / ab[oracle_idx]
        overheads_drawing.append(ov)
    if overheads_drawing:
        a = np.array(overheads_drawing)
        sys.stderr.write(
            f"  bucket(Drawing)={'%.1f%%' % (100*a.mean())} mean over oracle "
            f"(p50 {100*np.percentile(a, 50):.1f}%, p90 {100*np.percentile(a, 90):.1f}%)\n"
        )

    weights = {
        "n_inputs": int(Xe.shape[1]),
        "n_configs": int(n_configs),
        "config_names": {int(k): v for k, v in cell_names.items()},
        # `feat_cols` lists ONLY the raw zenanalyze features. The
        # bake script's `derive_extra_axes()` rebuilds the full
        # engineered axis list from this + n_inputs and produces
        # the schema_hash. Don't include the engineered columns
        # here or the layout-detection check fires twice.
        "feat_cols": list(FEAT_COLS),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "layers": [
            {"W": w.tolist(), "b": b.tolist()}
            for w, b in zip(student.coefs_, student.intercepts_)
        ],
        "activation": "relu",
        "schema_version_tag": "zenwebp.picker.v0.1.spike",
        "metrics": {
            "teacher": t_metrics,
            "student": s_metrics,
            "bucket_drawing_overhead_pct": (
                float(100 * np.mean(overheads_drawing))
                if overheads_drawing else None
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(weights))
    n_params = sum(np.array(c).size + np.array(b).size
                   for c, b in zip(student.coefs_, student.intercepts_))
    sys.stderr.write(
        f"\nwrote {args.out} ({args.out.stat().st_size} bytes JSON, "
        f"{n_params} weights ~ {n_params * 4 / 1024:.1f} KB f32)\n"
    )


if __name__ == "__main__":
    main()
