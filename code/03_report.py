# -*- coding: utf-8 -*-
"""
Step 3 of 3 – aggregate the probe outputs into one JSON summary (plus figures).

Reads what the first two steps produced and writes to ``{out}/``:

    summary.json                        the single result file of this run
    per_layer_test_<level>.csv          ACC / F1 / AUROC of *every* layer on the test split
    subtype_auroc_<level>.csv           probe AUROC per bias sub-type
    subtype_auroc_<level>.png           the same as a bar chart
    layer_curve_<level>.png             validation & test curves with l* marked

``summary.json`` layout::

    {
      "generated_at": "...", "config": "...", "backbone": "...",
      "features": {"pooling": "last", "max_length": 1536, "input_mode": "task"},
      "runs": [                                # one flat row per level
        {"level": "dialogue", "method": "linear_probe", "layer": 31, "n_test": 928,
         "accuracy": ..., "f1": ..., "auroc": ..., "accuracy_ci95": [...], ...}
      ],
      "levels": {
        "dialogue": {
          "level_name": "Dialogue level",
          "split_sizes": {"train": 7752, "val": 1012, "test": 928},
          "layer_selection": {"criterion": "val_acc", "best_layer": 31, ...},
          "probe":          {"accuracy": ..., "confusion_matrix": {...}, ...},
          "probe_hparams":  {...},
          "per_layer":      [{"layer": .., "val_acc": .., "test_auroc": ..}, ...],
          "subtype_auroc":  [{"subtype": .., "n_pos": .., "auroc": ..}, ...]
        },
        "sentence": {...}
      }
    }

The sub-type breakdown uses each positive sample's own bias category
(``error_family`` / ``contradiction_type`` at dialogue level,
``bias_family`` / ``ambiguity_type`` / ``intensity_type`` at sentence level)
against all clean negatives.

Examples
--------
    python 03_report.py
    python 03_report.py --levels dialogue
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import probe_utils as U  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate the probe results into summary.json")
    ap.add_argument("--config", default=os.path.join(_HERE, "probe_config.json"))
    ap.add_argument("--levels", nargs="+", default=None)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="read/write the smoke output directory (config.smoke_out_dir)")
    return ap.parse_args()


# -------------------------------------------------------------------------
# Romanised sub-type names for the figures only: ``summary.json`` keeps the
# original names, while charts use ASCII labels because the default matplotlib
# fonts have no CJK glyphs and would render empty boxes.
PLOT_LABELS = {
    "isolated_entity": "isolated entity",
    "contradiction:symptom_yes_no": "contradiction:\nsymptom yes/no",
    "contradiction:time_inconsistent": "contradiction:\ntime",
    "contradiction:severity_frequency_inconsistent": "contradiction:\nseverity/frequency",
    "ambiguity:语义模糊": "ambiguity:\nvague wording",
    "ambiguity:逻辑不一致": "ambiguity:\nlogical conflict",
    "intensity:症状表达强度": "intensity:\nsymptom intensity",
    "intensity:证据呈现偏向": "intensity:\nbiased evidence",
    "intensity:推断合理化": "intensity:\nunjustified inference",
    "intensity:外部来源触发": "intensity:\nexternal trigger",
}


def plot_label(subtype: str) -> str:
    return PLOT_LABELS.get(subtype, subtype.replace(":", ":\n"))


def subtype_table(cfg: Dict[str, Any], level: str, fdir: str) -> List[Dict[str, Any]]:
    """Probe AUROC per bias sub-type (each sub-type's positives vs all negatives)."""
    score_path = os.path.join(fdir, "test_scores.npz")
    if not os.path.exists(score_path):
        return []
    with np.load(score_path, allow_pickle=True) as z:
        sids = [str(s) for s in z["sample_ids"]]
        y = z["labels"].astype(np.int64)
        p = z["probs"].astype(np.float64)
    row_by_id: Dict[str, Dict[str, Any]] = {}
    for r in U.read_jsonl(U.data_path(cfg, level, "test")):
        row_by_id[U.sample_key(r)] = r

    neg = y == 0
    groups: Dict[str, List[int]] = {}
    for i, sid in enumerate(sids):
        if y[i] != 1:
            continue
        r = row_by_id.get(sid)
        key = U.bias_subtype(cfg, level, r) if r else "unknown"
        groups.setdefault(key, []).append(i)

    out = []
    for key in sorted(groups):
        idx = np.array(groups[key] + list(np.where(neg)[0]), dtype=np.int64)
        auc = U.auroc(y[idx], p[idx])
        out.append({"subtype": key, "n_pos": len(groups[key]),
                    "n_neg": int(neg.sum()), "auroc": auc})
    return out


def plot_subtypes(rows: List[Dict[str, Any]], png: str, title: str) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["subtype"] for r in rows]
        vals = [r["auroc"] if r["auroc"] is not None else 0.0 for r in rows]
        fig, ax = plt.subplots(figsize=(max(5.2, 1.5 * len(rows) + 2.2), 4.0))
        ax.bar(range(len(vals)), vals, color="#55A868")
        ax.axhline(0.5, color="#888888", ls="--", lw=1)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{plot_label(n)}\n(n={r['n_pos']})" for n, r in zip(names, rows)],
                           fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probe AUROC")
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(png, dpi=200)
        plt.close(fig)
    except Exception as e:                                            # pragma: no cover
        print(f"[warn] could not draw {png}: {e}")


def plot_curve(rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]], best: int,
               png: str, title: str) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        layers = [r["layer"] for r in rows]
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.plot(layers, [r["val_acc"] for r in rows], marker="o", ms=3.5,
                color="#3B6FB6", label="ACC (val)")
        if any(r.get("val_auroc") is not None for r in rows):
            ax.plot(layers, [r.get("val_auroc") for r in rows], marker="^", ms=3.5,
                    color="#C0392B", label="AUROC (val)")
        if test_rows:
            tl = [r["layer"] for r in test_rows]
            ax.plot(tl, [r["test_acc"] for r in test_rows], marker="o", ms=3, ls="--",
                    color="#7FA8D9", label="ACC (test)")
            if any(r.get("test_auroc") is not None for r in test_rows):
                ax.plot(tl, [r.get("test_auroc") for r in test_rows], marker="^", ms=3, ls="--",
                        color="#E08A7F", label="AUROC (test)")
        ax.axvline(best, color="#444444", ls="--", lw=1, alpha=0.7)
        ax.annotate(f"l*={best}", xy=(best, 0.5), xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color="#444444")
        ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Bias detection score")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8.5, ncol=2)
        fig.tight_layout()
        fig.savefig(png, dpi=200)
        plt.close(fig)
    except Exception as e:                                            # pragma: no cover
        print(f"[warn] could not draw {png}: {e}")


def merge_per_layer(rows: List[Dict[str, Any]],
                    test_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One entry per layer with the validation and test metrics side by side."""
    test_by_layer = {int(r["layer"]): r for r in (test_rows or [])}
    out = []
    for r in rows or []:
        layer = int(r["layer"])
        t = test_by_layer.get(layer, {})
        out.append({
            "layer": layer,
            "feature_index": r.get("feature_index", layer),
            "val_acc": r.get("val_acc"), "val_precision": r.get("val_precision"),
            "val_recall": r.get("val_recall"), "val_f1": r.get("val_f1"),
            "val_auroc": r.get("val_auroc"),
            "test_acc": t.get("test_acc"), "test_f1": t.get("test_f1"),
            "test_auroc": t.get("test_auroc"),
        })
    return out


# -------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = U.load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    elif args.smoke:
        cfg["out_dir"] = cfg.get("smoke_out_dir", cfg["out_dir"] + "_smoke")
    levels = args.levels or U.levels_of(cfg)

    summary: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg["_config_path"],
        "out_dir": os.path.abspath(cfg["out_dir"]),
        "levels": {},
        "backbone": None,
        "features": {},
        "runs": [],
    }

    for level in levels:
        fdir = U.out_dir(cfg, level)
        probe_path = os.path.join(fdir, "probe.json")
        if not os.path.exists(probe_path):
            print(f"[skip] {level}: {probe_path} not found (run 02_probe.py first)")
            continue
        res = U.load_json(probe_path)
        level_name = res.get("level_name", level)
        data_meta = res.get("data") or {}
        test = res.get("test") or {}
        summary["backbone"] = summary["backbone"] or res.get("model_features")
        if not summary["features"]:
            summary["features"] = {k: data_meta.get(k)
                                   for k in ("pooling", "max_length", "input_mode")}

        sub = subtype_table(cfg, level, fdir)
        test_rows = res.get("per_layer_test") or []
        per_layer = merge_per_layer(res.get("per_layer") or [], test_rows)

        entry = {
            "level_name": level_name,
            "backbone": res.get("model_features"),
            "probe_json": probe_path,
            "split_sizes": {"train": data_meta.get("n_train"), "val": data_meta.get("n_val"),
                            "test": data_meta.get("n_test")},
            "layer_selection": {
                "criterion": res.get("select_by"),
                "best_layer": res.get("best_layer"),
                "best_layer_feature_index": res.get("best_layer_feature_index"),
                "best_layer_val_acc": res.get("best_layer_val_acc"),
                "best_layer_val_auroc": res.get("best_layer_val_auroc"),
                "mean_val_acc_over_layers": res.get("mean_val_acc_over_layers"),
                "gain_over_mean_val_acc": res.get("gain_over_mean_val_acc"),
            },
            # the selected probe, evaluated once on the held-out test split
            "probe": res.get("test"),
            "probe_hparams": res.get("probe_hparams"),
            "per_layer": per_layer,
            "subtype_auroc": sub,
        }
        summary["levels"][level] = entry
        summary["runs"].append({
            "level": level, "level_name": level_name,
            "method": "linear_probe", "layer": res.get("best_layer"),
            "n_test": test.get("n"),
            "accuracy": test.get("accuracy"),
            "precision": test.get("precision"),
            "recall": test.get("recall"),
            "f1": test.get("f1"),
            "auroc": test.get("auroc"),
            "accuracy_ci95": test.get("accuracy_ci95"),
            "f1_ci95": test.get("f1_ci95"),
            "auroc_ci95": test.get("auroc_ci95"),
        })

        U.save_json(os.path.join(cfg["out_dir"], f"summary_{level}.json"), entry)

        # ---- per-layer test csv ----
        if test_rows:
            with open(os.path.join(cfg["out_dir"], f"per_layer_test_{level}.csv"), "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["layer", "test_acc", "test_f1", "test_auroc"])
                w.writeheader()
                for r in test_rows:
                    w.writerow({k: r.get(k) for k in w.fieldnames})

        # ---- sub-type csv + png ----
        if sub:
            with open(os.path.join(cfg["out_dir"], f"subtype_auroc_{level}.csv"), "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["subtype", "n_pos", "n_neg", "auroc"])
                w.writeheader()
                for r in sub:
                    w.writerow(r)
            plot_subtypes(sub, os.path.join(cfg["out_dir"], f"subtype_auroc_{level}.png"),
                          f"{level_name}: probe AUROC by bias sub-type")

        plot_curve(res.get("per_layer") or [], test_rows, int(res.get("best_layer") or 0),
                   os.path.join(cfg["out_dir"], f"layer_curve_{level}.png"),
                   f"{level_name}: layer-wise bias detection")

        print(f"  {level_name}: l*={res.get('best_layer')} n_test={test.get('n')} "
              f"ACC={test.get('accuracy')} F1={test.get('f1')} AUROC={test.get('auroc')}")

    json_path = os.path.join(cfg["out_dir"], "summary.json")
    U.save_json(json_path, summary)
    print(f"[ok] wrote {json_path}")


if __name__ == "__main__":
    main()
