# -*- coding: utf-8 -*-
"""
Aggregate the probe outputs into tables and figures.

Reads everything the first three steps produced and writes to ``{out}/``:

    summary.md / summary.csv            the paper's Table 1 block for this run
    per_layer_test_<level>.csv          ACC / F1 / AUROC of *every* layer on the test split
    subtype_auroc_<level>.csv           probe AUROC per bias sub-type
    subtype_auroc_<level>.png           the same as a bar chart
    layer_curve_<level>.png             validation & test curves with l* marked

The sub-type breakdown uses each positive sample's own bias category
(``error_family`` / ``contradiction_type`` at dialogue level,
``bias_family`` / ``ambiguity_type`` / ``intensity_type`` at sentence level)
against all clean negatives, i.e. the fine-grained analysis mentioned in the
paper (Sec. 3.2, "characterise different bias categories").

Examples
--------
    python 04_report.py
    python 04_report.py --levels dialogue
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import probe_utils as U  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate the probe results")
    ap.add_argument("--config", default=os.path.join(_HERE, "probe_config.json"))
    ap.add_argument("--levels", nargs="+", default=None)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="read/write the smoke output directory (config.smoke_out_dir)")
    return ap.parse_args()


# -------------------------------------------------------------------------
# Romanised sub-type names for the figures.  The CSV/Markdown tables keep the
# original Chinese sub-type names; charts use ASCII labels only, because the
# default matplotlib fonts have no CJK glyphs and would render empty boxes.
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


def fnum(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ci(pair: Optional[List[float]], nd: int = 3) -> str:
    if not pair:
        return "—"
    return f"[{pair[0]:.{nd}f},{pair[1]:.{nd}f}]"


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


# -------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = U.load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    elif args.smoke:
        cfg["out_dir"] = cfg.get("smoke_out_dir", cfg["out_dir"] + "_smoke")
    levels = args.levels or U.levels_of(cfg)

    summary: Dict[str, Any] = {"levels": {}, "backbone": None, "settings": {}}
    md: List[str] = ["# Patient reporting bias detection", ""]

    for level in levels:
        fdir = U.out_dir(cfg, level)
        probe_path = os.path.join(fdir, "probe.json")
        if not os.path.exists(probe_path):
            print(f"[skip] {level}: {probe_path} not found (run 02_probe.py first)")
            continue
        res = U.load_json(probe_path)
        level_name = res.get("level_name", level)
        summary["backbone"] = summary["backbone"] or res.get("model_features")
        if not summary["settings"] and res.get("data"):
            summary["settings"] = res["data"]

        prompt: Dict[str, Any] = {}
        for split, fname in (("test", "prompt_test.json"), ("test", "prompt.json")):
            p = os.path.join(fdir, fname)
            if os.path.exists(p):
                prompt = U.load_json(p)
                break

        sub = subtype_table(cfg, level, fdir)
        test_rows = res.get("per_layer_test") or []

        summary["levels"][level] = {
            "level_name": level_name,
            "backbone": res.get("model_features"),
            "n_train": res["data"]["n_train"], "n_val": res["data"]["n_val"],
            "n_test": res["data"]["n_test"],
            "select_by": res.get("select_by"),
            "best_layer": res.get("best_layer"),
            "best_layer_val_acc": res.get("best_layer_val_acc"),
            "best_layer_val_auroc": res.get("best_layer_val_auroc"),
            "mean_val_acc_over_layers": res.get("mean_val_acc_over_layers"),
            "gain_over_mean_val_acc": res.get("gain_over_mean_val_acc"),
            "probe_test": res.get("test"),
            "prompt_test_generate": (prompt.get("test_generate") if prompt else None),
            "prompt_test_score": (prompt.get("test_score") if prompt else None),
            "prompt_n_parse_fail": (prompt.get("n_parse_fail") if prompt else None),
            "per_layer_test": test_rows,
            "subtype_auroc": sub,
        }
        U.save_json(os.path.join(cfg["out_dir"], f"summary_{level}.json"),
                    summary["levels"][level])

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

        # ---- markdown block ----
        t = res["test"]
        md += [f"## {level_name}", "",
               f"- test samples: **{res['data']['n_test']}** "
               f"(train {res['data']['n_train']} / val {res['data']['n_val']}); "
               f"backbone `{os.path.basename(str(res.get('model_features')))}`; "
               f"pooling `{res['data'].get('pooling')}`, max_length `{res['data'].get('max_length')}`",
               f"- selected layer **l\\* = {res.get('best_layer')}** "
               f"(by `{res.get('select_by')}`; val ACC {fnum(res.get('best_layer_val_acc'))}, "
               f"val AUROC {fnum(res.get('best_layer_val_auroc'))})",
               f"- mean validation ACC over all layers: "
               f"{fnum(res.get('mean_val_acc_over_layers'))} "
               f"→ gain of l\\*: {fnum(res.get('gain_over_mean_val_acc'))}",
               "", "| Method | ACC | Precision | Recall | F1 | AUROC |", "|---|---|---|---|---|---|",
               f"| Linear probe (ours, l\\*={res.get('best_layer')}) | {fnum(t['accuracy'])} | "
               f"{fnum(t['precision'])} | {fnum(t['recall'])} | {fnum(t['f1'])} | "
               f"{fnum(t.get('auroc'))} |"]
        if prompt and prompt.get("test_generate"):
            g = prompt["test_generate"]
            md.append(f"| Prompt classifier (zero-shot) | {fnum(g['accuracy'])} | "
                      f"{fnum(g['precision'])} | {fnum(g['recall'])} | {fnum(g['f1'])} | "
                      f"{fnum(g.get('auroc'))} |")
        if prompt and prompt.get("test_score"):
            s = prompt["test_score"]
            md.append(f"| Prompt classifier (logit score) | {fnum(s['accuracy'])} | "
                      f"{fnum(s['precision'])} | {fnum(s['recall'])} | {fnum(s['f1'])} | "
                      f"{fnum(s.get('auroc'))} |")
        md += ["",
               f"95% bootstrap CIs (1000 resamples): ACC {ci(t.get('accuracy_ci95'))}, "
               f"F1 {ci(t.get('f1_ci95'))}, AUROC {ci(t.get('auroc_ci95'))}. "
               f"Confusion matrix: {t['confusion_matrix']}."]
        if prompt:
            md.append(f"Prompt baseline JSON parse failures: {prompt.get('n_parse_fail')}.")
        if sub:
            md += ["", "Probe AUROC per bias sub-type (sub-type positives vs all clean negatives):",
                   "", "| Sub-type | #pos | AUROC |", "|---|---|---|"]
            for r in sub:
                md.append(f"| {r['subtype']} | {r['n_pos']} | {fnum(r['auroc'])} |")
        md.append("")

    # ---- global csv/markdown ----
    keys = ["level", "method", "layer", "n_test", "acc", "acc_ci", "precision", "recall", "f1",
            "f1_ci", "auroc", "auroc_ci"]
    csv_path = os.path.join(cfg["out_dir"], "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for level, d in summary["levels"].items():
            for method, m, layer in (("linear_probe", d["probe_test"], d["best_layer"]),
                                     ("prompt_generate", d["prompt_test_generate"], ""),
                                     ("prompt_score", d["prompt_test_score"], "")):
                if not m:
                    continue
                w.writerow([d["level_name"], method, layer, m.get("n"),
                            m.get("accuracy"), ci(m.get("accuracy_ci95")),
                            m.get("precision"), m.get("recall"), m.get("f1"),
                            ci(m.get("f1_ci95")), m.get("auroc"), ci(m.get("auroc_ci95"))])

    header = ["# Patient reporting bias detection (auto-generated)", "",
              f"Backbone: `{summary['backbone']}`  ",
              f"Features: pooling `{summary['settings'].get('pooling')}`, "
              f"max_length `{summary['settings'].get('max_length')}`, "
              f"input mode `{summary['settings'].get('input_mode')}`  ",
              "Metrics on the held-out test split. The probe probability is the "
              "sigmoid output at threshold 0.5; the prompt-classifier AUROC comes from the "
              "0/1 logit score (threshold-free).", ""]
    with open(os.path.join(cfg["out_dir"], "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(header + md))
    U.save_json(os.path.join(cfg["out_dir"], "summary_all.json"), summary)

    print(f"[ok] wrote {os.path.join(cfg['out_dir'], 'summary.md')} and summary.csv")
    for level, d in summary["levels"].items():
        t = d["probe_test"]
        print(f"  {d['level_name']:16s} l*={d['best_layer']:<3} ACC={fnum(t['accuracy'])} "
              f"F1={fnum(t['f1'])} AUROC={fnum(t.get('auroc'))}")


if __name__ == "__main__":
    main()
