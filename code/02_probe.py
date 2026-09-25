# -*- coding: utf-8 -*-
"""
Step 2 of 3 – linear bias probe (paper Sec. 2.4, Eq. 3-4 and Table 1).

For **every** layer l we train the linear probe

    p_hat^(l)(x) = sigmoid( w^(l)T r^(l)(x) + b^(l) )                       (Eq. 3)

with binary cross-entropy over reliable / biased narratives

    L^(l) = -y log p_hat - (1 - y) log(1 - p_hat)                          (Eq. 4)

and select the best layer ``l*`` on the validation split (default: accuracy).
The selected probe is then evaluated once on the held-out test split and
reported with ACC / Precision / Recall / F1 / AUROC and percentile bootstrap
95% confidence intervals.  Every per-layer validation score is kept as well, so
the layer-wise curve of paper Fig. 4(a) can be re-plotted.

Outputs (``{out}/{level}/``):

    probe.json    per-layer validation metrics, l*, test metrics (+CI), hyper-params
    per_layer.csv     the same per-layer table in CSV form
    probe_ckpt.pt     l* weights, standardisation statistics, threshold
    test_scores.npz   test-set probabilities of l*
    layer_curve_<level>.png   validation ACC / AUROC per layer, l* marked

Resume support: one JSONL line is appended per finished layer, so an interrupted
run continues where it stopped.  ``--overwrite`` retrains everything.

Examples
--------
    python 02_probe.py --levels dialogue sentence
    python 02_probe.py --levels sentence --select_by val_auroc
    python 02_probe.py --levels dialogue --layers 15,31   # refit selected layers
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import probe_utils as U  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="linear bias probe, one probe per layer")
    ap.add_argument("--config", default=os.path.join(_HERE, "probe_config.json"))
    ap.add_argument("--levels", nargs="+", default=None)
    ap.add_argument("--layers", default="", help="comma separated feature indices; default all")
    ap.add_argument("--lr", type=float, default=-1)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--weight_decay", type=float, default=-1)
    ap.add_argument("--no_standardize", action="store_true",
                    help="disable train-set z-scoring of the hidden states")
    ap.add_argument("--select_by", default="", choices=["", "val_acc", "val_auroc"],
                    help="layer-selection criterion (default from config: val_acc)")
    ap.add_argument("--device", default="cpu", help="probes are tiny: CPU is fine")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="read/write the smoke output directory (config.smoke_out_dir)")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


# -------------------------------------------------------------------------
# probing
# -------------------------------------------------------------------------
def standardize(train: np.ndarray, others: List[np.ndarray]):
    """z-score with **train** statistics only (no leakage into val/test)."""
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, [(x - mean) / std for x in others], mean, std


def train_probe(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray,
                lr: float, bs: int, epochs: int, wd: float, device: str, seed: int):
    """Fit the linear probe with AdamW + BCEWithLogits; return (model, val_probs)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = nn.Linear(xtr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    X = torch.from_numpy(np.ascontiguousarray(xtr, dtype=np.float32))
    Y = torch.from_numpy(np.asarray(ytr, dtype=np.float32)).unsqueeze(1)
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(X[idx].to(device)), Y[idx].to(device))
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(
            model(torch.from_numpy(np.ascontiguousarray(xva, dtype=np.float32)).to(device))
        ).cpu().numpy().ravel()
    return model, pv


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = U.load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    elif args.smoke:
        cfg["out_dir"] = cfg.get("smoke_out_dir", cfg["out_dir"] + "_smoke")
    pcfg = cfg["probe"]
    lr = args.lr if args.lr > 0 else float(pcfg["lr"])
    bs = args.batch_size or int(pcfg["batch_size"])
    epochs = args.epochs or int(pcfg["epochs"])
    wd = args.weight_decay if args.weight_decay >= 0 else float(pcfg["weight_decay"])
    threshold = float(pcfg.get("threshold", 0.5))
    select_by = args.select_by or str(pcfg.get("layer_selection", "val_acc"))
    standardize_on = (not args.no_standardize) and bool(pcfg.get("standardize", True))
    levels = args.levels or U.levels_of(cfg)
    seed = int(cfg["seed"])

    import torch

    for level in levels:
        fdir = U.out_dir(cfg, level)
        level_name = U.task_of(cfg, level)["display_name"]
        needed = [os.path.join(fdir, f"features_{sp}.npz") for sp in ("train", "val", "test")]
        miss = [p for p in needed if not os.path.exists(p)]
        if miss:
            print(f"\n[skip] {level_name}: missing {[os.path.basename(p) for p in miss]}; "
                  f"run 01_extract_features.py --levels {level} first")
            continue

        tr = U.load_features(needed[0])
        va = U.load_features(needed[1])
        te = U.load_features(needed[2])
        n_layers = tr["features"].shape[1]
        keep = (tr["meta"] or {}).get("layer_labels")
        layer_labels = [int(x) for x in keep] if isinstance(keep, list) and len(keep) == n_layers \
            else list(range(n_layers))
        feature_ids = sorted({int(x) for x in args.layers.split(",") if x.strip()}) if args.layers \
            else list(range(n_layers))

        print(f"\n===== bias probe | {level_name} | "
              f"train={tr['features'].shape} val={va['features'].shape} "
              f"test={te['features'].shape} =====")
        print(f"[features] model={tr['meta'].get('model_path')} "
              f"pooling={tr['meta'].get('pooling')} max_length={tr['meta'].get('max_length')} "
              f"truncated={tr['meta'].get('n_truncated')}")

        # ---- per-layer resume: one jsonl line per finished layer ----
        progress = os.path.join(fdir, f"per_layer_{level}.jsonl")
        if args.overwrite and os.path.exists(progress):
            os.remove(progress)
        done = {int(r["feature_index"]): r for r in U.read_jsonl(progress)}
        remaining = [li for li in feature_ids if li not in done]
        print(f"[resume] {len(done)} layers already trained, {len(remaining)} to go")

        for li in remaining:
            xtr, xva = tr["features"][:, li, :], va["features"][:, li, :]
            if standardize_on:
                xtr_s, (xva_s,), _, _ = standardize(xtr, [xva])
            else:
                xtr_s, xva_s = xtr, xva
            _, pv = train_probe(xtr_s, tr["labels"], xva_s, va["labels"], lr, bs, epochs, wd,
                                args.device, seed)
            m = U.binary_metrics(va["labels"], pv, threshold)
            row = {"layer": layer_labels[li], "feature_index": li,
                   "val_acc": m["accuracy"], "val_f1": m["f1"], "val_precision": m["precision"],
                   "val_recall": m["recall"], "val_auroc": U.auroc(va["labels"], pv)}
            U.append_jsonl_sync(progress, [row])
            print(f"  layer {layer_labels[li]:3d}  val ACC={m['accuracy']:.4f} "
                  f"F1={m['f1']:.4f} AUROC={row['val_auroc']}", flush=True)

        rows = sorted(({int(r["feature_index"]): r for r in U.read_jsonl(progress)}).values(),
                      key=lambda r: r["feature_index"])
        missing_layers = [li for li in feature_ids if li not in {r["feature_index"] for r in rows}]
        if missing_layers:
            print(f"[partial] {len(missing_layers)} layers still missing – re-run to resume")
            continue

        # ---- select l* on the validation split, then refit that layer for the test run ----
        def _key(metric: str):
            if metric == "val_auroc":
                return lambda r: (r.get("val_auroc") if r.get("val_auroc") is not None else -1.0,
                                  -r["feature_index"])
            return lambda r: (r["val_acc"], -r["feature_index"])       # ties -> earlier layer

        best = max(rows, key=_key(select_by))
        li, l_star = int(best["feature_index"]), int(best["layer"])
        xtr, xva = tr["features"][:, li, :], va["features"][:, li, :]
        if standardize_on:
            xtr_s, (xva_s,), mean, std = standardize(xtr, [xva])
        else:
            mean = np.zeros((1, xtr.shape[1]), dtype=np.float32)
            std = np.ones((1, xtr.shape[1]), dtype=np.float32)
            xtr_s, xva_s = xtr, xva
        model, _ = train_probe(xtr_s, tr["labels"], xva_s, va["labels"], lr, bs, epochs, wd,
                               args.device, seed)
        model.eval()
        xt = (te["features"][:, li, :] - mean) / std
        with torch.no_grad():
            pt = torch.sigmoid(
                model(torch.from_numpy(np.ascontiguousarray(xt, dtype=np.float32)).to(args.device))
            ).cpu().numpy().ravel()

        test = U.binary_metrics(te["labels"], pt, threshold)
        test["auroc"] = U.auroc(te["labels"], pt)
        test["accuracy_ci95"] = U.bootstrap_ci(te["labels"], pt, "accuracy", threshold)
        test["precision_ci95"] = U.bootstrap_ci(te["labels"], pt, "precision", threshold)
        test["recall_ci95"] = U.bootstrap_ci(te["labels"], pt, "recall", threshold)
        test["f1_ci95"] = U.bootstrap_ci(te["labels"], pt, "f1", threshold)
        test["auroc_ci95"] = U.bootstrap_ci(te["labels"], pt, "auroc", threshold)
        U.print_metrics(f"bias probe {level_name} (layer {l_star})", test)

        # ---- layer-wise test metrics (needed for the paper's Fig. 4(a)-style curve) ----
        per_layer_test = []
        for r in rows:
            li2 = int(r["feature_index"])
            xa, xb = tr["features"][:, li2, :], te["features"][:, li2, :]
            if standardize_on:
                xa_s, (xb_s,), _, _ = standardize(xa, [xb])
            else:
                xa_s, xb_s = xa, xb
            _, pv2 = train_probe(xa_s, tr["labels"], xb_s, te["labels"], lr, bs, epochs, wd,
                                 args.device, seed)
            m2 = U.binary_metrics(te["labels"], pv2, threshold)
            per_layer_test.append({"layer": r["layer"], "feature_index": li2,
                                   "test_acc": m2["accuracy"], "test_f1": m2["f1"],
                                   "test_auroc": U.auroc(te["labels"], pv2)})

        accs = [r["val_acc"] for r in rows]
        result: Dict[str, Any] = {
            "level": level, "level_name": level_name,
            "method": "linear_probe", "reference": "paper Sec. 2.4 / Table 1",
            "model_features": tr["meta"].get("model_path"),
            "num_feature_layers": n_layers,
            "layer_labels": layer_labels,
            "select_by": select_by,
            "best_layer": l_star,
            "best_layer_feature_index": li,
            "best_layer_val_acc": best["val_acc"],
            "best_layer_val_auroc": best.get("val_auroc"),
            "mean_val_acc_over_layers": float(np.mean(accs)) if accs else None,
            "gain_over_mean_val_acc": float(best["val_acc"] - np.mean(accs)) if accs else None,
            "test": test,
            "per_layer": rows,
            "per_layer_test": per_layer_test,
            "probe_hparams": {"lr": lr, "batch_size": bs, "epochs": epochs, "weight_decay": wd,
                              "standardize": standardize_on, "threshold": threshold,
                              "optimizer": "AdamW", "loss": "BCEWithLogits", "seed": seed},
            "data": {"n_train": int(len(tr["labels"])), "n_val": int(len(va["labels"])),
                     "n_test": int(len(te["labels"])),
                     "input_mode": tr["meta"].get("input_mode"),
                     "pooling": tr["meta"].get("pooling"),
                     "max_length": tr["meta"].get("max_length"),
                     "n_truncated_test": (te["meta"] or {}).get("n_truncated")},
        }
        U.save_json(os.path.join(fdir, "probe.json"), result)

        with open(os.path.join(fdir, "per_layer.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["layer", "feature_index", "val_acc", "val_f1",
                                              "val_precision", "val_recall", "val_auroc"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

        torch.save({"state_dict": model.state_dict(), "layer": l_star, "layer_index": li,
                    "mean": mean, "std": std, "threshold": threshold, "level": level,
                    "standardize": standardize_on, "pooling": tr["meta"].get("pooling"),
                    "model_path": tr["meta"].get("model_path"), "input_mode": tr["meta"].get("input_mode")},
                   os.path.join(fdir, "probe_ckpt.pt"))
        np.savez_compressed(os.path.join(fdir, "test_scores.npz"),
                            sample_ids=np.array(te["sample_ids"], dtype=object),
                            labels=te["labels"], probs=pt,
                            layer=l_star, threshold=threshold)

        # ---- layer-wise curve (validation ACC / AUROC) ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            layers = [r["layer"] for r in rows]
            fig, ax = plt.subplots(figsize=(8.0, 4.2))
            ax.plot(layers, [r["val_acc"] for r in rows], marker="o", ms=3.5,
                    color="#3B6FB6", label="Accuracy (val)")
            if any(r.get("val_auroc") is not None for r in rows):
                ax.plot(layers, [r.get("val_auroc") for r in rows], marker="^", ms=3.5,
                        color="#C0392B", label="AUROC (val)")
            ax.axvline(l_star, color="#444444", ls="--", lw=1, alpha=0.8)
            ax.annotate(f"l*={l_star}", xy=(l_star, best["val_acc"]), xytext=(4, -14),
                        textcoords="offset points", color="#444444", fontsize=9)
            ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
            ax.set_xlabel("Transformer layer")
            ax.set_ylabel("Validation score")
            ax.set_title(f"{level_name}: layer-wise bias detection ({base_name(tr)})")
            ax.grid(alpha=0.25)
            ax.legend(loc="lower right", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(fdir, f"layer_curve_{level}.png"), dpi=200)
            plt.close(fig)
        except Exception as e:                                        # pragma: no cover
            print(f"[warn] could not draw the layer curve: {e}")

        print(f"[ok] {level_name}: l*={l_star} -> probe.json / per_layer.csv / "
              f"probe_ckpt.pt / test_scores.npz")


def base_name(tr: Dict[str, Any]) -> str:
    p = str(tr.get("meta", {}).get("model_path") or "")
    return os.path.basename(p.rstrip("/")) or "backbone"


if __name__ == "__main__":
    main()
