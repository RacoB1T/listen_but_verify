# -*- coding: utf-8 -*-
"""
Step 3 of 3 – zero-shot prompting baseline with the same frozen LLM
(paper Sec. 3.1 / Table 1, the "direct prompting" rows).

The backbone is asked the *identical* t1 question the probe's features come
from, and we measure it in two ways:

  generate  greedy decoding -> parse the JSON answer -> 0/1
            => ACC / Precision / Recall / F1 (the metrics of Table 1)
  score     append the assistant prefix ``{"<field>": `` and compare the logits
            of the two tokens ``0`` and ``1`` (softmax over them)
            => a threshold-free P(bias) for AUROC, directly comparable to the
               probe's sigmoid probability

Everything is cached per sample in ``prompt_predictions_<level>_<split>.jsonl``, so
an interrupted run resumes and the metrics are always rebuilt from the complete
prediction file.

Examples
--------
    python 03_prompt_baseline.py --levels dialogue sentence
    python 03_prompt_baseline.py --levels sentence --smoke \\
        --model_path /path/to/Qwen3.5-0.8B --device cpu --dtype float32
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import probe_utils as U  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="zero-shot prompting baseline")
    ap.add_argument("--config", default=os.path.join(_HERE, "probe_config.json"))
    ap.add_argument("--levels", nargs="+", default=None)
    ap.add_argument("--splits", nargs="+", default=["test"],
                    help="splits to run the baseline on (default: test)")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--device_map", default="")
    ap.add_argument("--dtype", default="", choices=["", "float32", "float16", "bfloat16"])
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_generate", action="store_true", help="skip the generation path")
    ap.add_argument("--no_score", action="store_true", help="skip the logit-scoring path")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


# -------------------------------------------------------------------------
# model
# -------------------------------------------------------------------------
def load_causal_lm(model_path: str, dtype: str, device: str, device_map: str = "",
                   trust_remote_code: bool = True, max_memory: Optional[Dict[Any, Any]] = None):
    """Load the LM head model described by ``config.architectures`` (e.g. Qwen3.5)."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"          # keep the tail (chat suffix / "目标句: ...")

    hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    arch = (getattr(hf_cfg, "architectures", None) or [None])[0]
    cls = getattr(transformers, arch, None) if arch else None
    if not isinstance(cls, type):
        cls = AutoModelForCausalLM

    kw: Dict[str, Any] = dict(trust_remote_code=trust_remote_code, dtype=dt, low_cpu_mem_usage=True)
    if device_map:
        dm: Any = device_map
        if isinstance(device_map, str) and device_map.strip().startswith("{"):
            dm = json.loads(device_map)
        if max_memory:
            kw["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in max_memory.items()}
        model = cls.from_pretrained(model_path, device_map=dm, **kw)
        dev = None
        U.log_device_map(model, dm)
    else:
        model = cls.from_pretrained(model_path, **kw)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            print("[warn] CUDA not available -> falling back to CPU")
            device = "cpu"
        model.to(device)
        dev = device
    model.eval()
    return model, tok, dev, cls.__name__


def parse_binary(text: str, field: str) -> Optional[int]:
    """Extract the 0/1 answer from a model response (tolerates a leading think block)."""
    if not isinstance(text, str) or not text.strip():
        return None
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()

    def coerce(obj) -> Optional[int]:
        if not isinstance(obj, dict):
            return None
        v = obj.get(field)
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, (int, float)) and int(v) in (0, 1):
            return int(v)
        if isinstance(v, str) and v.strip() in ("0", "1"):
            return int(v.strip())
        return None

    candidates = [s] + [m.group(0) for m in re.finditer(r"\{[^{}]*\}", s)][::-1]
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        r = coerce(obj)
        if r is not None:
            return r
    m = re.search(rf'"?{re.escape(field)}"?\s*[:：]\s*"?([01])"?', s)
    return int(m.group(1)) if m else None


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
    mcfg = cfg["prompt_classifier"]
    bs = args.batch_size or int(mcfg["batch_size"])
    max_new = args.max_new_tokens or int(mcfg["max_new_tokens"])
    do_gen = (not args.no_generate) and bool(mcfg.get("do_generate", True))
    do_score = (not args.no_score) and bool(mcfg.get("do_score", True))
    levels = args.levels or U.levels_of(cfg)
    model_path = args.model_path or cfg["model_path"]
    dtype = args.dtype or cfg["dtype"]
    seed = int(cfg["seed"])

    model, tok, dev, cls_name = load_causal_lm(model_path, dtype, args.device or cfg["device"],
                                               args.device_map,
                                               max_memory=cfg.get("max_memory") or None)
    print(f"[model] {cls_name} | {model_path} | "
          f"device={dev if dev else args.device_map} | dtype={dtype}")

    import torch

    ids0 = tok.encode("0", add_special_tokens=False)
    ids1 = tok.encode("1", add_special_tokens=False)
    if not ids0 or not ids1:
        raise RuntimeError("could not encode the answer tokens '0'/'1'")
    print(f"[score] token ids: '0' -> {ids0}, '1' -> {ids1}")

    for level in levels:
        level_name = U.task_of(cfg, level)["display_name"]
        field = U.task_of(cfg, level).get("pred_field") or U.task_of(cfg, level)["label_field"]
        for split in args.splits:
            rows = U.load_rows(cfg, level, split)
            limit = args.limit or (
                int(cfg["smoke"][f"limit_{split}"]) if args.smoke and f"limit_{split}" in cfg["smoke"]
                else 0)
            if limit:
                rows = U.stratified_take(cfg, level, rows, limit, seed=seed)

            pairs, golds, sids = [], [], []
            for r in rows:
                y = U.label_of(cfg, level, r)
                if y is None:
                    continue
                system, user = U.build_prompt(cfg, level, r,
                                              input_mode=cfg.get("input_mode", "task"))
                pairs.append((system, user))
                golds.append(y)
                sids.append(U.sample_key(r))

            pred_path = os.path.join(U.out_dir(cfg, level),
                                     f"prompt_predictions_{level}_{split}.jsonl")
            if args.overwrite and os.path.exists(pred_path):
                os.remove(pred_path)
            existing = {r["sample_id"]: r for r in U.read_jsonl(pred_path)}
            todo = [i for i, sid in enumerate(sids) if sid not in existing]
            print(f"\n===== prompt baseline | {level_name}/{split} | n={len(pairs)} "
                  f"done={len(pairs) - len(todo)} todo={len(todo)} (field={field}) =====")

            model.generation_config.pad_token_id = tok.pad_token_id
            prefix = '{"' + str(field) + '": '
            prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            max_len = args.max_length or int(cfg["max_length"])
            score_budget = max(16, max_len - len(prefix_ids))  # reserve room for the prefix
            n_trunc_gen = n_trunc_score = 0
            t0 = time.time()

            with torch.no_grad():
                for i in range(0, len(todo), bs):
                    chunk = todo[i:i + bs]
                    texts = [U.render_input_text(tok, pairs[j][0], pairs[j][1],
                                                 prompt_style=cfg.get("prompt_style", "chat"),
                                                 enable_thinking=cfg.get("enable_thinking", False))
                             for j in chunk]
                    preds_c: List[Optional[int]] = [None] * len(chunk)
                    raws_c: List[str] = [""] * len(chunk)
                    p1_c: List[Optional[float]] = [None] * len(chunk)

                    if do_gen:
                        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                                  max_length=max_len)
                        n_trunc_gen += int((enc["attention_mask"].sum(dim=1) >= max_len).sum().item())
                        if dev is not None:
                            enc = {k: v.to(dev) for k, v in enc.items()}
                        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
                        gen = out[:, enc["input_ids"].shape[1]:]
                        for t, g in enumerate(gen):
                            txt = tok.decode(g, skip_special_tokens=True)
                            raws_c[t] = txt
                            preds_c[t] = parse_binary(txt, field)

                    if do_score:
                        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                                  max_length=score_budget)
                        n_trunc_score += int(
                            (enc["attention_mask"].sum(dim=1) >= score_budget).sum().item())
                        enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
                        pid = torch.tensor(prefix_ids, dtype=enc["input_ids"].dtype).unsqueeze(0)
                        pid = pid.repeat(enc["input_ids"].shape[0], 1)
                        enc["input_ids"] = torch.cat([enc["input_ids"], pid], dim=1)
                        enc["attention_mask"] = torch.cat(
                            [enc["attention_mask"], torch.ones_like(pid)], dim=1)
                        if dev is not None:
                            enc = {k: v.to(dev) for k, v in enc.items()}
                        logits = model(**enc).logits[:, -1, :]
                        p1 = torch.softmax(
                            torch.stack([logits[:, ids0[0]], logits[:, ids1[0]]], dim=-1),
                            dim=-1)[:, 1]
                        for t, v in enumerate(p1.float().cpu().tolist()):
                            p1_c[t] = float(v)

                    U.append_jsonl_sync(pred_path, [
                        {"sample_id": sids[j], "gold": golds[j], "pred": preds_c[t],
                         "p_bias": p1_c[t], "raw": raws_c[t]} for t, j in enumerate(chunk)])
                    nd = len(pairs) - len(todo) + min(i + bs, len(todo))
                    print(f"  [{nd}/{len(pairs)}] {time.time() - t0:.0f}s", flush=True)

            # ---- rebuild the metrics from the complete prediction file ----
            rec = {r["sample_id"]: r for r in U.read_jsonl(pred_path)}
            missing = [s for s in sids if s not in rec]
            if missing:
                print(f"[partial] {len(missing)} predictions still missing – re-run to resume")
                continue
            preds = [rec[s].get("pred") for s in sids]
            p1s = [rec[s].get("p_bias") for s in sids]
            n_bad = sum(1 for p in preds if p is None)
            gen_ok = [i for i in range(len(preds)) if preds[i] is not None]

            result: Dict[str, Any] = {
                "level": level, "level_name": level_name, "split": split,
                "method": "prompt_classifier", "reference": "paper Table 1 (direct prompting)",
                "model": model_path, "n": len(pairs), "n_parse_fail": n_bad, "field": field,
                "prompt": "identical t1 template as the probe features",
                "max_length": max_len, "score_budget": score_budget,
                "truncation_side": str(getattr(tok, "truncation_side", "right")),
                "n_truncated_generate": n_trunc_gen, "n_truncated_score": n_trunc_score,
            }
            if n_trunc_gen or n_trunc_score:
                print(f"[warn] truncated: generate {n_trunc_gen}/{len(pairs)}, "
                      f"score {n_trunc_score}/{len(pairs)}")
            if gen_ok:
                y = [golds[i] for i in gen_ok]
                p = [float(preds[i]) for i in gen_ok]
                m = U.binary_metrics(y, p, 0.5)
                m["accuracy_ci95"] = U.bootstrap_ci(y, p, "accuracy", 0.5)
                m["f1_ci95"] = U.bootstrap_ci(y, p, "f1", 0.5)
                if do_score:
                    ps = [p1s[i] for i in gen_ok if p1s[i] is not None]
                    if len(ps) == len(y):
                        m["auroc"] = U.auroc(y, ps)
                        m["auroc_ci95"] = U.bootstrap_ci(y, ps, "auroc", 0.5)
                result["test_generate"] = m
                U.print_metrics(f"prompt baseline {level_name} (generate)", m)
            if do_score:
                ok = [i for i in range(len(p1s)) if p1s[i] is not None]
                if ok:
                    y = [golds[i] for i in ok]
                    ps = [p1s[i] for i in ok]
                    ms = U.binary_metrics(y, ps, 0.5)
                    ms["auroc"] = U.auroc(y, ps)
                    ms["accuracy_ci95"] = U.bootstrap_ci(y, ps, "accuracy", 0.5)
                    ms["f1_ci95"] = U.bootstrap_ci(y, ps, "f1", 0.5)
                    ms["auroc_ci95"] = U.bootstrap_ci(y, ps, "auroc", 0.5)
                    result["test_score"] = ms
                    U.print_metrics(f"prompt baseline {level_name} (score)", ms)

            U.save_json(os.path.join(U.out_dir(cfg, level), f"prompt_{split}.json"), result)
            print(f"[ok] {level_name}/{split} -> prompt_{split}.json "
                  f"({len(sids)} predictions in {pred_path})")


if __name__ == "__main__":
    main()
