# -*- coding: utf-8 -*-
"""
Step 1 of 3 – extract layer-wise hidden states from a frozen LLM  (paper Sec. 2.3 / 3.1.2).

For every sample of every split we run one forward pass through the frozen
backbone with ``output_hidden_states=True`` and store the pooled representation
of *each* layer:

    hidden_states[0]    -> embedding output
    hidden_states[1..L] -> Transformer layers 1..L
    r^(l)(x) = h_n^(l)                             (paper Eq. 2, last token)

The result is written to ``{out}/{level}/features_{split}.npz``:

    features   float16 [n_samples, L+1, hidden_size]
    labels     int8    [n_samples]
    sample_ids object  [n_samples]
    meta       json    (model, dtype, max_length, pooling, data fingerprint, ...)

Resume support
--------------
Work is checkpointed **per batch** into ``{out}/{level}/_shards_{split}/``:
``index.jsonl`` (ordered sample list), ``done.txt`` (finished ids) and one
``shard_XXXXXX.npz`` per batch.  Re-running the same command after an interrupt
continues automatically; once a split is complete the shards are merged and
de-duplicated by ``sample_id``.  ``--overwrite`` starts from scratch.

A *fingerprint* (model / dtype / max_length / pooling / input_mode / data hash)
travels with every npz.  Change any of these and the old features are refused
instead of being silently reused – mixing pooling schemes across train/val/test
is the classic way to obtain a meaningless AUROC.

Examples
--------
    # full run, two visible GPUs
    CUDA_VISIBLE_DEVICES=0,2 python 01_extract_features.py --levels dialogue sentence

    # quick end-to-end check (100 samples/split, CPU, small model)
    python 01_extract_features.py --levels sentence --smoke \\
        --model_path /path/to/Qwen3.5-0.8B --device cpu --dtype float32

    # cheaper: keep only the best layer once 02_probe.py has reported l*
    python 01_extract_features.py --levels dialogue --keep_layers 15
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import probe_utils as U  # noqa: E402


# -------------------------------------------------------------------------
# arguments
# -------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Extract per-layer hidden states with a frozen LLM")
    ap.add_argument("--config", default=os.path.join(_HERE, "probe_config.json"))
    ap.add_argument("--levels", nargs="+", default=None,
                    help="subset of levels (dialogue / sentence); default = config.levels")
    ap.add_argument("--splits", nargs="+", default=None, help="default = config.splits")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--device_map", default="",
                    help="'auto' / 'balanced' / 'sequential' or a JSON dict for accelerate sharding")
    ap.add_argument("--dtype", default="", choices=["", "float32", "float16", "bfloat16"])
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument("--pooling", default="", choices=["", "last", "mean"])
    ap.add_argument("--input_mode", default="", choices=["", "task", "context"])
    ap.add_argument("--keep_layers", default="",
                    help="comma separated layer indices to store, e.g. '15' or '0,15,32' "
                         "(0 = embedding output). Default: store all L+1 tensors")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep N label-stratified samples per split (quick experiments)")
    ap.add_argument("--smoke", action="store_true", help="use the limits from config.smoke")
    ap.add_argument("--out_dir", default="", help="override config.out_dir")
    ap.add_argument("--overwrite", action="store_true", help="ignore existing results and shards")
    ap.add_argument("--keep_shards", action="store_true", help="keep the shard directory after merging")
    ap.add_argument("--allow_fingerprint_mismatch", action="store_true",
                    help="skip (instead of aborting on) features extracted with other settings")
    return ap.parse_args()


# -------------------------------------------------------------------------
# small helpers
# -------------------------------------------------------------------------
def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def _append_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")
        f.flush()
        os.fsync(f.fileno())


def _save_shard(path: str, vec: np.ndarray, labels: List[int], sids: List[str]) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp,
                        features=vec.astype(np.float16),
                        labels=np.array(labels, dtype=np.int64),
                        sample_ids=np.array(sids, dtype=object))
    os.replace(tmp, path)


def merge_shards(shard_dir: str, index: List[Dict[str, Any]], out_npz: str,
                 meta: Dict[str, Any]) -> Tuple[int, int, int]:
    """Merge all shards in ``index`` order, de-duplicate by sample_id, write one npz."""
    by_id: Dict[str, Tuple[np.ndarray, int]] = {}
    for sh in sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz"))):
        try:
            with np.load(sh, allow_pickle=True) as d:
                ids = [str(x) for x in d["sample_ids"]]
                F, L = d["features"], d["labels"]
                for i, sid in enumerate(ids):
                    by_id[sid] = (F[i], int(L[i]))   # later shards win (de-duplication)
        except Exception as e:                        # pragma: no cover
            print(f"[warn] skipping corrupted shard {sh}: {e}")
    missing = [x["sample_id"] for x in index if x["sample_id"] not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} samples are still missing at merge time "
                           f"(e.g. {missing[:3]}); re-run the same command to resume")
    feats = np.stack([by_id[x["sample_id"]][0] for x in index], axis=0)
    labels = np.array([by_id[x["sample_id"]][1] for x in index], dtype=np.int64)
    sids = [x["sample_id"] for x in index]
    meta = dict(meta)
    meta.update({"n_samples": int(feats.shape[0]),
                 "num_feature_layers": int(feats.shape[1]),
                 "hidden_size": int(feats.shape[2]),
                 "layer_labels": sorted({int(x) for x in meta.get("layer_labels", [])})
                 if meta.get("layer_labels") else list(range(int(feats.shape[1])))})
    U.save_features(out_npz, feats, labels, sids, str(meta.get("split", "")), meta)
    return feats.shape[0], feats.shape[1], feats.shape[2]


def build_fingerprint(cfg: Dict[str, Any], level: str, split: str, args,
                      model_path: str, dtype: str, max_length: int, pooling: str,
                      input_mode: str, tokenizer) -> Dict[str, Any]:
    limit = args.limit or (int(cfg["smoke"][f"limit_{split}"]) if args.smoke else 0)
    return {
        "model_path": os.path.abspath(model_path),
        "dtype": dtype,
        "max_length": int(max_length),
        "pooling": pooling,
        "input_mode": input_mode,
        "prompt_style": cfg.get("prompt_style", "chat"),
        "enable_thinking": cfg.get("enable_thinking", False),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "right")),
        "keep_layers": str(args.keep_layers or "all"),
        "level": level,
        "split": split,
        "data_sha1": U.sha1_of_file(U.data_path(cfg, level, split)),
        "limit": int(limit),
    }


def _fp_diff(old: Any, new: Dict[str, Any]) -> List[str]:
    if not isinstance(old, dict):
        return ["no fingerprint in the old file (produced by an older version) -> --overwrite"]
    return [f"{k}: old={old.get(k)!r} -> new={new.get(k)!r}"
            for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = U.load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    elif args.smoke:
        # keep smoke artefacts away from the official results directory
        cfg["out_dir"] = cfg.get("smoke_out_dir", cfg["out_dir"] + "_smoke")
    levels = args.levels or U.levels_of(cfg)
    splits = args.splits or cfg["splits"]
    batch_size = args.batch_size or int(cfg["batch_size"])
    max_length = args.max_length or int(cfg["max_length"])
    dtype = args.dtype or cfg["dtype"]
    device = args.device or cfg["device"]
    pooling = args.pooling or cfg["pooling"]
    input_mode = args.input_mode or cfg["input_mode"]
    model_path = args.model_path or cfg["model_path"]
    prompt_style = cfg.get("prompt_style", "chat")
    enable_thinking = cfg.get("enable_thinking", False)
    seed = int(cfg["seed"])

    keep_layers: List[int] = []
    if args.keep_layers.strip():
        keep_layers = sorted({int(x) for x in args.keep_layers.split(",") if x.strip() != ""})

    import torch

    model, tok, dev = U.load_model_and_tokenizer(
        model_path, device=device, dtype=dtype, device_map=args.device_map,
        max_memory=cfg.get("max_memory") or None)
    print(f"[model] {model_path} | device={dev if dev else args.device_map} | dtype={dtype} | "
          f"batch={batch_size} | max_length={max_length} | pooling={pooling} | "
          f"input_mode={input_mode} | truncation_side={tok.truncation_side}")

    for level in levels:
        level_name = U.task_of(cfg, level)["display_name"]
        for split in splits:
            d = U.out_dir(cfg, level)
            out_npz = os.path.join(d, f"features_{split}.npz")
            fp = build_fingerprint(cfg, level, split, args, model_path, dtype, max_length,
                                   pooling, input_mode, tok)

            # ---- refuse to reuse features extracted with different settings ----
            if os.path.exists(out_npz) and not args.overwrite:
                old_fp = U.peek_meta(out_npz).get("fingerprint")
                if old_fp == fp:
                    print(f"[skip] {out_npz} already exists and matches the current settings")
                    continue
                msg = (f"\n[fingerprint-mismatch] {out_npz} was extracted with other settings:\n    "
                       + "\n    ".join(_fp_diff(old_fp, fp)))
                if args.allow_fingerprint_mismatch:
                    print(msg + "\n    (--allow_fingerprint_mismatch: skipping this split)")
                    continue
                raise SystemExit(msg + "\n    Fix: re-run with --overwrite.")

            shard_dir = os.path.join(d, f"_shards_{split}")
            index_path = os.path.join(shard_dir, "index.jsonl")
            done_path = os.path.join(shard_dir, "done.txt")
            trunc_path = os.path.join(shard_dir, "truncated.txt")
            fp_path = os.path.join(shard_dir, "fingerprint.json")

            if args.overwrite and os.path.isdir(shard_dir):
                shutil.rmtree(shard_dir)

            if os.path.isdir(shard_dir) and os.path.exists(fp_path) and not args.overwrite:
                old_fp = U.load_json(fp_path)
                if old_fp != fp:
                    msg = (f"\n[fingerprint-mismatch] the resume directory {shard_dir} belongs to "
                           f"another configuration:\n    " + "\n    ".join(_fp_diff(old_fp, fp)))
                    if args.allow_fingerprint_mismatch:
                        print(msg + "\n    discarding the shards and starting over")
                        shutil.rmtree(shard_dir, ignore_errors=True)
                    else:
                        raise SystemExit(msg + "\n    Fix: re-run with --overwrite.")
            if not os.path.exists(fp_path):
                os.makedirs(shard_dir, exist_ok=True)
                U.save_json(fp_path, fp)

            # ---- ordered sample index ----
            if os.path.exists(index_path):
                index = U.read_jsonl(index_path)
            else:
                limit = args.limit or (int(cfg["smoke"][f"limit_{split}"]) if args.smoke else 0)
                rows = U.load_rows(cfg, level, split, limit=limit, seed=seed)
                index = []
                for r in rows:
                    y = U.label_of(cfg, level, r)
                    if y is None:
                        continue
                    index.append({"sample_id": U.sample_key(r), "label": y,
                                  "variant": r.get("variant"),
                                  "bias_subtype": U.bias_subtype(cfg, level, r),
                                  "dialog_key": r.get("dialog_key"),
                                  "sentence_key": r.get("sentence_key")})
                U.write_jsonl(index_path, index)

            by_id = {x["sample_id"]: x for x in index}
            done = set(_read_lines(done_path))
            pending = [x["sample_id"] for x in index if x["sample_id"] not in done]
            print(f"[{level_name}/{split}] total={len(index)} done={len(index) - len(pending)} "
                  f"pending={len(pending)}")
            n_truncated = 0

            if pending:
                rows_all = U.load_rows(cfg, level, split)
                row_by_sid = {}
                for r in rows_all:
                    sid = U.sample_key(r)
                    if sid in by_id:
                        row_by_sid[sid] = r

                t0 = time.time()
                shard_no = len(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
                oom = tuple({getattr(torch, "OutOfMemoryError", RuntimeError),
                             getattr(torch.cuda, "OutOfMemoryError", RuntimeError)})

                def extract_chunk(chunk: List[str]) -> np.ndarray:
                    """Forward one chunk; on CUDA OOM split it in half and retry recursively."""
                    nonlocal n_truncated
                    texts = []
                    for sid in chunk:
                        r = row_by_sid.get(sid)
                        if r is None:
                            raise RuntimeError(f"raw row for sample {sid} not found")
                        system, user = U.build_prompt(cfg, level, r, input_mode=input_mode)
                        texts.append(U.render_input_text(tok, system, user,
                                                         prompt_style=prompt_style,
                                                         enable_thinking=enable_thinking))
                    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                              max_length=max_length)
                    hit = (enc["attention_mask"].sum(dim=1) >= max_length).tolist()
                    if any(hit):
                        _append_lines(trunc_path, [chunk[j] for j, h in enumerate(hit) if h])
                        n_truncated += sum(1 for h in hit if h)
                    if dev is not None:
                        enc = {k: v.to(dev) for k, v in enc.items()}
                    try:
                        with torch.no_grad():
                            out = model(input_ids=enc["input_ids"],
                                        attention_mask=enc["attention_mask"],
                                        output_hidden_states=True)
                        vec = U.last_token_hidden_states(out.hidden_states, enc["attention_mask"],
                                                         pooling=pooling)     # [B, L+1, d]
                        if keep_layers:
                            vec = vec[:, keep_layers, :]
                        return vec
                    except oom:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if len(chunk) <= 1:
                            raise RuntimeError(
                                f"[OOM] a single sample still does not fit "
                                f"(max_length={max_length}, pooling={pooling}). Lower "
                                f"--max_length (e.g. 768), use a smaller model, or shard with "
                                f"--device_map balanced.") from None
                        mid = len(chunk) // 2
                        print(f"    [OOM] batch={len(chunk)} -> retry as {mid}+{len(chunk) - mid}",
                              flush=True)
                        return np.concatenate([extract_chunk(chunk[:mid]),
                                               extract_chunk(chunk[mid:])], axis=0)

                for i in range(0, len(pending), batch_size):
                    chunk = pending[i:i + batch_size]
                    vec = extract_chunk(chunk)
                    _save_shard(os.path.join(shard_dir, f"shard_{shard_no:06d}.npz"), vec,
                                [int(by_id[s]["label"]) for s in chunk], chunk)
                    _append_lines(done_path, chunk)      # progress only after a durable shard
                    shard_no += 1
                    nd = len(index) - len(pending) + min(i + batch_size, len(pending))
                    print(f"  [{nd}/{len(index)}] "
                          f"{(min(i + batch_size, len(pending)) / max(time.time() - t0, 1e-6)):.2f} "
                          f"samples/s", flush=True)

                n_tr = len(set(_read_lines(trunc_path)))
                if n_tr:
                    print(f"[warn] {n_tr}/{len(index)} samples exceeded max_length={max_length} "
                          f"and were LEFT-truncated; consider a larger --max_length")
                else:
                    print(f"[ok] no sample was truncated (max_length={max_length} is enough)")

            missing = [x["sample_id"] for x in index
                       if x["sample_id"] not in set(_read_lines(done_path))]
            if missing:
                print(f"[partial] {len(missing)} samples still pending – re-run to resume")
                continue

            meta = {
                "level": level, "level_name": level_name, "split": split,
                "task": "patient_reporting_bias_detection",
                "model_path": model_path, "dtype": dtype, "max_length": max_length,
                "pooling": pooling, "input_mode": input_mode, "prompt_style": prompt_style,
                "truncation_side": str(getattr(tok, "truncation_side", "right")),
                "keep_layers": keep_layers or "all",
                "layer_labels": keep_layers,       # filled with 0..L when empty
                "n_truncated": len(set(_read_lines(trunc_path))),
                "fingerprint": fp,
            }
            n, L, h = merge_shards(shard_dir, index, out_npz, meta)
            U.write_jsonl(os.path.join(d, f"manifest_{split}.jsonl"), index)
            print(f"[ok] {level_name}/{split}: features=({n}, {L}, {h}) -> {out_npz}")
            if not args.keep_shards:
                shutil.rmtree(shard_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
