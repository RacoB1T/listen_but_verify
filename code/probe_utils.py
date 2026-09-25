# -*- coding: utf-8 -*-
"""
Shared utilities for the patient-reporting-bias probe (paper Sec. 2.3 / 2.4 / 3.2).

This module is deliberately **self-contained**: it does not import anything from
the authors' private working tree.  Everything the detection pipeline needs is here:

  * json / jsonl / npz I/O;
  * the prompt definitions (``probe_config.json``) and the dialogue renderer,
    which reproduces the format used for the zero-shot LLM baselines in the
    paper (``[turn] role: utterance`` + the t1 JSON instruction template), so the
    probe and the prompting baseline always see exactly the same input;
  * dataset loading and label handling;
  * frozen-LLM loading and last-token hidden-state extraction.

Layer-index convention (identical to the paper, Sec. 2.3 Eq. 2):

    hidden_states[0]  -> embedding output
    hidden_states[l]  -> output of Transformer layer l   (1 <= l <= L)
    => a forward pass returns L+1 tensors

so a feature array has shape ``[n_samples, L+1, hidden_size]`` and
``feature index == layer number``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "probe_config.json")

# The released sentence-level data stores the four intensity sub-types as single
# letters; these are their names (paper Sec. 2.2: "intensity and inference bias").
INTENSITY_TYPE_NAMES = {
    "A": "症状表达强度",   # symptom intensity / exaggeration
    "B": "证据呈现偏向",   # biased presentation of evidence
    "C": "推断合理化",     # unjustified inference / rationalisation
    "D": "外部来源触发",   # triggered by external information
}


# =========================================================================
# 1. basic IO
# =========================================================================
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a jsonl file; a missing file yields an empty list (resume-friendly)."""
    out: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl_sync(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Append rows and fsync, so an interrupted run can always resume."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha1_of_file(path: str) -> Optional[str]:
    """Content fingerprint (first 16 hex chars) used to detect silent data swaps."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# =========================================================================
# 2. configuration
# =========================================================================
def load_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    cfg = load_json(path)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_repo_root"] = os.path.dirname(_HERE)          # <repo>/code -> <repo>
    cfg["_code_dir"] = _HERE
    # environment overrides, handy for smoke runs / shared clusters
    if os.environ.get("PRB_DATA_DIR", "").strip():
        cfg["data_dir"] = os.environ["PRB_DATA_DIR"].strip()
    if os.environ.get("PRB_OUT_DIR", "").strip():
        cfg["out_dir"] = os.environ["PRB_OUT_DIR"].strip()
    if os.environ.get("PRB_MODEL_PATH", "").strip():
        cfg["model_path"] = os.environ["PRB_MODEL_PATH"].strip()
    return cfg


def levels_of(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg.get("levels") or list(cfg["tasks"].keys()))


def task_of(cfg: Dict[str, Any], level: str) -> Dict[str, Any]:
    if level not in cfg["tasks"]:
        raise KeyError(f"unknown level {level!r}; available: {list(cfg['tasks'])}")
    return cfg["tasks"][level]


def data_path(cfg: Dict[str, Any], level: str, split: str) -> str:
    task = task_of(cfg, level)
    return os.path.join(cfg["data_dir"], task["data_subdir"], f"{split}.jsonl")


def out_dir(cfg: Dict[str, Any], level: str, *parts: str) -> str:
    p = os.path.join(cfg["out_dir"], task_of(cfg, level)["out_subdir"], *parts)
    os.makedirs(p, exist_ok=True)
    return p


# =========================================================================
# 3. prompts  (reproduces the evaluation prompt of the zero-shot baselines)
# =========================================================================
_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run (including newlines) into a single space."""
    return _WS_RE.sub(" ", text).strip()


def make_dialog_text(turns: Sequence[Dict[str, Any]], include_role: bool = True,
                     normalize_ws: bool = True) -> str:
    """Render a dialogue as the baseline prompt sees it:

        [0] Patients: 我最近总是腹胀……
        [1] Doctor: 您好，请问持续多久了？

    Role names in the released data are ``Patients`` / ``Doctor``.
    """
    lines: List[str] = []
    for i, t in enumerate(turns or []):
        role = t.get("id", "")
        sent = t.get("Sentence", "")
        if not isinstance(sent, str):
            sent = str(sent)
        if normalize_ws:
            sent = normalize_whitespace(sent)
        lines.append(f"[{i}] {role}: {sent}" if include_role else f"[{i}] {sent}")
    return "\n".join(lines)


def build_prompt(cfg: Dict[str, Any], level: str, sample: Dict[str, Any],
                 input_mode: str = "task") -> Tuple[str, str]:
    """Return ``(system, user)`` for one sample.

    ``input_mode="task"``    – the t1 instruction template (default; identical to
                               the prompt of the prompting baselines in Table 1).
    ``input_mode="context"`` – raw dialogue context (+ target sentence) with no
                               task instruction, i.e. the pure "frozen feature
                               extractor" setting of the paper.
    """
    task = task_of(cfg, level)
    dlg = cfg.get("dialog_format", {}) or {}
    dialog_text = make_dialog_text(
        sample.get("turns") or [],
        include_role=bool(dlg.get("include_role", True)),
        normalize_ws=bool(dlg.get("normalize_whitespace", True)),
    )
    if input_mode == "context":
        user = dialog_text
        tgt = sample.get("target_sentence")
        if isinstance(tgt, str) and tgt.strip():
            user = user + "\n\n目标句: " + tgt.strip()
        return "", user

    pr = task["prompt"]
    system = pr.get("system", "")
    user = pr["template"].format(
        DIALOG=dialog_text,
        TARGET_IDX=sample.get("target_idx", ""),
        TARGET_SPEAKER=sample.get("target_speaker", ""),
        TARGET_SENTENCE=sample.get("target_sentence", ""),
    )
    return system, user


def render_input_text(tokenizer: Any, system: str, user: str,
                      prompt_style: str = "chat",
                      enable_thinking: Optional[bool] = False) -> str:
    """Turn (system, user) into the exact string fed to the model.

    With ``prompt_style="chat"`` the tokenizer chat template is applied together
    with ``add_generation_prompt=True``; this makes the final token a stable
    prompt suffix (the assistant header), which is what the paper pools over
    ("the hidden state of the final token").  ``enable_thinking=False`` keeps the
    hybrid-thinking templates from emitting a thinking block first.
    """
    if prompt_style == "chat" and getattr(tokenizer, "chat_template", None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        kw = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            try:
                return tokenizer.apply_chat_template(msgs, enable_thinking=enable_thinking, **kw)
            except Exception:
                pass  # template without the flag -> fall through
        try:
            return tokenizer.apply_chat_template(msgs, **kw)
        except Exception:
            pass
    return (system + "\n\n" + user) if system else user


# =========================================================================
# 4. data / labels
# =========================================================================
def load_rows(cfg: Dict[str, Any], level: str, split: str,
              limit: int = 0, seed: int = 42) -> List[Dict[str, Any]]:
    """Load one split; ``limit > 0`` keeps a label-stratified subset."""
    rows = read_jsonl(data_path(cfg, level, split))
    if limit and limit > 0 and limit < len(rows):
        rows = stratified_take(cfg, level, rows, limit, seed=seed)
    return rows


def sample_key(row: Dict[str, Any]) -> str:
    for k in ("sample_id", "sentence_key", "dialog_key"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def label_of(cfg: Dict[str, Any], level: str, row: Dict[str, Any]) -> Optional[int]:
    field = task_of(cfg, level)["label_field"]
    v = row.get(field)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def variant_of(row: Dict[str, Any]) -> str:
    v = row.get("variant")
    return str(v) if isinstance(v, str) else ""


def bias_subtype(cfg: Dict[str, Any], level: str, row: Dict[str, Any]) -> str:
    """Fine-grained bias category of a positive sample (used for per-type AUROC).

    dialogue level: ``isolated_entity`` / ``contradiction:<type>``
    sentence level: ``ambiguity:<type>`` / ``intensity:<type>``

    The four intensity sub-types are stored as the letters A/B/C/D in the
    released data, so they are mapped back to their names here (see
    ``data/README.md`` for the mapping).
    """
    if level in ("dialogue", "dialogue_level"):
        fam = row.get("error_family") or "none"
        sub = row.get("contradiction_type")
        return f"{fam}:{sub}" if sub else str(fam)
    fam = row.get("bias_family") or "none"
    sub = row.get("ambiguity_type") or row.get("intensity_type")
    if not sub:
        return str(fam)
    sub = str(sub)
    if fam == "intensity":
        sub = INTENSITY_TYPE_NAMES.get(sub.strip().upper(), sub)
    return f"{fam}:{sub}"


def stratified_take(cfg: Dict[str, Any], level: str, rows: List[Dict[str, Any]],
                    n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Keep ``n`` rows with a 1:1 positive/negative ratio (deterministic)."""
    if not n or n <= 0 or n >= len(rows):
        return list(rows)
    pos = [r for r in rows if label_of(cfg, level, r) == 1]
    neg = [r for r in rows if label_of(cfg, level, r) == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = max(1, n // 2)
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def dataset_stats(cfg: Dict[str, Any], level: str) -> Dict[str, Any]:
    """Counts per split / label / variant / bias subtype – for the README table."""
    stats: Dict[str, Any] = {}
    for split in cfg.get("splits", ["train", "val", "test"]):
        rows = read_jsonl(data_path(cfg, level, split))
        by_label = {0: 0, 1: 0}
        by_variant: Dict[str, int] = {}
        by_subtype: Dict[str, int] = {}
        for r in rows:
            y = label_of(cfg, level, r)
            if y in by_label:
                by_label[y] += 1
            by_variant[variant_of(r)] = by_variant.get(variant_of(r), 0) + 1
            if y == 1:
                s = bias_subtype(cfg, level, r)
                by_subtype[s] = by_subtype.get(s, 0) + 1
        stats[split] = {"n": len(rows), "label": by_label,
                        "variant": by_variant, "pos_subtype": by_subtype}
    return stats


# =========================================================================
# 5. frozen LLM + hidden states
# =========================================================================
_DTYPE_MAP_NAME = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


def _torch_dtype(name: str):
    import torch

    if name not in _DTYPE_MAP_NAME:
        raise ValueError(f"dtype must be one of {list(_DTYPE_MAP_NAME)}, got {name!r}")
    return getattr(torch, name)


def load_model_and_tokenizer(model_path: str, device: str = "auto", dtype: str = "float16",
                             device_map: str = "", trust_remote_code: bool = True,
                             max_memory: Optional[Dict[Any, Any]] = None):
    """Load a frozen base model (no LM head) plus its tokenizer.

    ``device_map`` (``auto``/``balanced``/``sequential``/a JSON dict) switches to
    accelerate sharding; otherwise the model is moved to ``device`` as a whole.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # T1 fix: keep the TAIL when truncating.  The chat-template suffix (and the
    # "目标句: ..." tail at sentence level) lives at the END of the sequence, so
    # right truncation would destroy the last-token semantics.
    tok.truncation_side = "left"

    kw: Dict[str, Any] = dict(trust_remote_code=trust_remote_code,
                              dtype=_torch_dtype(dtype), low_cpu_mem_usage=True)
    if device_map:
        import json as _json

        dm: Any = device_map
        if isinstance(device_map, str) and device_map.strip().startswith("{"):
            dm = _json.loads(device_map)
        if max_memory:
            kw["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in max_memory.items()}
        model = AutoModel.from_pretrained(model_path, device_map=dm, **kw)
        dev = None
        log_device_map(model, dm)
    else:
        model = AutoModel.from_pretrained(model_path, **kw)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            print("[warn] CUDA not available -> falling back to CPU")
            device = "cpu"
        model.to(device)
        dev = device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, dev


def log_device_map(model: Any, requested: Any = None) -> None:
    """Print how accelerate sharded the model (silent single-GPU mistakes are common)."""
    hm = getattr(model, "hf_device_map", None)
    if not isinstance(hm, dict) or not hm:
        print(f"[multi-gpu] no hf_device_map (device_map={requested}); the whole model "
              f"probably sits on one GPU. Use --device_map balanced to force sharding.")
        return
    cnt: Dict[str, int] = {}
    for v in hm.values():
        cnt[str(v)] = cnt.get(str(v), 0) + 1
    print("[multi-gpu] shards -> " + ", ".join(f"{k}: {v} modules" for k, v in sorted(cnt.items())))


def last_token_hidden_states(hidden_states, attention_mask, pooling: str = "last") -> np.ndarray:
    """Pool every layer's hidden states into ``[B, L+1, d]``.

    ``pooling="last"``  – representation of the final token           (paper Eq. 2)
    ``pooling="mean"``  – mean over the non-padding tokens

    Each layer may live on a different GPU under ``device_map`` sharding, while
    ``attention_mask`` usually stays on CPU, so the mask is moved per layer.
    """
    import torch

    out = []
    for h in hidden_states:
        am = attention_mask.to(h.device)
        if pooling == "last":
            idx = am.sum(dim=1).clamp(min=1) - 1                     # [B]
            b = torch.arange(h.shape[0], device=h.device)
            vec = h[b, idx, :]
        elif pooling == "mean":
            m = am.unsqueeze(-1).to(h.dtype)
            vec = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        out.append(vec.float().cpu())
    return torch.stack(out, dim=1).numpy()                          # [B, L+1, d]


def save_features(path: str, features: np.ndarray, labels: np.ndarray,
                  sample_ids: Sequence[str], split: str, meta: Dict[str, Any]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        features=features.astype(np.float16),
        labels=labels.astype(np.int8),
        sample_ids=np.array(list(sample_ids), dtype=object),
        split=np.array([split], dtype=object),
        meta=np.array([json.dumps(meta, ensure_ascii=False)], dtype=object),
    )
    os.replace(tmp, path)


def load_features(path: str) -> Dict[str, Any]:
    """Load ``features_{split}.npz`` into a dict (features as float32 in memory)."""
    with np.load(path, allow_pickle=True) as data:
        meta = json.loads(str(data["meta"][0])) if "meta" in data else {}
        return {
            "features": data["features"].astype(np.float32),
            "labels": data["labels"].astype(np.int64),
            "sample_ids": [str(x) for x in data["sample_ids"]],
            "split": str(data["split"][0]) if "split" in data else "",
            "meta": meta,
        }


def peek_meta(path: str) -> Dict[str, Any]:
    """Read only the ``meta`` blob of a features npz (avoids loading GB of arrays)."""
    try:
        with np.load(path, allow_pickle=True) as z:
            if "meta" not in z:
                return {}
            raw = z["meta"]
            try:
                s = str(raw[0])
            except Exception:
                s = str(raw.item())
            s = s.strip()
            return json.loads(s) if s.startswith("{") else {}
    except Exception:
        return {}


# =========================================================================
# 6. binary metrics
# =========================================================================
def binary_metrics(y: Sequence[int], p: Sequence[float], thr: float = 0.5) -> Dict[str, Any]:
    """ACC / Precision / Recall / F1 / confusion matrix at a fixed threshold."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p).astype(np.float64)
    pred = (p > thr).astype(np.int64)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    n = len(y)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"n": n, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "threshold": thr}


def auroc(y: Sequence[int], p: Sequence[float]) -> Optional[float]:
    y = np.asarray(y).astype(np.int64)
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, np.asarray(p).astype(np.float64)))


def bootstrap_ci(y: Sequence[int], p: Sequence[float], metric: str = "accuracy",
                 thr: float = 0.5, n_boot: int = 1000, seed: int = 42) -> Optional[List[float]]:
    """Percentile bootstrap 95% CI for accuracy / f1 / precision / recall / auroc."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p).astype(np.float64)
    n = len(y)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        v = auroc(yy, pp) if metric == "auroc" else binary_metrics(yy, pp, thr).get(metric)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def print_metrics(tag: str, m: Dict[str, Any]) -> None:
    keys = ["n", "accuracy", "precision", "recall", "f1", "auroc"]
    s = "  ".join(f"{k}={m[k]:.4f}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}"
                  for k in keys if k in m)
    print(f"[{tag}] {s}")
