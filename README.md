# Listen, but Verify: Detecting Patient Reporting Bias in Medical Dialogue LLMs

Code and data for the paper:

**Listen, but Verify: Detecting Patient Reporting Bias in Medical Dialogue LLMs**

![](figure/framework.jpg)

LLM-based medical dialogue systems often implicitly assume that patients can accurately and consistently describe their health conditions. In real-world interactions, however, patient reports may contain exaggerations, omissions, contradictions, or subjective interpretations of clinical evidence — a phenomenon we refer to as **patient reporting bias**.


---

## Repository layout

```
.
├── README.md
├── LICENSE                       # MIT (code); see the data notice in the LICENSE file
├── requirements.txt
├── code/
│   ├── probe_config.json         # datasets, prompt templates, hyper-parameters
│   ├── probe_utils.py            # shared utilities: IO, prompts, data, frozen LLM, metrics
│   ├── 01_extract_features.py    # step 1 – layer-wise hidden states of a frozen LLM
│   ├── 02_probe.py               # step 2 – train one linear probe per layer, select l*, test
│   ├── 03_report.py              # step 3 – summary.json, layer curves, per-sub-type AUROC
│   └── run.sh                    # one-command pipeline (all three steps)
├── data/
│   ├── dialogue/{train,val,test}.jsonl
│   └── sentence/{train,val,test}.jsonl
└── results/                      # empty; every artefact is written here
```

Everything is plain Python + PyTorch/Transformers. The only external artefact is the
frozen backbone (Qwen3.5-9B-Instruct in the paper), whose path you pass on the command
line.

---

## Setup

Python ≥ 3.9, PyTorch ≥ 2.0, `transformers`, `numpy`, `scikit-learn`, `matplotlib`
(`accelerate` only for multi-GPU sharding):

```bash
git clone https://github.com/RacoB1T/listen_but_verify.git && cd listen_but_verify
pip install -r requirements.txt

# any Hugging Face causal LM works; the paper uses Qwen3.5-9B-Instruct
huggingface-cli download Qwen/Qwen3.5-9B-Instruct --local-dir /models/Qwen3.5-9B-Instruct
```

Point the code at the weights with `--model_path`, the `MODEL=` variable, or
`"model_path"` in `code/probe_config.json`.

---

## Quick start

A full run needs a GPU and a few hours — feature extraction dominates; probe training is
CPU-only and takes minutes. To check the installation quickly, run the smoke
configuration on CPU with a small backbone:

```bash
cd code
SMOKE=1 LIMIT=8 MODEL=/models/Qwen3.5-0.8B DEVICE=cpu DTYPE=float32 BATCH=4 bash run.sh

cat ../results_smoke/summary.json
```

Smoke runs read and write `results_smoke/` and only prove that the pipeline works —
they are **not** results.

---

## Data

Built on **MedDG**. Every reliable patient narrative is paired with a biased twin
produced by one controlled rewrite of the patient side only, so the two differ in
reporting style but describe the same case.

| Level | Unit | Label field | Train / val / test | Total |
|---|---|---|---|---|
| Dialogue | one dialogue variant | `has_misreport` | 7,752 / 1,012 / 928 | 9,692 |
| Sentence | one target sentence | `has_bias` | 7,270 / 950 / 952 | 9,172 |

Each split is 1:1 positive/negative, and bias is annotated with its type at both levels
(isolated entity / cross-turn contradiction for dialogues, intensity and inference /
vagueness for sentences) in fields such as `error_family`, `contradiction_type`,
`bias_family`, `intensity_type` and `ambiguity_type` — see the JSONL files for the exact
schema and per-type counts.

---

## Running

### One command

```bash
cd code
MODEL=/models/Qwen3.5-9B-Instruct bash run.sh                       # both levels
MODEL=/models/Qwen3.5-9B-Instruct LEVELS="sentence" bash run.sh    # one level only
```

Environment variables: `MODEL`, `LEVELS`, `OUT_DIR`, `BATCH`, `MAX_LEN`, `DTYPE`,
`DEVICE`, `DEVICE_MAP`, `POOLING`, `LIMIT`, `SMOKE`, `OVERWRITE`, `PYTHON`
(see the header of `code/run.sh`).

### Step by step

```bash
cd code
export MODEL=/models/Qwen3.5-9B-Instruct

python 01_extract_features.py --levels dialogue sentence --model_path $MODEL --batch_size 8
python 02_probe.py --levels dialogue sentence                       # CPU is fine
python 03_report.py
```

### Multi-GPU

`CUDA_VISIBLE_DEVICES=0,2` only makes two GPUs *visible*; a model that fits on one card
is not sharded automatically.

```bash
# one process per GPU, one level each – near-linear speed-up (recommended)
CUDA_VISIBLE_DEVICES=0 LEVELS="dialogue" bash run.sh &
CUDA_VISIBLE_DEVICES=2 LEVELS="sentence" bash run.sh &
wait

# or one process across both GPUs (layer-wise sharding; slower, fits bigger models)
CUDA_VISIBLE_DEVICES=0,2 python 01_extract_features.py --levels sentence \
    --model_path $MODEL --device_map balanced --batch_size 8
```

---

## Outputs

Everything lands in `results/` (`--out_dir` / `OUT_DIR=` to change it):

```
results/
├── summary.json                         # the result file of this run (start here)
├── summary_<level>.json                 # the same, per level
├── per_layer_test_<level>.csv           # ACC / F1 / AUROC of every layer on the test split
├── subtype_auroc_<level>.{csv,png}      # probe AUROC per bias sub-type
├── layer_curve_<level>.png              # layer-wise curves with l* marked
└── {dialogue,sentence}_probe/
    ├── features_{train,val,test}.npz    # float16 [n, L+1, d] + labels + sample_ids + meta
    ├── manifest_{split}.jsonl           # sample_id / label / variant / bias_subtype
    ├── probe.json                       # per-layer val metrics, l*, test metrics (+CI), hparams
    ├── per_layer.csv                    # the same per-layer table as CSV
    ├── probe_ckpt.pt / test_scores.npz  # l* weights and test probabilities
    └── layer_curve_<level>.png          # written by step 2
```

`summary.json` is a single JSON document; the headline numbers are in `runs`, the
details per level under `levels`:

```jsonc
{
  "generated_at": "2026-01-01 12:00:00",
  "backbone": "/models/Qwen3.5-9B-Instruct",
  "features": {"pooling": "last", "max_length": 1536, "input_mode": "task"},
  "runs": [                                   // one flat row per level – feed straight to pandas
    {"level": "dialogue", "method": "linear_probe", "layer": 31, "n_test": 928,
     "accuracy": 0.7004, "precision": 0.7330, "recall": 0.5837, "f1": 0.6499,
     "auroc": 0.7578, "accuracy_ci95": [...], "f1_ci95": [...], "auroc_ci95": [...]}
  ],
  "levels": {
    "dialogue": {
      "split_sizes":     {"train": 7752, "val": 1012, "test": 928},
      "layer_selection": {"criterion": "val_acc", "best_layer": 31,
                          "best_layer_val_acc": 0.7045, "best_layer_val_auroc": 0.7875},
      "probe":           {"accuracy": 0.7004, "f1": 0.6499, "auroc": 0.7578,
                          "confusion_matrix": {"tp": 258, "tn": 392, "fp": 94, "fn": 184}},
      "probe_hparams":   {...},
      "per_layer":       [{"layer": 0, "val_acc": ..., "test_auroc": ...}, ...],
      "subtype_auroc":   [{"subtype": "isolated_entity", "n_pos": 250, "auroc": ...}, ...]
    },
    "sentence": { ... }
  }
}
```

---

## License and ethics

The **code** is MIT licensed (see [`LICENSE`](LICENSE)). The **data** are derived from
MedDG and released for non-commercial research use only, subject to the terms of the
original corpus. The dialogues are simulated clinical consultations — not real patient
records — and must never be used for diagnosis or any clinical decision.
