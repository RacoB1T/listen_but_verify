#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command bias detection pipeline: extract features -> train the probe ->
# aggregate the results.
#
#   bash code/run.sh                            # full data, settings from config
#   SMOKE=1 MODEL=/path/to/Qwen3.5-0.8B DEVICE=cpu DTYPE=float32 bash code/run.sh
#   MODEL=/path/to/Qwen3.5-9B LEVELS="dialogue" bash code/run.sh
#
# Environment variables
#   MODEL        path to the frozen backbone            (default: config.model_path)
#   LEVELS       "dialogue sentence"                    (default: config.levels)
#   OUT_DIR      output root                            (default: config.out_dir)
#   BATCH        extraction batch size                   (default: config.batch_size)
#   MAX_LEN      maximum sequence length                 (default: config.max_length)
#   DTYPE        float16 | bfloat16 | float32            (default: config.dtype)
#   DEVICE       cuda:0 | cpu                            (default: config.device)
#   DEVICE_MAP   auto | balanced | sequential | JSON     (default: single device)
#   POOLING      last | mean                             (default: config.pooling)
#   SMOKE        1 = quick small-sample run (limits from config.smoke)
#   LIMIT        keep N label-stratified samples per split (0 = all)
#   OVERWRITE    1 = ignore everything that already exists
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL="${MODEL:-}"
LEVELS="${LEVELS:-dialogue sentence}"
OUT_DIR="${OUT_DIR:-}"
BATCH="${BATCH:-}"
MAX_LEN="${MAX_LEN:-}"
DTYPE="${DTYPE:-}"
DEVICE="${DEVICE:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
POOLING="${POOLING:-}"
SMOKE="${SMOKE:-0}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
PYTHON="${PYTHON:-python3}"

COMMON=(--levels $LEVELS)
[[ -n "$OUT_DIR" ]] && COMMON+=(--out_dir "$OUT_DIR")

# step 1 (extraction) accepts every knob
EXTRACT=("${COMMON[@]}")
[[ -n "$MODEL" ]]      && EXTRACT+=(--model_path "$MODEL")
[[ -n "$BATCH" ]]      && EXTRACT+=(--batch_size "$BATCH")
[[ -n "$MAX_LEN" ]]    && EXTRACT+=(--max_length "$MAX_LEN")
[[ -n "$DTYPE" ]]      && EXTRACT+=(--dtype "$DTYPE")
[[ -n "$DEVICE" ]]     && EXTRACT+=(--device "$DEVICE")
[[ -n "$DEVICE_MAP" ]] && EXTRACT+=(--device_map "$DEVICE_MAP")
[[ -n "$POOLING" ]]    && EXTRACT+=(--pooling "$POOLING")
[[ "$SMOKE" == "1" ]]  && EXTRACT+=(--smoke)
[[ "$LIMIT" != "0" ]]  && EXTRACT+=(--limit "$LIMIT")
[[ "$OVERWRITE" == "1" ]] && EXTRACT+=(--overwrite)

# step 2 (probe training) has its own, smaller set of knobs
PROBE_ARGS=("${COMMON[@]}")
[[ "$SMOKE" == "1" ]]  && PROBE_ARGS+=(--smoke)
[[ "$OVERWRITE" == "1" ]] && PROBE_ARGS+=(--overwrite)

# step 3 (aggregation) only reads what the first two steps produced
REPORT=("${COMMON[@]}")
[[ "$SMOKE" == "1" ]] && REPORT+=(--smoke)

# reduce fragmentation-driven CUDA OOM during the extraction pass
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=============================================================="
echo " bias detection pipeline"
echo "   levels      : $LEVELS"
echo "   model       : ${MODEL:-<from config>}"
echo "   output      : ${OUT_DIR:-<from config>}"
echo "   smoke=$SMOKE limit=$LIMIT overwrite=$OVERWRITE"
echo "=============================================================="

echo
echo "--- [1/3] extracting per-layer hidden states (the slow step) ---"
"$PYTHON" 01_extract_features.py "${EXTRACT[@]}"

echo
echo "--- [2/3] training the linear probe on every layer ---"
"$PYTHON" 02_probe.py "${PROBE_ARGS[@]}"

echo
echo "--- [3/3] aggregating the results ---"
"$PYTHON" 03_report.py "${REPORT[@]}"

echo
DEFAULT_OUT="../results"
[[ "$SMOKE" == "1" ]] && DEFAULT_OUT="../results_smoke"
echo "done -> ${OUT_DIR:-$DEFAULT_OUT}/summary.json"
