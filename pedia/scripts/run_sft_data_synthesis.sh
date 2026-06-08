#!/usr/bin/env bash
# Synthesize SFT data from ReasonMap-Plus.
#
# Defaults:
#   raw HF data:    ./pedia_data/raw/reasonmap_plus/data/train*.parquet
#   prepared data:  ./pedia_data/raw/reasonmap_plus_prepared/train.parquet
#   sampler model:  gpt-5-2025-08-07 via OpenAI-compatible API
#   augment model:  ./pedia_model/Qwen3-VL-235B-A22B-Thinking served by vLLM
#   sample rate:    0.01 (1% example run)
#   outputs:        ./outputs/trajectories/reason_map_plus*
#                   ./pedia_data/pedia_sft
set -euo pipefail

PEDIA_MODEL="${PEDIA_MODEL:-./pedia_model}"
PEDIA_DATA="${PEDIA_DATA:-./pedia_data}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
RAW_DATASET_DIR="${RAW_DATASET_DIR:-${PEDIA_DATA}/raw/reasonmap_plus}"
RAW_DATASET_PATH="${RAW_DATASET_PATH:-${RAW_DATASET_DIR}/data/${DATASET_SPLIT}.parquet}"
PREPARED_DATASET_DIR="${PREPARED_DATASET_DIR:-${PEDIA_DATA}/raw/reasonmap_plus_prepared}"
DATASET_PATH="${DATASET_PATH:-${PREPARED_DATASET_DIR}/${DATASET_SPLIT}.parquet}"
FORCE_PREPARE_DATASET="${FORCE_PREPARE_DATASET:-0}"
DATASET_NAME="${DATASET_NAME:-reason_map_plus}"
SFT_DATASET_NAME="${SFT_DATASET_NAME:-reasonmap_plus}"
SFT_DATA_SOURCE="${SFT_DATA_SOURCE:-reasonmap_plus}"

TRAJ_ROOT="${TRAJ_ROOT:-./outputs/trajectories}"
TRAJ_DIR="${TRAJ_DIR:-${TRAJ_ROOT}/${DATASET_NAME}}"
AUG_DIR="${AUG_DIR:-${TRAJ_ROOT}/${DATASET_NAME}_augmented}"
SFT_OUT_DIR="${SFT_OUT_DIR:-${PEDIA_DATA}/pedia_sft}"

# Stage 1: iterative sampling with an OpenAI-compatible API model.
API_KEY="${API_KEY:-${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}}"
API_BASE="${API_BASE:-https://api.openai.com/v1}"
SAMPLING_MODEL="${SAMPLING_MODEL:-gpt-5-2025-08-07}"
SAMPLING_MODEL_TYPE="${SAMPLING_MODEL_TYPE:-OpenAI}"
SAMPLE_RATE="${SAMPLE_RATE:-0.01}"
N_TRAJECTORIES="${N_TRAJECTORIES:-1}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-32}"
MAX_ITERATIVE_ROUNDS="${MAX_ITERATIVE_ROUNDS:-4}"

# Stage 2: filter + diversify with a local vLLM endpoint.
AUG_MODEL_PATH="${AUG_MODEL_PATH:-${PEDIA_MODEL}/Qwen3-VL-235B-A22B-Thinking}"
AUG_MODEL_NAME="${AUG_MODEL_NAME:-$(basename "$AUG_MODEL_PATH")}"
VLLM_PORT="${VLLM_PORT:-8000}"
AUG_API_BASE="${AUG_API_BASE:-http://127.0.0.1:${VLLM_PORT}/v1}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-5}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VLLM_LOG="${VLLM_LOG:-/tmp/log/vllm_${AUG_MODEL_NAME}_sft_synthesis.log}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ALLOWED_MEDIA_PATH="${ALLOWED_MEDIA_PATH:-$(pwd)}"

JUDGE_API_KEY="${JUDGE_API_KEY:-$API_KEY}"
JUDGE_API_BASE="${JUDGE_API_BASE:-$API_BASE}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
AUG_API_KEY="${AUG_API_KEY:-$API_KEY}"
AUG_MODEL="${AUG_MODEL:-$AUG_MODEL_PATH}"
AUG_MAX_CONCURRENT="${AUG_MAX_CONCURRENT:-16}"
AUG_MAX_LLM_WORKERS="${AUG_MAX_LLM_WORKERS:-32}"
AUG_TEMPERATURE="${AUG_TEMPERATURE:-1.0}"
AUG_REQUESTS_PER_MINUTE="${AUG_REQUESTS_PER_MINUTE:-128}"

ENABLE_TOOLS="${ENABLE_TOOLS:-map general}"
RAY_PORT="${RAY_PORT:-6379}"
TOOL_NUM_GPUS="${TOOL_NUM_GPUS:-4}"
TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NODE_RESOURCE="${NODE_RESOURCE:-tool_agent}"

: "${API_KEY:?Set API_KEY, JUDGE_API_KEY, or OPENAI_API_KEY before running.}"
: "${JUDGE_API_KEY:?Set JUDGE_API_KEY, API_KEY, or OPENAI_API_KEY before running.}"

mkdir -p "$TRAJ_DIR" "$AUG_DIR" "$SFT_OUT_DIR" "$(dirname "$VLLM_LOG")"

VLLM_PID=""
STARTED_RAY=0

stop_ray_if_started() {
    if [ "$STARTED_RAY" = "1" ]; then
        echo "[run_sft_data_synthesis] stopping Ray head on :$RAY_PORT"
        ray stop --force >/dev/null 2>&1 || true
        STARTED_RAY=0
    fi
}

cleanup() {
    if [ -n "$VLLM_PID" ]; then
        echo "[run_sft_data_synthesis] stopping vLLM pid=$VLLM_PID"
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    stop_ray_if_started
}
trap cleanup EXIT

unset ROCR_VISIBLE_DEVICES
export PEDIA_DATA PEDIA_MODEL

if [ "$FORCE_PREPARE_DATASET" = "1" ] || [ ! -f "$DATASET_PATH" ]; then
    echo "[run_sft_data_synthesis] preparing ReasonMap-Plus data: $RAW_DATASET_PATH -> $DATASET_PATH"
    RAW_DATASET_DIR="$RAW_DATASET_DIR" \
    RAW_DATASET_PATH="$RAW_DATASET_PATH" \
    DATASET_SPLIT="$DATASET_SPLIT" \
    DATASET_PATH="$DATASET_PATH" \
    python - <<'PY'
import base64
import binascii
import io
import os
from pathlib import Path

from datasets import Dataset, Image, load_dataset

raw_root = Path(os.environ["RAW_DATASET_DIR"])
raw_path = Path(os.environ["RAW_DATASET_PATH"])
split = os.environ["DATASET_SPLIT"]
out_path = Path(os.environ["DATASET_PATH"])
allowed_types = {"Counting1", "Counting2", "Counting3", "TorF1", "TorF2"}
raw_files = [raw_path]

if not raw_path.exists():
    shard_dir = raw_path.parent
    candidates = sorted(shard_dir.glob(f"{split}-*.parquet"))
    if candidates:
        raw_files = candidates
        raw_path = candidates[0]
    else:
        raise SystemExit(
            f"Raw ReasonMap-Plus parquet not found: {raw_path}. "
            "Download FSCCS/ReasonMap-Plus to RAW_DATASET_DIR first."
        )


def _candidate_paths(value: str):
    clean = value[2:] if value.startswith("./") else value
    path = Path(value)
    if path.is_absolute():
        yield path
    yield raw_path.parent / value
    yield raw_path.parent / clean
    yield raw_root / value
    yield raw_root / clean


def _image_bytes(value):
    if isinstance(value, dict) and value.get("bytes"):
        return value["bytes"]
    if isinstance(value, dict) and value.get("path"):
        value = value["path"]
    if hasattr(value, "save"):
        buffer = io.BytesIO()
        value.save(buffer, format=getattr(value, "format", None) or "PNG")
        return buffer.getvalue()
    if value is None:
        raise FileNotFoundError("missing figure/image value")

    text = str(value)
    for candidate in _candidate_paths(text):
        if candidate.is_file():
            return candidate.read_bytes()

    encoded = text.split(",", 1)[1] if text.startswith("data:image") and "," in text else text
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        decoded = b""
    if decoded.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
        return decoded

    searched = ", ".join(str(path) for path in _candidate_paths(text))
    raise FileNotFoundError(
        f"Could not resolve ReasonMap-Plus figure/image '{text}'. "
        f"Searched: {searched}"
    )


dataset = load_dataset("parquet", data_files=[str(path) for path in raw_files], split="train")
records = []
for row_index, item in enumerate(dataset):
    qtype = str(item.get("type", ""))
    if qtype not in allowed_types:
        continue

    record = dict(item)
    record["id"] = str(
        item.get("id")
        or item.get("question_id")
        or f"reason_map_plus_{row_index}"
    )
    image_value = record.get("image") or record.get("figure")
    record["image"] = {"bytes": _image_bytes(image_value), "path": None}
    records.append(record)

if not records:
    raise SystemExit(
        f"No supported ReasonMap-Plus rows found in {raw_path}; "
        f"expected types: {', '.join(sorted(allowed_types))}."
    )

out_path.parent.mkdir(parents=True, exist_ok=True)
prepared = Dataset.from_list(records).cast_column("image", Image(decode=False))
prepared.to_parquet(str(out_path))
print(f"[run_sft_data_synthesis] prepared {len(records)} rows at {out_path}")
PY
fi

if ray status --address="127.0.0.1:${RAY_PORT}" >/dev/null 2>&1; then
    echo "[run_sft_data_synthesis] reusing existing Ray head on :$RAY_PORT"
else
    echo "[run_sft_data_synthesis] starting Ray head on GPUs $TOOL_CUDA_VISIBLE_DEVICES (${NODE_RESOURCE}=$TOOL_NUM_GPUS)"
    CUDA_VISIBLE_DEVICES="$TOOL_CUDA_VISIBLE_DEVICES" ray start --head --port="$RAY_PORT" \
        --num-gpus="$TOOL_NUM_GPUS" --resources="{\"${NODE_RESOURCE}\":${TOOL_NUM_GPUS}}"
    STARTED_RAY=1
fi

python -m pedia.scripts.iterative_sampling_generate \
    --api_key "$API_KEY" \
    --api_base "$API_BASE" \
    --model_name_or_path "$SAMPLING_MODEL" \
    --model_type "$SAMPLING_MODEL_TYPE" \
    --dataset_path "$DATASET_PATH" \
    --dataset_split "$DATASET_SPLIT" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$TRAJ_DIR" \
    --sample_rate "$SAMPLE_RATE" \
    --n_trajectories "$N_TRAJECTORIES" \
    --max_concurrent_requests "$MAX_CONCURRENT_REQUESTS" \
    --max_iterative_rounds "$MAX_ITERATIVE_ROUNDS" \
    --judge_model "$JUDGE_MODEL" \
    --judge_api_key "$JUDGE_API_KEY" \
    --judge_api_base "$JUDGE_API_BASE" \
    --node_resource "$NODE_RESOURCE" \
    --enable_tools $ENABLE_TOOLS

stop_ray_if_started

export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
echo "[run_sft_data_synthesis] launching vLLM on GPUs $VLLM_CUDA_VISIBLE_DEVICES dp=$DP_SIZE tp=$TP_SIZE model=$AUG_MODEL_PATH log=$VLLM_LOG"
CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$AUG_MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --data-parallel-size "$DP_SIZE" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype auto \
    --allowed-local-media-path "$ALLOWED_MEDIA_PATH" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES_PER_PROMPT}}" \
    $EXTRA_VLLM_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "[run_sft_data_synthesis] vLLM pid=$VLLM_PID - waiting for http://127.0.0.1:${VLLM_PORT}/v1/models"
until curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[run_sft_data_synthesis] vLLM process died - last 30 log lines:"
        tail -n 30 "$VLLM_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "[run_sft_data_synthesis] vLLM endpoint ready"

python -m pedia.data_preprocess.augment_traj_data \
    --src-dir "$TRAJ_DIR" \
    --dst-dir "$AUG_DIR" \
    --api-base "$AUG_API_BASE" \
    --api-key "$AUG_API_KEY" \
    --model "$AUG_MODEL" \
    --judge-api-base "$JUDGE_API_BASE" \
    --judge-model "$JUDGE_MODEL" \
    --judge-api-key "$JUDGE_API_KEY" \
    --filter-wrong-answers \
    --filter-answer-leakage \
    --leakage-check-mode full \
    --filter-brute-force \
    --filter-tool-mismatch \
    --max-concurrent "$AUG_MAX_CONCURRENT" \
    --max-llm-workers "$AUG_MAX_LLM_WORKERS" \
    --temperature "$AUG_TEMPERATURE" \
    --requests-per-minute "$AUG_REQUESTS_PER_MINUTE" \
    --reuse-filter

python -m pedia.data_preprocess.convert_trajectory_to_sft \
    --src_dir "$AUG_DIR" \
    --dst_dir "$SFT_OUT_DIR" \
    --dataset_name "$SFT_DATASET_NAME" \
    --data_source "$SFT_DATA_SOURCE" \
    --enable_tools $ENABLE_TOOLS

echo "[run_sft_data_synthesis] done - SFT data at $SFT_OUT_DIR"
