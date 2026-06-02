#!/usr/bin/env bash
# Run inference on one dataset (default: visual_probe_easy).
#
# 1. Starts a local Ray head for tool actors on GPUs 0-3.
# 2. Launches vLLM on GPUs 4-7 in the background.
# 3. Waits for the OpenAI-compatible endpoint to come up.
# 4. Runs async_generate_with_tool_call_api against the dataset. The script
#    creates Ray tool actors itself via ToolRouter; no HTTP tool server is used.
# 5. Stops vLLM and the Ray head it started on exit.
#
# The dataset id auto-resolves to its parquet path + eval template via
# geo_edit.eval_datasets.DATASET_REGISTRY.
#
# Examples:
#   bash geo_edit/scripts/run_inference.sh                       # visual_probe_easy
#   DATASET=reason_map bash geo_edit/scripts/run_inference.sh    # any registered id
#   VLLM_PORT=8001 bash geo_edit/scripts/run_inference.sh        # tweak vLLM
set -euo pipefail

# ─── inference config ───
DATASET="${DATASET:-visual_probe_easy}"
PEDIA_DATA="${PEDIA_DATA:-./pedia_data}"
PEDIA_MODEL="${PEDIA_MODEL:-./pedia_model}"
MODEL_PATH="${MODEL_PATH:-${PEDIA_MODEL}/PEDIA_8B_v1}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/eval_results}"
OUT_DIR="${OUTPUT_ROOT}/${DATASET}/${MODEL_NAME}"

# ─── vLLM config ───
VLLM_PORT="${VLLM_PORT:-8000}"
API_BASE="${API_BASE:-http://127.0.0.1:${VLLM_PORT}}"
DP_SIZE="${DP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-5}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VLLM_LOG="${VLLM_LOG:-/tmp/log/vllm_${MODEL_NAME}.log}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-4,5,6,7}"

# ─── tool config (overridable via env) ───
USE_TOOLS="${USE_TOOLS:-auto}"
ENABLE_TOOLS="${ENABLE_TOOLS:-map general}"
RAY_PORT="${RAY_PORT:-6379}"
TOOL_NUM_GPUS="${TOOL_NUM_GPUS:-4}"
TOOL_CUDA_VISIBLE_DEVICES="${TOOL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NODE_RESOURCE="${NODE_RESOURCE:-tool_agent}"

mkdir -p "$OUT_DIR" "$(dirname "$VLLM_LOG")"

VLLM_PID=""
STARTED_RAY=0

cleanup() {
    if [ -n "$VLLM_PID" ]; then
        echo "[run_inference] stopping vLLM pid=$VLLM_PID"
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    if [ "$STARTED_RAY" = "1" ]; then
        echo "[run_inference] stopping Ray head on :$RAY_PORT"
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

export PEDIA_DATA PEDIA_MODEL

# ─── 1. Start local Ray head for ToolRouter-created actors ───
if [ "$USE_TOOLS" != "direct" ]; then
    if ray status --address="127.0.0.1:${RAY_PORT}" >/dev/null 2>&1; then
        echo "[run_inference] reusing existing Ray head on :$RAY_PORT"
    else
        echo "[run_inference] starting Ray head on GPUs $TOOL_CUDA_VISIBLE_DEVICES (${NODE_RESOURCE}=$TOOL_NUM_GPUS)"
        CUDA_VISIBLE_DEVICES="$TOOL_CUDA_VISIBLE_DEVICES" ray start --head --port="$RAY_PORT" \
            --num-gpus="$TOOL_NUM_GPUS" --resources="{\"${NODE_RESOURCE}\":${TOOL_NUM_GPUS}}"
        STARTED_RAY=1
    fi
fi

# ─── 2. Background-launch vLLM ───
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
echo "[run_inference] launching vLLM on GPUs $VLLM_CUDA_VISIBLE_DEVICES dp=$DP_SIZE tp=$TP_SIZE model=$MODEL_PATH log=$VLLM_LOG"
CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --data-parallel-size "$DP_SIZE" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype auto \
    --allowed-local-media-path "$PEDIA_DATA" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES_PER_PROMPT}}" \
    $EXTRA_VLLM_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# ─── 3. Wait for endpoint ───
echo "[run_inference] vLLM pid=$VLLM_PID — waiting for http://127.0.0.1:${VLLM_PORT}/v1/models"
until curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[run_inference] vLLM process died — last 30 log lines:"
        tail -n 30 "$VLLM_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "[run_inference] vLLM endpoint ready"

# ─── 4. Run inference ───
python -m geo_edit.scripts.async_generate_with_tool_call_api \
    --dataset "$DATASET" \
    --data_root "$PEDIA_DATA" \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --model_type vLLM --api_base "$API_BASE" \
    --temperature 0 --sample_rate 1.0 \
    --use_tools "$USE_TOOLS" --enable_tools $ENABLE_TOOLS \
    --node_resource "$NODE_RESOURCE" \
    --max_concurrent_requests 64 --max_tool_calls 10 \
    --no_image_compression

echo "[run_inference] done — output at $OUT_DIR"
