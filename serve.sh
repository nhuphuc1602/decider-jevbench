#!/usr/bin/env bash
# One-command serving of the decider for JevBench (TypeSafe /v1/systemone wire format).
#
#   ./serve.sh                       # downloads google/gemma-4-31B-it @ the pinned revision from Hugging Face, serves on :8040
#   MODEL=/path/to/snapshot ./serve.sh   # or point at a local copy of that exact revision
#   QUANT=fp8 ./serve.sh                 # on-the-fly FP8 weights (vLLM), for cards with less than ~80 GB free
#
# Needs one GPU with >= 80 GB (bf16 weights 62.5 GB + paged KV for 90k-token prompts via chunked prefill) and the pinned
# Python packages from requirements.txt (vllm 0.31.0 brings torch 2.13.0).  Two processes: stock `vllm serve` (OpenAI-compatible
# chat completions with logprobs) and the shim `decider.server`, which builds the prompt, reads the option-letter logprobs of ONE
# forward pass, applies the per-type temperatures from decider_config.json and answers /v1/systemone.  Nothing is generated
# (max_tokens=1, output_tokens=0).
set -euo pipefail
cd "$(dirname "$0")"
PORT=${PORT:-8040}; VPORT=${VPORT:-8891}; GPU=${GPU:-0}; CFG=${CFG:-decider_config.json}
# QUANT=fp8 serves the same weights with vLLM's on-the-fly FP8 (needed only if the card has < ~80 GB free; see README)
QARG=(); [ -n "${QUANT:-}" ] && QARG=(--quantization "$QUANT")
REPO=$(python3 -c "import json; print(json.load(open('$CFG'))['base_model']['repo'])")
REV=$(python3 -c "import json; print(json.load(open('$CFG'))['base_model']['revision'])")
STYLE=$(python3 -c "import json; print(json.load(open('$CFG'))['prompt_style'])")
MODEL=${MODEL:-$REPO}
export CUDA_VISIBLE_DEVICES=$GPU
REVARG=(); [ "$MODEL" = "$REPO" ] && REVARG=(--revision "$REV")
vllm serve "$MODEL" "${REVARG[@]}" "${QARG[@]}" --served-model-name decider --host 127.0.0.1 --port "$VPORT" \
    --max-model-len ${MAXLEN:-90000} --gpu-memory-utilization ${GPU_UTIL:-0.92} --enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 8 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$VPORT/v1/models" >/dev/null; do sleep 5; kill -0 $VLLM_PID 2>/dev/null || { echo "vllm exited"; exit 1; }; done
exec python3 -m decider.server --vllm "http://127.0.0.1:$VPORT" --model decider --name decider \
    --host 0.0.0.0 --port "$PORT" --prompt-style "$STYLE" --calib "$CFG"
