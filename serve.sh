#!/usr/bin/env bash
# One-command serving of the decider for JevBench (TypeSafe /v1/systemone wire format).
#
#   ./serve.sh                       # downloads google/gemma-4-31B-it @ the pinned revision from Hugging Face, serves on :8040
#   MODEL=/path/to/snapshot ./serve.sh   # or point at a local copy of that exact revision
#   QUANT=none ./serve.sh                # bf16 weights instead of the default FP8 (needs ~96 GB, or MAXLEN=11000 on 80 GB)
#
# Needs one GPU with >= 80 GB (bf16 weights 62.5 GB + paged KV for 90k-token prompts via chunked prefill) and the pinned
# Python packages from requirements.txt (vllm 0.31.0 brings torch 2.13.0).  Two processes: stock `vllm serve` (OpenAI-compatible
# chat completions with logprobs) and the shim `decider.server`, which builds the prompt, reads the option-letter logprobs of ONE
# forward pass, applies the per-type temperatures from decider_config.json and answers /v1/systemone.  Nothing is generated
# (max_tokens=1, output_tokens=0).
set -euo pipefail
cd "$(dirname "$0")"
PORT=${PORT:-8040}; VPORT=${VPORT:-8891}; GPU=${GPU:-0}; CFG=${CFG:-decider_config.json}
# Weights precision: FP8 by default (vLLM on-the-fly quantisation of the same bf16 checkpoint). This is the measured and calibrated
# configuration and the one that fits an 80 GB card with the full 90k-token context. QUANT=none serves bf16 (needs ~96 GB or
# MAXLEN<=11000 on 80 GB); the calibration file was fitted on the FP8 path, so bf16 is not the submitted configuration.
QUANT=${QUANT:-fp8}; QARG=(); [ "$QUANT" != "none" ] && QARG=(--quantization "$QUANT")
REPO=$(python3 -c "import json; print(json.load(open('$CFG'))['base_model']['repo'])")
REV=$(python3 -c "import json; print(json.load(open('$CFG'))['base_model']['revision'])")
STYLE=$(python3 -c "import json; print(json.load(open('$CFG'))['prompt_style'])")
MODEL=${MODEL:-$REPO}
export CUDA_VISIBLE_DEVICES=$GPU
# No just-in-time kernel builds at start-up: vLLM's FlashInfer sampler compiles CUDA ops with nvcc on first use, which fails on hosts
# without a CUDA toolkit (the wheels need only the driver). The PyTorch sampler is used instead; the readout only needs logprobs.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
REVARG=(); [ "$MODEL" = "$REPO" ] && REVARG=(--revision "$REV")
vllm serve "$MODEL" "${REVARG[@]}" "${QARG[@]}" --served-model-name decider --host 127.0.0.1 --port "$VPORT" \
    --max-model-len ${MAXLEN:-90000} --gpu-memory-utilization ${GPU_UTIL:-0.92} --enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 8 \
    --max-logprobs 64 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
ready() { python3 - "$VPORT" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=3); sys.exit(0)
except Exception:
    sys.exit(1)
PY
}
until ready; do sleep 5; kill -0 $VLLM_PID 2>/dev/null || { echo "vllm exited"; exit 1; }; done
exec python3 -m decider.server --vllm "http://127.0.0.1:$VPORT" --model decider --name decider \
    --host 0.0.0.0 --port "$PORT" --prompt-style "$STYLE" --calib "$CFG"
