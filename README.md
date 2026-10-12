# Decider — a Jev-class typed-decision system on frozen Gemma-4-31B-it

One forward pass of **google/gemma-4-31B-it** (bf16, pinned revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, Apache-2.0) per
decision. The option letters' next-token probabilities at the answer position are read out, tempered per question type, and returned
as the decision distribution. Nothing is generated. No fine-tuning: the weights are Google's, unmodified; everything specific to this
entry is the prompt scaffold, the readout and the calibration policy in `decider_config.json`.

## Wire format

`POST /v1/systemone` (TypeSafe-compatible), one or more questions per request, each answered by one forward pass:

```json
{"state": "<string or JSON>", "model": "decider",
 "questions": {"decision": {"type": "choice|noul|score", "instructions": "...", "criteria": {...} | [...]}}}
```
Response: `{"answers": {"decision": {...}}, "usage": {"input_tokens": N, "output_tokens": 0}, "warnings": []}` where
`noul` -> `{"type": "noul", "noul": P(yes), "confidence"}`, `choice` -> `{"choice", "confidence", "probabilities": {label: p}}` keyed by
the request's labels, `score` -> `{"score": E[level], "confidence", "probabilities": {"0": p0, "1": p1, ...}}` keyed by level index.
Distributions sum to 1. `GET /health` -> 200. Up to 26 options; states of any length are answered (above 90k tokens the state's head
and tail are kept and `warnings` carries `state_truncated`); a question is never refused with 422 unless it is malformed.

## Run (one GPU with >= 80 GB; measured on an NVIDIA B200 and on an H100 80GB)

System prerequisites: Linux x86-64, an NVIDIA driver that supports CUDA 13 (the vllm 0.31.0 wheels are built against CUDA 13.0 and need
no CUDA toolkit), `gcc`, and **Python 3.12 with its development headers** (`python3.12-dev`; vLLM's torch.compile/Triton path compiles a
small C helper at start-up and fails with `Python.h: No such file or directory` without them). A `uv`-managed interpreter
(`uv python install 3.12`) ships the headers and works without root.

```bash
python3.12 -m venv .venv && . .venv/bin/activate     # or: uv venv --python 3.12 .venv && . .venv/bin/activate
pip install -r requirements.txt          # vllm==0.31.0 (brings torch 2.13.0 / CUDA 13 wheels), fastapi, httpx, uvicorn  (~8 GB)
./serve.sh                               # downloads the pinned revision from Hugging Face on first use (62.5 GB); serves on :8040
curl -s localhost:8040/health
```
`serve.sh` starts stock `vllm serve google/gemma-4-31B-it --revision 842da379... --served-model-name decider --max-model-len 90000
--enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 8` and then `python3 -m decider.server`, which reads the prompt
style and the calibration policy from `decider_config.json`. `MODEL=/path/to/snapshot ./serve.sh` serves a local copy of the same
revision; `GPU=<index>`, `PORT`, `VPORT`, `GPU_UTIL` (default 0.92; use 0.85 on a 96 GB card shared with anything else) are optional.
Weights are 62.5 GB in bf16; nothing else is downloaded.

## What one decision does (`decider/prompt.py`, `decider/server.py`)

1. Prompt (`cygnet-short` style, a "Cygnet"/deck-31B-type scaffold): a single user turn
   `Choose exactly one option; reply with its letter only.` + blank line + the state + blank line + the question's instructions +
   `Options:` with one line per option `A. <description>` ... + `Answer with the letter of exactly one option, and nothing else:`.
   Option order is the request's order (choice: `criteria` order; noul: the request's own true/false order; score: level index order).
2. Readout: one chat completion with `max_tokens=1`, `logprobs` + `top_logprobs=20`, structured output restricted to the option
   letters; the probability mass of each letter (all surface forms of that letter summed) is renormalised over the letters.
3. Calibration policy (`decider_config.json`): per-type temperature `p_i^(1/T) / sum` with `choice` 5.9 (`choice_prob` 4.9 for choice
   questions whose instructions ask for probabilities), `noul` 7.2, `score` 1.0; then the noul commit floor: P(yes) is pushed out of the
   interval (0.1, 0.9) to its nearer side, so no noul answer lands in the 0.2-0.8 abstention band.
4. `usage.input_tokens` is vLLM's prompt token count for that forward pass; `output_tokens` is 0.

## Calibration: how the temperatures were chosen (disclosure)

- No JevBench item, public or sealed, was used to fit anything. The 231 public items were used once, after fitting, to measure
  (see RESULTS.md).
- Fitting data: our own generated decision items (rule-generated items with computed gold and LLM-authored items whose gold was
  accepted by two independent model families), split into an easy pool and two frozen-model-hard holdouts. For each type the fit set is
  a mixture of hard and easy items whose accuracy matches the published competence of this same frozen recipe on the full JevBench pool
  (deck-31B, v1.6.1), so that the temperatures are tuned for the difficulty the sealed items have rather than for our easier dev items.
- Objective per type: chance-corrected competence + the calibration term of the JevBench method (choice: top-label ECE over hard
  items, 10 bins; noul: ECE of P(yes); score: 50*RPS + 100*top-level ECE), with the score temperature fixed at 1 because the score
  competence uses the expected level. Raw outputs were taken from this very serving path (vLLM logprobs), not from a different readout.
- `results/calibration_fit.json` records the fitted policy and the files it was fitted on.

## Cost basis

Self-hosted open weights, no tariff -> the OpenRouter list price of the exact base model (google/gemma-4-31b-it: $0.09 per million
input tokens, $0.34 per million output tokens) x measured `usage.input_tokens`; output tokens are always 0. Measured mean 651 input
tokens per decision on the public items -> $0.0586 per 1,000 decisions (cap $0.0646). Latency: serial p50 67 ms on a B200 through the
official CLI (adjusted 2x + 0.15 s = 0.28 s; cap 1.233 s).

## Repository layout

| path | role |
|---|---|
| `serve.sh` | one-command serving (vLLM + shim) |
| `decider/prompt.py` | prompt scaffold and option normalisation |
| `decider/server.py` | the `/v1/systemone` shim (FastAPI) |
| `decider_config.json` | base model + revision, prompt style, temperatures, noul commit floor, limits, calibration disclosure |
| `requirements.txt` | pinned versions |
| `RESULTS.md`, `results/` | own public-set measurements (official CLI output) and the calibration fit record |
| `LICENSE`, `NOTICE` | Apache-2.0 (code); Gemma 4 is Apache-2.0 |

## Contact

nhuphuc1602@gmail.com — GitHub `nhuphuc1602`.
