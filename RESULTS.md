# Own measurements (not official)

Serving path exactly as `serve.sh` (vLLM 0.31.0, FP8, `decider.server`, prompt `cygnet-short`, temperatures from `decider_config.json`),
one NVIDIA H100 80GB, official JevBench CLI (`jevbench.cli run --adapter typesafe`, serial, 231 public items, 0 failures).

| | value |
|---|---|
| correct (argmax) | 213 / 231 |
| latency, serial, H100 | p50 34 ms, p95 228 ms raw -> adjusted p50 2x + 0.15 s = 0.22 s (cap 1.233 s) |
| mean `usage.input_tokens` | 651 -> $0.0586 per 1,000 decisions at the OpenRouter list price of gemma-4-31b-it ($0.09/M input; 0 output tokens; cap $0.0646) |
| noul abstentions (P(yes) in 0.2-0.8) | 0 |
| local v1.6 scorer on this run | Intelligence 74.9, Calibration 89.1, Capability 82.0 (public half only; see the note below) |

Files: `results/public_231_official_cli.jsonl` (the CLI's per-item results of this run), `results/calibration_fit.json` (the fitted policy
and the files it was fitted on). Calibration was fitted on our own generated decision items only (frozen-hard holdouts mixed with easier
items so that per-type accuracy matches the published competence of the same frozen recipe); never on JevBench items.

Note on the public-set numbers: the local Intelligence estimate moves by several points between otherwise equivalent runs because the
public set has only 6 hard score items (one hedged item moves the score competence by ~14 points) and the choice calibration term is an
ECE over 67 hard items in 10 bins. Over bf16 and FP8 serving on a B200 and an H100 we saw 212-214 correct, Intelligence 74.9-80.2 and
Calibration 88.2-92.2 for the same recipe. The sealed pool (1,200 items) averages this out.

## Reproducibility checks

- Clean clone, fresh Python 3.12 environment from `requirements.txt`, B200, bf16: 231/231 answered, 213 correct, p50 53 ms.
- H100 80GB, FP8 (the submitted configuration): 231/231 answered, 213 correct, p50 34 ms; KV cache 175k tokens at 90k max length.
- H100 80GB, bf16: starts only with `MAXLEN<=11000` (9.4 GiB of KV cache left); 212 correct with that limit on the public items, which
  are short; the sealed pool contains states of up to ~80k tokens, hence FP8.
