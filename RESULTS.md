# Own measurements (not official)

Serving path exactly as `serve.sh` (vLLM 0.31.0 + `decider.server`, prompt `cygnet-short`, temperatures from `decider_config.json`),
one NVIDIA B200, official JevBench CLI (`jevbench.cli run --adapter typesafe`, serial, 231 public items, 0 failures).

| | value |
|---|---|
| correct (argmax) | 213 / 231 (easy 48/48, standard 70/72, hard 95/111) |
| local v1.6 scorer | Intelligence 80.1, Calibration 90.0, Capability 85.1 (public half only; the sealed half is unknown to us) |
| per type CC / calibration | choice 86.3 / 86.3, noul 88.6 / 96.1, score 65.5 / 87.7 |
| latency, serial | p50 67 ms, p95 176 ms raw -> adjusted p50 2x+0.15 s = 0.28 s (cap 1.233 s) |
| mean `usage.input_tokens` | 651 -> $0.0586 per 1,000 decisions at the OpenRouter list price of gemma-4-31b-it ($0.09/M input; 0 output tokens; cap $0.0646) |
| noul abstentions (P(yes) in 0.2-0.8) | 0 |

Files: `results/public_231_official_cli.jsonl` (the CLI's per-item results), `results/calibration_fit.json` (the fitted policy and what it was fitted on).
Calibration was fitted on our own generated decision items only (frozen-hard holdouts mixed with easier items so that per-type accuracy
matches the published competence of the same frozen recipe); never on JevBench items. Expect the public-set numbers above to be
noisier than the board's: the choice calibration term is an ECE over 67 hard items in 10 bins and moves by several points between runs.

## Clean-clone check (2026-10-12)

The repository was copied to a scratch directory, a fresh Python 3.12 environment was created from `requirements.txt` (uv; vllm 0.31.0,
torch 2.13.0), `./serve.sh` was started with `MODEL=` pointing at a local copy of the pinned revision, and the official CLI was run
against it: 231/231 answered, 0 failures, 213 correct, local v1.6 scorer I 80.2 / C 89.0 / cap 84.6, p50 53 ms, p95 178 ms. (The
calibration term differs by ±1 point between otherwise identical runs: vLLM batching is not bit-reproducible and the public choice ECE is
computed over 67 hard items.)
