"""JevBench decider: /v1/systemone shim over a stock vLLM server (TypeSafe wire format).

  vllm serve google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --served-model-name decider --port 8891 \
      --max-model-len 90000 --gpu-memory-utilization 0.92 --enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 8
  python -m decider.server --vllm http://127.0.0.1:8891 --model decider --port 8040 --prompt-style cygnet-short --calib decider_config.json

Readout: one chat completion per decision with max_tokens=1, logprobs + top_logprobs=20 and structured output restricted to the
option letters; the probability mass of each letter (all surface tokens of that letter summed) is renormalised over the letters,
tempered per type (p^(1/T)) with the temperatures in --calib, and for noul pushed out of the abstention band by the commit floor.
usage.input_tokens is vLLM's prompt_tokens; output_tokens is 0.  Long states are never refused: on a context-length error the
state's head and tail are kept and the request retried (warning "state_truncated").  --route (optional) serves different vLLM
model names per question type; not used by the submitted configuration.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .prompt import CYGNET_LETTERS, CYGNET_SYSTEM, SHORT_INSTRUCTION, cygnet_options, cygnet_prompt

TOK_LETTER = re.compile(r"^[\s(\[{'\"]*([A-Za-z])[\s.,:)\]}'\"]*$")
app = FastAPI()
S: dict = {}


def messages_for(style, state, question):
    items = cygnet_options(question)
    user = cygnet_prompt(state, question.get("instructions") or "", items)
    if style == "cygnet-short":
        msgs = [{"role": "user", "content": SHORT_INSTRUCTION + user}]
    elif style == "cygnet-nosys":
        msgs = [{"role": "user", "content": user}]
    else:
        msgs = [{"role": "system", "content": CYGNET_SYSTEM}, {"role": "user", "content": user}]
    return msgs, items


def letter_mass(top_logprobs, n):
    acc = [0.0] * n
    for d in top_logprobs:
        m = TOK_LETTER.match(d.get("token", ""))
        if not m:
            continue
        i = CYGNET_LETTERS.find(m.group(1).upper())
        if 0 <= i < n:
            acc[i] += math.exp(d["logprob"])
    tot = sum(acc)
    return [x / tot for x in acc] if tot > 0 else None


def temper(p, T):
    z = [max(x, 1e-12) ** (1.0 / T) for x in p]
    s = sum(z)
    return [x / s for x in z]


def commit(p_yes, c):
    if c <= 0.5:
        return p_yes
    if 0.5 <= p_yes < c:
        return c
    if 1 - c < p_yes < 0.5:
        return 1 - c
    return p_yes


@app.get("/health")
def health():
    return {"status": "ok", "model": S["name"]}


@app.post("/v1/systemone")
async def systemone(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        return JSONResponse({"error": "questions must be a non-empty object"}, status_code=422)
    state = body.get("state", "")
    answers, total_tokens = {}, 0
    truncated = False
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score"):
            return JSONResponse({"error": f"question {qid!r}: unsupported type"}, status_code=422)
        try:
            msgs, items = messages_for(S["style"], state, q)
        except ValueError as e:
            return JSONResponse({"error": f"question {qid!r}: {e}"}, status_code=422)
        n = len(items)
        if n > len(CYGNET_LETTERS):
            return JSONResponse({"error": f"{n} options exceed 26"}, status_code=422)
        letters = list(CYGNET_LETTERS[:n])
        rt = S["routes"].get(q["type"], S["default"])
        payload = {"model": rt["model"], "messages": msgs, "max_tokens": 1, "temperature": 1.0, "logprobs": True,
                   "top_logprobs": 20, "structured_outputs": {"choice": letters},
                   "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = await S["client"].post(S["vllm"] + "/v1/chat/completions", json=payload, timeout=S["timeout"])
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": f"vLLM unreachable: {e}"}, status_code=502)
        # never refuse a long state: on a context-limit error, keep the head and tail of the state and retry (like Quyet)
        truncated = False
        for attempt in range(6):
            if r.status_code == 200:
                break
            detail = r.text[:300]
            if r.status_code == 400 and re.search(r"maximum context length|max_model_len|too long", detail, re.I):
                st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
                keep = int(len(st) * 0.7)
                st = st[: keep // 2] + "\n[...]\n" + st[-(keep - keep // 2):]
                state = st
                truncated = True
                msgs, items = messages_for(S["style"], state, q)
                payload["messages"] = msgs
                try:
                    r = await S["client"].post(S["vllm"] + "/v1/chat/completions", json=payload, timeout=S["timeout"])
                except Exception as e:  # noqa: BLE001
                    return JSONResponse({"error": f"vLLM unreachable: {e}"}, status_code=502)
                continue
            return JSONResponse({"error": f"vLLM HTTP {r.status_code}: {detail}"}, status_code=502)
        if r.status_code != 200:
            return JSONResponse({"error": f"vLLM HTTP {r.status_code}: {r.text[:300]}"}, status_code=502)
        d = r.json()
        try:
            top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            return JSONResponse({"error": "vLLM returned no logprobs"}, status_code=502)
        p = letter_mass(top, n)
        if p is None:
            return JSONResponse({"error": "no option letter in the top logprobs"}, status_code=502)
        qtype = q["type"]
        temps = rt["temps"]
        bucket = "choice_prob" if (qtype == "choice" and re.search(r"probabilit", q.get("instructions") or "", re.I) and "choice_prob" in temps) else qtype
        p = temper(p, temps.get(bucket, temps.get(qtype, 1.0)))
        keys = [k for k, _ in items]
        total_tokens += int((d.get("usage") or {}).get("prompt_tokens") or 0)
        if qtype == "noul":
            p_yes = p[[str(k).lower() for k in keys].index("true")]
            p_yes = commit(p_yes, rt["noul_commit"])
            answers[qid] = {"type": "noul", "noul": round(p_yes, 6), "confidence": round(max(p_yes, 1 - p_yes), 6)}
        elif qtype == "choice":
            i = max(range(n), key=p.__getitem__)
            answers[qid] = {"type": "choice", "choice": keys[i], "confidence": round(p[i], 6),
                            "probabilities": {k: round(x, 6) for k, x in zip(keys, p)}}
        else:
            ev = sum(i * x for i, x in enumerate(p))
            answers[qid] = {"type": "score", "score": round(ev, 6), "confidence": round(max(p), 6),
                            "legend": {k: t for k, t in items}, "probabilities": {k: round(x, 6) for k, x in zip(keys, p)}}
    warnings = [{"code": "state_truncated"}] if truncated else []
    return {"model": S["name"], "answers": answers, "usage": {"input_tokens": total_tokens, "output_tokens": 0}, "warnings": warnings}


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default="http://127.0.0.1:8890")
    ap.add_argument("--model", default="decider", help="served-model-name on the vLLM side")
    ap.add_argument("--name", default="decider")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--prompt-style", default="cygnet-short", choices=["cygnet", "cygnet-short", "cygnet-nosys"])
    ap.add_argument("--calib", default=None, help="json with temperatures {type: T} and noul_commit")
    ap.add_argument("--timeout", type=float, default=110.0)
    ap.add_argument("--route", default=None, help="json: {routes: {type: key}, models: {key: {model, calib}}} -> per-type served model + calibration (type-routed adapter serving)")
    args = ap.parse_args()
    cal = json.load(open(args.calib)) if args.calib else {}
    default = {"model": args.model, "temps": cal.get("temperatures", {}), "noul_commit": float(cal.get("noul_commit", 0.0))}
    routes = {}
    if args.route:
        rc = json.load(open(args.route))
        models = {}
        for key, m in rc["models"].items():
            c = json.load(open(m["calib"])) if m.get("calib") else {}
            models[key] = {"model": m["model"], "temps": c.get("temperatures", {}), "noul_commit": float(c.get("noul_commit", 0.0))}
        routes = {t: models[k] for t, k in rc["routes"].items()}
    S.update(vllm=args.vllm.rstrip("/"), model=args.model, name=args.name, style=args.prompt_style, default=default, routes=routes,
             timeout=args.timeout, client=httpx.AsyncClient())
    print(f"[shim] vLLM {S['vllm']} default {default}; prompt {args.prompt_style}; routes " +
          ", ".join(f"{t}->{r['model']} T{r['temps']} c{r['noul_commit']}" for t, r in routes.items()), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
