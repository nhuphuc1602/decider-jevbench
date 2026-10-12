"""Prompt construction for the decider (Cygnet-style scaffold), shared by every serving path.

The user turn is: state, blank line, the question's instructions, blank line, "Options:" with one lettered line per option
(A., B., ...), blank line, "Answer with the letter of exactly one option, and nothing else:".  Prompt styles:
  cygnet       : system prompt CYGNET_SYSTEM + the user turn
  cygnet-short : no system prompt; the user turn is prefixed with SHORT_INSTRUCTION   (the submitted configuration)
  cygnet-nosys : the bare user turn
Option order follows the request: choice = the order of `criteria`; noul = the request's own true/false order
(JevBench sends criteria with sorted keys, i.e. false first); score = level index order.
"""
from __future__ import annotations

import json

CYGNET_SYSTEM = ("You are a calibration engine. You never answer in prose. You are given a state, a question and "
                 "a numbered set of options, and you choose exactly one option. You reply with that option's "
                 "LETTER and nothing else \u2014 a single character, no words, no punctuation, no explanation.")
CYGNET_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SHORT_INSTRUCTION = "Choose exactly one option; reply with its letter only.\n\n"


def cygnet_options(question: dict):
    qtype, criteria = question["type"], question.get("criteria")
    if qtype == "score":
        if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
            raise ValueError("score needs a list of at least two level descriptions")
        items = [(str(i), v) for i, v in enumerate(criteria)]
    elif qtype == "noul":
        crit = dict(criteria) if isinstance(criteria, dict) else {}
        # the benchmark may send a noul without criteria; fall back to neutral texts (never refuse a valid question)
        items = [("true", crit.get("true") or "Yes / true"), ("false", crit.get("false") or "No / false")]
        if "false" in crit and "true" in crit and list(crit)[0] == "false":
            items = items[::-1]  # keep the request's own order, as the Cygnet recipe does
    else:
        if isinstance(criteria, dict):
            items = [(str(k), v if v is not None else str(k)) for k, v in criteria.items()]
        elif isinstance(criteria, (list, tuple)):
            items = [(str(k), str(k)) for k in criteria]
        else:
            raise ValueError("choice needs criteria as {label: description} or a list of labels")
        if len({k for k, _ in items}) != len(items):
            raise ValueError("choice labels must be unique")
    if not items or len(items) > len(CYGNET_LETTERS):
        raise ValueError(f"{len(items)} options")
    return items


def cygnet_prompt(state, instructions, items):
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    lines = [state_text.rstrip(), "", (instructions or "").rstrip(), "", "Options:"]
    for letter, (_lab, desc) in zip(CYGNET_LETTERS, items):
        lines.append(f"{letter}. {desc}")
    lines += ["", "Answer with the letter of exactly one option, and nothing else:"]
    return "\n".join(lines)
