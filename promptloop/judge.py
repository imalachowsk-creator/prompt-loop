"""LLM-as-judge: a second model grades each answer against the retrieved
documentation, and we measure how often it agrees with the keyword metric.

Why both? The keyword metric is cheap and deterministic but brittle (a correct
answer phrased differently fails). A judge understands meaning but can be
wrong or inconsistent. Where they disagree is exactly where a human should look.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

from .evaluate import CaseResult

JUDGE_SYSTEM = """You are an ANSWER JUDGE for a documentation assistant.
You get the documentation CONTEXT the assistant saw, the QUESTION and the ANSWER.
Decide:
- correct: true if the answer correctly answers the question using the context.
  If the context does not contain the answer, the ONLY correct answer is saying
  it does not know.
- grounded: true if every claim in the answer is supported by the context.
Reply with JSON only, e.g. {"correct": true, "grounded": true, "reason": "short reason"}"""


@dataclass
class Judgement:
    id: str
    correct: bool
    grounded: bool
    reason: str
    metric_pass: bool   # what the keyword metric said

    @property
    def agrees(self) -> bool:
        return self.correct == self.metric_pass


def parse_judgement(text: str) -> tuple[bool, bool, str]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return bool(d.get("correct")), bool(d.get("grounded")), str(d.get("reason", ""))
        except json.JSONDecodeError:
            pass
    # fallback for models that do not return valid JSON
    low = text.lower()
    correct = bool(re.search(r'"?correct"?\s*:\s*true', low))
    grounded = bool(re.search(r'"?grounded"?\s*:\s*true', low))
    return correct, grounded, "unparsed judge output"


def metric_pass(r: CaseResult) -> bool:
    return r.keyword_score == 1.0 if r.answerable else r.refused


def judge_results(judge_llm, results: list[CaseResult], progress: bool = True) -> list[Judgement]:
    out = []
    for n, r in enumerate(results, 1):
        if progress:
            print(f"\r  judging [{n}/{len(results)}]", end="", file=sys.stderr, flush=True)
        msg = f"CONTEXT:\n{r.context or '(none)'}\n\nQUESTION: {r.question}\n\nANSWER: {r.answer}"
        correct, grounded, reason = parse_judgement(judge_llm.chat(JUDGE_SYSTEM, msg))
        out.append(Judgement(r.id, correct, grounded, reason, metric_pass(r)))
    if progress:
        print("\r" + " " * 30 + "\r", end="", file=sys.stderr, flush=True)
    return out


def agreement_summary(judgements: list[Judgement]) -> str:
    n = len(judgements)
    both = sum(j.correct and j.metric_pass for j in judgements)
    neither = sum(not j.correct and not j.metric_pass for j in judgements)
    only_judge = sum(j.correct and not j.metric_pass for j in judgements)
    only_metric = sum(not j.correct and j.metric_pass for j in judgements)
    return (f"judge: correct={sum(j.correct for j in judgements) / n:.0%}  "
            f"grounded={sum(j.grounded for j in judgements) / n:.0%}  "
            f"agreement with keyword metric={(both + neither) / n:.0%}\n"
            f"        both pass={both}  both fail={neither}  "
            f"only judge passes={only_judge}  only metric passes={only_metric}")
