"""Evaluation: run the agent over a test set and score every answer.

Each case in the JSONL test set has:
  question      - what the user asks
  must_include  - strings a correct answer must contain
  source        - knowledge-base file the answer comes from, or null if the
                  question is out of scope and the agent should refuse

Scores (0..1) per case:
  answerable:  0.7 * share of must_include found + 0.3 * (cites the right source)
  out of scope: 1 if the agent refuses, else 0
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REFUSAL_MARKERS = ["don't know", "do not know", "not in the documentation",
                   "not covered", "nie wiem"]


@dataclass
class CaseResult:
    id: str
    question: str
    answer: str
    score: float
    keyword_score: float
    cited: bool
    refused: bool
    answerable: bool
    retrieved: list[str]
    context: str = ""

    @property
    def failed(self) -> bool:
        return self.score < 1.0


def load_cases(path: str | Path, split: str = "all") -> list[dict]:
    """split: "train", "test" or "all". Cases without a split field count as train."""
    with open(path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if split == "all":
        return cases
    return [c for c in cases if c.get("split", "train") == split]


def is_refusal(answer: str) -> bool:
    a = answer.lower().replace("’", "'")
    return any(m in a for m in REFUSAL_MARKERS)


def score_case(case: dict, answer: str, retrieved: list[str], context: str = "") -> CaseResult:
    refused = is_refusal(answer)
    answerable = case.get("source") is not None
    if answerable:
        needed = case["must_include"]
        # "a|b" means: either "a" or "b" counts as correct
        found = sum(1 for s in needed
                    if any(alt.lower() in answer.lower() for alt in s.split("|")))
        kw = found / len(needed) if needed else 1.0
        # accept "[json#parsing-json]" as well as "Source: json#parsing-json"
        cited = f"[{case['source']}" in answer or f"{case['source']}#" in answer
        score = 0.7 * kw + 0.3 * cited
    else:
        kw, cited = 1.0, False
        score = 1.0 if refused else 0.0
    return CaseResult(case["id"], case["question"], answer, round(score, 3),
                      round(kw, 3), cited, refused, answerable, retrieved, context)


@dataclass
class EvalReport:
    prompt_version: str
    score: float
    accuracy: float          # share of answerable cases with all keywords
    citation_rate: float     # share of answerable cases citing the right source
    refusal_rate: float      # share of out-of-scope cases correctly refused
    results: list[CaseResult]

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if r.failed]

    def summary(self) -> str:
        return (f"{self.prompt_version}: score={self.score:.3f}  "
                f"accuracy={self.accuracy:.0%}  citations={self.citation_rate:.0%}  "
                f"refusals={self.refusal_rate:.0%}  failures={len(self.failures())}")

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_eval(agent, cases: list[dict], prompt_version: str = "?",
             progress: bool = False) -> EvalReport:
    results = []
    for n, case in enumerate(cases, 1):
        if progress:
            print(f"\r  evaluating {prompt_version} [{n}/{len(cases)}]", end="",
                  file=sys.stderr, flush=True)
        out = agent.ask(case["question"])
        results.append(score_case(case, out.answer, out.retrieved, out.context))
    if progress:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr, flush=True)
    ans = [r for r in results if r.answerable]
    oos = [r for r in results if not r.answerable]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return EvalReport(
        prompt_version=prompt_version,
        score=round(mean([r.score for r in results]), 3),
        accuracy=mean([r.keyword_score == 1.0 for r in ans]),
        citation_rate=mean([r.cited for r in ans]),
        refusal_rate=mean([r.refused for r in oos]),
        results=results,
    )
