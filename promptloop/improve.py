"""The improvement loop: EVAL -> FEEDBACK -> new PROMPT -> RUN -> EVAL ...

1. Evaluate the current prompt.
2. Turn failures and human feedback into a report with concrete learnings.
3. Ask an LLM (the "optimizer") to rewrite the prompt using that report.
4. Evaluate the candidate prompt on the same test set.
5. Gate: keep the candidate only if the score improves by at least `min_gain`.
6. Log every step to runs/history.jsonl so changes are traceable.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .agent import DocsAgent
from .evaluate import CaseResult, EvalReport, run_eval

OPTIMIZER_SYSTEM = """You are a PROMPT OPTIMIZER for a documentation question-answering agent.
You receive the agent's current system prompt, a list of failed test cases and
human feedback. Rewrite the system prompt so the agent avoids these failures.
Keep what already works. Write general rules, never answers to specific questions.
Return ONLY the new system prompt text, with no explanation and no code fences."""


def diagnose(r: CaseResult) -> str | None:
    """Turn one failed case into a reusable rule (a 'learning')."""
    if not r.answerable and not r.refused:
        return ("If the context does not contain the answer, reply exactly: "
                "I don't know based on the documentation.")
    if r.answerable and r.refused:
        return "Answer whenever the context contains relevant information, even partially."
    if r.answerable and not r.cited:
        return "Always cite the source tag of the context you used in square brackets, e.g. [json#parsing-json]."
    if r.answerable and r.keyword_score < 1.0:
        return "Include the exact function, method or parameter names from the context."
    return None


def load_feedback(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_report(prompt: str, report: EvalReport, feedback: list[dict], max_examples: int = 8,
                 rejected: list[tuple[str, float]] | None = None) -> str:
    failures = report.failures()
    lines = [f"CURRENT PROMPT:\n{prompt}\n", f"FAILURES: {len(failures)} of {len(report.results)} cases"]
    for r in failures[:max_examples]:
        lines.append(f"- Q: {r.question}\n  A: {r.answer[:200]}\n  expected_source: "
                     f"{'out of scope (should refuse)' if not r.answerable else 'see retrieved ' + str(r.retrieved[:1])}")
    rules = sorted({rule for r in failures if (rule := diagnose(r))})
    lines.append("\nLEARNINGS FROM FAILURES:")
    lines += [f"SUGGESTED RULE: {rule}" for rule in rules]
    if feedback:
        lines.append("\nHUMAN FEEDBACK:")
        lines += [f"- ({fb.get('source', '?')}) {fb['note']}" for fb in feedback]
    if rejected:
        # Without this, a deterministic optimizer would propose the same
        # rejected prompt again and the loop would get stuck.
        lines.append(f"\nREJECTED ATTEMPTS (current score {report.score:.3f}) - these did NOT "
                     "improve the score, try a clearly different approach:")
        lines += [f"--- attempt scored {score:.3f}:\n{text.strip()}" for text, score in rejected]
    return "\n".join(lines)


def clean_prompt(text: str) -> str:
    text = re.sub(r"^```\w*\n|```$", "", text.strip()).strip()
    return text + "\n"


def latest_version(prompts_dir: Path) -> int:
    nums = [int(m.group(1)) for p in prompts_dir.glob("v*.md") if (m := re.match(r"v(\d+)$", p.stem))]
    return max(nums, default=0)


def improve(llm, optimizer_llm, retriever, cases, prompts_dir="prompts", runs_dir="runs",
            feedback_path="feedback/feedback.jsonl", iterations: int = 3,
            min_gain: float = 0.01, log=print) -> EvalReport:
    prompts_dir, runs_dir = Path(prompts_dir), Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    feedback = load_feedback(feedback_path)

    version = latest_version(prompts_dir)
    if version == 0:
        raise FileNotFoundError(f"No prompts/v1.md found in {prompts_dir}")
    prompt = (prompts_dir / f"v{version}.md").read_text(encoding="utf-8")
    best = run_eval(DocsAgent(llm, retriever, prompt), cases, f"v{version}", progress=True)
    log(f"baseline  {best.summary()}")
    _save_report(runs_dir, best)
    rejected: list[tuple[str, float]] = []

    for i in range(1, iterations + 1):
        if not best.failures():
            log("no failures left, stopping")
            break
        report_text = build_report(prompt, best, feedback, rejected=rejected)
        candidate = clean_prompt(optimizer_llm.chat(OPTIMIZER_SYSTEM, report_text))
        if candidate.strip() == prompt.strip() or any(candidate.strip() == r.strip() for r, _ in rejected):
            log(f"iter {i}    optimizer repeated an earlier prompt, skipping evaluation")
            continue
        cand_report = run_eval(DocsAgent(llm, retriever, candidate), cases,
                               f"v{version + 1}?", progress=True)
        gain = cand_report.score - best.score
        accepted = gain >= min_gain
        log(f"iter {i}    {cand_report.summary()}  gain={gain:+.3f}  "
            f"{'ACCEPTED' if accepted else 'rejected'}")
        _log_history(runs_dir, {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "iteration": i, "from_version": f"v{version}",
            "old_score": best.score, "new_score": cand_report.score,
            "gain": round(gain, 3), "accepted": accepted,
            "learnings": [line[16:] for line in report_text.splitlines() if line.startswith("SUGGESTED RULE")],
        })
        if not accepted:
            rejected.append((candidate, cand_report.score))
        if accepted:
            version += 1
            prompt = candidate
            rejected = []
            (prompts_dir / f"v{version}.md").write_text(prompt, encoding="utf-8")
            cand_report.prompt_version = f"v{version}"
            best = cand_report
            _save_report(runs_dir, best)
    log(f"final     {best.summary()}")
    return best


def _save_report(runs_dir: Path, report: EvalReport):
    path = runs_dir / f"eval_{report.prompt_version}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _log_history(runs_dir: Path, entry: dict):
    with open(runs_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
