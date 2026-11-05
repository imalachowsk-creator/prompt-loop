"""Command line interface.

  python -m promptloop ask "How do I read JSON from a file?"
  python -m promptloop eval
  python -m promptloop improve --iterations 3
  python -m promptloop history

Add --llm mock to run without Ollama (deterministic fake model).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import DocsAgent
from .evaluate import load_cases, run_eval
from .improve import improve, latest_version
from .judge import agreement_summary, judge_results
from .llm import LLMError, get_llm
from .retriever import BM25Retriever


def main(argv=None):
    p = argparse.ArgumentParser(prog="promptloop", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", choices=["ollama", "mock"], default="ollama")
    p.add_argument("--model", help="Ollama model name (default: llama3.2:3b)")
    p.add_argument("--kb", default="knowledge_base")
    p.add_argument("--prompts", default="prompts")
    p.add_argument("--cases", default="eval/questions.jsonl")
    p.add_argument("--runs", default="runs")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="ask the agent one question")
    a.add_argument("question")
    a.add_argument("--version", type=int, help="prompt version (default: latest)")

    e = sub.add_parser("eval", help="evaluate a prompt version")
    e.add_argument("--version", type=int)
    e.add_argument("--min-score", type=float, default=0.0,
                   help="exit with code 1 if the score is below this (CI gate)")
    e.add_argument("-v", "--verbose", action="store_true")
    e.add_argument("--split", choices=["train", "test", "all"], default="all")
    e.add_argument("--judge", action="store_true",
                   help="also grade answers with an LLM judge and report agreement")

    i = sub.add_parser("improve", help="run the improvement loop")
    i.add_argument("--iterations", type=int, default=3)
    i.add_argument("--min-gain", type=float, default=0.01)
    i.add_argument("--feedback", default="feedback/feedback.jsonl")
    i.add_argument("--no-test", action="store_true",
                   help="optimize on all cases instead of train, skip held-out test")

    sub.add_parser("history", help="show accepted/rejected prompt changes")

    args = p.parse_args(argv)
    llm = get_llm(args.llm, **({"model": args.model} if args.model else {}))
    prompts = Path(args.prompts)

    def load_prompt(v):
        v = v or latest_version(prompts)
        return f"v{v}", (prompts / f"v{v}.md").read_text(encoding="utf-8")

    try:
        if args.cmd == "ask":
            name, prompt = load_prompt(args.version)
            out = DocsAgent(llm, BM25Retriever.from_dir(args.kb), prompt).ask(args.question)
            print(out.answer)
            print(f"\n(prompt {name}, retrieved: {', '.join(out.retrieved) or 'nothing'})")

        elif args.cmd == "eval":
            name, prompt = load_prompt(args.version)
            agent = DocsAgent(llm, BM25Retriever.from_dir(args.kb), prompt)
            report = run_eval(agent, load_cases(args.cases, args.split), name, progress=True)
            judgements = {}
            if args.judge:
                judgements = {j.id: j for j in judge_results(llm, report.results)}
            if args.verbose:
                for r in report.results:
                    mark = "ok  " if not r.failed else "FAIL"
                    why = [] if not r.failed else (
                        ["no citation"] * (r.answerable and not r.cited)
                        + ["missing keywords"] * (r.answerable and r.keyword_score < 1)
                        + ["should refuse"] * (not r.answerable and not r.refused))
                    j = judgements.get(r.id)
                    if j:
                        why.append(f"judge: {'correct' if j.correct else 'WRONG'}"
                                   + ("" if j.agrees else " (disagrees)"))
                    flat = " ".join(r.answer.split())
                    print(f"{mark} {r.id} {r.score:.2f}  {r.question}"
                          + (f"   <- {', '.join(why)}" if why else "")
                          + f"\n      -> {flat[:160]}{'...' if len(flat) > 160 else ''}")
            print(report.summary())
            if judgements:
                print(agreement_summary(list(judgements.values())))
            if report.score < args.min_score:
                print(f"score below gate {args.min_score}", file=sys.stderr)
                return 1

        elif args.cmd == "improve":
            # The agent answers deterministically (temperature 0) so evaluations are
            # comparable; the optimizer gets some randomness to explore new prompts.
            opt_kwargs = {"temperature": 0.7, **({"model": args.model} if args.model else {})}
            optimizer = get_llm(args.llm, **opt_kwargs)
            train = load_cases(args.cases, "all" if args.no_test else "train")
            test = None if args.no_test else load_cases(args.cases, "test")
            print(f"optimizing on {len(train)} cases"
                  + (f", held-out test: {len(test)} cases" if test else ""))
            improve(llm, optimizer, BM25Retriever.from_dir(args.kb), train,
                    prompts_dir=prompts, runs_dir=args.runs, feedback_path=args.feedback,
                    iterations=args.iterations, min_gain=args.min_gain, test_cases=test)

        elif args.cmd == "history":
            path = Path(args.runs) / "history.jsonl"
            if not path.exists():
                print("no history yet - run `improve` first")
                return 0
            for line in path.read_text(encoding="utf-8").splitlines():
                h = json.loads(line)
                status = "ACCEPTED" if h["accepted"] else "rejected"
                print(f"{h['time']}  {h['from_version']}  {h['old_score']:.3f} -> "
                      f"{h['new_score']:.3f}  {status}")
                for rule in h["learnings"]:
                    print(f"    learning: {rule}")
    except LLMError as err:
        print(err, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
