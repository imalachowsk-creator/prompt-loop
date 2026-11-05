# promptloop

A small framework for **continuous prompt improvement in agentic workflows**.
A documentation-QA agent answers Python questions from a knowledge base; the
framework evaluates it, turns failures and human feedback into reusable
learnings, lets an LLM rewrite the agent's prompt, and **only keeps a new prompt
if the evaluation score goes up**.

```
        ┌──────── PROMPT (prompts/vN.md) ◄──────┐
        ▼                                        │
      RUN  agent answers the eval set      optimizer LLM rewrites
        │                                   the prompt from learnings
        ▼                                        │
      EVAL  score: keywords, citations,          │
            refusals on out-of-scope       FEEDBACK  failures + human notes
        │                                   (feedback/feedback.jsonl)
        └──── gate: accept only if score ↑ ──────┘
```

Runs fully locally with [Ollama](https://ollama.com). No paid API, no external
Python dependencies (standard library only).

## Example run

Real run with `llama3.2:3b` on a MacBook Air, starting from a deliberately weak
prompt (`prompts/v1.md`):

```
baseline  v1:  score=0.554  accuracy=95%  citations=0%   refusals=0%    failures=24
iter 1    v2?: score=0.808  accuracy=95%  citations=35%  refusals=100%  gain=+0.254  ACCEPTED
iter 2    v3?: score=0.808  ...                                          gain=+0.000  rejected
iter 3    v3?: score=0.808  ...                                          gain=+0.000  rejected
final     v2:  score=0.808
```

One iteration stopped the agent from inventing answers to out-of-scope
questions (refusals 0% → 100%) and introduced source citations (0% → 35%).

Second run, after fixing the issues described below (metric slightly revised,
so the v2 baseline is re-scored):

```
baseline  v2:  score=0.837  accuracy=100%  citations=35%  refusals=100%  failures=13
iter 1    v3?: score=0.938  accuracy=100%  citations=75%  refusals=100%  gain=+0.101  ACCEPTED
iter 2    v4?: score=0.812  accuracy=100%  citations=25%  refusals=100%  gain=-0.126  rejected
iter 3    v4?: score=0.888  accuracy=100%  citations=55%  refusals=100%  gain=-0.050  rejected
final     v3:  score=0.938
```

The optimizer now explores different candidates instead of repeating one, and
the gate rejects the two that would have made the agent worse.
**Overall: 0.554 → 0.938** in two runs; every prompt version is in `prompts/`.

## What the first real run taught me

- **The loop got stuck.** Iterations 2 and 3 scored identically to three
  decimals. With temperature 0 and an unchanged best prompt, the optimizer got
  the same input and produced the same rejected prompt. Fix: rejected attempts
  are now fed back to the optimizer ("these did not help, try a different
  approach"), the optimizer runs at temperature 0.7 while the agent stays at 0
  so evaluations remain comparable, and exact repeats are skipped.
- **One failure was the metric, not the model.** The agent correctly said that
  `replace()` returns a new string but was marked wrong for not using the word
  "immutable". The eval set now supports alternative correct answers
  (`"immutable|new string"`).
- **Citations are the hard part for a 3B model.** Reading failures with
  `eval -v` showed the model mostly skips the source or invents one
  ("Python docs"). The failure reasons are now printed next to each case.

## Quick start

```bash
# 1. install Ollama and pull a small model
ollama pull llama3.2:3b

# 2. ask the agent
python -m promptloop ask "How do I keep Polish letters when writing JSON?"

# 3. evaluate the current prompt (-v shows every case)
python -m promptloop eval -v

# 4. run the improvement loop
python -m promptloop improve --iterations 3

# 5. see which prompt changes were accepted and why
python -m promptloop history
```

Use another model with `--model qwen2.5:7b`. Add `--llm mock` to any command to
run without Ollama.

## How it works

| Component | File | What it does |
|---|---|---|
| LLM client | `promptloop/llm.py` | Ollama over HTTP + deterministic mock for tests/CI |
| Retriever | `promptloop/retriever.py` | BM25 over markdown chunks, implemented from scratch |
| Agent | `promptloop/agent.py` | Retrieval-augmented answering with source tags |
| Evaluation | `promptloop/evaluate.py` | Per-case scoring and aggregate metrics |
| Improvement loop | `promptloop/improve.py` | Learnings → optimizer → gated acceptance → history |

**Evaluation metrics** (`eval/questions.jsonl`, 20 in-scope + 4 out-of-scope questions)
- *accuracy* – answer contains the expected function/parameter names
- *citation rate* – answer cites the correct knowledge-base section
- *refusal rate* – agent says "I don't know" for questions outside the docs
- *score* – weighted mean used by the acceptance gate

**Learnings.** Each failure is mapped to a general rule (e.g. "cite the source
tag"), and human feedback such as support threads or review comments
(`feedback/feedback.jsonl`) is added to the optimizer's input. The optimizer is
told to write general rules, not answers to specific test questions.

**Traceability.** Every iteration is logged to `runs/history.jsonl` (old/new
score, accepted or rejected, learnings used); each accepted prompt is saved as
a new version file, so changes can be reviewed like code.

**CI.** GitHub Actions runs unit tests and an evaluation gate on every push.

## Limitations and next steps

- Keyword-based scoring is simple; next step is an LLM-as-judge metric and comparing it with the keyword metric.
- The eval set is small, so a held-out test split is needed to detect overfitting the prompt to the eval set.
- Add a tool (e.g. running a Python snippet) to turn the QA agent into a multi-step agent.
- Run several models (Llama, Qwen, Mistral) and compare how much each benefits from the loop.

## Author

Ignacy Małachowski – AI & Cognitive Science, University of Warsaw.
