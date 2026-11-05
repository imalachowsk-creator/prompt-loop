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

## Results (honest measurement with a held-out test split)

`llama3.2:3b` as the agent, `qwen2.5:7b` as the optimizer, starting from the
weak `prompts/v1.md`. The loop learned on 18 train questions; 18 test questions
were never shown to it.

| Prompt | Train score | Held-out test score | Refusals (test) | What changed |
|---|---|---|---|---|
| v1 | 0.544 | – | 0% | baseline |
| v2 | 0.728 | – | – | added "cite sources" and "say I don't know" rules |
| v3 | 0.744 | **0.767** | **100%** | sharper citation rule |

The test score is not lower than the train score, so the prompt generalizes
instead of memorizing the training questions. Hallucinated answers to
out-of-scope questions dropped from 100% to 0%. Citations stayed low (7–13%):
the 3B agent does not reliably follow the citation format, and optimizing for
refusals pushed it further down - a trade-off between two metrics. Prompt
versions are in `prompts_split/`.

**Takeaway:** not everything should be fixed in the prompt. The retriever
already knows which section the context came from, so in a production system
the code would attach citations and the prompt would focus on what code cannot
do (answering correctly and refusing when the docs are silent).

## Earlier runs (before the train/test split)

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
This 0.938 turned out to be **inflated**: it was measured on the same questions
the loop learned from. The train/test split above gives the honest number.

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
- **The optimizer never saw hallucinations.** After the split, refusals stayed
  at 0%. The report showed the optimizer only the first 8 failures, and the
  out-of-scope questions sit at the end of the eval file. Examples are now
  picked round-robin across failure types, and a unit test checks it. With
  that fix (and a bigger optimizer model) refusals went 0% → 100%.

## Quick start

```bash
# 1. install Ollama and pull a small model
ollama pull llama3.2:3b

# 2. ask the agent
python -m promptloop ask "How do I keep Polish letters when writing JSON?"

# 3. evaluate the current prompt (-v shows every case and why it failed)
python -m promptloop eval -v

# 4. run the improvement loop (learns on train, reports held-out test)
python -m promptloop improve --iterations 3

# 5. grade the test split with an LLM judge and compare it with the keyword metric
python -m promptloop eval --split test --judge -v

# 6. see which prompt changes were accepted and why
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
| LLM judge | `promptloop/judge.py` | Second model grades answers; agreement with keyword metric |

**Evaluation metrics** (`eval/questions.jsonl`, 30 in-scope + 6 out-of-scope questions)
- *accuracy* – answer contains the expected function/parameter names
- *citation rate* – answer cites the correct knowledge-base section
- *refusal rate* – agent says "I don't know" for questions outside the docs
- *score* – weighted mean used by the acceptance gate

**Train/test split.** The 36 questions are split 50/50 (both halves contain
every documentation file and out-of-scope questions). The loop sees failures
only from *train* and accepts or rejects prompts only on the *train* score. The
*test* split is scored for every accepted prompt but never shown to the
optimizer, so the test score shows whether a prompt really generalizes or just
memorized the training questions. A unit test checks that no test question
ever reaches the optimizer.

**LLM-as-judge.** `eval --judge` asks a model to grade each answer as
*correct* and *grounded* in the retrieved documentation (JSON output, with a
fallback parser for small models that break the format). The report shows how
often the judge agrees with the keyword metric and lists the disagreements -
cases where either the metric is too strict or the judge is too lenient, which
is where a human reviewer should look first.

**Learnings.** Each failure is mapped to a general rule (e.g. "cite the source
tag"), and human feedback such as support threads or review comments
(`feedback/feedback.jsonl`) is added to the optimizer's input. The optimizer is
told to write general rules, not answers to specific test questions.

**Traceability.** Every iteration is logged to `runs/history.jsonl` (old/new
score, accepted or rejected, learnings used); each accepted prompt is saved as
a new version file, so changes can be reviewed like code.

**CI.** GitHub Actions runs unit tests and an evaluation gate on every push.

## Limitations and next steps

- Use human labels on a sample of answers to measure which grader (keyword metric or LLM judge) is closer to a person.
- The judge is the same small model as the agent; a stronger or different judge model would reduce self-preference bias.
- Read feedback directly from GitHub issue and pull-request comments instead of a JSONL file.
- Add a tool (e.g. running a Python snippet) to turn the QA agent into a multi-step agent.
- Run several models (Llama, Qwen, Mistral) and compare how much each benefits from the loop.

## Author

Ignacy Małachowski – AI & Cognitive Science, University of Warsaw.
