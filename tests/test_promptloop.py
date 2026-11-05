import shutil
from pathlib import Path

from promptloop.agent import DocsAgent
from promptloop.evaluate import is_refusal, load_cases, run_eval, score_case
from promptloop.improve import clean_prompt, diagnose, improve
from promptloop.llm import MockLLM
from promptloop.retriever import BM25Retriever

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "knowledge_base"
CASES = ROOT / "eval" / "questions.jsonl"


def test_retriever_finds_right_file():
    r = BM25Retriever.from_dir(KB)
    top = r.search("How do I parse invalid JSON JSONDecodeError", k=1)[0][0]
    assert top.source.startswith("json#")


def test_retriever_top3_hit_rate():
    r = BM25Retriever.from_dir(KB)
    cases = [c for c in load_cases(CASES) if c["source"]]
    hits = sum(any(ch.source.startswith(c["source"] + "#") for ch, _ in r.search(c["question"], k=3))
               for c in cases)
    assert hits / len(cases) >= 0.9


def test_scoring():
    case = {"id": "x", "question": "?", "must_include": ["loads"], "source": "json"}
    assert score_case(case, "Use json.loads [json#parsing-json]", []).score == 1.0
    assert score_case(case, "Use json.loads", []).score == 0.7
    oos = {"id": "y", "question": "?", "must_include": [], "source": None}
    assert score_case(oos, "I don't know based on the documentation.", []).score == 1.0
    assert score_case(oos, "Canberra.", []).score == 0.0


def test_refusal_detection():
    assert is_refusal("I don’t know.")
    assert not is_refusal("Use pathlib.")


def test_diagnose_gives_rules():
    case = {"id": "y", "question": "?", "must_include": [], "source": None}
    assert "I don't know" in diagnose(score_case(case, "Canberra.", []))


def test_clean_prompt_strips_fences():
    assert clean_prompt("```\nBe nice.\n```") == "Be nice.\n"


def test_improve_loop_raises_score(tmp_path):
    prompts = tmp_path / "prompts"
    shutil.copytree(ROOT / "prompts", prompts)
    for extra in prompts.glob("v*.md"):
        if extra.stem != "v1":
            extra.unlink()
    llm = MockLLM()
    retriever = BM25Retriever.from_dir(KB)
    cases = load_cases(CASES)
    baseline = run_eval(DocsAgent(llm, retriever, (prompts / "v1.md").read_text()), cases)
    final = improve(llm, llm, retriever, cases, prompts_dir=prompts, runs_dir=tmp_path / "runs",
                    feedback_path=ROOT / "feedback" / "feedback.jsonl", iterations=2, log=lambda *_: None)
    assert final.score > baseline.score
    assert (prompts / "v2.md").exists()
    assert (tmp_path / "runs" / "history.jsonl").exists()


def test_keyword_alternatives_and_loose_citation():
    case = {"id": "x", "question": "?", "must_include": ["immutable|new string"], "source": "strings"}
    assert score_case(case, "It returns a new string. Source: strings#immutability", []).score == 1.0


def test_report_lists_rejected_attempts():
    from promptloop.improve import build_report
    llm, r = MockLLM(), BM25Retriever.from_dir(KB)
    report = run_eval(DocsAgent(llm, r, "Answer."), load_cases(CASES)[:3])
    text = build_report("Answer.", report, [], rejected=[("Bad prompt", 0.1)])
    assert "REJECTED ATTEMPTS" in text and "Bad prompt" in text


def test_split_is_disjoint_and_balanced():
    train, test = load_cases(CASES, "train"), load_cases(CASES, "test")
    assert not {c["id"] for c in train} & {c["id"] for c in test}
    assert len(train) == len(test)
    assert any(c["source"] is None for c in test)  # out-of-scope questions in test too


def test_improve_never_shows_test_cases_to_optimizer(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    shutil.copy(ROOT / "prompts" / "v1.md", prompts / "v1.md")
    seen = []

    class SpyOptimizer(MockLLM):
        def chat(self, system, user):
            seen.append(user)
            return super().chat(system, user)

    test = load_cases(CASES, "test")
    improve(MockLLM(), SpyOptimizer(), BM25Retriever.from_dir(KB), load_cases(CASES, "train"),
            prompts_dir=prompts, runs_dir=tmp_path / "runs", feedback_path=tmp_path / "none.jsonl",
            iterations=1, test_cases=test, log=lambda *_: None)
    assert seen and not any(c["question"] in u for c in test for u in seen)


def test_parse_judgement():
    from promptloop.judge import parse_judgement
    assert parse_judgement('Sure! {"correct": true, "grounded": false, "reason": "x"}') == (True, False, "x")
    assert parse_judgement('"correct": true, "grounded": true')[:2] == (True, True)
    assert parse_judgement("no idea")[:2] == (False, False)


def test_judge_agreement():
    from promptloop.judge import agreement_summary, judge_results
    llm, r = MockLLM(), BM25Retriever.from_dir(KB)
    report = run_eval(DocsAgent(llm, r, "Answer. Cite sources. Say I don't know if unsure."),
                      load_cases(CASES, "test"))
    js = judge_results(llm, report.results, progress=False)
    assert len(js) == len(report.results)
    assert "agreement with keyword metric" in agreement_summary(js)
