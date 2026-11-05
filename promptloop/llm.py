"""LLM clients.

OllamaLLM talks to a locally running Ollama server (https://ollama.com).
MockLLM is a deterministic fake used in tests and CI, where no model is available.
Both expose the same method: chat(system, user) -> str.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.0, timeout: int = 120):
        self.model = model or os.environ.get("PROMPTLOOP_MODEL", "llama3.2:3b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Is it running? (`ollama serve`)\n{e}"
            ) from e
        return data["message"]["content"].strip()


class MockLLM:
    """Deterministic stand-in for a real model.

    - As an answering agent: copies the sentences from the context that share
      the most words with the question, and cites their source tags.
      If the system prompt says to refuse when unsure, it answers
      "I don't know" when overlap is very low.
    - As a prompt optimizer: appends the suggested rules from the failure
      report to the prompt.
    """

    def chat(self, system: str, user: str) -> str:
        if "PROMPT OPTIMIZER" in system:
            return self._improve(user)
        if "ANSWER JUDGE" in system:
            return self._judge(user)
        return self._answer(system, user)

    def _judge(self, user: str) -> str:
        context = user.split("CONTEXT:")[1].split("QUESTION:")[0]
        question = user.split("QUESTION:")[1].split("ANSWER:")[0]
        answer = user.split("ANSWER:")[1]
        ctx, q, a = self._words(context), self._words(question), self._words(answer)
        if "know" in a and len(q & ctx) < 2:
            ok = True                       # correct refusal
        else:
            ok = len(a & ctx) >= 5 and len(q & ctx) >= 2
        return json.dumps({"correct": ok, "grounded": ok, "reason": "word overlap"})

    @staticmethod
    def _words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z_][a-z0-9_]+", text.lower()) if len(w) > 2}

    def _answer(self, system: str, user: str) -> str:
        question = user.split("QUESTION:")[-1]
        q = self._words(question)
        best, best_score = [], 0
        for line in user.split("\n"):
            m = re.match(r"\[(.+?)\]\s*(.*)", line)
            if not m:
                continue
            source, text = m.groups()
            score = len(q & self._words(text))
            if score > best_score:
                best, best_score = [(source, text)], score
            elif score == best_score and score > 0:
                best.append((source, text))
        refuse = "i don't know" in system.lower()
        if best_score < 2 and refuse:
            return "I don't know based on the documentation."
        if not best:
            return "No relevant information."
        source, text = best[0]
        cite = f" [{source}]" if "cite" in system.lower() else ""
        return text + cite

    def _improve(self, user: str) -> str:
        current = user.split("CURRENT PROMPT:")[1].split("FAILURES:")[0].strip()
        rules = re.findall(r"SUGGESTED RULE: (.+)", user)
        new = current
        for rule in rules:
            if rule not in new:
                new += f"\n- {rule}"
        return new


def get_llm(name: str = "ollama", **kwargs):
    if name == "mock":
        return MockLLM()
    return OllamaLLM(**kwargs)
