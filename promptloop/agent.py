"""The documentation-QA agent: retrieve relevant chunks, then ask the LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from .retriever import BM25Retriever


@dataclass
class AgentAnswer:
    question: str
    answer: str
    retrieved: list[str] = field(default_factory=list)
    context: str = ""


class DocsAgent:
    def __init__(self, llm, retriever: BM25Retriever, system_prompt: str, top_k: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.top_k = top_k

    def build_user_message(self, question: str, hits) -> str:
        context = "\n".join(f"[{chunk.source}] {chunk.text}" for chunk, _ in hits)
        return f"CONTEXT:\n{context or '(no relevant documents found)'}\n\nQUESTION: {question}"

    def ask(self, question: str) -> AgentAnswer:
        hits = self.retriever.search(question, k=self.top_k)
        reply = self.llm.chat(self.system_prompt, self.build_user_message(question, hits))
        context = "\n".join(f"[{c.source}] {c.text}" for c, _ in hits)
        return AgentAnswer(question, reply, [c.source for c, _ in hits], context)
