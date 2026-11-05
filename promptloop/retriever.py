"""A small BM25 retriever over markdown files (no external libraries).

Each markdown file is split into chunks at `## ` headings. A chunk keeps its
source as "<file>#<heading>" so the agent can cite where an answer came from.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set("""a an and are as at be by can do does for from how i in is it
of on or that the this to use used using what when which with you your""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z_][a-z0-9_]*", text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    source: str
    text: str


def load_chunks(kb_dir: str | Path) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        heading, lines = "intro", []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if lines:
                    chunks.append(Chunk(f"{path.stem}#{heading}", " ".join(lines)))
                heading, lines = line[3:].strip().lower().replace(" ", "-"), []
            elif line.strip() and not line.startswith("# "):
                lines.append(line.strip())
        if lines:
            chunks.append(Chunk(f"{path.stem}#{heading}", " ".join(lines)))
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.source + " " + c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_dir(cls, kb_dir: str | Path) -> "BM25Retriever":
        return cls(load_chunks(kb_dir))

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for chunk, doc, length in zip(self.chunks, self.docs, self.lengths):
            score = 0.0
            for term in q:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self.idf[term] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
