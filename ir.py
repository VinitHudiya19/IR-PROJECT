"""Classical IR: text cleaning, inverted index, BM25 ranking, result highlighting."""
import html
import math
import re
from collections import Counter
from functools import lru_cache

from nltk.stem import PorterStemmer

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "hers", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "may", "might", "more", "most", "no", "not", "of", "on", "or",
    "our", "ours", "she", "should", "so", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your",
}
_stemmer = PorterStemmer()


@lru_cache(maxsize=100_000)
def stem(word):
    return _stemmer.stem(word)


def tokenize(text):
    """Lowercase words and numbers."""
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def preprocess(text):
    """Tokenize, drop stop words, stem (clustering -> cluster)."""
    return [stem(w) for w in tokenize(text) if w not in STOP_WORDS]


def build_index(texts):
    """Build an inverted index from {document_id: text}."""
    ids = sorted(texts)
    counts = {}
    inverted = {}

    for doc in ids:
        words = tokenize(texts[doc])
        counts[doc] = Counter(stem(w) for w in words if w not in STOP_WORDS)
        for term in counts[doc]:
            inverted.setdefault(term, set()).add(doc)

    lengths = {doc: sum(counts[doc].values()) for doc in ids}
    avg_length = sum(lengths.values()) / len(ids) if ids else 0

    return {
        "ids": ids,
        "counts": counts,
        "inverted": inverted,
        "lengths": lengths,
        "avg_length": avg_length,
    }


def bm25(index, query, k1=1.5, b=0.75):
    """Return [(document_id, score)] in best-first order."""
    terms = set(preprocess(query))
    if not terms or not index["avg_length"]:
        return []

    n = len(index["ids"])
    scores = {}

    for term in terms:
        docs = index["inverted"].get(term, ())
        idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
        for doc in docs:
            tf = index["counts"][doc][term]
            norm = k1 * (1 - b + b * index["lengths"][doc] / index["avg_length"])
            scores[doc] = scores.get(doc, 0.0) + idf * tf * (k1 + 1) / (tf + norm)

    return sorted(scores.items(), key=lambda item: -item[1])


def best_chunk(query, chunks):
    """The chunk that contains the most query words."""
    wanted = set(preprocess(query))
    score = lambda c: sum(1 for w in tokenize(c["text"]) if stem(w) in wanted)
    return max(chunks, key=score) if chunks else None


def highlight(text, query, max_chars=450):
    """Safe HTML snippet with query words in <mark>."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    wanted = set(preprocess(query))
    hits = [m.start() for m in re.finditer(r"[A-Za-z0-9]+", text) if stem(m.group().lower()) in wanted]
    start = max(0, hits[0] - 100) if hits else 0
    snippet = text[start:start + max_chars]
    parts = re.split(r"([A-Za-z0-9]+)", snippet)
    body = "".join(
        f"<mark>{p}</mark>" if i % 2 and stem(p.lower()) in wanted else html.escape(p)
        for i, p in enumerate(parts)
    )
    return ("… " if start else "") + body + (" …" if start + max_chars < len(text) else "")
