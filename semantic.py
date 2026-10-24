"""Semantic search (sentence embeddings), hybrid fusion (RRF) and the search dispatcher."""
import numpy as np

import library
from ir import best_chunk, bm25

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_SIMILARITY = 0.2  # ignore weak semantic matches; tune with the Evaluate page
METHODS = ["Hybrid", "BM25", "Semantic"]
_model = None


def embed(texts):
    """Unit-length sentence embeddings (the model downloads on first use)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    vectors = _model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def ensure_embeddings():
    """Embed chunks that do not have a vector yet."""
    todo = library.chunks_missing_vectors()
    for i in range(0, len(todo), 256):
        batch = todo[i:i + 256]
        vectors = embed([c["text"] for c in batch])
        library.save_vectors([(v.tobytes(), c["id"]) for c, v in zip(batch, vectors)])


def chunk_similarities(query):
    """(chunks, cosine similarity of each chunk to the query)."""
    ensure_embeddings()
    rows = library.load_chunks(vectors=True)
    if not rows:
        return [], np.zeros(0)
    matrix = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
    return rows, matrix @ embed([query])[0]


def semantic_search(query):
    """Papers ranked by their best-matching chunk."""
    rows, sims = chunk_similarities(query)
    best = {}
    for row, sim in zip(rows, sims):
        if sim >= MIN_SIMILARITY and sim > best.get(row["paper_id"], {}).get("score", -1):
            best[row["paper_id"]] = {"id": row["paper_id"], "score": float(sim),
                                     "page": row["page"], "passage": row["text"]}
    return sorted(best.values(), key=lambda hit: -hit["score"])


def rrf(*rankings, k=60):
    """Reciprocal Rank Fusion. Each ranking is a list of ids, best first."""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


def search(method, query, papers, index, limit=50):
    """Ranked results: dicts with id, title, file_name, score, page, passage."""
    query = (query or "").strip()
    if not query:
        return []
    hits = {}  # semantic hits by paper id (they carry their own passage)
    if method == "BM25":
        ranking = bm25(index, query)
    elif method == "Semantic":
        hits = {h["id"]: h for h in semantic_search(query)}
        ranking = [(i, h["score"]) for i, h in hits.items()]
    elif method == "Hybrid":
        hits = {h["id"]: h for h in semantic_search(query)}
        lexical = [i for i, _ in bm25(index, query)[:limit]]
        ranking = rrf(lexical, list(hits)[:limit])
    else:
        raise ValueError(f"unknown method: {method}")

    by_id = {p["id"]: p for p in papers}
    chunks = None
    results = []
    for paper_id, score in ranking[:limit]:
        if paper_id not in by_id:
            continue
        if paper_id in hits:
            page, passage = hits[paper_id]["page"], hits[paper_id]["passage"]
        else:
            if chunks is None:
                chunks = {}
                for c in library.load_chunks():
                    chunks.setdefault(c["paper_id"], []).append(c)
            chunk = best_chunk(query, chunks.get(paper_id, []))
            page, passage = (chunk["page"], chunk["text"]) if chunk else (None, "")
        paper = by_id[paper_id]
        results.append({"id": paper_id, "title": paper["title"], "file_name": paper["file_name"],
                        "score": score, "page": page, "passage": passage})
    return results
