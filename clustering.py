"""Topic clustering: K-Means on paper embeddings; c-TF-IDF words (and optional Groq names) describe each topic."""
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import silhouette_score

import library
import semantic
from ir import stem

MAX_CHUNKS = 15  # embed the first ~2,500 words of a paper: skips references and appendices
STOP = set(ENGLISH_STOP_WORDS) | {
    "paper", "papers", "figure", "fig", "table", "using", "used", "use", "al", "also", "however",
    "thus", "show", "shown", "shows", "proposed", "propose", "approach", "method", "methods",
    "result", "results", "section", "arxiv", "http", "https", "www", "doi", "pp", "vol", "ieee",
    "acm", "proceedings", "conference", "journal", "university", "new", "based", "one", "two",
}


def paper_vectors():
    """One unit vector per paper = mean of its first chunk vectors."""
    semantic.ensure_embeddings()
    groups = {}
    for c in library.load_chunks(vectors=True):  # ordered by id = reading order
        group = groups.setdefault(c["paper_id"], [])
        if len(group) < MAX_CHUNKS:
            group.append(np.frombuffer(c["vec"], dtype=np.float32))
    ids = sorted(groups)
    vectors = np.vstack([np.mean(groups[i], axis=0) for i in ids])
    return ids, vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


def best_k(vectors):
    """(best number of topics, {k: silhouette}) for k = 2..8, using cosine silhouette."""
    n = len(vectors)
    scores = {}
    for k in range(2, min(8, n - 1) + 1):
        labels = KMeans(k, n_init=10, random_state=42).fit_predict(vectors)
        if len(set(labels)) == k:
            scores[k] = float(silhouette_score(vectors, labels, metric="cosine"))
    return (max(scores, key=scores.get) if scores else 2), scores


def body_text(text):
    """Cut the reference list so author names do not become topic words."""
    cuts = [m.start() for m in re.finditer(r"\n\s*references\s*\n", text, flags=re.I)]
    return text[:cuts[-1]] if cuts and cuts[-1] > len(text) * 0.3 else text


def topic_keywords(paper_docs, topics, top=6):
    """c-TF-IDF: words frequent in one topic but rare in the others.
    A word must appear in at least 2 papers of its topic (drops author names, typos)."""
    vec = CountVectorizer(stop_words=sorted(STOP), ngram_range=(1, 2), max_features=20000,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b")
    counts = vec.fit_transform(paper_docs).toarray().astype(float)  # papers x terms
    terms = vec.get_feature_names_out()
    topics = np.array(topics)
    ids = sorted(set(topics))
    topic_counts = np.vstack([counts[topics == t].sum(axis=0) for t in ids])
    in_papers = np.vstack([(counts[topics == t] > 0).sum(axis=0) for t in ids])
    sizes = np.array([(topics == t).sum() for t in ids])[:, None]
    topic_counts = topic_counts * (in_papers >= np.minimum(2, sizes))
    tf = topic_counts / (topic_counts.sum(axis=1, keepdims=True) + 1e-12)
    idf = np.log(1 + topic_counts.sum(axis=1).mean() / (topic_counts.sum(axis=0) + 1e-9))
    scores = tf * idf * np.where([" " in t for t in terms], 1.5, 1.0)  # favour phrases: "cold start"
    words = []
    for row in scores:
        picked, used = [], set()
        for j in np.argsort(-row):
            if row[j] <= 0 or len(picked) == top:
                break
            stems = {stem(w) for w in terms[j].split()}
            if stems & used:  # skip repeats: index / indexing / index compression
                continue
            picked.append(str(terms[j]))
            used |= stems
        words.append(picked)
    return words


def cluster_papers(papers, texts, k=None, namer=None):
    """Return (DataFrame with one row per paper, info dict).
    namer(keywords, titles) -> topic name; optional (the app uses Groq)."""
    ids, vectors = paper_vectors()
    by_id = {p["id"]: p for p in papers}
    keep = [n for n, i in enumerate(ids) if i in by_id]
    ids, vectors = [ids[n] for n in keep], vectors[keep]
    if len(ids) < 2:
        raise ValueError("Add at least 2 papers.")
    scores = {}
    if k is None:
        k, scores = best_k(vectors) if len(ids) >= 4 else (2, {})
    k = min(k, len(ids))
    raw = KMeans(k, n_init=10, random_state=42).fit_predict(vectors)

    # number topics 1..m, biggest first
    order = [t for t, _ in sorted(pd.Series(raw).value_counts().items(), key=lambda x: -x[1])]
    topics = [order.index(r) + 1 for r in raw]
    m = len(order)

    try:
        words = topic_keywords([body_text(texts[i])[:30000] for i in ids], topics)
    except ValueError:  # no usable words
        words = [[] for _ in range(m)]
    keywords = {t: ", ".join(words[t - 1]) for t in range(1, m + 1)}

    similarity = np.zeros(len(ids))
    for topic in range(1, m + 1):
        mask = np.array([t == topic for t in topics])
        centre = vectors[mask].mean(axis=0)
        similarity[mask] = vectors[mask] @ (centre / (np.linalg.norm(centre) + 1e-12))

    labels, naming_error = {}, None
    for topic in range(1, m + 1):
        name = ""
        if namer:
            try:
                titles = [by_id[i]["title"] for i, t in zip(ids, topics) if t == topic]
                name = namer(words[topic - 1], titles)
            except Exception as exc:
                naming_error = naming_error or str(exc)
        labels[topic] = name or " · ".join(words[topic - 1][:3]) or f"Topic {topic}"

    coords = PCA(n_components=2, random_state=42).fit_transform(vectors) if len(ids) >= 3 else np.zeros((len(ids), 2))
    df = pd.DataFrame({
        "Topic": topics, "Label": [labels[t] for t in topics],
        "Paper": [by_id[i]["title"] for i in ids], "File": [by_id[i]["file_name"] for i in ids],
        "Similarity": similarity.round(3), "Keywords": [keywords[t] for t in topics],
        "x": coords[:, 0], "y": coords[:, 1],
    }).sort_values(["Topic", "Similarity"], ascending=[True, False], kind="stable")
    silhouette = float(silhouette_score(vectors, raw, metric="cosine")) if 1 < m < len(ids) else None
    return df, {"k": m, "silhouette": silhouette, "scores": scores, "keywords": keywords,
                "labels": labels, "naming_error": naming_error}
