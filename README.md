# ResearchSphere

A small search engine for your own research papers (PDFs). Built with Python and Streamlit for an Information Retrieval mini project.

## Features

| Page | What it does | IR concept |
|---|---|---|
| Search → BM25 | Keyword ranking | tokenizing, stop words, stemming, inverted index, BM25 |
| Search → Semantic | finds similar meaning | sentence embeddings, cosine similarity |
| Search → Hybrid | BM25 + Semantic merged | Reciprocal Rank Fusion |
| Ask papers | cited answers from your papers | RAG (retrieve, then generate) |
| Topics | groups papers into named topics | K-Means, silhouette, c-TF-IDF keywords, optional AI names |
| Evaluate | measures search quality | Precision@k, Recall@k, MRR |
| History | past searches | |

## Files (7)

| File | Job |
|---|---|
| `ir.py` | classical IR: cleaning, stemming, inverted index, BM25 functions, highlighting |
| `library.py` | SQLite database, PDF import, chunks, history |
| `semantic.py` | embeddings, semantic search, RRF, the `search()` function |
| `rag.py` | passage retrieval + Groq answer with citation check |
| `clustering.py` | topic clustering |
| `evaluate.py` | evaluation metrics |
| `app.py` | the Streamlit interface |

## Run (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

First semantic search downloads the `all-MiniLM-L6-v2` model (about 90 MB) once. After that it runs offline.

For **Ask papers**: copy `.env.example` to `.env`, paste your Groq key, restart the app.

## Demo script (7 minutes)

1. Import 5 to 6 PDFs from the sidebar.
2. **BM25**: search `compressing postings`. Stemming lets it match "compression".
3. **Semantic vs BM25**: search a paraphrase, e.g. `how do search engines store word locations`. BM25 misses it, Semantic finds it.
4. **Hybrid**: same query, explain RRF in one line.
5. **Evaluate**: enter 5 to 8 lines like `query | paper.pdf`, show the table.
6. **Ask papers**: ask a question, open the sources.
7. **Topics**: run clustering, show the topic cards, the silhouette chart and the map.

## How the important parts work

- **BM25** scores a paper by how often query words appear (term frequency), how rare they are in the collection (IDF), and normalises for paper length.
- **RRF** merges two rankings by adding `1 / (60 + rank)` for each list, so no score calibration is needed.
- **Semantic search** splits pages into ~180-word chunks, embeds them, and compares the query with each chunk by cosine similarity. A paper is ranked by its best chunk. Matches below 0.2 similarity are ignored (`MIN_SIMILARITY` in `semantic.py`).
- **Topics:** each paper is one vector (mean of the embeddings of its first ~2,500 words, so references and appendices do not matter). K-Means groups them; the number of topics is the one with the best cosine silhouette. Topic keywords use **c-TF-IDF**: words that are frequent in one topic but rare in the others (reference lists are cut, repeated word forms are merged). With a Groq key, a short readable name is written for each topic; otherwise the top keywords are the name.
- **RAG** answers only from retrieved passages. Any citation label the model invents is replaced by `[unverified]`.

## Limits

- Scanned PDFs (no selectable text) are rejected. There is no OCR.
- Titles are guessed from PDF metadata or the first page and can be wrong. The file name is always shown.
- No search system is 100% accurate. Use the **Evaluate** page to measure it on your papers.
- Ask papers (and AI topic names) send text to Groq. Do not use it for confidential papers.
