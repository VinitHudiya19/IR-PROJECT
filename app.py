"""ResearchSphere: search, ask and explore your research papers."""
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

import library
import rag
import semantic
from clustering import cluster_papers
from evaluate import evaluate, parse_judgments
from ir import build_index, highlight

load_dotenv()
st.set_page_config(page_title="ResearchSphere", page_icon="📚", layout="wide")
library.init()

HINTS = {
    "Hybrid": "Best overall: merges keyword (BM25) and meaning-based (semantic) rankings with Reciprocal Rank Fusion.",
    "BM25": "Keyword ranking. Words are stemmed, so 'clustering' also finds 'cluster'.",
    "Semantic": "Finds passages with a similar meaning, even when the words differ.",
}
SCORES = {"Hybrid": "fusion score", "BM25": "BM25 score", "Semantic": "cosine similarity"}


def get_index(papers):
    """Rebuild the search index only when the library changes."""
    ids = tuple(p["id"] for p in papers)
    if st.session_state.get("index_ids") != ids:
        with st.spinner("Building search index..."):
            st.session_state.index = build_index(library.get_texts())
        st.session_state.index_ids = ids
    return st.session_state.index


# ---------------- sidebar ----------------
papers = library.get_papers()
with st.sidebar:
    st.title("ResearchSphere")
    page = st.radio("Go to", ["Search", "Ask papers", "Topics", "Evaluate", "History"], label_visibility="collapsed")
    st.divider()
    st.subheader("Add papers")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Import", type="primary", disabled=not files):
        for f in files:
            try:
                if library.import_pdf(f.getvalue(), f.name) is None:
                    st.info(f"Already in library: {f.name}")
                else:
                    st.success(f"Added: {f.name}")
            except Exception as exc:
                st.warning(f"Skipped {f.name}: {exc}")
        papers = library.get_papers()
    if papers:
        with st.expander(f"Library ({len(papers)} papers)"):
            for p in papers:
                st.caption(f"{p['title'][:70]}  \n`{p['file_name']}`")
            doomed = st.selectbox("Remove a paper", papers, index=None, format_func=lambda p: p["file_name"])
            if doomed and st.button("Remove"):
                library.delete_paper(doomed["id"])
                st.rerun()
    env_key, env_model = rag.settings()
    with st.expander("Groq (optional)", expanded=False):
        typed_key = st.text_input("API key", type="password", help="Or set GROQ_API_KEY in the .env file.")
        groq_model = st.text_input("Model", value=env_model)
    groq_key = typed_key.strip() or env_key
    st.caption("Everything stays on this computer. Only 'Ask papers' and AI topic names send text to Groq.")

# ---------------- search ----------------
if page == "Search":
    st.header("Search papers")
    with st.form("search"):
        query = st.text_input("Query", placeholder="e.g. how are postings lists compressed")
        c1, c2 = st.columns([3, 1])
        method = c1.selectbox("Method", semantic.METHODS)
        limit = c2.selectbox("Results", [5, 10, 20], index=1)
        go = st.form_submit_button("Search", type="primary")
    st.caption(HINTS[method])
    if go and query.strip():
        if not papers:
            st.info("Add some PDFs first (sidebar).")
        else:
            try:
                with st.spinner("Searching... (the first semantic search downloads the model and embeds your papers)"):
                    results = semantic.search(method, query, papers, get_index(papers), limit)
                library.save_search(query.strip(), method, len(results))
                st.subheader(f"{len(results)} result(s)")
                if not results:
                    st.info("No matching papers. Try other words or another method.")
                for rank, r in enumerate(results, 1):
                    with st.container(border=True):
                        st.markdown(f"**{rank}. {r['title']}**")
                        st.caption(f"{r['file_name']} · page {r['page']} · {SCORES[method]}: {r['score']:.3f}")
                        st.markdown(highlight(r["passage"], query), unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Search failed: {exc}")

# ---------------- ask (RAG) ----------------
elif page == "Ask papers":
    st.header("Ask your papers")
    st.caption("Finds the best passages in your papers, then Groq writes an answer that cites them. Always check the sources.")
    question = st.text_area("Your question", height=90, placeholder="What are the main approaches to document clustering?")
    count = st.slider("Passages to use", 3, 8, 5)
    if st.button("Answer", type="primary"):
        if not question.strip() or not papers:
            st.warning("Add papers and type a question.")
        else:
            try:
                with st.spinner("Finding passages..."):
                    contexts = rag.retrieve(question, count)
                if not contexts:
                    st.warning("No relevant passages found. Try rephrasing the question.")
                else:
                    with st.spinner("Writing the answer..."):
                        reply = rag.answer(question, contexts, groq_key, groq_model.strip())
                    st.subheader("Answer")
                    st.markdown(reply)
                    st.subheader("Sources")
                    for c in contexts:
                        with st.expander(f"[{c['label']}] {c['title']} · page {c['page']}"):
                            st.caption(c["file_name"])
                            st.write(c["text"])
            except Exception as exc:
                st.error(str(exc))

# ---------------- topics ----------------
elif page == "Topics":
    st.header("Topic clustering")
    st.caption("Papers are embedded and grouped with K-Means. Each topic gets distinctive keywords (c-TF-IDF) and, optionally, an AI-written name.")
    if len(papers) < 2:
        st.info("Add at least 2 papers.")
    else:
        max_k = min(8, len(papers))
        auto = st.checkbox("Choose the number of topics automatically", value=True)
        k = None if auto else (st.slider("Number of topics", 2, max_k, min(3, max_k)) if max_k > 2 else 2)
        use_ai = st.checkbox("Name topics with Groq", value=bool(groq_key), disabled=not groq_key,
                             help="Needs a Groq key (sidebar). Without it, topics are named by their top keywords.")
        ids = tuple(p["id"] for p in papers)
        if st.button("Find topics", type="primary"):
            namer = (lambda words, titles: rag.name_topic(words, titles, groq_key, groq_model.strip())) if use_ai else None
            try:
                with st.spinner("Embedding and clustering..."):
                    df, info = cluster_papers(papers, library.get_texts(), k, namer)
                st.session_state.topics = (ids, df, info)
            except Exception as exc:
                st.error(str(exc))
        saved = st.session_state.get("topics")
        if saved and saved[0] == ids:
            _, df, info = saved
            if info["naming_error"]:
                st.warning(f"AI naming failed, showing keyword names instead: {info['naming_error']}")
            m1, m2 = st.columns(2)
            m1.metric("Topics", info["k"])
            m2.metric("Silhouette (higher is better)", "n/a" if info["silhouette"] is None else f"{info['silhouette']:.2f}")
            for topic, label in info["labels"].items():
                rows = df[df["Topic"] == topic]
                with st.container(border=True):
                    st.markdown(f"**Topic {topic}: {label}** · {len(rows)} paper(s)")
                    st.caption("Keywords: " + (info["keywords"][topic] or "-"))
                    for _, r in rows.head(5).iterrows():
                        st.markdown(f"- {r['Paper']}  \n  `{r['File']}` · closeness to topic centre {r['Similarity']:.2f}")
            st.plotly_chart(px.scatter(df, x="x", y="y", color="Label", hover_name="Paper",
                                       title="Paper map (2-D view, for orientation only)"))
            st.dataframe(df[["Topic", "Label", "Paper", "File", "Similarity", "Keywords"]], hide_index=True)
            st.download_button("Download CSV", df.drop(columns=["x", "y"]).to_csv(index=False), "topics.csv", "text/csv")

# ---------------- evaluate ----------------
elif page == "Evaluate":
    st.header("Evaluate search quality")
    st.caption("Tell the app which papers are relevant for a few queries. It measures how well each method finds them.")
    with st.expander("Your file names"):
        st.code("\n".join(p["file_name"] for p in papers) or "(no papers yet)")
    text = st.text_area("One line per query:  query | relevant1.pdf, relevant2.pdf", height=170,
                        placeholder="how are postings lists compressed | index_compression.pdf\ngrouping similar documents | clustering_survey.pdf")
    k = st.slider("k (top results to check)", 1, 10, 3)
    if st.button("Run evaluation", type="primary"):
        judgments, problems = parse_judgments(text, [p["file_name"] for p in papers])
        for problem in problems:
            st.warning(problem)
        if not judgments:
            st.error("No valid lines found. Use the format: query | file.pdf")
        else:
            try:
                index = get_index(papers)
                rank = lambda method: (lambda q: [r["file_name"] for r in semantic.search(method, q, papers, index, 50)])
                with st.spinner("Evaluating..."):
                    table = evaluate(judgments, {m: rank(m) for m in ("BM25", "Semantic", "Hybrid")}, k)
                st.caption(f"{len(judgments)} query(ies), averaged. Higher is better (max 1.0).")
                st.dataframe(table.round(3), hide_index=True)
                st.bar_chart(table.set_index("Method"))
            except Exception as exc:
                st.error(str(exc))

# ---------------- history ----------------
elif page == "History":
    st.header("Search history")
    history = library.get_history()
    if not history:
        st.info("No searches yet.")
    else:
        st.dataframe(pd.DataFrame(history), hide_index=True)
        if st.button("Clear history"):
            library.clear_history()
            st.rerun()
