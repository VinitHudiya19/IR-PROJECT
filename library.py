"""SQLite storage (papers, chunks, history) and PDF import."""
import hashlib
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

try:
    import pymupdf
except ImportError:  # older PyMuPDF
    import fitz as pymupdf

DB_PATH = Path(os.getenv("RS_DB_PATH") or Path(__file__).parent / "data" / "researchsphere.db")


@contextmanager
def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        with con:  # commit on success
            yield con
    finally:
        con.close()


def init():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY, file_name TEXT NOT NULL, title TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE, text TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            page INTEGER NOT NULL, text TEXT NOT NULL, vec BLOB);
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY, query TEXT NOT NULL, method TEXT NOT NULL,
            results INTEGER NOT NULL, searched TEXT DEFAULT CURRENT_TIMESTAMP);
        """)


# ---------- PDF import ----------
def _guess_title(meta_title, first_page, file_name):
    meta_title = " ".join(meta_title.split())
    junk = {"untitled", "microsoft word", "document"}
    if len(meta_title) >= 8 and meta_title.lower() not in junk and not meta_title.lower().endswith((".doc", ".docx", ".pdf", ".tex")):
        return meta_title[:200]
    for line in first_page.splitlines()[:8]:
        line = " ".join(line.split())
        if 15 <= len(line) <= 150:
            return line
    return Path(file_name).stem.replace("_", " ").replace("-", " ")


def read_pdf(raw, file_name):
    """Return (title, [(page_number, text)]). Raises ValueError for bad files."""
    if not raw.startswith(b"%PDF"):
        raise ValueError("not a PDF file")
    pages = []
    with pymupdf.open(stream=raw, filetype="pdf") as doc:
        meta_title = (doc.metadata or {}).get("title") or ""
        for number, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if text:
                pages.append((number, text))
    if not pages:
        raise ValueError("no selectable text (scanned PDF? OCR is not supported)")
    return _guess_title(meta_title, pages[0][1], file_name), pages


def split_chunks(text, size=180, overlap=30):
    """Split text into overlapping chunks of about `size` words."""
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks


def import_pdf(raw, file_name):
    """Store a PDF. Returns the paper id, or None if it is already in the library."""
    file_hash = hashlib.sha256(raw).hexdigest()
    with db() as con:
        if con.execute("SELECT 1 FROM papers WHERE file_hash = ?", (file_hash,)).fetchone():
            return None
    title, pages = read_pdf(raw, file_name)
    with db() as con:
        paper_id = con.execute(
            "INSERT INTO papers (file_name, title, file_hash, text) VALUES (?, ?, ?, ?)",
            (Path(file_name).name, title, file_hash, "\n".join(t for _, t in pages)),
        ).lastrowid
        con.executemany(
            "INSERT INTO chunks (paper_id, page, text) VALUES (?, ?, ?)",
            [(paper_id, page, c) for page, text in pages for c in split_chunks(text)],
        )
    return paper_id


# ---------- reading ----------
def get_papers():
    with db() as con:
        return [dict(r) for r in con.execute("SELECT id, file_name, title FROM papers ORDER BY id")]


def get_texts():
    """{paper_id: title + full text} for building the search index."""
    with db() as con:
        return {r["id"]: r["title"] + "\n" + r["text"] for r in con.execute("SELECT id, title, text FROM papers")}


def load_chunks(vectors=False):
    """All chunks as dicts. With vectors=True only embedded chunks, including 'vec'."""
    sql = "SELECT id, paper_id, page, text" + (", vec FROM chunks WHERE vec IS NOT NULL" if vectors else " FROM chunks")
    with db() as con:
        return [dict(r) for r in con.execute(sql + " ORDER BY id")]


def chunks_missing_vectors():
    with db() as con:
        return [dict(r) for r in con.execute("SELECT id, text FROM chunks WHERE vec IS NULL ORDER BY id")]


def save_vectors(pairs):
    """pairs: [(vector_bytes, chunk_id)]"""
    with db() as con:
        con.executemany("UPDATE chunks SET vec = ? WHERE id = ?", pairs)


def delete_paper(paper_id):
    with db() as con:
        con.execute("DELETE FROM papers WHERE id = ?", (paper_id,))


# ---------- history ----------
def save_search(query, method, results):
    with db() as con:
        con.execute("INSERT INTO history (query, method, results) VALUES (?, ?, ?)", (query, method, results))


def get_history(limit=200):
    with db() as con:
        return [dict(r) for r in con.execute(
            "SELECT searched AS Time, query AS Query, method AS Method, results AS Results "
            "FROM history ORDER BY id DESC LIMIT ?", (limit,))]


def clear_history():
    with db() as con:
        con.execute("DELETE FROM history")
