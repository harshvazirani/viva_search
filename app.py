"""Viva Quick Search — local semantic search over an uploaded Q&A .docx file.

Upload a Q&A document, type 2–5 keywords, get the best-matching Q&A block.
No LLM, no chat — pure retrieval with sentence-transformers embeddings and a
FAISS index. Disk persistence across refreshes is opt-in via the
"Remember across refreshes" checkbox; when enabled, the file is cached at
``~/.cache/viva-search/``. Leave it off on shared hosting.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import streamlit as st
from docx import Document
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Local persistence — remember the last uploaded .docx across refreshes
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "viva-search"
CACHE_POINTER = CACHE_DIR / "latest.json"


def _save_to_cache(file_bytes: bytes, filename: str) -> str:
    """Persist the uploaded file and remember it as the latest."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.md5(file_bytes).hexdigest()
    (CACHE_DIR / f"{file_hash}.docx").write_bytes(file_bytes)
    CACHE_POINTER.write_text(json.dumps({"hash": file_hash, "name": filename}))
    return file_hash


def _load_from_cache() -> Optional[Tuple[bytes, str, str]]:
    """Return (bytes, filename, hash) for the last uploaded file, if any."""
    if not CACHE_POINTER.exists():
        return None
    try:
        meta = json.loads(CACHE_POINTER.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    path = CACHE_DIR / f"{meta.get('hash', '')}.docx"
    if not path.exists():
        return None
    return path.read_bytes(), meta.get("name", "previous.docx"), meta["hash"]


def _clear_cache() -> None:
    """Forget the saved document (used when the user clicks Clear)."""
    if CACHE_POINTER.exists():
        CACHE_POINTER.unlink()
    for p in CACHE_DIR.glob("*.docx"):
        p.unlink()


MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 6
MAX_WORDS_PER_CHUNK = 500


# ---------------------------------------------------------------------------
# .docx parsing
# ---------------------------------------------------------------------------

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract full text from a .docx file, preserving line breaks."""
    doc = Document(io.BytesIO(file_bytes))
    lines: List[str] = []
    for para in doc.paragraphs:
        # Keep empty paragraphs so blank lines between Q/A blocks survive.
        lines.append(para.text)
    # Some Q&A docs stash content in tables — pull that too.
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    lines.append(para.text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Q/A extraction — robust to real-world messy formatting
# ---------------------------------------------------------------------------

# Question boundary: line start with Q:, Q -, Question:, or "1. " numbering.
# Whitespace allowed between letter and separator so "q - foo" works.
_Q_BOUNDARY = re.compile(
    r"(?:^|\n)[ \t]*(?:Question[ \t]*[:\-]?|Q[ \t]*[:\-]|\d+\.\s)",
    re.IGNORECASE,
)

# Answer marker: anchored to line start so "A:" inside prose can't split.
_A_MARKER = re.compile(
    r"(?m)^[ \t]*A(?:ns(?:wer)?)?[ \t]*[:\-]",
    re.IGNORECASE,
)

# Leading prefix to strip from a captured question. Longer alternative first
# so "Question:" isn't shortened to "uestion:" by a greedy "Q" match.
_Q_PREFIX = re.compile(
    r"^(?:Question[ \t]*[:\-\s]*|Q[ \t]*[:\-\s]*|\d+\.\s*)",
    re.IGNORECASE,
)
_A_PREFIX = re.compile(
    r"^A(?:ns(?:wer)?)?[ \t]*[:\-\s]*",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Normalize whitespace without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n"))
    return text.strip()


def extract_qa_pairs(text: str) -> List[str]:
    """Split text into one ``Q: ...\\nA: ...`` string per Q/A block.

    Strategy:
      1. Normalize whitespace.
      2. Detect question boundaries (Q:, Question:, numbered).
      3. Inside each block, split on the first line-start A: marker. If none,
         fall back to "first line = question, rest = answer".
      4. Drop genuinely empty halves so the index stays clean.
    """
    text = _normalize(text)
    if not text:
        return []

    matches = list(_Q_BOUNDARY.finditer(text))
    if not matches:
        return []

    pairs: List[str] = []
    for i, m in enumerate(matches):
        # Skip the leading newline captured by (?:^|\n) but keep the marker.
        start = m.start() + (1 if text[m.start()] == "\n" else 0)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue

        a_split = _A_MARKER.split(chunk, maxsplit=1)
        if len(a_split) == 2:
            question, answer = a_split[0], a_split[1]
        else:
            # Fallback: no explicit A: marker — first line is question, rest is answer.
            lines = chunk.split("\n", 1)
            if len(lines) < 2:
                continue
            question, answer = lines[0], lines[1]

        question = _Q_PREFIX.sub("", question, count=1).strip()
        answer = _A_PREFIX.sub("", answer, count=1)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

        if len(question) >= 3 and len(answer) >= 1:
            pairs.append(f"Q: {question}\nA: {answer}")

    return pairs


# ---------------------------------------------------------------------------
# Chunking — split very long Q/A blocks while preserving the question as context
# ---------------------------------------------------------------------------

def _split_sentences(body: str) -> List[str]:
    """Paragraph-then-sentence split; preserves paragraph boundaries."""
    segments: List[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        para_one_line = re.sub(r"\s*\n\s*", " ", para)
        segments.extend(s for s in re.split(r"(?<=[.!?])\s+", para_one_line) if s)
    return segments


def _split_long_chunk(chunk: str, max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
    """If a Q/A block exceeds ``max_words``, split the answer while keeping the
    question as a prefix on every sub-chunk so retrieval context is preserved."""
    if len(chunk.split()) <= max_words:
        return [chunk]

    a_match = re.search(r"(?m)^A[:\-]", chunk)
    if a_match:
        q_header = chunk[: a_match.start()].rstrip()
        answer_body = chunk[a_match.start():].strip()
    else:
        q_header = ""
        answer_body = chunk

    segments = _split_sentences(answer_body) or [answer_body]
    parts: List[str] = []
    buf: List[str] = []
    buf_words = 0
    for seg in segments:
        seg_words = len(seg.split())
        if buf and buf_words + seg_words > max_words:
            body = " ".join(buf)
            parts.append(f"{q_header}\n{body}".strip() if q_header else body)
            buf, buf_words = [seg], seg_words
        else:
            buf.append(seg)
            buf_words += seg_words
    if buf:
        body = " ".join(buf)
        parts.append(f"{q_header}\n{body}".strip() if q_header else body)
    return parts or [chunk]


def build_chunks(text: str) -> List[str]:
    chunks: List[str] = []
    for block in extract_qa_pairs(text):
        chunks.extend(_split_long_chunk(block))
    return chunks


# ---------------------------------------------------------------------------
# Model + index
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def build_index(docs: List[str], model: SentenceTransformer) -> faiss.Index:
    embeddings = model.encode(
        docs,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(
    query: str,
    docs: List[str],
    index: faiss.Index,
    model: SentenceTransformer,
    k: int = TOP_K,
) -> List[str]:
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    _distances, ids = index.search(q_emb, min(k, len(docs)))
    return [docs[i] for i in ids[0] if i >= 0]


# ---------------------------------------------------------------------------
# Result rendering helpers
# ---------------------------------------------------------------------------

def _split_qa(chunk: str) -> tuple[str, str]:
    """Split a stored chunk into (question, answer), with markers stripped."""
    lines = chunk.split("\n", 1)
    q = lines[0]
    a = lines[1] if len(lines) > 1 else ""
    q = re.sub(r"^\s*Q[:\-\s]*", "", q, flags=re.IGNORECASE).strip()
    a = re.sub(r"^\s*A(?:ns(?:wer)?)?[:\-\s]*", "", a, flags=re.IGNORECASE).strip()
    return q, a


def _md_preserve_breaks(text: str) -> str:
    """Keep paragraph breaks (\\n\\n) and convert single \\n to markdown hard breaks."""
    paragraphs = text.split("\n\n")
    return "\n\n".join(p.replace("\n", "  \n") for p in paragraphs)


def _answer_html(a: str) -> str:
    """Convert a plain-text answer into HTML-safe paragraph markup."""
    out_paragraphs = []
    for para in a.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        out_paragraphs.append(
            "<p>" + html.escape(para).replace("\n", "<br>") + "</p>"
        )
    return "".join(out_paragraphs)


def _render_best_match(chunk: str) -> None:
    """Large bordered card — the primary result, with bigger body text."""
    q, a = _split_qa(chunk)
    with st.container(border=True):
        st.markdown(
            f'<div class="best-question">{html.escape(q)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="best-answer">{_answer_html(a)}</div>',
            unsafe_allow_html=True,
        )


def _render_secondary(chunk: str, rank: int) -> None:
    """Collapsed expander — matches best-match body size when opened."""
    q, a = _split_qa(chunk)
    with st.expander(f"**#{rank}** — {q}", expanded=False):
        st.markdown(
            f'<div class="best-answer">{_answer_html(a)}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Viva Quick Search", layout="centered")

# Custom styling — typography, spacing, and best-match emphasis.
st.markdown(
    """
    <style>
    /* Content container — narrower for prose readability, more top padding */
    section.main > div.block-container {
        max-width: 820px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* Title: tighter tracking, bolder weight */
    section.main h1 {
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem !important;
    }

    /* Section eyebrows (##### Best match / Other matches / Browse all) */
    section.main h5 {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        opacity: 0.55;
        margin-top: 2rem;
        margin-bottom: 0.75rem;
    }

    /* Larger, friendlier search input */
    [data-testid="stTextInput"] input {
        font-size: 1.05rem;
        padding: 0.85rem 1rem;
        border-radius: 10px;
    }

    /* Best-match question heading */
    .best-question {
        font-size: 1.35rem;
        font-weight: 600;
        line-height: 1.35;
        letter-spacing: -0.015em;
        margin-bottom: 0.85rem;
    }

    /* Best-match answer body — this is the main size bump */
    .best-answer {
        font-size: 1.15rem;
        line-height: 1.75;
    }
    .best-answer p {
        margin: 0 0 0.9em 0;
    }
    .best-answer p:last-child {
        margin-bottom: 0;
    }

    /* Bordered containers — softer corners and a bit more breathing room */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
    }

    /* Expanders — subtle, clickable feel */
    [data-testid="stExpander"] {
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary {
        padding-top: 0.55rem;
        padding-bottom: 0.55rem;
        font-size: 0.95rem;
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] h3 {
        margin-top: 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.7;
    }

    /* Hide Streamlit's auto-generated anchor links on markdown headings */
    h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a,
    h1 a[href^="#"], h2 a[href^="#"], h3 a[href^="#"],
    h4 a[href^="#"], h5 a[href^="#"], h6 a[href^="#"],
    [data-testid="stHeadingActionElements"],
    [data-testid="StyledLinkIconContainer"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

cached_on_disk = _load_from_cache()

with st.sidebar:
    st.markdown("### Document")
    uploaded = st.file_uploader(
        "Upload Q&A document (.docx)",
        type=["docx"],
        label_visibility="collapsed",
    )
    sidebar_info = st.empty()

    remember = st.checkbox(
        "Remember across refreshes",
        value=cached_on_disk is not None,
        help=(
            "Saves your uploaded file to local disk so refreshes don't clear "
            "it. Only enable this on your own machine — on shared hosting "
            "(e.g. Streamlit Cloud) a shared filesystem can leak files "
            "between visitors."
        ),
    )

    # If the user turned the checkbox off, forget any previously saved file now.
    if not remember and cached_on_disk is not None:
        _clear_cache()
        cached_on_disk = None

    st.markdown("---")
    st.caption(
        "Local semantic search. No LLM, no chat — your document stays in "
        "this session."
    )

st.title("Viva Quick Search")

# Resolve the active file:
#   - a fresh upload always wins;
#   - otherwise use the disk cache only if the user has opted in.
active_bytes: Optional[bytes] = None
active_name: Optional[str] = None
active_hash: Optional[str] = None
from_cache = False

if uploaded is not None:
    active_bytes = uploaded.getvalue()
    active_name = uploaded.name
    if remember:
        active_hash = _save_to_cache(active_bytes, active_name)
    else:
        active_hash = hashlib.md5(active_bytes).hexdigest()
elif remember and cached_on_disk is not None:
    active_bytes, active_name, active_hash = cached_on_disk
    from_cache = True

if active_bytes is None:
    st.info("Please upload a .docx file in the sidebar to begin.")

    st.markdown("#### How to format your Q&A document")
    st.markdown(
        "Write each question on a line starting with `Q:` and its answer on a "
        "line starting with `A:`. Separate pairs with a blank line."
    )
    st.code(
        "Q: What is the main contribution of your thesis?\n"
        "A: The primary contribution is a unified framework that combines\n"
        "retrieval and re-ranking under a single contrastive loss.\n"
        "\n"
        "Q: What are the limitations of your work?\n"
        "A: The modest dataset size limits statistical power. External\n"
        "validity is restricted to the single domain we evaluated on.",
        language="text",
    )

    st.markdown(
        "**Tips for best results**\n"
        "- Keep each Q&A pair short and self-contained.\n"
        "- Answers can span multiple paragraphs — paragraph breaks are preserved.\n"
        "- Include synonyms in parentheses after the question to broaden "
        "keyword matching, e.g. `Q: Why did you choose method X? "
        "(method choice, approach selection)`.\n"
        "- Very long answers (>500 words) are split automatically; the question "
        "is repeated on each sub-chunk so context is preserved."
    )

    with st.expander("Other accepted formats"):
        st.markdown(
            "The parser is tolerant of common variations:\n"
            "- `Question:` / `Answer:` instead of `Q:` / `A:`\n"
            "- Numbered questions (`1. What is ...`)\n"
            "- Dash separators (`Q - ...` / `A - ...`)\n"
            "- Extra whitespace around the markers is fine\n"
            "- Mixed case (`q:` / `a:` / `QUESTION:`) is fine\n"
            "\n"
            "If a pair is missing the `A:` marker, the parser falls back to "
            "treating the first line as the question and the rest as the answer."
        )
    st.stop()

# Re-index only when the active file actually changes.
if st.session_state.get("file_hash") != active_hash:
    try:
        raw_text = extract_text_from_docx(active_bytes)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to parse .docx file: {e}")
        st.stop()

    docs = build_chunks(raw_text)
    if not docs:
        st.warning(
            "No Q&A pairs detected. Make sure the document uses `Q:` and `A:` "
            "markers at the start of lines."
        )
        st.stop()

    model = load_model()
    with st.spinner(f"Indexing {len(docs)} Q&A chunks..."):
        index = build_index(docs, model)

    st.session_state.file_hash = active_hash
    st.session_state.docs = docs
    st.session_state.index = index
    st.session_state.doc_info = {"name": active_name, "chunks": len(docs)}

docs = st.session_state.docs
index = st.session_state.index
model = load_model()

# Populate the sidebar's info placeholder now that the doc is ready.
with sidebar_info.container():
    info = st.session_state["doc_info"]
    st.caption(f"**{info['name']}**")
    st.caption(f"{info['chunks']} Q&A chunks indexed")
    if from_cache:
        st.caption("_Loaded from local cache_")

query = st.text_input(
    "Search",
    placeholder="Type keywords: method choice, limitations, future work...",
    label_visibility="collapsed",
)

if query.strip():
    # Search mode — show best match prominently, others in expanders.
    results = search(query, docs, index, model)
    if results:
        st.markdown("##### Best match")
        _render_best_match(results[0])
        if len(results) > 1:
            st.markdown("")
            st.markdown("##### Other matches")
            for i, chunk in enumerate(results[1:], start=2):
                _render_secondary(chunk, rank=i)
    else:
        st.caption("No matches.")
else:
    # Browse mode — show all Q&A pairs as collapsed expanders.
    st.caption(
        f"{len(docs)} Q&A pairs indexed. Type keywords above to search, "
        "or browse all pairs below."
    )
    st.markdown("##### Browse all")
    for i, chunk in enumerate(docs, start=1):
        _render_secondary(chunk, rank=i)
