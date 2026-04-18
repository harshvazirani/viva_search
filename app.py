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
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
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
DEFAULT_TOP_K = 6
MAX_TOP_K = 20
MAX_WORDS_PER_CHUNK = 500


# ---------------------------------------------------------------------------
# .docx parsing
# ---------------------------------------------------------------------------

# WordprocessingML namespace prefix used on every tag in a .docx body.
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Each paragraph's text is prefixed with ⟪PAGE=N⟫ during extraction so page
# info survives normalization and Q/A splitting. Uncommon Unicode brackets
# make false positives in user content essentially impossible.
_PAGE_MARKER = re.compile(r"⟪PAGE=(\d+)⟫")


def _count_page_breaks(element) -> int:
    """Number of page boundaries that occur inside a WordML element.

    ``w:lastRenderedPageBreak`` is written by Word/LibreOffice after each
    save-and-render — it's absent if the file was generated programmatically
    and never opened in a word processor. ``w:br w:type='page'`` is the
    explicit page break a user inserts with Ctrl-Enter.
    """
    count = 0
    for child in element.iter():
        if child.tag == f"{_W_NS}lastRenderedPageBreak":
            count += 1
        elif child.tag == f"{_W_NS}br" and child.get(f"{_W_NS}type") == "page":
            count += 1
    return count


def _iter_paragraphs_in_order(doc):
    """Yield every paragraph in document order, flattening tables into cells."""
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            for row in Table(child, doc).rows:
                for cell in row.cells:
                    yield from cell.paragraphs


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract full text, prefixing each paragraph with a ⟪PAGE=N⟫ marker.

    Page numbers come from ``w:lastRenderedPageBreak`` + explicit page breaks
    in document order. If the file has no break info (never rendered), every
    paragraph is tagged page 1 and the UI hides the page badge.
    """
    doc = Document(io.BytesIO(file_bytes))
    lines: List[str] = []
    current_page = 1
    for para in _iter_paragraphs_in_order(doc):
        lines.append(f"⟪PAGE={current_page}⟫{para.text}")
        current_page += _count_page_breaks(para._element)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Q/A extraction — robust to real-world messy formatting
# ---------------------------------------------------------------------------

# Question boundary: line start with Q:, Q -, Question:, or "1. " numbering.
# Whitespace allowed between letter and separator so "q - foo" works.
# The optional ⟪PAGE=N⟫ group lets the boundary survive after page tagging.
_Q_BOUNDARY = re.compile(
    r"(?:^|\n)(?:⟪PAGE=\d+⟫)?[ \t]*"
    r"(?:Question[ \t]*[:\-]?|Q[ \t]*[:\-]|\d+\.\s)",
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


def extract_qa_pairs(text: str) -> List[Tuple[str, int]]:
    """Split text into ``(Q: ...\\nA: ..., page_number)`` tuples per Q/A block.

    Strategy:
      1. Normalize whitespace (⟪PAGE=N⟫ markers survive intact).
      2. Detect question boundaries (Q:, Question:, numbered).
      3. Read the first ⟪PAGE=N⟫ inside each block — that's the page the
         question line starts on.
      4. Strip all page markers, then split on the first line-start A: marker.
         If none, fall back to "first line = question, rest = answer".
      5. Drop genuinely empty halves so the index stays clean.
    """
    text = _normalize(text)
    if not text:
        return []

    matches = list(_Q_BOUNDARY.finditer(text))
    if not matches:
        return []

    pairs: List[Tuple[str, int]] = []
    for i, m in enumerate(matches):
        # Skip the leading newline captured by (?:^|\n) but keep the marker.
        start = m.start() + (1 if text[m.start()] == "\n" else 0)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue

        page_match = _PAGE_MARKER.search(chunk)
        page = int(page_match.group(1)) if page_match else 1
        chunk = _PAGE_MARKER.sub("", chunk).strip()
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
            pairs.append((f"Q: {question}\nA: {answer}", page))

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


def build_chunks(text: str) -> Tuple[List[str], List[int]]:
    """Return parallel lists: Q&A text chunks and the page each one starts on.

    When one Q/A block is split into multiple sub-chunks for length, each
    sub-chunk inherits the original block's starting page — good enough for
    "jump to this section of the PDF" UX.
    """
    chunks: List[str] = []
    pages: List[int] = []
    for block, page in extract_qa_pairs(text):
        for sub in _split_long_chunk(block):
            chunks.append(sub)
            pages.append(page)
    return chunks, pages


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
    k: int = DEFAULT_TOP_K,
) -> List[int]:
    """Return the indices of the top-k matching docs, best first."""
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    _distances, ids = index.search(q_emb, min(k, len(docs)))
    return [int(i) for i in ids[0] if i >= 0]


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


def _render_best_match(chunk: str, page: Optional[int]) -> None:
    """Large bordered card — the primary result, with bigger body text."""
    q, a = _split_qa(chunk)
    with st.container(border=True):
        if page is not None:
            st.markdown(
                f'<div class="page-badge">Page {page}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="best-question">{html.escape(q)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="best-answer">{_answer_html(a)}</div>',
            unsafe_allow_html=True,
        )


def _render_secondary(chunk: str, rank: int, page: Optional[int]) -> None:
    """Collapsed row — uses a native <details> element so we can attach a
    CSS-driven hover tooltip (`data-page` + `::after`). Streamlit's built-in
    expander doesn't expose a tooltip API in this version, and the browser's
    native `title=` tooltip has a long delay that feels broken."""
    q, a = _split_qa(chunk)
    tooltip_attr = f' data-page="Page {page}"' if page is not None else ""
    label = f"<strong>#{rank}</strong> — {html.escape(q)}"
    body = _answer_html(a)
    st.markdown(
        f'<details class="qa-secondary"{tooltip_attr}>'
        f'<summary>{label}</summary>'
        f'<div class="best-answer qa-secondary-body">{body}</div>'
        f'</details>',
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

    /* Page badge — small pill above the best-match question */
    .page-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.7rem;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        background: rgba(127, 127, 127, 0.12);
        opacity: 0.95;
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

    /* Custom <details> rows used for secondary matches — styled to match
       Streamlit's own expander so the two coexist visually. */
    details.qa-secondary {
        position: relative;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 0.55rem 1rem;
        margin-bottom: 0.5rem;
        background: transparent;
    }
    details.qa-secondary > summary {
        cursor: pointer;
        list-style: none;
        font-size: 0.95rem;
        padding: 0.1rem 0;
    }
    details.qa-secondary > summary::-webkit-details-marker { display: none; }
    details.qa-secondary > summary::before {
        content: "▸";
        display: inline-block;
        width: 1rem;
        margin-right: 0.35rem;
        opacity: 0.55;
        transition: transform 0.15s ease;
    }
    details.qa-secondary[open] > summary::before {
        transform: rotate(90deg);
    }
    details.qa-secondary > .qa-secondary-body {
        margin-top: 0.75rem;
    }

    /* Instant CSS tooltip — reveals "Page N" in the top-right corner on hover.
       `title=` was too slow (browsers delay ~1s); `data-page` + ::after fires
       immediately and is also visible while the row is expanded. */
    details.qa-secondary[data-page]::after {
        content: attr(data-page);
        position: absolute;
        top: 0.5rem;
        right: 0.75rem;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background: rgba(0, 0, 0, 0.75);
        color: #fff;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.12s ease;
    }
    details.qa-secondary[data-page]:hover::after,
    details.qa-secondary[data-page]:focus-within::after {
        opacity: 1;
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

    docs, pages = build_chunks(raw_text)
    if not docs:
        st.warning(
            "No Q&A pairs detected. Make sure the document uses `Q:` and `A:` "
            "markers at the start of lines."
        )
        st.stop()

    model = load_model()
    with st.spinner(f"Indexing {len(docs)} Q&A chunks..."):
        index = build_index(docs, model)

    # Only surface pages if the document actually has break info. A doc that
    # was never rendered has every paragraph tagged page 1 — showing "Page 1"
    # on every result would be noise, not signal.
    page_count = max(pages) if pages else 1
    has_pages = page_count > 1

    st.session_state.file_hash = active_hash
    st.session_state.docs = docs
    st.session_state.pages = pages
    st.session_state.index = index
    st.session_state.doc_info = {
        "name": active_name,
        "chunks": len(docs),
        "has_pages": has_pages,
        "page_count": page_count,
    }

docs = st.session_state.docs
pages = st.session_state.pages
index = st.session_state.index
has_pages = st.session_state.doc_info["has_pages"]
model = load_model()

# Populate the sidebar's info placeholder now that the doc is ready.
with sidebar_info.container():
    info = st.session_state["doc_info"]
    st.caption(f"**{info['name']}**")
    st.caption(f"{info['chunks']} Q&A chunks indexed")
    if info["has_pages"]:
        n = info["page_count"]
        st.caption(f"_{n} pages detected_" if n != 1 else "_1 page detected_")
    if from_cache:
        st.caption("_Loaded from local cache_")

    # Let the user tune how many matches to show. Cap at MAX_TOP_K so the
    # results list stays scannable; also cap at chunk count (can't return
    # more matches than exist).
    _k_max = max(1, min(MAX_TOP_K, info["chunks"]))
    _k_default = min(DEFAULT_TOP_K, _k_max)
    # Clamp any stale session value left over from a larger previous doc,
    # otherwise Streamlit raises when the saved value exceeds max_value.
    if st.session_state.get("top_k", 0) > _k_max:
        st.session_state["top_k"] = _k_max
    if _k_max > 1:
        top_k = st.slider(
            "Matches to show",
            min_value=1,
            max_value=_k_max,
            value=_k_default,
            key="top_k",
        )
    else:
        top_k = 1

query = st.text_input(
    "Search",
    placeholder="Type keywords: method choice, limitations, future work...",
    label_visibility="collapsed",
)

def _page_for(idx: int) -> Optional[int]:
    return pages[idx] if has_pages else None


if query.strip():
    # Search mode — show best match prominently, others in expanders.
    result_ids = search(query, docs, index, model, k=top_k)
    if result_ids:
        st.markdown("##### Best match")
        _render_best_match(docs[result_ids[0]], _page_for(result_ids[0]))
        if len(result_ids) > 1:
            st.markdown("")
            st.markdown("##### Other matches")
            for rank, idx in enumerate(result_ids[1:], start=2):
                _render_secondary(docs[idx], rank=rank, page=_page_for(idx))
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
        _render_secondary(chunk, rank=i, page=_page_for(i - 1))
