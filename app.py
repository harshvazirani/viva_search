"""Notes Lookup — local semantic search over an uploaded Q&A .docx file.

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
from rank_bm25 import BM25Okapi
from rapidfuzz import process as rf_process
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from sentence_transformers import SentenceTransformer
from st_keyup import st_keyup


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
DEFAULT_TOP_K = 6
MAX_TOP_K = 20
MAX_WORDS_PER_CHUNK = 500


# ---------------------------------------------------------------------------
# .docx parsing
# ---------------------------------------------------------------------------

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Page tag prefixed onto each paragraph so page info survives normalization
# and Q/A splitting. Uncommon Unicode brackets make false positives in user
# content essentially impossible.
_PAGE_MARKER = re.compile(r"⟪PAGE=(\d+)⟫")

# Sentinel that flags a paragraph as a list/bullet item. Stripped before
# search; the answer renderer turns runs of these into a real <ul>.
_BULLET_PREFIX = "⟦•⟧ "


def _paragraph_page_breaks(element, use_rendered: bool) -> Tuple[int, int]:
    """Count page breaks in a paragraph, split by position relative to text.

    Word writes ``<w:lastRenderedPageBreak/>`` at the start of the first run
    of the first paragraph on each new page — for **both** automatic and
    manual page breaks. If we also counted the user's ``<w:br type="page"/>``,
    every manual break would be counted twice. So:

      * ``use_rendered=True`` (doc was opened in Word at least once) → trust
        only ``lastRenderedPageBreak`` markers; ignore ``<w:br>``.
      * ``use_rendered=False`` (never rendered) → count explicit ``<w:br>``.

    Splitting into ``leading`` (before any visible text in the paragraph)
    and ``trailing`` (after text) lets us advance the page counter at the
    right moment instead of mis-attributing the boundary paragraph.
    """
    target = (
        f"{_W_NS}lastRenderedPageBreak" if use_rendered else None
    )
    leading = 0
    trailing = 0
    seen_text = False
    for node in element.iter():
        tag = node.tag
        if tag == f"{_W_NS}t" and node.text:
            seen_text = True
        elif tag == f"{_W_NS}tab":
            seen_text = True
        elif use_rendered and tag == target:
            if seen_text:
                trailing += 1
            else:
                leading += 1
        elif (
            not use_rendered
            and tag == f"{_W_NS}br"
            and node.get(f"{_W_NS}type") == "page"
        ):
            if seen_text:
                trailing += 1
            else:
                leading += 1
    return leading, trailing


def _is_list_paragraph(para: Paragraph) -> bool:
    """True if the paragraph is a bulleted/numbered list item in Word."""
    if para._element.find(f".//{_W_NS}numPr") is not None:
        return True
    style = para.style.name if para.style is not None else ""
    return style == "List Paragraph"


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
    """Extract text, prefixing each paragraph with a ⟪PAGE=N⟫ tag and
    marking list items with a sentinel so the renderer can rebuild bullets.

    Page numbers come from ``w:lastRenderedPageBreak`` + explicit page breaks
    in document order. If the file has no break info (never rendered), every
    paragraph is tagged page 1 and the UI hides the page badge.
    """
    doc = Document(io.BytesIO(file_bytes))
    # Decide once per doc which break signal to trust. lastRenderedPageBreak
    # covers both auto and manual breaks — but only exists if the doc has
    # been opened in Word/LibreOffice at least once.
    body_xml = doc.element.body
    use_rendered = body_xml.find(f".//{_W_NS}lastRenderedPageBreak") is not None
    lines: List[str] = []
    current_page = 1
    for para in _iter_paragraphs_in_order(doc):
        leading, trailing = _paragraph_page_breaks(para._element, use_rendered)
        current_page += leading
        body = para.text
        if _is_list_paragraph(para) and body.strip():
            body = f"{_BULLET_PREFIX}{body.lstrip()}"
        lines.append(f"⟪PAGE={current_page}⟫{body}")
        current_page += trailing
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Q/A extraction — robust to real-world messy formatting
# ---------------------------------------------------------------------------

# Question boundary: line start with Q:, Q -, Question:, or "1. " numbering.
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
    """Split text into ``(Q: ...\\nA: ..., page_number)`` tuples per block.

    Page comes from the first ⟪PAGE=N⟫ marker inside the block (= the page
    the question line starts on). Markers are stripped from the stored
    chunk; bullet sentinels survive into rendering.
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

        # Page = where the answer body starts. Word writes
        # ``lastRenderedPageBreak`` at the first run that physically lands on
        # the new page, which is often the first answer bullet — even when
        # the Q line itself was visually pushed to that new page. Anchoring
        # on the answer body matches what the user sees in Word.
        a_anchor = re.search(
            r"(?m)^(?:⟪PAGE=\d+⟫)?[ \t]*A(?:ns(?:wer)?)?[ \t]*[:\-]",
            chunk,
        )
        search_from = a_anchor.end() if a_anchor else 0
        page_match = (
            _PAGE_MARKER.search(chunk, search_from)
            or _PAGE_MARKER.search(chunk)
        )
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
    """Return parallel lists: Q&A chunks and the page each one starts on."""
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


def build_q_index(docs: List[str], model: SentenceTransformer) -> faiss.Index:
    """A second FAISS index over just the question line of each chunk.

    Embedding the full chunk dilutes the question's signal across a long
    answer body, so a paraphrase query like "method choice" can lose to a
    chunk whose answer happens to mention method choices in passing. The
    Q line is short and intent-bearing — usually carrying the parenthetical
    keyword tags — so a separate index over it gives the semantic side a
    Q-prioritized ranking that mirrors the BM25 Q-line boost.
    """
    q_lines = [d.split("\n", 1)[0] for d in docs]
    embeddings = model.encode(
        q_lines,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# Tokenizer for BM25. Lowercases, splits on non-alphanumerics, strips
# leading zeros from pure-digit tokens (so "Case 03" matches "Case 4"),
# and folds simple English plurals so "limitations" ↔ "limitation",
# "cases" ↔ "case", "studies" ↔ "study". Real morphological stemming
# (Porter, Snowball) would handle more edge cases at the cost of an
# extra dependency; the rules below cover the high-frequency forms
# in this doc without misfiring on Latin/Greek -is/-us/-ss endings.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NO_STEM_ENDINGS = {"ss", "us", "is", "os", "as"}


def _stem(token: str) -> str:
    if len(token) <= 4 or token.isdigit():
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("s") and token[-2:] not in _NO_STEM_ENDINGS:
        return token[:-1]
    return token


def _tokenize(text: str) -> List[str]:
    out: List[str] = []
    for t in _TOKEN_RE.findall(text.lower()):
        if t.isdigit():
            out.append(t.lstrip("0") or "0")
        else:
            out.append(_stem(t))
    return out


# Doc-specific acronym ↔ phrase synonyms. Tokens here are already
# stemmed forms (e.g. "service" not "services"). Forward expansion
# always: a query containing the acronym also matches the phrase.
# Reverse expansion when the query contains every spelled-out word:
# searching "mental health" also matches chunks tagged "(MH findings)".
_TOKEN_SYNONYMS = {
    "mh": ["mental", "health"],
    "phc": ["primary", "health", "care"],
    "ems": ["emergency", "medical", "service"],
    "hrh": ["human", "resource", "health"],
    "chw": ["community", "health", "worker"],
    "tnc": ["trauma", "nurse", "coordinator"],
    "fgd": ["focus", "group", "discussion"],
    "po": ["participant", "observation"],
}


def _expand_synonyms(tokens: List[str]) -> List[str]:
    expanded = list(tokens)
    token_set = set(tokens)
    for acro, words in _TOKEN_SYNONYMS.items():
        if acro in token_set:
            expanded.extend(words)
        elif all(w in token_set for w in words):
            expanded.append(acro)
    return expanded


_Q_LINE_BOOST = 3  # extra copies of Q-line tokens added to the BM25 corpus


def build_bm25(docs: List[str]) -> BM25Okapi:
    """Index each chunk with its question line repeated, so Q-line matches
    dominate answer-body coincidences. The user's search keywords live in
    the Q line (often as parenthetical tags) — without this boost, a long
    answer that happens to contain a section header like "Chapter 5 —
    Primary Health Care" can outscore the actual "Can you summarise
    Chapter 5?" question whose Q line has only one mention.

    Also stashes per-doc Q-line token sets on the returned object for use
    by the identifier prefilter — the user's intent for a query like
    "Chapter 3" is to find chunks tagged with "Chapter 3" in the Q line,
    not chunks that happen to mention "3" somewhere in their answer body.
    """
    corpus: List[List[str]] = []
    q_line_tokens: List[set] = []
    chunk_token_lists: List[List[str]] = []
    q_line_token_lists: List[List[str]] = []
    for d in docs:
        full = _tokenize(d)
        q_line = d.split("\n", 1)[0]
        q_toks = _tokenize(q_line)
        corpus.append(full + q_toks * _Q_LINE_BOOST)
        q_line_tokens.append(set(q_toks))
        chunk_token_lists.append(full)
        q_line_token_lists.append(q_toks)
    bm25 = BM25Okapi(corpus)
    bm25.q_line_tokens = q_line_tokens  # type: ignore[attr-defined]
    bm25.chunk_tokens = chunk_token_lists  # type: ignore[attr-defined]
    bm25.q_line_token_lists = q_line_token_lists  # type: ignore[attr-defined]
    vocab: set[str] = set()
    for toks in chunk_token_lists:
        vocab.update(toks)
    bm25.vocab = vocab  # type: ignore[attr-defined]
    return bm25


def _fuzzy_correct(tokens: List[str], vocab: set[str]) -> List[str]:
    """Replace unknown tokens with the closest vocab token within edit-distance
    bounds. Short tokens get a tighter cutoff so "cat" doesn't snap to "rat".
    Known tokens, digits, and very short tokens pass through unchanged."""
    out: List[str] = []
    for t in tokens:
        if len(t) < 4 or t.isdigit() or t in vocab:
            out.append(t)
            continue
        # score_cutoff is rapidfuzz WRatio (0-100). 82 ≈ 1 typo on a 6-letter
        # word; 88 for shorter tokens to avoid wild snaps.
        cutoff = 88 if len(t) <= 5 else 82
        match = rf_process.extractOne(t, vocab, score_cutoff=cutoff)
        out.append(match[0] if match else t)
    return out


def _phrase_indices(token_seq: List[str], phrase: List[str]) -> bool:
    """True if phrase appears as a contiguous subsequence of token_seq."""
    if len(phrase) < 2 or len(token_seq) < len(phrase):
        return False
    m = len(phrase)
    for i in range(len(token_seq) - m + 1):
        if token_seq[i:i + m] == phrase:
            return True
    return False


# Reciprocal Rank Fusion constant. 60 is the value from the original Cormack
# et al. RRF paper and the de-facto default; small enough that top ranks
# dominate, large enough that a doc ranked highly by one retriever but
# missed by the other still surfaces.
_RRF_K = 60


def search(
    query: str,
    docs: List[str],
    index: faiss.Index,
    q_index: faiss.Index,
    bm25: BM25Okapi,
    model: SentenceTransformer,
    k: int = TOP_K,
) -> List[int]:
    """Hybrid search: fuse FAISS (semantic) and BM25 (lexical) with RRF.

    Pure semantic search misses short identifier queries like "Case 4" —
    embeddings dilute single-token signals across many "case"-mentioning
    chunks. BM25 nails those; FAISS handles paraphrases. RRF combines them
    without needing to calibrate score scales between the two retrievers.
    """
    n = len(docs)
    if n == 0:
        return []
    pool = min(n, max(k * 3, 20))

    # Identifier prefilter. Embeddings can't tell "Chapter 5" from "Chapter 10"
    # — MiniLM treats the digit as low-information, so the actual Chapter-5
    # Q&A can land at semantic rank ~100 while "Chapter 10" wins rank 1. RRF
    # then averages BM25's correct top hit back down. When the query carries
    # digit tokens (chapter/case numbers, percentages, years), restrict the
    # candidate set to chunks that contain every digit. BM25's doc_freqs is
    # already a per-doc token→count map, so this is a free O(n) lookup.
    q_tokens = _tokenize(query)
    # Typo tolerance: snap unknown tokens to the closest vocab match. Only
    # affects tokens that aren't already in the corpus, so exact matches
    # remain untouched and the user's literal spelling wins when it exists.
    q_tokens = _fuzzy_correct(q_tokens, bm25.vocab)  # type: ignore[attr-defined]
    q_digits = [t for t in q_tokens if t.isdigit()]
    allowed: Optional[set[int]] = None
    if q_digits:
        # Strict pass: require every digit in the chunk's Q line. The user's
        # tags ("Chapter 3 summary", "Case 4, maturity") sit there, so this
        # matches their mental model and excludes chunks that merely
        # reference the identifier in passing.
        q_line_strict = {
            i for i in range(n)
            if all(d in bm25.q_line_tokens[i] for d in q_digits)
        }
        if q_line_strict:
            allowed = q_line_strict
        else:
            # Loose fallback: digit just has to appear somewhere in the
            # chunk. Avoids returning nothing when the identifier isn't
            # tagged in any Q line (e.g. searching a year that only
            # surfaces in answer bodies).
            chunk_loose = {
                i for i in range(n)
                if all(d in bm25.doc_freqs[i] for d in q_digits)
            }
            if chunk_loose:
                allowed = chunk_loose

    # When digits act as the prefilter, drop them from the BM25 query so the
    # score reflects the conceptual term only ("chapter", "case"). Otherwise
    # short chunks heavy in digits ("3.5/5", "Case 05") win on length-
    # normalized term frequency over the chunk that's actually about
    # Chapter 5. If the query is digits only, keep them — there's nothing
    # else to score on.
    bm25_query = (
        [t for t in q_tokens if not t.isdigit()]
        if allowed is not None and any(not t.isdigit() for t in q_tokens)
        else q_tokens
    )
    # Acronym/phrase synonym expansion. "mental health" picks up chunks
    # tagged "(MH findings)" and vice versa; the lexical-presence anchor
    # downstream then keeps those chunks in the semantic ranking too.
    bm25_query = _expand_synonyms(bm25_query)
    bm25_scores = bm25.get_scores(bm25_query)

    # Phrase boost. "risk assessment" should outrank chunks where "risk"
    # and "assessment" appear in different sentences. Detect chunks where
    # the (stemmed) query tokens appear contiguously, with extra weight
    # for a Q-line phrase hit since tags are the user's primary search
    # surface. Only multi-token queries qualify; single tokens have
    # nothing to "phrase".
    phrase_q = [t for t in q_tokens if not t.isdigit()]
    phrase_chunk_hits: set = set()
    phrase_qline_hits: set = set()
    if len(phrase_q) >= 2:
        phrase_chunk_hits = {
            i for i in range(n)
            if _phrase_indices(bm25.chunk_tokens[i], phrase_q)
        }
        phrase_qline_hits = {
            i for i in range(n)
            if _phrase_indices(bm25.q_line_token_lists[i], phrase_q)
        }

    # Identifier queries: rank prefiltered chunks by BM25 alone (semantic is
    # net-noise once the user has typed an explicit identifier — see
    # comment on `allowed` above). The strict pass returns the right top
    # hit; we then pad with full hybrid results so the user still sees
    # related context as secondary matches. Without padding, "Chapter 4"
    # would return only the one chunk whose Q line tags Chapter 4.
    identifier_top: List[int] = []
    if allowed is not None:
        # Phrase hits in the Q line jump to the top of the identifier list
        # — e.g. "Case 13 power" should put the chunk whose Q line literally
        # contains "case 13" above other Case-13-tagged chunks.
        scored = [
            (
                int(i),
                bm25_scores[i] + (
                    2.0 if i in phrase_qline_hits
                    else 1.0 if i in phrase_chunk_hits
                    else 0.0
                ),
            )
            for i in allowed
            if bm25_scores[i] > 0
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        identifier_top = [i for i, _ in scored]
        if len(identifier_top) >= k:
            return identifier_top[:k]

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    # Larger semantic pool so the lexical-presence filter below has enough
    # candidates left when the matching chunks are BM25-rare.
    sem_pool = min(n, max(pool * 3, 60))
    _d1, full_ids = index.search(q_emb, sem_pool)
    full_semantic_ranking = [int(i) for i in full_ids[0] if i >= 0]
    _d2, q_ids = q_index.search(q_emb, sem_pool)
    q_semantic_ranking = [int(i) for i in q_ids[0] if i >= 0]

    # Anchor semantic rankings to lexical presence. If any chunk lexically
    # matches a query term, drop semantic candidates that don't — otherwise
    # a query like "reflexivity" picks a conceptually-adjacent "coenrolment"
    # chunk that doesn't even contain the word. When no chunk matches at
    # all, keep the full semantic ranking so true paraphrase queries
    # still work.
    if any(s > 0 for s in bm25_scores):
        full_semantic_ranking = [
            i for i in full_semantic_ranking if bm25_scores[i] > 0
        ][:pool]
        q_semantic_ranking = [
            i for i in q_semantic_ranking if bm25_scores[i] > 0
        ][:pool]
    else:
        full_semantic_ranking = full_semantic_ranking[:pool]
        q_semantic_ranking = q_semantic_ranking[:pool]
    # argsort descending; take only docs with positive score (a real lexical
    # hit). Zero-score docs would just be arbitrary tie-breaking noise.
    lexical_ranking = [
        int(i) for i in bm25_scores.argsort()[::-1]
        if bm25_scores[i] > 0
    ][:pool]

    # Three retrievers, fused with RRF. Q-line semantic and BM25 (Q-boosted)
    # both target the question's intent — when the user paraphrases a
    # tagged keyword, both reward the same chunk. Full-chunk semantic
    # backs them up for queries whose meaning lives in the answer body.
    fused: dict[int, float] = {}
    for rank, idx in enumerate(q_semantic_ranking):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (_RRF_K + rank)
    for rank, idx in enumerate(full_semantic_ranking):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (_RRF_K + rank)
    for rank, idx in enumerate(lexical_ranking):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (_RRF_K + rank)

    # Phrase bonus: scaled to be on the order of a top RRF rank slot
    # (1/_RRF_K ≈ 0.0167). A Q-line phrase match is worth a couple of
    # rank-1 slots, a body phrase match worth one — enough to lift the
    # phrase chunk past close competitors but not enough to override
    # multiple stronger lexical/semantic signals.
    for idx in phrase_qline_hits:
        fused[idx] = fused.get(idx, 0.0) + 0.05
    for idx in phrase_chunk_hits - phrase_qline_hits:
        fused[idx] = fused.get(idx, 0.0) + 0.02

    # Verbatim-substring bonus: if the user's literal query (case-insensitive,
    # whitespace-collapsed) appears in the chunk, give a strong push so exact
    # matches outrank stemmed/semantic near-misses. Q-line hits get the
    # heaviest boost since tags live there.
    raw_query = " ".join(query.lower().split())
    if len(raw_query) >= 3:
        for i in range(n):
            chunk = docs[i].lower()
            q_line = chunk.split("\n", 1)[0]
            if raw_query in q_line:
                fused[i] = fused.get(i, 0.0) + 0.12
            elif raw_query in chunk:
                fused[i] = fused.get(i, 0.0) + 0.06

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    hybrid_ranking = [idx for idx, _ in ordered]

    # Identifier-tagged chunks lead, then fill with hybrid results (deduped)
    # so the user gets related context after the strict matches. For
    # non-identifier queries, identifier_top is empty and this is just the
    # hybrid ranking.
    seen = set()
    result: List[int] = []
    for idx in identifier_top + hybrid_ranking:
        if idx in seen:
            continue
        seen.add(idx)
        result.append(idx)
        if len(result) >= k:
            break
    return result


# ---------------------------------------------------------------------------
# Result rendering helpers
# ---------------------------------------------------------------------------

def _split_qa(chunk: str) -> tuple[str, str]:
    """Split a stored chunk into (question, answer), with markers stripped."""
    lines = chunk.split("\n", 1)
    q = lines[0]
    a = lines[1] if len(lines) > 1 else ""
    q = re.sub(r"^\s*Q[:\-\s]*", "", q, flags=re.IGNORECASE)
    q = q.replace(_BULLET_PREFIX, "").strip()
    a = re.sub(r"^\s*A(?:ns(?:wer)?)?[:\-\s]*", "", a, flags=re.IGNORECASE).strip()
    return q, a


def _md_preserve_breaks(text: str) -> str:
    """Keep paragraph breaks (\\n\\n) and convert single \\n to markdown hard breaks."""
    paragraphs = text.split("\n\n")
    return "\n\n".join(p.replace("\n", "  \n") for p in paragraphs)


def _answer_html(a: str) -> str:
    """Convert a plain-text answer into HTML-safe paragraph + list markup.

    Lines tagged with the bullet sentinel become ``<li>`` items inside a
    single ``<ul>``; consecutive bullets stay grouped. Other lines render
    as ``<p>`` paragraphs as before.
    """
    out: List[str] = []
    bullet_buf: List[str] = []

    def flush_bullets() -> None:
        if not bullet_buf:
            return
        items = "".join(f"<li>{html.escape(b)}</li>" for b in bullet_buf)
        out.append(f"<ul class='answer-list'>{items}</ul>")
        bullet_buf.clear()

    for para in a.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for line in para.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(_BULLET_PREFIX):
                bullet_buf.append(stripped[len(_BULLET_PREFIX):].strip())
            else:
                flush_bullets()
                out.append(f"<p>{html.escape(stripped)}</p>")
        flush_bullets()
    return "".join(out)


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
    """Collapsed expander — matches best-match body size when opened."""
    q, a = _split_qa(chunk)
    suffix = f"  ·  Page {page}" if page is not None else ""
    with st.expander(f"**#{rank}** — {q}{suffix}", expanded=False):
        st.markdown(
            f'<div class="best-answer">{_answer_html(a)}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Notes Lookup", layout="centered")

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

    /* Bulleted answer lists — match best-answer typography */
    .best-answer ul.answer-list {
        margin: 0 0 0.9em 0;
        padding-left: 1.4rem;
    }
    .best-answer ul.answer-list li {
        margin-bottom: 0.4em;
    }
    .best-answer ul.answer-list li:last-child {
        margin-bottom: 0;
    }

    /* Best-match question heading */
    .best-question {
        font-size: 1.35rem;
        font-weight: 600;
        line-height: 1.35;
        letter-spacing: -0.015em;
        margin-bottom: 0.85rem;
    }

    /* Best-match answer body — this is the main size bump.
       Selectors are scoped so they also win inside expanders, where
       Streamlit otherwise drops the font size. */
    .best-answer,
    [data-testid="stExpander"] .best-answer,
    [data-testid="stExpander"] .best-answer p,
    [data-testid="stExpander"] .best-answer li {
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
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] details > summary,
    [data-testid="stExpander"] details > summary * {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        line-height: 1.35 !important;
        letter-spacing: -0.015em !important;
    }
    [data-testid="stExpander"] summary {
        padding-top: 0.7rem;
        padding-bottom: 0.7rem;
    }
    [data-testid="stExpander"] summary p {
        margin: 0;
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

st.title("Notes Lookup")

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
        q_index = build_q_index(docs, model)
        bm25 = build_bm25(docs)

    # Only surface pages if the document actually has break info. A doc that
    # was never rendered tags every paragraph page 1 — showing "Page 1"
    # everywhere would be noise, not signal.
    page_count = max(pages) if pages else 1
    has_pages = page_count > 1

    st.session_state.file_hash = active_hash
    st.session_state.docs = docs
    st.session_state.pages = pages
    st.session_state.index = index
    st.session_state.q_index = q_index
    st.session_state.bm25 = bm25
    st.session_state.doc_info = {
        "name": active_name,
        "chunks": len(docs),
        "has_pages": has_pages,
        "page_count": page_count,
    }

docs = st.session_state.docs
pages = st.session_state.pages
index = st.session_state.index
q_index = st.session_state.q_index
bm25 = st.session_state.bm25
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
    # results list stays scannable; also cap at chunk count.
    _k_max = max(1, min(MAX_TOP_K, info["chunks"]))
    _k_default = min(DEFAULT_TOP_K, _k_max)
    # Clamp any stale session value left over from a previous larger doc,
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

_MIN_LIVE_CHARS = 3
query = st_keyup(
    "Search",
    placeholder="Type keywords: method choice, limitations, future work...",
    debounce=180,
    key="search_box",
    label_visibility="collapsed",
) or ""

def _page_for(idx: int) -> Optional[int]:
    return pages[idx] if has_pages else None


_q = query.strip()
if _q and len(_q) >= _MIN_LIVE_CHARS:
    # Search mode — show best match prominently, others in expanders.
    result_ids = search(query, docs, index, q_index, bm25, model, k=top_k)
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
elif _q:
    st.caption(f"Keep typing — searching after {_MIN_LIVE_CHARS} characters.")
else:
    # Browse mode — show all Q&A pairs as collapsed expanders.
    st.caption(
        f"{len(docs)} Q&A pairs indexed. Type keywords above to search, "
        "or browse all pairs below."
    )
    st.markdown("##### Browse all")
    for i, chunk in enumerate(docs, start=1):
        _render_secondary(chunk, rank=i, page=_page_for(i - 1))
