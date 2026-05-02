# Notes Lookup

Minimal local semantic search over a personal Q&A notes file. Upload a `.docx` of Q&A-formatted notes, type 2–5 keywords, and get the best-matching block back instantly. No chat, no LLM generation, no persistence — pure retrieval over a FAISS index.

## Project layout

```
.
├── app.py            # Streamlit app (upload + parse + index + search)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container image definition
└── README.md         # This file
```

## Input document format

The uploaded `.docx` should contain Q&A pairs like this:

```
Q: Why did you choose method X? (method choice, approach selection)
A: Because it balances interpretability and accuracy...

Q: What are the limitations? (drawbacks, weaknesses)
A: The main limitations are...
```

The parser is tolerant: it accepts `Q:`, `Q -`, `Question:`, or numbered (`1. `) question starts, and `A:`, `A -`, `Answer:`, or `Ans:` answer markers. Extra whitespace, multi-paragraph answers, and Unicode noise are handled. Very long Q&A blocks (>500 words) are split into sub-chunks with the question repeated as context.

Tip: include synonyms in brackets inside the question. They ride along into the embedding and broaden keyword recall.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. The first launch downloads the `all-MiniLM-L6-v2` model (~90 MB). Upload a `.docx` file via the uploader; the search box appears once indexing finishes. Re-uploading the same file is a no-op (indexing is keyed on the file hash).

## Run with Docker

```bash
docker build -t notes-lookup .
docker run --rm -p 8501:8501 notes-lookup
```

Open `http://localhost:8501` and upload your `.docx` file.

## Example queries

Type short keyword phrases, not full sentences:

- `method choice`
- `limitations drawbacks`
- `future work extensions`
- `evaluation metrics`
- `main contribution`

Each query returns the best match plus two backup matches.

## How it works

1. You upload a `.docx` file via the Streamlit uploader.
2. `python-docx` extracts the full text (paragraphs and table cells), preserving line breaks.
3. A robust parser splits the text into Q&A blocks at question boundaries, cleans prefixes, and drops half-pairs.
4. Each Q&A becomes one chunk (very long ones are split into sentence-grouped sub-chunks, with the question repeated).
5. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and added to a FAISS `IndexFlatL2`.
6. Each query is embedded and the top 3 nearest neighbours are returned — one "Best match" and two "Other matches".

The model loads once per process (`@st.cache_resource`) and the index is cached per-file in `st.session_state`, so re-queries stay sub-200ms.

## What this app is not

- No LLM generation, no chat turns, no history.
- No external APIs.
- No persistence — the index lives in memory for the current session.
