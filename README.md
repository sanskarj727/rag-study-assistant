# RAG study assistant

A retrieval-augmented generation (RAG) assistant that answers questions using
your own documents (lecture notes, textbook PDFs, etc.) instead of relying
purely on an LLM's memorised training data.

Built step by step as a learning project. Each step is a real, working
module - nothing here is a stub or placeholder.

## Progress

- [x] **Step 1 - Ingestion**: load `.txt`/`.pdf` files and chunk them into
      overlapping pieces (`src/ingest.py`)
- [ ] **Step 2 - Embeddings**: turn chunks into vectors, understand
      cosine similarity from first principles
- [ ] **Step 3 - Vector store**: persist and search embeddings (ChromaDB)
- [ ] **Step 4 - Retrieval + generation**: combine retrieval with an LLM
      call to actually answer questions
- [ ] **Step 5 - Interface**: a simple Streamlit chat UI
- [ ] **Step 6 - Deploy**: put it online with a live demo link

## Setup

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Try step 1 yourself

```bash
python3 src/ingest.py
```

This loads `data/sample.txt`, splits it into overlapping 60-word chunks, and
prints each one so you can see exactly where the boundaries land and how the
overlap works.

Try swapping in one of your own PDFs: drop it in `data/`, then in a Python
shell or a new script:

```python
from src.ingest import load_document, chunk_text

text = load_document("data/your_notes.pdf")
chunks = chunk_text(text, chunk_size=300, overlap=50)
print(len(chunks), "chunks created")
```
