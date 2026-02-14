**YouTube QA Bot — Backend (RAG Pipeline)**

This repository contains the backend components for a Retrieval-Augmented Generation (RAG) based YouTube QA Bot. It provides a simple pipeline to index transcript embeddings into a FAISS vector store and run a retrieval + generation pipeline to answer user questions grounded in YouTube video content.

**Overview**
- **Purpose:**: Provide accurate, context-grounded answers about YouTube videos using vector retrieval and a generative model.
- **Core script:**: `rag_pipeline.py` — orchestrates embedding, indexing, and query-time retrieval + generation.
- **Vector store:**: `vectorstore/index.faiss` — FAISS index storing vector embeddings for video transcripts.

**Why this architecture?**
- **Explainable**: Retrieval steps make it easy to trace which parts of the video content influenced an answer.
- **Accurate**: The generative model is fed retrieved passages (context) to reduce hallucination.
- **Efficient**: FAISS provides fast similarity search for large transcript collections.

**Features**
- Build or load a FAISS vector index of video transcript embeddings.
- Query the index to retrieve top-k passages relevant to a question.
- Use a language model to generate a final answer grounded on retrieved context.

**Placeholders**

- [Video demo](https://drive.google.com/file/d/1y_v0Kgezjxj8aiY9OfcpulGqN89GDDYl/view?usp=sharing)

**Getting Started**
Prerequisites:
- Python 3.8+ installed
- A language model API key if required (e.g., OpenAI) — set as environment variable `OPENAI_API_KEY` or configure your provider in the code.

Install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the pipeline (basic usage):

```bash
python rag_pipeline.py
```

Notes:
- If a `vectorstore/index.faiss` file exists, the pipeline may load it instead of rebuilding the index.
- To rebuild the index from transcripts, modify the pipeline input section in `rag_pipeline.py` to point at your transcript source and run the script.

**How it works — high level**
1. Extract or load transcript text for each video.
2. Convert text passages into vector embeddings using an embedding model.
3. Index embeddings in FAISS (`vectorstore/index.faiss`).
4. At query time, embed the user question and retrieve nearest passages from FAISS.
5. Provide retrieved passages as context to a language model to produce a grounded answer.

This approach separates retrieval (factual grounding) from generation (answer composition) to improve reliability and traceability.

**Configuration**
- Inspect `requirements.txt` for Python package versions. Update or pin versions if needed.
- Common environment variables (example):

```
OPENAI_API_KEY=<your_api_key>
```

**Project Structure**
- `rag_pipeline.py` — main pipeline runner for indexing and querying.
- `requirements.txt` — Python dependencies.
- `vectorstore/` — holds FAISS index files; `index.faiss` is the default index file.

**Contributing**
- Add reproducible steps or tests when submitting changes.
- Keep changes small and focused; include a short description of intent in PRs.


