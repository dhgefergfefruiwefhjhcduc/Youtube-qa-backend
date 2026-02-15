**YouTube QA Bot — Backend (RAG Pipeline)**

This repository contains the backend for a Retrieval-Augmented Generation (RAG) YouTube QA Bot. The pipeline indexes YouTube video transcripts into a FAISS vector store and performs retrieval + generation to answer user questions grounded in the video's transcript.

**Overview**
- **Purpose:** Provide concise, context-grounded answers about YouTube videos using retrieval + generative models.
- **Main script:** `rag_pipeline.py` — creates/loads per-video FAISS indexes and runs interactive QA.
- **Vector store location:** `vector_store/{video_id}` — each video's FAISS files are saved under `vector_store`.

**Why this architecture**
- **Explainable:** Retriever returns source passages so you can trace which transcript segments influenced an answer.
- **Grounded:** The LLM receives retrieved context to reduce hallucinations.
- **Fast:** FAISS provides efficient nearest-neighbour search over embeddings.

**Key features**
- Fetches video transcript via `youtube-transcript-api`.
- Splits transcripts into chunks and embeds them with `sentence-transformers`.
- Stores/reloads embeddings using FAISS (per-video directory under `vector_store`).
- Interactive CLI: the script prompts for a YouTube URL and then accepts natural-language questions.

**Placeholders**

- [Video demo](https://drive.google.com/file/d/1y_v0Kgezjxj8aiY9OfcpulGqN89GDDYl/view?usp=drive_link)

**Getting started**
Prerequisites:
- Python 3.8+
- `HUGGINGFACEHUB_API_TOKEN` — required if using Hugging Face endpoints/models.

Install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the pipeline (interactive):

```bash
python rag_pipeline.py
```

Behavior notes:
- The script prompts for a full YouTube URL. It extracts the `v=` parameter to build the `video_id`.
- Per-video embeddings are saved in `vector_store/{video_id}`; if that directory exists the pipeline will load it instead of creating a new index.

**How it works — high level**
1. Fetch transcript for the provided YouTube video using `youtube-transcript-api`.
2. Split the transcript into chunks (via `langchain_text_splitters`).
3. Convert chunks to embeddings (`sentence-transformers` / Hugging Face embeddings).
4. Index embeddings with FAISS and save to `vector_store/{video_id}`.
5. At query time, embed the question, retrieve relevant chunks, then pass context + question to the LLM for a grounded answer.

**Configuration & environment variables**
- `HUGGINGFACEHUB_API_TOKEN` — token for Hugging Face API endpoints (used by `HuggingFaceEndpoint`/embeddings).
- `OPENAI_API_KEY` — only required if you change the LLM provider to OpenAI.

**Project structure**
- `rag_pipeline.py` — main interactive pipeline.
- `requirements.txt` — dependency list.
- `vector_store/` — per-video FAISS data directories are stored here.

**Troubleshooting**
- If transcripts fail to fetch, verify the video allows transcripts and network access.
- If model calls fail, confirm `HUGGINGFACEHUB_API_TOKEN` is set and valid.
- If FAISS load/save fails, check file permissions for `vector_store/`.

**Streamlit web app**

A minimal Streamlit UI is provided in `app.py` to load a video and ask questions from the browser.

Run the app locally:

```bash
python -m venv .venv
.
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app lets you paste a YouTube URL or video id, initializes the pipeline (may fetch and index the transcript), and provides a question box to query the video.

**Contributing**
- Open an issue or PR with clear reproduction steps.
- Keep changes focused and document any configuration additions.




