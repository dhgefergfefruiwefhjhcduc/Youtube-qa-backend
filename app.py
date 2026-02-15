import re
import streamlit as st

st.set_page_config(page_title="YouTube QA Bot (RAG)", layout="centered")

st.title("YouTube QA Bot — RAG Pipeline")

from rag_pipeline import RAGPipeline


def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    if not url_or_id:
        return ""
    # If full youtu.be short url
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", url_or_id)
    if m:
        return m.group(1)
    # v= parameter
    m = re.search(r"v=([A-Za-z0-9_-]{6,})", url_or_id)
    if m:
        return m.group(1)
    # If user pasted a raw id
    m = re.match(r"^[A-Za-z0-9_-]{6,}$", url_or_id)
    if m:
        return url_or_id
    return ""


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.video_id = None


with st.form("load_video_form"):
    video_input = st.text_input("YouTube URL or video ID", value="")
    load_btn = st.form_submit_button("Load video")

if load_btn:
    video_id = extract_video_id(video_input)
    if not video_id:
        st.error("Could not extract a video id. Paste a full YouTube URL or a video id.")
    else:
        st.session_state.video_id = video_id
        with st.spinner("Initializing pipeline (may fetch transcript and create embeddings)..."):
            try:
                st.session_state.pipeline = RAGPipeline(video_id)
                st.success(f"Loaded pipeline for video: {video_id}")
            except Exception as e:
                st.session_state.pipeline = None
                st.error(f"Failed to initialize pipeline: {e}")


if st.session_state.pipeline is not None:
    st.markdown("---")
    st.subheader("Ask a question about the video")
    question = st.text_input("Your question", key="question_input")
    if st.button("Ask",help="Ask a question about the video. The answer will be based solely on the video transcript, and will include timestamps if relevant."):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running retrieval and generation..."):
                try:
                    answer = st.session_state.pipeline.run(question)
                    st.markdown("**Rendered answer:**")
                    st.success(str(answer))
                except Exception as e:
                    st.error(f"Error running pipeline: {e}")

    st.markdown("---")
   
