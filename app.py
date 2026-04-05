import streamlit as st
import tempfile

from rag_pipeline import (
    build_index,
    multimodal_pdf_rag_pipeline
)

st.set_page_config(page_title="Multimodal RAG", layout="wide")

st.title("📄🔍 Multimodal PDF RAG (CLIP + FAISS + LLM)")


# =========================
# SESSION STATE
# =========================

if "index_built" not in st.session_state:
    st.session_state.index_built = False

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None


# =========================
# UPLOAD PDF
# =========================

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.session_state.pdf_path = pdf_path

    if st.button("🚀 Build Index"):
        with st.spinner("Building embeddings (CLIP + FAISS)... this may take time"):
            build_index(pdf_path)
            st.session_state.index_built = True

        st.success("Index built successfully!")


# =========================
# QUERY SECTION
# =========================

if st.session_state.index_built:

    st.subheader("Ask Questions")

    query = st.text_input("Enter your question")

    if st.button("Get Answer"):

        if query.strip():

            with st.spinner("Thinking..."):
                answer = multimodal_pdf_rag_pipeline(query)

            st.markdown("### Answer")
            st.write(answer)

        else:
            st.warning("Enter a valid question")


# =========================
# SIDEBAR
# =========================

