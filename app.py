import streamlit as st
from PIL import Image
import tempfile
import time

# Backend
from rag_pipeline import process_pdf, multimodal_pipeline

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Multimodal RAG Chat", layout="wide")

st.title("🧠 Multimodal RAG Chat (CLIP + BLIP + LLaMA)")
st.caption("PDF + Image AI Chat System 🚀 (No LLaVA, fully stable)")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

mode = st.sidebar.radio(
    "Select Mode",
    ["📄 PDF Chat", "🖼️ Image Search", "🔀 Multimodal"]
)

uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
uploaded_image = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# Clear chat
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------- PROCESS PDF ----------------
if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    st.sidebar.success("Processing PDF... ⏳")

    vector_store, image_data_store = process_pdf(pdf_path)

    st.session_state.vector_store = vector_store
    st.sidebar.success("✅ PDF Ready!")

# ---------------- IMAGE HANDLING ----------------
image_bytes = None
image = None

if uploaded_image:
    image_bytes = uploaded_image.getvalue()
    image = Image.open(uploaded_image)

    st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask something...")

if user_input:

    # ---------------- VALIDATION ----------------
    if mode == "📄 PDF Chat" and st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a PDF first")
        st.stop()

    if mode == "🖼️ Image Search" and image_bytes is None:
        st.warning("⚠️ Please upload an image first")
        st.stop()

    if mode == "🔀 Multimodal":
        if st.session_state.vector_store is None or image_bytes is None:
            st.warning("⚠️ Upload BOTH PDF and Image")
            st.stop()

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------------- ASSISTANT ----------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            try:
                vector_store = st.session_state.vector_store

                # -------- PDF MODE --------
                if mode == "📄 PDF Chat":
                    response, _ = multimodal_pipeline(
                        query=user_input,
                        vector_store=vector_store,
                        image_bytes=None
                    )

                # -------- IMAGE MODE (BLIP NOW) --------
                elif mode == "🖼️ Image Search":
                    response, _ = multimodal_pipeline(
                        query=user_input,
                        vector_store=None,
                        image_bytes=image_bytes
                    )

                # -------- MULTIMODAL MODE --------
                else:
                    response, _ = multimodal_pipeline(
                        query=user_input,
                        vector_store=vector_store,
                        image_bytes=image_bytes
                    )

                # ---------------- STREAMING ----------------
                placeholder = st.empty()
                streamed_text = ""

                for char in response:
                    streamed_text += char
                    placeholder.markdown(streamed_text)
                    time.sleep(0.01)

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                st.error(f"Error: {e}")