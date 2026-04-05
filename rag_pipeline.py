import fitz  # PyMuPDF
import numpy as np
import base64
import io
import torch

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# =========================
# GLOBAL MODEL CACHE
# =========================

_clip_model = None
_clip_processor = None
_llm = None
_model = None
_vector_store = None
_image_store = None


def load_clip():
    global _clip_model, _clip_processor

    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()

    return _clip_model, _clip_processor


def load_llm():
    global _llm, _model

    if _model is None:
        _llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            task="text-generation"
        )
        _model = ChatHuggingFace(llm=_llm)

    return _model


# =========================
# EMBEDDINGS
# =========================

def embed_image(image):
    clip_model, clip_processor = load_clip()

    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()


def embed_text(text):
    clip_model, clip_processor = load_clip()

    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()


# =========================
# BUILD INDEX (MAIN FIX)
# =========================

def build_index(pdf_path: str):
    global _vector_store, _image_store

    doc = fitz.open(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    all_docs = []
    all_embeddings = []
    image_store = {}

    for i, page in enumerate(doc):

        # -------- TEXT --------
        text = page.get_text()
        if text.strip():
            temp_doc = Document(
                page_content=text,
                metadata={"page": i, "type": "text"}
            )

            chunks = splitter.split_documents([temp_doc])

            for chunk in chunks:
                emb = embed_text(chunk.page_content)
                all_docs.append(chunk)
                all_embeddings.append(emb)

        # -------- IMAGES --------
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base = doc.extract_image(xref)
                image_bytes = base["image"]

                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # store base64
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                image_id = f"page_{i}_img_{img_index}"
                image_store[image_id] = img_base64

                emb = embed_image(pil_img)

                all_docs.append(
                    Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={"page": i, "type": "image", "image_id": image_id}
                    )
                )

                all_embeddings.append(emb)

            except Exception as e:
                print("Image error:", e)

    doc.close()

    embeddings_array = np.array(all_embeddings)

    # FAISS store
    _vector_store = FAISS.from_embeddings(
        text_embeddings=[
            (doc.page_content, emb)
            for doc, emb in zip(all_docs, embeddings_array)
        ],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs]
    )

    _image_store = image_store

    return _vector_store, _image_store


# =========================
# RETRIEVAL
# =========================

def retrieve(query, k=5):
    query_emb = embed_text(query)

    results = _vector_store.similarity_search_by_vector(
        embedding=query_emb,
        k=k
    )

    return results


# =========================
# PROMPT
# =========================

def create_message(query, docs):
    context = ""

    for d in docs:
        if d.metadata["type"] == "text":
            context += f"[Page {d.metadata['page']}]\n{d.page_content}\n\n"
        else:
            context += f"[Image: {d.metadata.get('image_id')}]\n"

    prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{query}
"""

    return [prompt]


# =========================
# PIPELINE
# =========================

def multimodal_pdf_rag_pipeline(query):
    global _model

    model = load_llm()

    docs = retrieve(query)

    message = create_message(query, docs)

    response = model.invoke(message)

    return response.content