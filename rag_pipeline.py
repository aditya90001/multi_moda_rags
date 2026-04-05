import fitz
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import numpy as np
import base64
import io

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ---------------- LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# ---------------- CLIP ----------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ---------------- BLIP (IMAGE UNDERSTANDING) ----------------
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model.eval()

# ---------------- EMBEDDING ----------------
def embed_text(text):
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


def embed_image(image):
    image = image.convert("RGB")

    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()

# ---------------- BLIP IMAGE CAPTION ----------------
def ask_blip(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = blip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = blip_model.generate(**inputs)

    caption = blip_processor.decode(output[0], skip_special_tokens=True)

    return caption

# ---------------- PDF PROCESS ----------------
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    all_docs = []
    all_embeddings = []
    image_data_store = {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for i, page in enumerate(doc):
        text = page.get_text()

        # -------- TEXT --------
        if text.strip():
            temp_doc = Document(
                page_content=text,
                metadata={"page": i, "type": "text"}
            )

            chunks = splitter.split_documents([temp_doc])

            for chunk in chunks:
                emb = embed_text(chunk.page_content)
                all_embeddings.append(emb)
                all_docs.append(chunk)

        # -------- IMAGES --------
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                image_id = f"page_{i}_img_{img_index}"

                # store image for UI
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                image_data_store[image_id] = base64.b64encode(
                    buffered.getvalue()
                ).decode()

                # CLIP embedding
                emb = embed_image(pil_image)
                all_embeddings.append(emb)

                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={
                        "page": i,
                        "type": "image",
                        "image_id": image_id
                    }
                )

                all_docs.append(image_doc)

            except:
                continue

    doc.close()

    embeddings_array = np.array(all_embeddings)

    vector_store = FAISS.from_embeddings(
        text_embeddings=[
            (doc.page_content, emb)
            for doc, emb in zip(all_docs, embeddings_array)
        ],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs]
    )

    return vector_store, image_data_store

# ---------------- RETRIEVE ----------------
def retrieve_multimodal(query, vector_store, k=5):
    query_emb = embed_text(query)

    return vector_store.similarity_search_by_vector(
        embedding=query_emb,
        k=k
    )

# ---------------- PROMPT ----------------
def create_text_message(query, context_docs):
    context = ""

    for doc in context_docs:
        if doc.metadata["type"] == "text":
            context += doc.page_content + "\n"
        else:
            context += f"[Image: {doc.metadata['image_id']}]\n"

    prompt = f"""
Answer the question based on the context:

{context}

Question: {query}
"""

    return [HumanMessage(content=prompt)]

# ---------------- MAIN PIPELINE ----------------
def multimodal_pipeline(query, vector_store=None, image_bytes=None):

    context_docs = []
    pdf_answer = ""

    # -------- PDF PART --------
    if vector_store is not None:
        context_docs = retrieve_multimodal(query, vector_store, k=5)
        message = create_text_message(query, context_docs)
        response = model.invoke(message)
        pdf_answer = response.content

    # -------- IMAGE PART (BLIP) --------
    image_answer = ""
    if image_bytes is not None:
        caption = ask_blip(image_bytes)
        image_answer = f"🖼️ Image Description: {caption}"

    # -------- FINAL RESPONSE --------
    if pdf_answer and image_answer:
        final_answer = f"""
📄 PDF Understanding:
{pdf_answer}

🖼️ Image Understanding:
{image_answer}
"""

    elif pdf_answer:
        final_answer = pdf_answer

    elif image_answer:
        final_answer = image_answer

    else:
        final_answer = "No input provided."

    return final_answer, context_docs