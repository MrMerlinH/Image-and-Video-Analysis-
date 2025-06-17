import streamlit as st
import sqlite3
import numpy as np
import torch
import clip
from PIL import Image
import io
import os

DB_PATH = "DB/cbvr.db"
KEYFRAME_FOLDER = "PythonProject/"  # folder where keyframe images are stored

# Load CLIP model once
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def fetch_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, keyframe_path, clip_embedding FROM shots WHERE clip_embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    ids, filenames, embeddings = [], [], []
    for row in rows:
        ids.append(row[0])
        filenames.append(row[1])
        emb = np.frombuffer(row[2], dtype=np.float32)
        embeddings.append(emb)
    return ids, filenames, np.array(embeddings)

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def search(query, model, device, ids, filenames, embeddings, top_k=5):
    with torch.no_grad():
        text_tokens = clip.tokenize([query]).to(device)
        text_embedding = model.encode_text(text_tokens).cpu().numpy()[0]

    sims = [cosine_similarity(text_embedding, emb) for emb in embeddings]
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = [(ids[i], filenames[i], sims[i]) for i in top_indices]
    return results

def load_image(path):
    return Image.open(path)

# Streamlit UI
st.title("Video Shot Search with CLIP")

model, preprocess, device = load_model()

query = st.text_input("Enter your search query", "")

if query:
    ids, filenames, embeddings = fetch_embeddings()

    if len(ids) == 0:
        st.warning("No embeddings found in the database.")
    else:
        results = search(query, model, device, ids, filenames, embeddings)

        st.write(f"Top {len(results)} results:")

        for idx, filename, score in results:
            img_path = os.path.join(filename)
            if os.path.exists(img_path):
                image = load_image(img_path)
                st.image(image, width=300)
            else:
                st.write(f"Image not found: {filename}")

            st.write(f"Shot ID: {idx}, Similarity: {score:.4f}")
            st.markdown("---")
