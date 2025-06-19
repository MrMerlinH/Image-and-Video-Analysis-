import os
import sqlite3
import torch
import clip
from PIL import Image
import numpy as np

DB_PATH = "DB/cbvr.db"
KEYFRAME_FOLDER = "testFrames"  # path to your keyframes
VIDEO_ID = 201  # adjust depending on your DB structure

def serialize_embedding(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()

def update_embedding_in_db(filename: str, embedding: np.ndarray, conn):
    print(filename)
    blob = serialize_embedding(embedding)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE shots
        SET clip_embedding = ?
        WHERE keyframe_path = ?
    """, (blob, f"testFrames/{filename}"))


def extract_and_store_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    image_files = sorted([
        f for f in os.listdir(KEYFRAME_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for img_file in image_files:
        image_path = os.path.join(KEYFRAME_FOLDER, img_file)
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy()[0]

            update_embedding_in_db(img_file, embedding, conn)
            # print(f"✅ Updated embedding for: {img_file}")

        except Exception as e:
            print(f"❌ Failed to process {img_file}: {e}")

    conn.commit()
    conn.close()
    print("\n📦 All embeddings updated in database.")

if __name__ == "__main__":
    extract_and_store_embeddings()
