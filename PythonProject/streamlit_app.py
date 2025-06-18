import streamlit as st
import sqlite3
import numpy as np
import torch
import clip
from PIL import Image
import io
import os
import requests

st.set_page_config(layout="wide")

logged_in = False
run_id = "c0c2f23f-dafe-4828-8f39-63f270688f6f"
DRES_URL = f"https://vbs.videobrowsing.org"
USERNAME = "TECHtalent09"
PASSWORD = "xhJ3T4Ct"

session = requests.Session()
def login_to_dres():
    session = requests.Session()
    response = session.post(f"{DRES_URL}/api/v2/login", json={
        "username": USERNAME,
        "password": PASSWORD
    })
    if response.status_code == 200:
        st.session_state["dres_session"] = session
        st.session_state["logged_in"] = True
        st.success("✅ Logged in to DRES")
    else:
        st.error("❌ Login failed")

def get_dres_session():
    if "dres_session" in st.session_state and st.session_state.get("logged_in", False):
        return st.session_state["dres_session"]
    else:
        login_to_dres()
        return st.session_state.get("dres_session", None)

# Automatically login if not done yet
session = get_dres_session()

if session:
    try:
        task_info = session.get(f"{DRES_URL}/api/v2/client/evaluation/list")
        if task_info.status_code == 200:
            tasks = task_info.json()
        else:
            st.warning("Could not fetch tasks.")
    except Exception as e:
        st.error(f"Error during DRES request: {e}")

task_info = session.get(f"{DRES_URL}/api/v2/client/evaluation/list").json()

evaluationId = task_info[0]["id"]


test = session.get(f"{DRES_URL}/api/v2/client/evaluation/currentTask/{evaluationId}").json()
st.write("Current Task ID: " + test["name"])




DB_PATH = "DB/cbvr.db"
KEYFRAME_FOLDER = "PythonProject/"  # folder where keyframe images are stored

@st.dialog("Selected Shot", width="large")
def play(video_path, start_time_sec):
    st.video(f"PythonProject/{video_path}", start_time=start_time_sec, autoplay=True, muted=True)



@st.dialog("Submitted Shot", width="large")
def submit(video_path, start_time_ms):
    vid = os.path.basename(video_path)
    vid = vid.split(".")[0]
    query = st.text_input("Optional Descriptor")
    submission_payload = {
        "answerSets": [
            {
                "answers": [
                    {
                        "text": query,
                        "mediaItemName": vid,
                        "mediaItemCollectionName": "IVADL",
                        "start": start_time_ms,
                        "end": start_time_ms
                    }
                ]
            }
        ]
    }
    if st.button("Submit"):
        resp = session.post(f"{DRES_URL}/api/v2/submit/{evaluationId}", json=submission_payload)
        st.write(resp.content)

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


ids, filenames, embeddings = fetch_embeddings()
if not query:
    conn = sqlite3.connect(DB_PATH)
    cols = st.columns(5)
    for i, (keyframeID, filename) in enumerate(zip(ids, filenames)):
        img_path = os.path.join(filename)
        if os.path.exists(img_path):
            with cols[i % 5]:
                st.image(Image.open(img_path), width=300, caption=f"Shot ID: {keyframeID}, Video Source: {1}")
                if st.button("Play", key=f"imgbtn_{keyframeID}", use_container_width=True):

                    cur = conn.cursor()
                    cur.execute("""
                                SELECT videos.file_path, shots.start_frame, videos.fps
                                FROM shots
                                         JOIN videos ON shots.video_id = videos.id
                                WHERE shots.id = ?
                                """, (keyframeID,))
                    result = cur.fetchone()
                    if result:
                        video_path, start_frame, fps = result
                        start_time_sec = start_frame / fps
                        play(video_path, start_time_sec)
                if st.button("Submit", key=f"imgsbmtbtn_{keyframeID}", use_container_width=True):
                    cur = conn.cursor()
                    cur.execute("""
                                SELECT videos.file_path, shots.start_frame, videos.fps
                                FROM shots
                                         JOIN videos ON shots.video_id = videos.id
                                WHERE shots.id = ?
                                """, (keyframeID,))
                    result = cur.fetchone()
                    if result:
                        video_path, start_frame, fps = result
                        start_time_ms = start_frame / fps * 1000
                        submit(video_path, start_time_ms)
    conn.close()
if query:
    conn = sqlite3.connect(DB_PATH)
    if len(ids) == 0:
        st.warning("No embeddings found in the database.")
    else:
        results = search(query, model, device, ids, filenames, embeddings, 20)

        st.write(f"Top {len(results)} results:")

        cols = st.columns(5)  # 3-column grid

        for i, (keyframeID, filename, score) in enumerate(results):
            img_path = os.path.join(filename)
            if os.path.exists(img_path):
                image = load_image(img_path)
                with cols[i % 5]:
                    st.image(image, width=300, caption=f"Shot ID: {keyframeID}, Similarity: {score}")
                    if st.button("Play", key=f"imgbtn_{keyframeID}", use_container_width=True):

                        cur = conn.cursor()
                        cur.execute("""
                                    SELECT videos.file_path, shots.start_frame, videos.fps
                                    FROM shots
                                    JOIN videos ON shots.video_id = videos.id
                                    WHERE shots.id = ?
                                    """, (keyframeID,))
                        result = cur.fetchone()
                        if result:
                            video_path, start_frame, fps = result
                            start_time_sec = start_frame / fps
                            play(video_path, start_time_sec)
                    if st.button("Submit", key=f"imgsbmtbtn_{keyframeID}", use_container_width=True):
                        cur = conn.cursor()
                        cur.execute("""
                                    SELECT videos.file_path, shots.start_frame, videos.fps
                                    FROM shots
                                             JOIN videos ON shots.video_id = videos.id
                                    WHERE shots.id = ?
                                    """, (keyframeID,))
                        result = cur.fetchone()
                        if result:
                            video_path, start_frame, fps = result
                            start_time_ms = start_frame / fps * 1000
                            submit(video_path, start_time_ms)
            else:
                with cols[i % 5]:
                    st.warning(f"Image not found: {filename}")
    conn.close()