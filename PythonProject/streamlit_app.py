import streamlit as st
import sqlite3
import numpy as np
import torch
import clip
from PIL import Image
import io
import os
import requests
from dotenv import load_dotenv

st.set_page_config(layout="wide")
load_dotenv()

logged_in = False
run_id = "e98c03c2-2042-441d-b3d1-ced103a91d21"
DRES_URL = f"https://vbs.videobrowsing.org"
USERNAME = os.getenv("DRES_USERNAME")
PASSWORD = os.getenv("DRES_PASSWORD")
print(USERNAME)
print(PASSWORD)
DB_PATH = "DB/cbvr.db"
KEYFRAME_FOLDER = "PythonProject/"

def get_video_name_to_id_map():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_path FROM videos")
    rows = cursor.fetchall()
    conn.close()
    return {os.path.basename(path).split(".")[0].lower(): vid for vid, path in rows}

def get_keyframes_for_video(video_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT shots.id, shots.keyframe_path
        FROM shots
        WHERE shots.video_id = ?
        ORDER BY shots.start_frame
    """, (video_id,))
    results = cursor.fetchall()
    conn.close()
    return results  # List of (shot_id, keyframe_path)


video_map = get_video_name_to_id_map()
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

ids = [item["id"] for item in task_info]
names = [item["name"] for item in task_info]

task = st.radio(
    "Select task",
    ids,
    captions=names,

)
evaluationId = task


activeTest = session.get(f"{DRES_URL}/api/v2/client/evaluation/currentTask/{evaluationId}").json()
st.write("Current Task ID: " + activeTest["name"])



@st.dialog("Selected Shot", width="large")
def play(video_path, start_time_sec):
    st.video(f"PythonProject/{video_path}", start_time=start_time_sec, autoplay=True, muted=True)



@st.dialog("Submitted Shot", width="large")
def submit(video_path, start_time_ms, end_time_ms, fps):
    vid = os.path.basename(video_path)
    vid = vid.split(".")[0]
    textInput = st.text_input("Text Input")

    st.write(f"Video ID: {vid}")
    st.write(f"Start Time: {start_time_ms:9.4f}ms | Start Frame: {(start_time_ms/1000)*fps}")
    st.write(f"End Time: {end_time_ms:9.4f}ms | End Frame: {(end_time_ms/1000)*fps}")
    col1, col2 = st.columns(2)
    with col1:
        startInput = st.text_input("Start Time")
    with col2:
        endInput = st.text_input("End Time")

    submission_payload = {
        "answerSets": [
            {
                "answers": [
                    {
                        "text": textInput,
                        "mediaItemName": vid,
                        "mediaItemCollectionName": "IVADL",
                        "start": startInput,
                        "end": endInput
                    }
                ]
            }
        ]
    }

    st.write(f"Submission Payload: {submission_payload}")
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
    cursor.execute("SELECT id, keyframe_path, clip_embedding, video_id FROM shots WHERE clip_embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    ids, filenames, embeddings , videoIDs = [], [], [], []
    for row in rows:
        ids.append(row[0])
        filenames.append(row[1])
        emb = np.frombuffer(row[2], dtype=np.float32)
        videoIDs.append(row[3])
        embeddings.append(emb)
    return ids, filenames, np.array(embeddings) , videoIDs

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def search(query, model, device, ids, filenames, embeddings, videoIDs,top_k=5 ):
    with torch.no_grad():
        text_tokens = clip.tokenize([query]).to(device)
        text_embedding = model.encode_text(text_tokens).cpu().numpy()[0]

    sims = [cosine_similarity(text_embedding, emb) for emb in embeddings]
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = [(ids[i], filenames[i], sims[i], videoIDs[i]) for i in top_indices]
    return results

def load_image(path):
    return Image.open(path)

model, preprocess, device = load_model()

query = st.text_input("Enter your search query or video name", "").strip()

ids, filenames, embeddings , videoIDs = fetch_embeddings()
if not query:
    conn = sqlite3.connect(DB_PATH)
    cols = st.columns(5)
    for i, (keyframeID, filename) in enumerate(zip(ids, filenames)):
        if i >= 2000:
            break
        img_path = os.path.join(filename)
        if os.path.exists(img_path):
            with cols[i % 5]:
                st.image(Image.open(img_path), width=300, caption=f"Shot ID: {keyframeID}, Video Source: {os.path.basename(str(filename)).split(".")[0]}")
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
                                SELECT videos.file_path, shots.start_frame, shots.end_frame, videos.fps
                                FROM shots
                                         JOIN videos ON shots.video_id = videos.id
                                WHERE shots.id = ?
                                """, (keyframeID,))
                    result = cur.fetchone()
                    if result:
                        video_path, start_frame, end_frame, fps = result
                        start_time_ms = (start_frame / fps) * 1000
                        end_time_ms = (end_frame / fps) * 1000
                        submit(video_path, start_time_ms, end_time_ms, fps)
    conn.close()
if query:
    if query.startswith("video:"):
        # Special case: Show all keyframes for a video by index
        try:
            index = int(query.split(":")[1].strip())
            if index < 1 or index > len(video_map)-1:
                st.warning(f"Index out of range. Enter 1 to {len(video_map)-1}.")
            else:
                selected_video_id = list(video_map.keys())[index - 1]  # convert to 0-based index
                selected_video_path = video_map[selected_video_id]

                st.subheader(f"Showing keyframes for DB_Video_ID {index}: `{1}`")

                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                #print(selected_video_id)
                cur.execute("SELECT id, keyframe_path FROM shots WHERE video_id = ?", (index,))
                shots = cur.fetchall()


                cols = st.columns(5)
                for i, (keyframe_id, keyframe_path) in enumerate(shots):
                    with cols[i % 5]:
                        if os.path.exists(keyframe_path):
                            st.image(load_image(keyframe_path), width=300, caption=f"Shot ID: {keyframe_id}")
                            if st.button("Play", key=f"imgbtn_{keyframe_id}", use_container_width=True):

                                cur = conn.cursor()
                                cur.execute("""
                                            SELECT videos.file_path, shots.start_frame, videos.fps
                                            FROM shots
                                                     JOIN videos ON shots.video_id = videos.id
                                            WHERE shots.id = ?
                                            """, (keyframe_id,))
                                result = cur.fetchone()
                                if result:
                                    video_path, start_frame, fps = result
                                    start_time_sec = start_frame / fps
                                    play(video_path, start_time_sec)
                            if st.button("Submit", key=f"imgsbmtbtn_{keyframe_id}", use_container_width=True):
                                cur = conn.cursor()
                                cur.execute("""
                                            SELECT videos.file_path, shots.start_frame, shots.end_frame, videos.fps
                                            FROM shots
                                                     JOIN videos ON shots.video_id = videos.id
                                            WHERE shots.id = ?
                                            """, (keyframe_id,))
                                result = cur.fetchone()
                                if result:
                                    video_path, start_frame, end_frame, fps = result
                                    start_time_ms = (start_frame / fps) * 1000
                                    end_time_ms = (end_frame / fps) * 1000
                                    submit(video_path, start_time_ms, end_time_ms, fps)
                        else:
                            st.warning(f"Missing keyframe: {keyframe_path}")
                conn.close()
        except ValueError:
            st.error("Invalid format. Use `video: <number>` (e.g., `video: 1`)")
    else:
        conn = sqlite3.connect(DB_PATH)
        if len(ids) == 0:
            st.warning("No embeddings found in the database.")
        else:
            results = search(query, model, device, ids, filenames, embeddings, videoIDs, 20)

            st.write(f"Top {len(results)} results:")

            cols = st.columns(5)  # 3-column grid

            for i, (keyframeID, filename, score, videoID) in enumerate(results):
                img_path = os.path.join(filename)
                if os.path.exists(img_path):
                    image = load_image(img_path)
                    with cols[i % 5]:
                        st.image(image, width=300, caption=f"Shot ID: {keyframeID}, Video File Name: {os.path.basename(str(filename)).split(".")[0]} Similarity: {score}, DB id video: {videoID}")
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
                                        SELECT videos.file_path, shots.start_frame, shots.end_frame, videos.fps
                                        FROM shots
                                                 JOIN videos ON shots.video_id = videos.id
                                        WHERE shots.id = ?
                                        """, (keyframeID,))
                            result = cur.fetchone()
                            if result:
                                video_path, start_frame, end_frame, fps = result
                                start_time_ms = (start_frame / fps) * 1000
                                end_time_ms = (end_frame / fps) * 1000
                                submit(video_path, start_time_ms, end_time_ms, fps)
                else:
                    with cols[i % 5]:
                        st.warning(f"Image not found: {filename}")
        conn.close()