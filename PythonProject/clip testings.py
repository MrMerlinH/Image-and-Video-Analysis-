
#pip install git+https://github.com/openai/CLIP.git
import clip
import cv2
#pip install torch torchvision
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faiss

import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
print(device)

# Load the model
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
image = preprocess(Image.open("susCat.png")).unsqueeze(0).to(device)  # Add batch dimension

# Tokenize the text
text = clip.tokenize(["a diagram", "a dog", "a cat", "a gun"]).to(device)  # Returns a tensor

# Run the model
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Compute cosine similarity
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()


def extract_frames(video_path, every_n_sec=1):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * every_n_sec)

    success, image = vidcap.read()
    frames = []
    timestamps = []
    count = 0

    while success:
        if count % interval == 0:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(preprocess(img).unsqueeze(0))
            timestamps.append(vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return torch.cat(frames).to(device), timestamps



def build_index(video_paths):
    index = faiss.IndexFlatL2(512)
    frame_meta = []

    for video_path in video_paths:
        frames, times = extract_frames(video_path)
        with torch.no_grad():
            embeddings = model.encode_image(frames).cpu().numpy()
        index.add(embeddings)
        frame_meta.extend([(video_path, t) for t in times])

    return index, frame_meta

def query_text(text_query, index, frame_meta, top_k=20):
    with torch.no_grad():
        text_tokens = clip.tokenize([text_query]).to(device)
        text_features = model.encode_text(text_tokens).cpu().numpy()

    D, I = index.search(text_features, top_k)

    results = []
    for idx in I[0]:
        video_path, timestamp = frame_meta[idx]
        results.append((video_path, timestamp))

    return results



video_paths = ["everestFrameN.mp4"]
index, metadata = build_index(video_paths)
results = query_text("a human face", index, metadata)

for path, time in results:
    print(f"Match at {time:.2f}s in {path}")


print("Label probs:", probs)  # [[0.992, 0.008]]
