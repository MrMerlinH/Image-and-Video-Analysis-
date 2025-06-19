import torch
import numpy as np
import cv2
from transnetv2_pytorch import TransNetV2
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.version.cuda)
print("Using device:", device)

# Load the model
model = TransNetV2()
model.load_state_dict(torch.load("transnetv2-weights/transnetv2-pytorch-weights.pth", map_location=device))
model.eval()


def shotDetection(path):

    cap = cv2.VideoCapture(path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (48, 27))  # Required input size
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print(f"Warning: No frames read from {path}")
        return []

    video_np = np.stack(frames)  # shape: (num_frames, 27, 48, 3)
    total_frames = len(video_np)

    # Parameters for chunking
    WINDOW_SIZE = 100
    STRIDE = 25

    predictions = []

    with torch.no_grad():
        for start in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            window_np = video_np[start:end]

            video_tensor = torch.from_numpy(window_np).unsqueeze(0).to(torch.uint8).to(device)

            scene_predictions = model(video_tensor)
            scores = torch.sigmoid(scene_predictions[0]).squeeze().cpu().numpy()

            # Stitch by tracking offset
            for i, score in enumerate(scores):
                frame_index = start + i
                if frame_index >= len(predictions):
                    predictions.append(score)
                else:
                    # Average overlapping predictions
                    predictions[frame_index] = (predictions[frame_index] + score) / 2.0

            # Cleanup per chunk
            del video_tensor, scene_predictions, scores
            if device.type == "cuda":
                torch.cuda.empty_cache()

    scene_scores = np.array(predictions)
    scene_changes = (scene_scores > 0.5).astype(np.uint8)
    change_indices = np.where(scene_changes == 1)[0].tolist()

    print("Detected scene changes at frames:", change_indices)

    # Final cleanup
    del video_np, predictions, scene_scores, scene_changes
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return change_indices
