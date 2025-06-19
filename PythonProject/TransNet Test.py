import torch
import numpy as np
import cv2
from transnetv2_pytorch import TransNetV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.version.cuda)
print("Using device:", device)

# Load the model
model = TransNetV2()
model.load_state_dict(torch.load("transnetv2-weights/transnetv2-pytorch-weights.pth", map_location=device))
model.eval()
# Read a video and convert it to a tensor
video_path = "everestFrameN.mp4"
cap = cv2.VideoCapture(video_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (48, 27))  # Required input size
    frames.append(frame)

cap.release()

# Convert frames to tensor
video_np = np.stack(frames)  # shape: (num_frames, 27, 48, 3)

video_tensor = torch.from_numpy(video_np).unsqueeze(0).to(torch.uint8)

# shape: (1, T, 27, 48, 3)

# Run inference
with (torch.no_grad()):
    video_tensor = video_tensor.to(device)
    #single_frame_pred, single_frame_features,
    scene_predictions = model(video_tensor)

    print(scene_predictions)

# Interpret output
#scene_scores = scene_predictions[0].numpy()
scene_scores = torch.sigmoid(scene_predictions[0]).squeeze().cpu().numpy()
scene_changes = (scene_scores > 0.5).astype(np.uint8)

print("Scene scores:", scene_scores[:50])
print("scene_changes",scene_changes)
change_indices = np.where(scene_changes == 1)[0].tolist()

print("Detected scene changes at frames:", change_indices)