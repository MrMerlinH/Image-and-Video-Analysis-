import os

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
    # Read a video and convert it to a tensor
    video_path = path
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
        # single_frame_pred, single_frame_features,
        scene_predictions = model(video_tensor)

        #print(scene_predictions)

    # Interpret output
    # scene_scores = scene_predictions[0].numpy()
    scene_scores = torch.sigmoid(scene_predictions[0]).squeeze().cpu().numpy()
    scene_changes = (scene_scores > 0.5).astype(np.uint8)

    change_indices = np.where(scene_changes == 1)[0].tolist()

    #print("Detected scene changes at frames:", change_indices)

    # Cleanup
    del video_np, video_tensor, scene_predictions, scene_scores, scene_changes
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return change_indices

for root, dirs, files in os.walk("videos"):

        full_path = "videos/V3C1_200/00009.mp4"#os.path.join(root, file)
        print(full_path)
        shot_boundaries = shotDetection(full_path)

        print(shot_boundaries)

        cap = cv2.VideoCapture(full_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # List to hold updated boundaries
        updated_boundaries = []



        # Process each shot
        for i in range(len(shot_boundaries) - 1):
            start = shot_boundaries[i]
            end = shot_boundaries[i + 1]
            updated_boundaries.append(start)

            duration = (end - start) / fps
            if duration > 7:
                # Add frames at 5-second intervals
                interval = int(fps * 7)
                intermediate = list(range(start + interval, end, interval))
                updated_boundaries.extend(intermediate)

        # Always include the last shot end
        updated_boundaries.append(shot_boundaries[-1])

        # Optional: sort and deduplicate
        updated_boundaries = sorted(set(updated_boundaries))
        print(updated_boundaries)
       # print(f"Final shot boundaries for {file}: {updated_boundaries}")