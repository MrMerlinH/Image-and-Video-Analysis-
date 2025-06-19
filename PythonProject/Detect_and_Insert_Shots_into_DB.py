import sqlite3
from unittest import result

import cv2

import TransNet_ShotD_Split

import os

# Set the folder you want to search
video_folder = "videos"

def save_middle_keyframes(video_path, shot_boundaries):
    cap = cv2.VideoCapture(video_path)
    prev_boundary = 0
    boundaries = [0] + shot_boundaries + [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1]

    # Connect to the SQLite database
    conn = sqlite3.connect("DB/cbvr.db")
    c = conn.cursor()
#


    # Normalize and prepare the input path
    #normalized_path = os.path.normpath(video_path)  # Convert to system's path format
    print(video_path)

    relative_path = video_path.replace("\\", "/",1)
    relative_path = "../" + relative_path

    # Optional: convert to Unix-style forward slashes to match DB (if needed)
    print(relative_path)

    # Execute query
    c.execute("SELECT videos.id FROM videos WHERE file_path = '"+str(relative_path)+"'")

    result = c.fetchone()

    print("ok = {}".format(relative_path))
    print("epic result"+str(result))

    video_id = result[0] if result else -1
    if video_id == -1:
        print("BRO WTF VIDEO IS NOT INSIDE DB: VIDEO PATH = {}".format(video_path))

    #shot detection and storing in db
    for i, boundary in enumerate(boundaries):
        middle_frame = (prev_boundary + boundary) // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if ret:
            filename = f"testFrames/{os.path.basename(video_path)}_keyframe_shot{i}_frame{middle_frame}.jpg"
            cv2.imwrite(filename, frame)
            #print(f"Saved keyframe for shot {i} at frame {middle_frame} as {filename}")
            c.execute("""
                           INSERT INTO shots (video_id, start_frame, end_frame, keyframe_path, clip_embedding)
                           VALUES (?, ?, ?, ?, NULL)
                           """, (video_id, prev_boundary, boundary, filename))
        else:
            print(f"Failed to read frame {middle_frame} for shot {i}")

        prev_boundary = boundary + 1  # next segment starts after this boundary

    cap.release()
    conn.commit()
    conn.close()


# Walk through the directory
for root, dirs, files in os.walk(video_folder):
    for file in files:
        full_path = os.path.join(root, file)
        print(full_path)
        shot_boundaries = TransNet_ShotD_Split.shotDetection(full_path)
        print(shot_boundaries)
        save_middle_keyframes(full_path, shot_boundaries)
