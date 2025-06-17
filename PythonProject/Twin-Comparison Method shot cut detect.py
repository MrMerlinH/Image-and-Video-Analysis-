from time import sleep

import cv2
import matplotlib
import numpy as np
# import tinydb as db
from matplotlib import pyplot as plt
import os
import sqlite3

#print("OpenCV version: " + cv.__version__)
#img = cv.imread("testimage.png") #loads image in BGR
#print("Size W x H = ", img.shape[1], "x", img.shape[0])
#print("Channels = ", img.shape[2])
#img[0:100, 0:50] = np.zeros((100, 50, 3), dtype="uint8")
#small = cv.resize(src=img, dsize=(180, 120))
#cv.imshow("Small", small)
#cv.imshow("Input Image", img)
#img2 = cv.imread("testimage.png",0)
#cv.waitKey(0)
#cv.destroyAllWindows()

vid = cv2.VideoCapture("videos/V3C1_200/everest.mp4")
hists = []
BINS = 64
ranges=[0, 256]
SAMPLING_INT = 250
frameCount = -1
frameIntervals = []
frames = []


def compute_histogram(frame, bins=64):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def euclidean_distance(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)


def twin_comparison_method(video_path, TH=0.5, TD=0.1):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return []

    prev_hist = compute_histogram(prev_frame)
    shot_boundaries = []
    diff_accum = 0
    frame_index = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_hist = compute_histogram(frame)
        diff = euclidean_distance(prev_hist, curr_hist)

        if diff > TH:
            shot_boundaries.append(frame_index)
            diff_accum = 0  # Reset for new shot
            # print("Shot Change at: " + str(frame_index))
            # cv2.imwrite(str(frame_index) + ".jpg", frame)  # Save the frame where the shot change occurs
        elif diff > TD:
            diff_accum += diff
            if diff_accum > TH:
                shot_boundaries.append(frame_index)
                # print("SLOW Shot Change at: " + str(frame_index))
                # cv2.imwrite(str(frame_index) + ".jpg", frame)  # Save the frame where the shot change occurs
                diff_accum = 0
        else:
            diff_accum = 0

        prev_hist = curr_hist
        frame_index += 1

    cap.release()
    save_middle_keyframes("videos/V3C1_200/everest.mp4", shot_boundaries)
    return shot_boundaries

def save_middle_keyframes(video_path, shot_boundaries):
    cap = cv2.VideoCapture(video_path)
    prev_boundary = 0
    boundaries = [0] + shot_boundaries + [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1]

    conn = sqlite3.connect("DB/cbvr.db")
    c = conn.cursor()


    for i, boundary in enumerate(boundaries):
        middle_frame = (prev_boundary + boundary) // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if ret:
            filename = f"testFrames/{os.path.basename(video_path)}_keyframe_shot{i}_frame{middle_frame}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved keyframe for shot {i} at frame {middle_frame} as {filename}")
            c.execute("""
                           INSERT INTO shots (video_id, start_frame, end_frame, keyframe_path, clip_embedding)
                           VALUES (?, ?, ?, ?, NULL)
                           """, (201, prev_boundary, boundary, filename))
        else:
            print(f"Failed to read frame {middle_frame} for shot {i}")

        prev_boundary = boundary + 1  # next segment starts after this boundary

    cap.release()
    conn.commit()
    conn.close()
print(twin_comparison_method("videos/V3C1_200/everest.mp4"))

#cv2.waitKey(0)

# closing all open windows
#cv2.destroyAllWindows()