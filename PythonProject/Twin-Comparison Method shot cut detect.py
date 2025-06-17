from time import sleep

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

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

vid = cv2.VideoCapture("everest.mp4")
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


def twin_comparison_method(video_path, TH=0.7, TD=0.2):
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
            print("Shot Change at: " + str(frame_index))
        elif diff > TD:
            diff_accum += diff
            if diff_accum > TH:
                shot_boundaries.append(frame_index)
                print("SLOW Shot Change at: " + str(frame_index))
                diff_accum = 0
        else:
            diff_accum = 0

        prev_hist = curr_hist
        frame_index += 1

    cap.release()
    return shot_boundaries

print(twin_comparison_method("everest.mp4"))

#cv2.waitKey(0)

# closing all open windows
#cv2.destroyAllWindows()