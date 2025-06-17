from time import sleep

import cv2 as cv
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

vid = cv.VideoCapture("videos/V3C1_200/everest.mp4")
hists = []
BINS = 64
ranges=[0, 256]
SAMPLING_INT = 250
frameCount = -1
frameIntervals = []
frames = []

while True:
    ret, frame = vid.read()
    if frame is None:
        break
    frameCount += 1
    if frameCount % SAMPLING_INT == 0:
        frameIntervals.append(frameCount)
        frames.append(frame)
        frameg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hist = cv.calcHist(images=[frameg], channels=[0],
        mask=None, histSize=[BINS], ranges=ranges)
        histt = cv.transpose(hist)
        hists.append(histt)
        cv.imshow(f"Frame {frameCount}", frameg)
        plt.plot(hist, label=f'{frameCount}')
plt.legend()
plt.show()

#k-means clustering
samples = np.zeros((len(hists), BINS)) #optional 2nd parameter dtype (default=float64)
i = 0
for h in hists:
    samples[i] = h
    i += 1

samples = np.float32(samples) #convert float64 to float32

numClusters = 6
flags = cv.KMEANS_RANDOM_CENTERS
#criteria is tuple of three parameters (type, max_iter, epsilon)
#type can by MAX_ITER, EPS, or MAX_ITER+EPS
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 10, 1.0)

compactness,labels,centers = cv.kmeans(data=samples, K=numClusters, bestLabels=None, criteria=criteria, attempts=10, flags=flags)
for i in range(0,len(labels)):
    lbl = labels[i][0]
    print("frame: "+str(i)+" Group: "+str(lbl))
    cv.imwrite(f'output/{lbl}_{frameIntervals[i]}.jpg', frames[i])

vid.release()

cv.waitKey(0)

# closing all open windows
cv.destroyAllWindows()