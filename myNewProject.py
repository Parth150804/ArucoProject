import cv2
import cv2.aruco as aruco
import numpy as np
import os
import ArucoModule as arm

cap = cv2.VideoCapture(0)
augDics = arm.loadAugImages('.\Images')

while True:
    success, img = cap.read()
    arucoFound = arm.findArucoMarker(img)

    # Loop throgh all the markers and augment each one
    if (len(arucoFound[0]) != 0):
        for bbox, id in zip(arucoFound[0], arucoFound[1]):
            if int(id) in augDics.keys():
                img = arm.augmentAruco(bbox, id, img, augDics[int(id)])

    cv2.imshow("Image", img)
    cv2.waitKey(1)
