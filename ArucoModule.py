import cv2
import cv2.aruco as aruco
import numpy as np
import os


def loadAugImages(path):
    """
    path: folder in which all the marker images with ids are stored
    return: dictionary with keys as the id and values as the augment image 
    """

    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total Number of Markers Detected: ", noOfMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug

        # 23: image23
        # 40: image40

    return augDics

def findArucoMarker(img, markerSize = 6, totalMarkers = 250, draw = True):
    """
    img: image in which to find the aruco markers
    markerSize: the size of the markers
    totalMarkers: total number of markers that compose the dictionary
    draw: flag to draw bbox around markers detected
    return: bounding boxes adn id numbers of markers detected
    """

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, 
                                               arucoDict, parameters = arucoParam)
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]


def augmentAruco(bbox, id, img, imgAug, drawId = True):
    """
    bbox: the four corner points of the box
    id: marker id of the corresponding box used only for display
    img: the final image on which to draw
    imgAug: the image that will be overlapped on the marker
    drawId: flag to display the id of the detected markers
    return: image with the augment image overlaid
    """

    # Four corner points
    tl = int(bbox[0][0][0]), int(bbox[0][0][1])
    tr = int(bbox[0][1][0]), int(bbox[0][1][1])
    br = int(bbox[0][2][0]), int(bbox[0][2][1])
    bl = int(bbox[0][3][0]), int(bbox[0][3][1])
 
    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))      ## Basically we are filling the image (at every position) with black color
    imgOut = img + imgOut

    if drawId:
        cv2.putText(imgOut, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, 
                    (255, 0, 255), 2)


    return imgOut

def main():
    cap = cv2.VideoCapture(0)
    # imgAug = cv2.imread("D:\B.Tech EE1 IITD\Coding Practice\ArucoProject\Images\23.jpg")
    augDics = loadAugImages('.\Images')
    while True:
        success, img = cap.read()
        arucoFound = findArucoMarker(img)

        # Loop throgh all the markers and augment each one
        if (len(arucoFound[0]) != 0):
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDics.keys():
                    img = augmentAruco(bbox, id, img, augDics[int(id)])

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()