import cv2
import cv2.aruco as aruco
import numpy as np

# Load Aruco Dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Define clues for each Aruco marker
clues = {
    0: "Go to the red door.",
    1: "Look under the table.",
    2: "Check behind the painting.",
    3: "Search in the garden.",
    4: "Congratulations! You've found all the clues!"
}

# Initialize video capture
cap = cv2.VideoCapture(0)

# Track found markers
found_markers = set()

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            # Draw a bounding box around the detected marker
            aruco.drawDetectedMarkers(frame, corners)

            # Get the center point of the marker
            c = corners[i][0]
            center = (int(c[:, 0].mean()), int(c[:, 1].mean()))

            # Display the clue
            if marker_id not in found_markers:
                found_markers.add(marker_id)
                if marker_id in clues:
                    clue_text = clues[marker_id]
                    cv2.putText(frame, clue_text, (center[0] - 100, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Check if all markers have been found
            if len(found_markers) == len(clues):
                cv2.putText(frame, "Treasure Hunt Completed!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                break

    # Display the resulting frame
    cv2.imshow('Treasure Hunt', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
