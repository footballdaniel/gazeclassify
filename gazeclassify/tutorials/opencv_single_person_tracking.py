import os
from dataclasses import field
from typing import Optional, Tuple, List

import cv2  # type: ignore

from gazeclassify.service.model_loader import ModelLoader

# Sources
# https://github.com/faizancodes/NBA-Pose-Estimation-Analysis
# https://cv-tricks.com/pose-estimation/using-deep-learning-in-opencv/

classifier = ModelLoader(
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel",
    "gazeclassify_data"
)
classifier.download_if_not_available()

# Specify the paths for the 2 files
protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"

weightsFile = os.path.expanduser("~/gazeclassify_data/") + "pose_iter_160000.caffemodel"
# weightsFile = "pose/mpi/pose_iter_160000.caffemodel"

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Read image
frame = cv2.imread("../example_data/frame.jpg", cv2.IMREAD_UNCHANGED)

# Resize image to analyze
frame = cv2.resize(frame, (600, 600))
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

# Analysis size
width = 300
height = 300

inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height), (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)
output = net.forward()

output.shape  # 1,44,46,46 array!

H = output.shape[2]
W = output.shape[3]
threshold = 0.1

points: List[Optional[Tuple[int, int]]] = field(default_factory=list)
points = []

nPoints = 15
for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold:
        cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                    lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

frame = cv2.resize(frame, (1000, 1000))
cv2.imshow("Output-Keypoints", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
