# https://cv-tricks.com/pose-estimation/using-deep-learning-in-opencv/
import os
from dataclasses import field
from typing import Optional, Tuple, List

import cv2  # type: ignore

from gazeclassify.core.services.analysis import ModelLoader

OPENPOSE_URL = "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose_iter_160000.caffemodel"


# https://github.com/faizancodes/NBA-Pose-Estimation-Analysis
# Basketball

def test_Analyze_image_coco() -> None:
    # ModelLoader("http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
    #             "~/gazeclassify_data/").download_if_not_available("pose_iter_440000.caffemodel")

    ModelLoader("http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel",
                "~/gazeclassify_data/").download_if_not_available("pose_iter_160000.caffemodel")

    # Specify the paths for the 2 files
    protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"

    weightsFile = os.path.expanduser("~/gazeclassify_data/") + "pose_iter_160000.caffemodel"
    # weightsFile = "pose/mpi/pose_iter_160000.caffemodel"

    # Read the network into Memory
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Read image
    frame = cv2.imread("frame.png", cv2.IMREAD_UNCHANGED)

    # resize
    width = 500
    height = 500
    frame = cv2.resize(frame, (width, height))

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    output.shape  # 1,44,46,46 array!

    # Plot
    H = output.shape[2]
    W = output.shape[3]
    threshold = 0.1

    # Empty list to store the detected keypoints
    points: List[Optional[Tuple[int, int]]] = field(default_factory=list)
    nPoints = 15
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (width * point[0]) / W
        y = (height * point[1]) / H

        if prob > threshold:
            cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                        lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    cv2.imshow("Output-Keypoints", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    assert 1 == 1
