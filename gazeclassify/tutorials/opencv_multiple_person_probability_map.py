# https://github.com/spmallick/learnopencv/blob/master/OpenPose/OpenPoseImage.py
import os

import cv2  # type: ignore
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

from gazeclassify.service.model_loader import ModelLoader

classifier = ModelLoader(
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel",
    "gazeclassify_data"
)
classifier.download_if_not_available()
protoFile = os.path.expanduser("~/gazeclassify_data/") + "pose_deploy_linevec.prototxt"
weightsFile = os.path.expanduser("~/gazeclassify_data/") + "pose_iter_440000.caffemodel"

nPoints = 18
POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
              [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

frame = cv2.imread("../example_data/humans.jpeg")
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.05

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# if args.device == "cpu":
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
# elif args.device == "gpu":
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#     print("Using GPU device")

# input image dimensions for the network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward() # shape is 1,keypoints,pixelwidht, pixelheight

H = output.shape[2]
W = output.shape[3]

i = 0 # is the head
probMap = output[0, i, :, :] #

# show image
probMap = cv2.resize(probMap, (frameWidth, frameHeight))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.imshow(probMap, alpha=0.6)
plt.show()
