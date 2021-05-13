# Source: https://www.programmersought.com/article/7432128222/
import os

import cv2

from gazeclassify.core.services.analysis import ModelLoader

ModelLoader("http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
            "~/gazeclassify_data/").download_if_not_available("pose_iter_440000.caffemodel")

frame = cv2.imread("../../tests/data/frame.jpg", cv2.IMREAD_UNCHANGED)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

inHeight = 368
inWidth = int((inHeight / frameHeight) * frameWidth)
frame = cv2.resize(frame, (300, 300))

inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (300, 300),
                                (0, 0, 0), swapRB=False, crop=False)

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = os.path.expanduser("~/gazeclassify_data/") + "pose_iter_440000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

net.setInput(inpBlob)
output = net.forward()

i = 0
probMap = output[0, i, :, :]
# probMap = cv2.resize(probMap, (frameWidth, frameHeight))
import matplotlib.pyplot as plt

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.imshow(probMap, alpha=0.6)

print("finished")
