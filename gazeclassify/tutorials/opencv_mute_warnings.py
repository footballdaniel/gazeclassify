from pathlib import Path

import cv2  # type: ignore

from gazeclassify.service.model_loader import ModelLoader


def main() -> None:
    model_weights = download_model_weights()
    model_prototype = download_model_prototype()

    frame = cv2.imread("../example_data/frame.png")
    net = cv2.dnn.readNetFromCaffe(str(model_prototype.file_path), str(model_weights.file_path))

    def has_cuda() -> int:
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                return 1
            else:
                return 0
        except:
            return 0

    cuda_devices = has_cuda()

    if cuda_devices > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    inHeight = 368
    inWidth = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    classified_image = net.forward()

    nPoints = 15
    tag_id = 1
    probMap = classified_image[0, tag_id, :, :]
    probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
    cv2.imshow('Keypoints', probMap)
    cv2.waitKey(0)


def download_model_weights() -> ModelLoader:
    model_weights = ModelLoader(
        "https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/raw/master/caffe_models/openpose/caffe_model/pose_iter_440000.caffemodel",
        str(Path("gazeclassify_data/models")))
    model_weights.download_if_not_available()
    return model_weights


def download_model_prototype() -> ModelLoader:
    model_prototype = ModelLoader(
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/OpenPose-Multi-Person/pose/coco/pose_deploy_linevec.prototxt",
         str(Path("gazeclassify_data/models")))
    model_prototype.download_if_not_available()
    return model_prototype

if __name__ == "__main__":
    main()
