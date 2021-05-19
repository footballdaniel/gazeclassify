# https://learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose/
# https://github.com/spmallick/learnopencv/blob/master/OpenPose-Multi-Person/multi-person-openpose.py
import logging
import os
from dataclasses import dataclass
from typing import Union

import cv2  # type: ignore
import numpy as np  # type: ignore

from gazeclassify.core.services.analysis import Analysis
from gazeclassify.core.services.gaze_distance import DistanceToPoint
from gazeclassify.core.services.opencv_classifier import OpenCVClassifier
from gazeclassify.core.services.results import InstanceClassification, FrameResults


@dataclass
class VideoWriter:
    target_file: str

    def initiate(self, video_width: int, video_height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.result_video = cv2.VideoWriter(self.target_file, fourcc, 10, (video_width, video_height))

    def write(self, frame: np.ndarray) -> None:
        self.result_video.write(frame)

    def release(self) -> None:
        self.result_video.release()


@dataclass
class VideoReader:
    target_file: str

    @property
    def has_frame(self) -> Union[bool, np.ndarray]:
        return self._has_frame

    def initiate(self) -> None:
        self.capture = cv2.VideoCapture(self.target_file)

    def next_frame(self) -> np.ndarray:
        self._has_frame, frame = self.capture.read()
        return frame


@dataclass
class InstanceSegmentation:
    analysis: Analysis
    model_url: str = "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"
    proto_file_url: str = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt"


    def classify(self, name: str) -> None:

        writer = self._setup_video_writer(name)
        reader = self._setup_video_reader()

        for idx, record in enumerate(self.analysis.dataset.records):

            frame = reader.next_frame()

            if not reader.has_frame:
                logging.error("Video has ended prematurely")

            classifier = OpenCVClassifier(model_url=self.model_url, proto_file_url=self.proto_file_url)
            classifier.download_model()
            classifier.classify_frame(frame)

            frameClone, personwiseKeypoints, keypoints_list = self.classify_frame(frame)

            # Write video out
            writer.write(frameClone)

            # Getting the keypoints per person
            POSE_PAIRS = self._get_pose_pairs()
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])

                    first_point = (B[0], A[0])
                    second_point = (B[1], A[1])

            # MY STUFF: I Manually append the points
            # Getting the keypoints per person
            POSE_PAIRS = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6],
                          [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12],
                          [13, 13], [14, 14], [15, 15], [16, 16], [17, 17],
                          [18, 18]]

            ## Append to results
            keypointsMapping = ['Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Left Shoulder', 'Left Elbow',
                                'Left Wrist', 'Right Hip', 'Right Knee',
                                'Right Ankle', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye',
                                'Right Ear', 'Left Ear']

            results = []
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])

                    point_x = B[0]
                    point_y = A[0]

                    distance = DistanceToPoint(point_x, point_y).distance_2d(record.gaze.x, record.gaze.y)
                    results.append(InstanceClassification(distance, keypointsMapping[i], n))

                    # if i == 14:
                    #     cv2.circle(frameClone,(point_x, point_y),10,[0, 100, 255])
                    #     cv2.imshow("image", frameClone)
                    #     cv2.waitKey(0)

            frame_results = FrameResults(idx, name, results)
            self.analysis.results.append(frame_results)

            idx += 1

            if idx == 2:
                print("DEBUG STOP")
                break

        writer.release()
        reader.capture.release()
        cv2.destroyAllWindows()

    def _get_pose_pairs(self):
        POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                      [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                      [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                      [2, 17], [5, 16]]
        return POSE_PAIRS

    def _setup_video_reader(self):
        source_file = str(self.analysis.dataset.world_video.file)
        reader = VideoReader(source_file)
        reader.initiate()

        return reader

    def _setup_video_writer(self, name):
        video_target = os.path.expanduser("~/gazeclassify_data/") + f"{name}.mp4"
        logging.info(f"Writing export video to: {video_target}")
        world_video = self.analysis.dataset.world_video
        writer = VideoWriter(video_target)
        writer.initiate(world_video.width, world_video.height)
        return writer

    def classify_frame(self, frame, threshold: float = 0.1):

        self._proto_file = os.path.expanduser("~/gazeclassify_data/") + "pose_deploy_linevec.prototxt"
        self._weights_file = os.path.expanduser("~/gazeclassify_data/") + "pose_iter_440000.caffemodel"


        mapIdx = self._get_mapping_part_affinity_fields()

        classified_image, frame_height, frame_width = self.classify_with_dnn(frame)

        detected_keypoints, keypoints_list = self.detect_keypoints(frame, classified_image, threshold)

        valid_pairs, invalid_pairs = self.getValidPairs(classified_image, frame_width, frame_height, detected_keypoints)
        personwiseKeypoints = self.getPersonwiseKeypoints(valid_pairs, invalid_pairs,
                                                          keypoints_list)

        POSE_PAIRS = self._get_pose_pairs()
        colors = self._get_colors()
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)


        return frame, personwiseKeypoints, keypoints_list

    def classify_with_dnn(self, frame):
        net = cv2.dnn.readNetFromCaffe(self._proto_file, self._weights_file)
        # if args.device == "cpu":
        # net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        # print("Using CPU device")
        # elif args.device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        inHeight = 368
        inWidth = int((inHeight / frame_height) * frame_width)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        classified_image = net.forward()
        return classified_image, frame_height, frame_width

    def detect_keypoints(self, frame, output, threshold):
        nPoints = 18
        keypointsMapping = self._get_keypoints_mapping()

        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        for part in range(nPoints):
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            keypoints = self.getKeypoints(probMap, threshold)
            print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)
        return detected_keypoints, keypoints_list

    def _get_colors(self):
        colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
                  [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
                  [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
        return colors

    def _get_mapping_part_affinity_fields(self):
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
                  [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                  [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
                  [37, 38], [45, 46]]
        return mapIdx

    def _get_keypoints_mapping(self):
        keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee',
                            'R-Ank',
                            'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
        return keypointsMapping

    def getKeypoints(self, probMap, threshold=0.1):
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

        mapMask = np.uint8(mapSmooth > threshold)
        keypoints = []

        # find the blobs
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints

    # Find valid connections between the different joints of a all persons present
    def getValidPairs(self, output, frameWidth, frameHeight, detected_keypoints):

        POSE_PAIRS = self._get_pose_pairs()
        mapIdx = self._get_mapping_part_affinity_fields()
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        # loop for every POSE_PAIR
        for k in range(len(mapIdx)):
            # A->B constitute a limb
            pafA = output[0, mapIdx[k][0], :, :]
            pafB = output[0, mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (frameWidth, frameHeight))
            pafB = cv2.resize(pafB, (frameWidth, frameHeight))

            # Find the keypoints for the first and second limb
            candA = detected_keypoints[POSE_PAIRS[k][0]]
            candB = detected_keypoints[POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)

            # If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid

            if (nA != 0 and nB != 0):
                valid_pair = np.zeros((0, 3))
                for i in range(nA):
                    max_j = -1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                               pafB[
                                                   int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)

                        # Check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                        if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else:  # If no keypoints are detected
                print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs

    # This function creates a list of keypoints belonging to each person
    # For each detected valid pair, it assigns the joint(s) to a person
    def getPersonwiseKeypoints(self, valid_pairs, invalid_pairs, keypoints_list):
        POSE_PAIRS = self._get_pose_pairs()
        mapIdx = self._get_mapping_part_affinity_fields()

        # the last number in each row is the overall score
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(mapIdx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:, 0]
                partBs = valid_pairs[k][:, 1]
                indexA, indexB = np.array(POSE_PAIRS[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + \
                                                               valid_pairs[k][i][
                                                                   2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints
