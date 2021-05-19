import os
from dataclasses import dataclass, field
from typing import List, cast, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from gazeclassify.core.model.dataset import DataRecord
from gazeclassify.core.services.gaze_distance import DistanceToPoint
from gazeclassify.core.services.model_loader import ModelLoader
from gazeclassify.core.services.results import InstanceClassification, Classification


@dataclass
class OpenCVClassifier:
    model_url: str = "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"
    proto_file_url: str = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt"

    def download_model(self) -> None:
        weights_model = ModelLoader(self.model_url, "~/gazeclassify_data/")
        weights_model.download_if_not_available("pose_iter_440000.caffemodel")

        proto_model = ModelLoader(self.proto_file_url, "~/gazeclassify_data/models")
        proto_model.download_if_not_available("pose_deploy_linevec.prototxt")

        self.weights_file = weights_model.file_path
        self._proto_file = proto_model.file_path

    def gaze_distance_to_instance(self, record: DataRecord) -> List[Classification]:
        POSE_PAIRS = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6],
                      [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12],
                      [13, 13], [14, 14], [15, 15], [16, 16], [17, 17],
                      [18, 18]]
        keypointsMapping = self._get_keypoints_mapping()
        results = []
        for i in range(17):
            for n in range(len(self.personwise_keypoints)):
                index = self.personwise_keypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(self.keypoints_list[index.astype(int), 0])
                A = np.int32(self.keypoints_list[index.astype(int), 1])

                point_x = B[0]
                point_y = A[0]

                distance = DistanceToPoint(point_x, point_y).distance_2d(record.gaze.x, record.gaze.y)
                classification = InstanceClassification(distance, keypointsMapping[i], n)
                results.append(classification)
        return results  # type: ignore

    def classify_frame(self, frame: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        self._proto_file = os.path.expanduser("~/gazeclassify_data/") + "pose_deploy_linevec.prototxt"
        self._weights_file = os.path.expanduser("~/gazeclassify_data/") + "pose_iter_440000.caffemodel"

        self._frame_width = frame.shape[1]
        self._frame_height = frame.shape[0]

        classified_image = self._classify_with_dnn(frame)
        self._detect_keypoints(classified_image, threshold)
        self._get_valid_keypoint_pairs(classified_image)
        self._get_personwise_keypoints()

        visualized_frame = self._visualize(frame)

        return visualized_frame

    def _get_keypoints_mapping(self) -> List[str]:
        keypointsMapping = ['Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Left Shoulder', 'Left Elbow',
                            'Left Wrist', 'Right Hip', 'Right Knee',
                            'Right Ankle', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye',
                            'Right Ear', 'Left Ear']
        return keypointsMapping

    def _get_pose_pairs(self) -> List[List[int]]:
        POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                      [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                      [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                      [2, 17], [5, 16]]
        return POSE_PAIRS

    def _visualize(self, frame: np.ndarray) -> np.ndarray:
        POSE_PAIRS = self._get_pose_pairs()
        colors = self._get_colors()
        for i in range(17):
            for n in range(len(self.personwise_keypoints)):
                index = self.personwise_keypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(self.keypoints_list[index.astype(int), 0])
                A = np.int32(self.keypoints_list[index.astype(int), 1])
                cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
        return frame

    def _classify_with_dnn(self, frame: np.ndarray) -> np.ndarray:
        net = cv2.dnn.readNetFromCaffe(self._proto_file, self._weights_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        inHeight = 368
        inWidth = int((inHeight / self._frame_height) * self._frame_width)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        classified_image = net.forward()
        return classified_image

    def _detect_keypoints(self, classified_image: np.ndarray, threshold: float) -> None:
        nPoints = 18
        keypointsMapping = self._get_keypoints_mapping_raw()

        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        for part in range(nPoints):
            probMap = classified_image[0, part, :, :]
            probMap = cv2.resize(probMap, (self._frame_width, self._frame_height))
            keypoints = self._get_keypoints(probMap, threshold)
            print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1
            detected_keypoints.append(keypoints_with_id)
        self.keypoints_list = keypoints_list
        self.detected_keypoints = detected_keypoints

    def _get_colors(self) -> List[List[int]]:
        colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
                  [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
                  [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
        return colors

    def _get_mapping_part_affinity_fields(self) -> List[List[int]]:
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
                  [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                  [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
                  [37, 38], [45, 46]]
        return mapIdx

    def _get_keypoints_mapping_raw(self) -> List[str]:
        keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee',
                            'R-Ank',
                            'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
        return keypointsMapping

    def _get_keypoints(self, probMap: np.ndarray, threshold: float =0.1) -> List[Tuple[int]]:
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

        mask = np.uint8(mapSmooth > threshold)
        keypoints = []

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            blobMask = np.zeros(mask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints

    def _get_valid_keypoint_pairs(self, classified_image: np.ndarray) -> None:
        POSE_PAIRS = self._get_pose_pairs()
        mapIdx = self._get_mapping_part_affinity_fields()
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7

        for k in range(len(mapIdx)):
            # A->B constitute a limb
            pafA = classified_image[0, mapIdx[k][0], :, :]
            pafB = classified_image[0, mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (self._frame_width, self._frame_height))
            pafB = cv2.resize(pafB, (self._frame_width, self._frame_height))

            # Find the keypoints for the first and second limb
            candidate_a = self.detected_keypoints[POSE_PAIRS[k][0]]
            candidate_b = self.detected_keypoints[POSE_PAIRS[k][1]]
            nA = len(candidate_a)
            nB = len(candidate_b)

            if (nA != 0 and nB != 0):
                valid_pair = np.zeros((0, 3))
                for i in range(nA):
                    max_j = -1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candidate_b[j][:2], candidate_a[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candidate_a[i][0], candidate_b[j][0], num=n_interp_samples),
                                                np.linspace(candidate_a[i][1], candidate_b[j][1],
                                                            num=n_interp_samples)))
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
                                maxScore = avg_paf_score # type: ignore
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[candidate_a[i][3], candidate_b[max_j][3], maxScore]],  # type: ignore
                                               axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else:  # If no keypoints are detected
                print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])
        self.valid_pairs = valid_pairs
        self.invalid_pairs = invalid_pairs

    def _get_personwise_keypoints(self) -> None:
        # This function creates a list of keypoints belonging to each person
        # For each detected valid pair, it assigns the joint(s) to a person
        POSE_PAIRS = self._get_pose_pairs()
        mapIdx = self._get_mapping_part_affinity_fields()

        # the last number in each row is the overall score
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(mapIdx)):
            if k not in self.invalid_pairs:
                partAs = self.valid_pairs[k][:, 0]
                partBs = self.valid_pairs[k][:, 1]
                indexA, indexB = np.array(POSE_PAIRS[k])

                for i in range(len(self.valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += self.keypoints_list[partBs[i].astype(int), 2] + \
                                                               self.valid_pairs[k][i][2]

                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(self.keypoints_list[self.valid_pairs[k][i, :2].astype(int), 2]) + self.valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])

        self.personwise_keypoints = personwiseKeypoints
