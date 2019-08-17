#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util


class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)


class FaceDetect:
    def __init__(self):
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT = './tensorflow-face-detection/model/frozen_inference_graph_face.pb'

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = './tensorflow-face-detection/protos/face_label_map.pbtxt'

        self.NUM_CLASSES = 2

        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.tDetector = TensoflowFaceDector(self.PATH_TO_CKPT)

    def calc_reward(self, boxes, h, w, score):
        reward = 0
        centers = []
        for j, box in enumerate(boxes):
            for i, face in enumerate(box):
                if score[j, i] > 0.7:
                    center = (int((face[3] + face[1]) * w / 2), int((face[2] + face[0]) * h / 2))
                    dx = (center[0] - w / 2) ** 2
                    dy = (center[1] - h / 2) ** 2
                    reward += np.exp(-dx / 10000) * score[j, i]
                    centers.append(center)
        return centers, reward


# if __name__ == "__main__":
#     camID = 0
#
#     tDetector = TensoflowFaceDector(PATH_TO_CKPT)
#
#     cap = cv2.VideoCapture(camID)
#     windowNotSet = True
#
#     while True:
#         ret, image = cap.read()
#         if ret == 0:
#             break
#
#         [h, w] = image.shape[:2]
#         image = cv2.flip(image, 1)
#
#         (boxes, scores, classes, num_detections) = tDetector.run(image)
#
#         centers, reward = calc_reward(boxes, h, w, scores)
#         print('Reward =', reward)
#         if reward:
#             avg_center = np.average(centers, axis=0).astype(int)
#
#             for center in centers:
#                 image = cv2.circle(image, center, 10, (0, 0, 255), -1)
#
#             image = cv2.circle(image, tuple(avg_center), 10, (0, 255, 0), -1)
#
#         if windowNotSet is True:
#             cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
#             windowNotSet = False
#
#         cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
#         k = cv2.waitKey(1) & 0xff
#         if k == ord('q') or k == 27:
#             break
#
#     cap.release()


