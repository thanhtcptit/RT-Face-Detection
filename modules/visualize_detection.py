import os
import sys
import glob
import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.face_model import align_face


def visualize_face_detection(model, img):
    bboxs, landmarks = model.detect_face(img)

    if bboxs is not None:
        detect_img = img.copy()
        for i in range(bboxs.shape[0]):
            box = bboxs[i].astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(detect_img, (box[0], box[1]), (box[2], box[3]),
                          color, 2)
            if landmarks is not None:
                landmark = landmarks[i].astype(np.int)
                for l in range(landmark.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(detect_img, (landmark[l][0], landmark[l][1]),
                               1, color, 2)
    return detect_img, bboxs, landmarks


def visualize_face_align(model, img):
    detection_img, bboxs, landmarks = visualize_face_detection(
        model, img)
    img_h, img_w, _ = img.shape

    figure = plt.figure(figsize=(8, 8))
    num_cols = len(bboxs) if len(bboxs) < 3 else 3
    num_rows = np.ceil(len(bboxs) / num_cols) if len(bboxs) else 0
    ind = 0
    pad = 10

    for bbox, landmark in zip(bboxs, landmarks):
        aligned_face = align_face(img, bbox, landmark)

        ind += 1
        ax = figure.add_subplot(num_rows, num_cols, ind)
        ax.imshow(aligned_face[:, :, ::-1])
        ax.tick_params(axis='both', which='both', bottom=False, left=False,
                            labelbottom=False, labelleft=False)
    faces_img_path = '/media/disk2/thanhtc/o2o/data/debug/tmp.jpg'
    plt.savefig(faces_img_path, bbox_inches='tight', pad_inches=0.4)
    plt.close()

    figure = plt.figure(figsize=(16, 16))
    ax = figure.add_subplot(1, 2, 1)
    ax.imshow(detection_img[:, :, ::-1])
    ax.tick_params(axis='both', which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax = figure.add_subplot(1, 2, 2)
    ax.imshow(cv2.imread(faces_img_path)[:, :, ::-1])
    ax.tick_params(axis='both', which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    plt.savefig(faces_img_path, bbox_inches='tight', pad_inches=0.4)
    plt.close()
    return cv2.imread(faces_img_path)


def visualize_face_recognition(model, image, dist_thresh=1.0,
                               output_size=(640, 480)):
    detection_image = cv2.resize(image, output_size)
    bboxs, identities = model.predict_identity(detection_image, dist_thresh)
    if bboxs is None:
        return image

    for box, identity in zip(bboxs, identities):
        box = box.astype(np.int)
        color = (0, 0, 255)
        cv2.rectangle(detection_image, (box[0], box[1]), (box[2], box[3]),
                      color, 2)

        if identity:
            cv2.putText(detection_image, identity,
                        (box[0], box[3] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                        1, cv2.LINE_AA)
    return detection_image
