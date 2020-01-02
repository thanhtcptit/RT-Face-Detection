import os
import sys
import glob
import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.face_model import align_face
from modules.utils import build_video_from_images


def draw_detection_results(img, bboxs, landmarks, identities=None):
    for i in range(len(bboxs)):
        box = bboxs[i].astype(np.int)
        color = (0, 0, 255)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                      color, 2)
        if landmarks is not None:
            landmark = landmarks[i].astype(np.int)
            for l in range(landmark.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(img, (landmark[l][0], landmark[l][1]),
                           1, color, 2)
        if identities is not None and identities[i] is not None and \
                'UNK' not in identities[i]:
            cv2.putText(img, identities[i],
                        (box[0], box[3] + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255),
                        2, cv2.LINE_AA)
    return img


def draw_person_info(img, metadata, identities):
    data_dir = '/media/disk2/thanhtc/o2o/data/demo/images'
    img_h, img_w, _ = img.shape
    panel_img = np.ones(shape=(img_h, 620, 3), dtype=np.uint8) * 255

    start_x = 70
    start_y = 70
    margin = 30
    ava_img_shape = (130, 130)

    identities = [i for i in identities if i is not None]
    identities = sorted(identities)

    for i, identity in enumerate(identities):
        if 'UNK' in identity:
            continue
        ava_img_path = os.path.join(data_dir, f'{identity}/{identity}_1.jpg')
        ava_img = cv2.resize(cv2.imread(ava_img_path), ava_img_shape)
        name = metadata[identity]['name']
        fb = metadata[identity]['FB']

        x0 = start_x
        x1 = x0 + ava_img_shape[0]
        y0 = start_y + i * ava_img_shape[1] + i * margin
        y1 = y0 + ava_img_shape[1]
        panel_img[y0: y1, x0: x1, :] = ava_img

        cv2.putText(panel_img, f'ID: {identity}', (x1 + 20, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                    1, cv2.LINE_AA)
        cv2.putText(panel_img, name, (x1 + 20, y0 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                    1, cv2.LINE_AA)
        cv2.putText(panel_img, fb, (x1 + 20, y0 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    1, cv2.LINE_AA)

    return panel_img


def visualize_face_detection(model, img, bbox_thresh=0.8):
    bboxs, landmarks = model.detect_face(img, bbox_thresh=bbox_thresh)
    detection_img = draw_detection_results(img.copy(), bboxs, landmarks)
    return detection_img


def visualize_face_align(model, img, bbox_thresh=0.8):
    bboxs, landmarks = model.detect_face(img, bbox_thresh=bbox_thresh)

    detection_img = draw_detection_results(img.copy(), bboxs, landmarks)
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


def visualize_face_recognition(model, image, tracker=None, bbox_thresh=0.8,
                               dist_thresh=1.0, output_size=(640, 480),
                               metadata=None, viz_landmark=False,
                               viz_info=False):
    if output_size is None:
        detection_image = image
    else:
        detection_image = cv2.resize(image, output_size)

    bboxs, landmarks, identities = model.predict_identity(
        detection_image, bbox_thresh, dist_thresh)

    if bboxs is None:
        return None, None

    if tracker is not None:
        identities = tracker.update(bboxs, landmarks, identities)

    if metadata is not None and not viz_info:
        for i in range(len(bboxs)):
            if identities[i] is not None and identities[i] in metadata:
                identities[i] = metadata[identities[i]]['name']

    detection_image = draw_detection_results(
        detection_image, bboxs, landmarks if viz_landmark else None,
        identities)

    if viz_info:
        panel_img = draw_person_info(detection_image, metadata,
                                     identities)
        detection_image = np.hstack((detection_image, panel_img))
    return detection_image, identities


def build_demo_video(video_src, model, tracker, metadata):
    video_dir, video_file = os.path.split(video_src)
    video_name = os.path.splitext(video_file)[0]

    tmp_dir = '/media/disk2/thanhtc/o2o/data/debug/tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    video_capture = cv2.VideoCapture(video_src)
    n_frames = 0
    start_time = time.time()
    max_frames = -1

    while True:
        if video_capture.isOpened():
            status, frame = video_capture.read()
            if frame is not None:
                frame, identities = visualize_face_recognition(
                    model, frame, tracker=tracker, output_size=None,
                    bbox_thresh=0.3, dist_thresh=0.9, metadata=metadata)
                if n_frames % 30 == 0:
                    panel_img = draw_person_info(frame, metadata, identities)
                frame = np.hstack((frame, panel_img))
                cv2.imwrite(os.path.join(tmp_dir, f'img{n_frames}.png'), frame)
                n_frames += 1
            if not status or (max_frames > 0 and n_frames >= max_frames):
                break

    print(f'Time: {time.time() - start_time} - Num frames: {n_frames} '
          f'- Fps: {n_frames / (time.time() - start_time)}')

    output_video = os.path.join(video_dir, video_name + '_recog.mp4')
    if os.path.exists(output_video):
        os.remove(output_video)

    build_video_from_images(tmp_dir, output_video)
