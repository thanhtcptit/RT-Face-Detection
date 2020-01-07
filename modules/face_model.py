import os
import time
import codecs

import cv2
import numpy as np
from tqdm import tqdm
from skimage import transform
from nsds.common import Params

from modules.utils import append_json, run_command
from modules.simvec.vector_search import VectorSearch
from modules.retinaface.model import RetinaFace
from modules.arcface.model import Face2VecModel
from modules.simvec.client import AnnoyClient


def get_model(name, *args, **kwargs):
    name = name.lower()
    if name == 'retina':
        return RetinaFace(*args, **kwargs)
    elif name == 'arcface':
        return Face2VecModel(*args, **kwargs)
    else:
        raise ValueError(f'Unknown face detector ("{name}")')


def normalize_landmark(box, landmark):
    return np.array([[p[0] - box[0], p[1] - box[1]] for p in landmark],
                    dtype=np.float32)


def crop_face_and_pad(img, bbox, padding):
    img_h, img_w, _ = img.shape
    bbox = bbox.astype(np.int)
    x1 = max(bbox[0] - padding, 0)
    y1 = max(bbox[1] - padding, 0)
    x2 = min(bbox[2] + padding, img_w)
    y2 = min(bbox[3] + padding, img_h)
    pad_box = [x1, y1, x2, y2]
    return img[y1: y2, x1: x2, :], pad_box


def get_center_bbox(bboxs, image_shape):
    bboxs_size = []
    v_offsets = []
    h_offsets = []
    image_center = np.array(image_shape) / 2
    for box in bboxs:
        area = (box[2] - box[0]) * (box[3] - box[1])
        h_offsets.append((box[0] + box[2]) / 2 - image_center[1])
        v_offsets.append((box[1] + box[3]) / 2 - image_center[0])
        bboxs_size.append(area)

    bboxs_size = np.array(bboxs_size, dtype=np.int32)
    offsets = [np.abs(h) + np.abs(v) for h, v in zip(h_offsets, v_offsets)]
    center_index = np.argmax(bboxs_size - offsets)
    return center_index


def rotate_face_center(img, dst, output_size=(112, 112)):
    if output_size == (112, 96):
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
    elif output_size == (112, 112):
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
    elif output_size == (150, 150):
        src = np.array([
            [51.287415, 69.23612],
            [98.48009, 68.97509],
            [75.03375, 96.075806],
            [55.646385, 123.7038],
            [94.72754, 123.48763]], dtype=np.float32)
    elif output_size == (160, 160):
        src = np.array([
            [54.706573, 73.85186],
            [105.045425, 73.573425],
            [80.036, 102.48086],
            [59.356144, 131.95071],
            [101.04271, 131.72014]], dtype=np.float32)
    elif output_size == (224, 224):
        src = np.array([
            [76.589195, 103.3926],
            [147.0636, 103.0028],
            [112.0504, 143.4732],
            [83.098595, 184.731],
            [141.4598, 184.4082]], dtype=np.float32)
    else:
        raise ValueError('Wrong destionation dimension')
    tform = transform.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    if M is None:
        return cv2.resize(face_img, output_size)
    face_img = cv2.warpAffine(img, M, output_size, borderValue=0.0)
    return face_img


def align_face(img, bbox, landmark, output_size=(112, 112), padding=10):
    if not isinstance(output_size, tuple):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif isinstance(output_size, list):
            output_size = tuple(output_size)
        else:
            raise ValueError('output_size must be tuple')

    face_img, pad_box = crop_face_and_pad(img, bbox, padding)
    landmark = normalize_landmark(pad_box, landmark)

    return rotate_face_center(face_img, landmark, output_size)


class FaceModelWrapper:
    def __init__(self, params):
        if 'detector' in params:
            detector_params = params['detector']
            detector_name = detector_params.pop('name')
            self._detector = get_model(
                detector_name, **detector_params)
        else:
            self._detector = None

        if 'featurizer' in params:
            featurizer_params = params['featurizer']
            featurizer_name = featurizer_params.pop('name')
            self._featurizer = get_model(
                featurizer_name, **featurizer_params)
        else:
            self._featurizer = None

        if 'vector_search' in params:
            vector_search_params = params['vector_search']
            self._top_k = vector_search_params.pop('top_k')
            if vector_search_params.pop('run_inplace'):
                self._vector_search = VectorSearch(
                    vector_search_params['data_file'],
                    vector_search_params['dims']
                )
            else:
                print('Start separate process for annoy')
                annoy_port = vector_search_params.pop('port')
                run_command("screen -S annoy -dm "
                            f"python modules/simvec/server.py -p {annoy_port} "
                            f"-f {vector_search_params['data_file']} "
                            f"-d {vector_search_params['dims']}")
                self._vector_search = AnnoyClient(annoy_port)
        else:
            self._vector_search = None

    @staticmethod
    def from_file(cfg_path):
        params = Params.from_file(cfg_path)
        return FaceModelWrapper(params)

    def detect_face(self, img, mode='many', bbox_thresh=0.8):
        assert self._detector is not None
        assert mode in ['single', 'many', 'center']
        bboxs, landmarks = self._detector.detect(img, threshold=bbox_thresh)

        if len(bboxs) == 0 or (len(bboxs) > 1 and mode == 'single'):
            return [], []
        elif len(bboxs) > 1 and mode == 'center':
            center_bboxs_index = get_center_bbox(bboxs, img.shape[:2])
            bboxs, landmarks = [bboxs[center_bboxs_index]], \
                [landmarks[center_bboxs_index]]
        return bboxs, landmarks

    def detect_and_align(self, img, output_size=(112, 112),
                         padding=10, mode='many', bbox_thresh=0.8):
        bboxs, landmarks = self.detect_face(img, mode=mode,
                                            bbox_thresh=bbox_thresh)
        if len(bboxs) == 0:
            return []

        aligned_faces = []
        for bbox, landmark in zip(bboxs, landmarks):
            aligned_faces.append(
                align_face(img, bbox, landmark, output_size, padding))
        return aligned_faces

    def detect_and_extract_embedding(self, img, output_size=(112, 112),
                                     padding=10, mode='many', align=True,
                                     bbox_thresh=0.8):
        assert self._featurizer is not None
        bboxs, landmarks = self.detect_face(img, mode=mode,
                                            bbox_thresh=bbox_thresh)
        if len(bboxs) == 0:
            return [], [], []

        aligned_faces = []
        for bbox, landmark in zip(bboxs, landmarks):
            if align:
                face_img = align_face(img, bbox, landmark, output_size,
                                      padding)
            else:
                face_img = crop_face_and_pad(img, bbox, padding)
                face_img = cv2.resize(face_img, output_size)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = np.transpose(face_img, [2, 0, 1])
            aligned_faces.append(face_img)
        aligned_faces = np.array(aligned_faces, dtype=np.float32)
        face_embs = self._featurizer.get_feature(aligned_faces)
        return bboxs, landmarks, face_embs

    def get_similar_images(self, img, bbox_thresh=0.8, dist_thresh=1.0):
        assert self._vector_search is not None
        bboxs, landmarks, embeddings = self.detect_and_extract_embedding(
            img, mode='center', bbox_thresh=bbox_thresh)
        if len(embeddings) == 0:
            return []

        ids, scores = self._vector_search.search(embeddings[0], self._top_k)
        if len(ids) == 0:
            return None
        return ids

    def predict_identity(self, img, bbox_thresh=0.8, dist_thresh=1.0):
        assert self._vector_search is not None
        bboxs, landmarks, embeddings = self.detect_and_extract_embedding(
            img, bbox_thresh=bbox_thresh)
        if len(embeddings) == 0:
            return [], [], []

        identities = []
        for emb in embeddings:
            ids, scores = self._vector_search.search(emb, self._top_k)
            if len(ids) == 0:
                return None, None, None
            pred_id = ids[0].split('_')[0]
            pred_dist = scores[0]
            if pred_dist <= dist_thresh:
                identities.append(pred_id)
            else:
                identities.append(None)
        return bboxs, landmarks, identities

    def detect_and_align_dataset(self, data_dir, output_dir,
                                 output_size=(112, 112), padding=10):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        for folder in tqdm(os.listdir(data_dir)):
            folder_dir = os.path.join(data_dir, folder)
            output_folder_dir = os.path.join(output_dir, folder)
            os.makedirs(output_folder_dir, exist_ok=True)
            for filename in os.listdir(folder_dir):
                img = cv2.imread(os.path.join(folder_dir, filename))
                aligned_faces = self.detect_and_align(
                    img, output_size, padding)
                if len(aligned_faces) == 0:
                    continue
                aligned_face = aligned_faces[0]
                output_img_path = os.path.join(output_folder_dir, filename)
                cv2.imwrite(output_img_path, aligned_face)

    def extract_face_embeddings_dataset(self, data_dir, output_path,
                                        output_size=(112, 112), padding=10,
                                        mode='single', align=True,
                                        append_uid=False):
        if os.path.exists(output_path):
            os.remove(output_path)

        f = codecs.open(output_path, 'a+')
        for folder in tqdm(os.listdir(data_dir)):
            folder_dir = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_dir):
                continue
            for filename in os.listdir(folder_dir):
                img = cv2.imread(os.path.join(folder_dir, filename))
                _, _, emb_vectors = self.detect_and_extract_embedding(
                    img, output_size=output_size, padding=padding, mode=mode,
                    align=align)
                if len(emb_vectors) == 0:
                    continue

                if append_uid:
                    filename = f'{folder}_{filename}'
                embs_data = []
                for emb in emb_vectors:
                    embs_data.append(
                        {'key': filename, 'embedding': emb.tolist()})

                append_json(f, embs_data)
        f.close()


if __name__ == '__main__':
    params = Params.from_file('config/streaming.json')
    face_model = FaceModelWrapper(params)
    img = cv2.imread('data/debug/phoebe.jpg')
    time.sleep(1)
    _, identities = face_model.predict_identity(img)
    print(identities)
