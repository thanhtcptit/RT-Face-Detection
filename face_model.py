import os

import cv2
import numpy as np
from skimage import transform
from nsds.common import Params

from vector_search import VectorSearch
from retinaface.model import RetinaFace
from arcface.model import Face2VecModel


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
    face_img, pad_box = crop_face_and_pad(img, bbox, padding)
    landmark = normalize_landmark(pad_box, landmark)

    return rotate_face_center(face_img, landmark, output_size)


class FaceModelWrapper:
    def __init__(self, params):
        detector_params = params['detector']
        detector_name = detector_params.pop('name')
        self._detector = get_model(
            detector_name, **detector_params)

        featurizer_params = params['featurizer']
        featurizer_name = featurizer_params.pop('name')
        self._featurizer = get_model(
            featurizer_name, **featurizer_params)

        vector_search_params = params['vector_search']
        self._vector_search = VectorSearch(**vector_search_params)

    @staticmethod
    def from_file(cfg_path):
        params = Params.from_file('config.json')
        return FaceModelWrapper(params)

    def detect_face(self, img):
        bboxs, landmarks = self._detector.detect(img)
        return bboxs, landmarks

    def detect_and_align(self, img, output_size=(112, 112),
                         padding=10, mode='many'):
        assert mode in ['single', 'many']
        if not isinstance(output_size, tuple):
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            elif isinstance(output_size, list):
                output_size = tuple(output_size)
            else:
                raise ValueError('output_size must be tuple')

        bboxs, landmarks = self.detect_face(img)
        if len(bboxs) == 0 or (len(bboxs) > 1 and mode == 'single'):
            return None
        aligned_faces = []
        for bbox, landmark in zip(bboxs, landmarks):
            aligned_faces.append(
                align_face(img, bbox, landmark, output_size, padding))
        return aligned_faces

    def detect_and_extract_embedding(self, img, output_size=(112, 112),
                                     padding=10, mode='many', align=True):
        assert mode in ['single', 'many']
        bboxs, landmarks = self.detect_face(img)
        if len(bboxs) == 0 or (len(bboxs) > 1 and mode == 'single'):
            return [None] * 3
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

    def predict_identity(self, img, threshold=1.0):
        bboxs, _, embeddings = self.detect_and_extract_embedding(img)
        if embeddings is None:
            return None, None

        identities = []
        for emb in embeddings:
            ids, scores = self._vector_search.search(emb)
            pred_id = list(ids)[0].split('_')[0]
            pred_dist = scores[0]
            if pred_dist <= threshold:
                identities.append(pred_id)
            else:
                identities.append(None)
        return bboxs, identities

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
                if aligned_faces is None:
                    continue
                aligned_face = aligned_faces[0]
                output_img_path = os.path.join(output_folder_dir, filename)
                cv2.imwrite(output_img_path, aligned_face)

    def extract_face_embeddings_dataset(self, data_dir, output_path,
                                        output_size=(112, 112), padding=10,
                                        mode='single', align=True):
        assert mode in ['single', 'many']
        if os.path.exists(output_path):
            os.remove(output_path)

        f = codecs.open(output_path, 'a+')
        for folder in tqdm(os.listdir(data_dir)):
            folder_dir = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_dir):
                continue
            for filename in os.listdir(folder_dir):
                img = cv2.imread(os.path.join(folder_dir, filename))
                _, _, emb = self.detect_and_extract_embedding(img, mode=mode)
                if emb is None:
                    continue
                embs_data = [{'key': filename, 'embedding': emb[0].tolist()}]
                append_json(f, embs_data)
        f.close()


if __name__ == '__main__':
    params = Params.from_file('config.json')
    face_model = FaceModelWrapper(params)
    img = cv2.imread('data/debug/phoebe.jpg')
    _, identities = face_model.predict_identity(img)
    print(identities)