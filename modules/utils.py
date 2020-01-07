import os
import json
import codecs
import base64
import requests
import subprocess
from io import BytesIO

import cv2
import imageio
import skimage
import numpy as np
from PIL import Image


def load_json(file_path):
    with codecs.open(file_path, 'r') as f:
        return json.load(f)


def append_json(f, data, close=False):
    """Append to an opened stream `f`"""
    assert not f.closed, "should only use this function when f is still opened"
    assert f.mode.startswith("a"), "should only use this function to append"
    for item in data:
        strs = json.dumps(item)
        f.write(str(strs) + '\n')
    if close and not f.closed:
        f.close()


def load_gif(gif_path):
    gif = imageio.mimread(gif_path)
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
    return imgs


def get_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except Exception as e:
        print(e)
        return None


def build_video_from_images(image_dir, output_path,
                            image_format='img%01d.png'):
    cmd = (f'ffmpeg -i {image_dir}/{image_format} -vcodec libx264 '
           f'-y {output_path}')
    run_command(cmd)


def encode_base64_from_file(image_path):
    return base64.b64encode(open(image_path, 'rb').read()).decode()


def encode_based64(image):
    flag, frame = cv2.imencode('.jpg', image)
    return base64.b64encode(frame).decode()


def decode_base64(base64_string):
    bytedata = base64.b64decode(base64_string)
    buffer = np.frombuffer(bytedata, dtype=np.uint8)
    img = cv2.imdecode(buffer, flags=1)
    return img


def maintain_aspect_ratio_resize(image, width=None, height=None,
                                 inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def distance(org_vector, target_vector, distance_metric=0):
    assert len(org_vector.shape) == len(target_vector.shape) == 2, \
        'Vector must be 2-dim'
    if distance_metric == 0:
        # Euclidian distance
        org_vector = org_vector / \
            np.linalg.norm(org_vector, axis=1, keepdims=True)
        target_vector = target_vector / \
            np.linalg.norm(target_vector, axis=1, keepdims=True)
        diff = np.subtract(org_vector, target_vector)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Cosine similarity distance
        dot = np.sum(np.multiply(org_vector, target_vector), axis=1)
        norm = np.linalg.norm(org_vector, axis=1) * \
            np.linalg.norm(target_vector, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist


def run_command(command):
    p = subprocess.Popen(command, shell=True)
    (output, err) = p.communicate()
    return output
