import os
import json
import requests
import subprocess
from io import BytesIO

import numpy as np
from PIL import Image


def append_json(f, data, close=False):
    """Append to an opened stream `f`"""
    assert not f.closed, "should only use this function when f is still opened"
    assert f.mode.startswith("a"), "should only use this function to append"
    for item in data:
        strs = json.dumps(item)
        f.write(str(strs) + '\n')
    if close and not f.closed:
        f.close()


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
    p = subprocess.Popen(cmd, shell=True)
    (output, err) = p.communicate()
    return output


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
