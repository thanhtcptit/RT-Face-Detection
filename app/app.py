import os
import time
import argparse

import cv2
from flask import Flask, render_template, Response

from nsds.common import Params
from modules.video_stream import VideoStreamWidget
from modules.face_model import FaceModelWrapper
from modules.visualize_detection import visualize_face_detection, \
    visualize_face_recognition

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('-p', '--port', type=int, default=5005)
    return parser.parse_args()


args = parse_args()
config = Params.from_file(args.config_path)

stream_config = config.pop('streaming')
vs = VideoStreamWidget(stream_config['src'], flip=stream_config['flip'])

model = FaceModelWrapper(config)

app = Flask(__name__)

vs.start()
time.sleep(0.2)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    count_frame = 0
    last = time.time()
    fps = 0

    while True:
        frame = vs.get_next_frames()
        if frame is None:
            continue

        detection_image = visualize_face_recognition(model, frame)
        # detection_image = visualize_face_detection(model, frame)

        count_frame += 1
        if time.time() - last >= 1:
            fps = count_frame / (time.time() - last)
            count_frame = 0
            last = time.time()

        cv2.putText(detection_image, 'FPS: %.3f' % fps, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)

        flag, frame = cv2.imencode('.jpg', detection_image)
        if not flag:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) +
               b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=args.port)
