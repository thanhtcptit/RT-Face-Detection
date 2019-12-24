import time

import cv2
from flask import Flask, render_template, Response

from video_stream import VideoStreamWidget
from face_model import FaceModelWrapper
from visualize_detection import visualize_face_detection, \
    visualize_face_recognition


model = FaceModelWrapper.from_file('config.json')

app = Flask(__name__)

# rtsp://admin:Tuan7110@192.168.1.64:554/ch1/main/av_stream
video_src = 'rtsp://admin:Tuan7110@192.168.1.64:554/ch1/main/av_stream'
vs = VideoStreamWidget(video_src, flip=True)
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
    app.run(host='0.0.0.0', threaded=True, port=5005)
