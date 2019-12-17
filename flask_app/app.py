import time

import cv2
from flask import Flask, render_template, Response

from video_stream import VideoStreamWidget
from RetinaFace.test import detect_face


app = Flask(__name__)

video_src = 'rtsp://admin:Tuan7110@192.168.1.64:554/ch1/main/av_stream' # rtsp://admin:Tuan7110@192.168.1.64:554/ch1/main/av_stream
video_stream_widget = VideoStreamWidget(video_src, flip=True)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    count_frame = 0
    last = 0
    fps = 0
    while True:
        frame = video_stream_widget.get_next_frames()
        if frame is not None:
            frame = detect_face(frame)

            count_frame += 1
            if time.time() - last >= 1:
                fps = count_frame / (time.time() - last)
                count_frame = 0
                last = time.time()

            cv2.putText(frame, f'FPS: {fps}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
            frame = cv2.imencode('.jpg', frame)[1].tostring()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5005)
