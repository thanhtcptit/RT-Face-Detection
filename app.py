import cv2
from flask import Flask, render_template, Response

from video_stream import VideoStreamWidget
from RetinaFace.test import detect_face


app = Flask(__name__)

stream_link = 'http://192.168.1.6:8080/video'
video_stream_widget = VideoStreamWidget(stream_link)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while True:
        frame = video_stream_widget.get_next_frames()
        if frame is not None:
            frame = detect_face(frame)
            frame = cv2.imencode('.jpg', frame)[1].tostring()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5005)
