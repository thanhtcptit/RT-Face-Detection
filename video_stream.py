import time
from threading import Thread

import cv2


class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.count_frame = 0
        self.last = 0
        self.fps = 0

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()

    def get_next_frames(self):
        if self.status:
            self.count_frame += 1
            if time.time() - self.last >= 1:
                self.fps = self.count_frame / (time.time() - self.last)
                self.count_frame = 0
                self.last = time.time()

            frame = self.maintain_aspect_ratio_resize(
                self.frame, width=640)
            cv2.putText(frame, f'FPS: {self.fps}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            self.status = None
            return frame
        return None

    def show_frame(self):
        frame = self.get_next_frames()
        if frame is not None:
            cv2.imshow('IPCam', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def maintain_aspect_ratio_resize(self, image, width=None, height=None,
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


if __name__ == '__main__':
    stream_link = 'http://192.168.1.6:8080/video'
    video_stream_widget = VideoStreamWidget(stream_link)
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError as ae:
            print(ae)
            pass
