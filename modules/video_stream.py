import time
from threading import Thread

import cv2


class VideoStreamWidget(object):
    def __init__(self, src, flip=False):
        self.capture = cv2.VideoCapture(src)
        self.flip = flip
        self.status = None

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()

    def get_next_frames(self):
        if self.status:
            frame = self.maintain_aspect_ratio_resize(
                self.frame, width=640)
            if self.flip:
                frame = cv2.flip(frame, -1)
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

    def release(self):
        self.capture.release()


if __name__ == '__main__':
    stream_link = 'rtsp://admin:Tuan7110@192.168.1.64:554/ch1/main/av_stream'
    video_stream_widget = VideoStreamWidget(stream_link)
    video_stream_widget.start()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError as ae:
            pass
