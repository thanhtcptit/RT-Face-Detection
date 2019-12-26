FROM tensorflow/tensorflow:1.13.2-gpu-py3

RUN apt-get update && \
    apt-get install -y build-essential libsm6 libxext6 libxrender-dev screen

COPY requirements.txt /project/

RUN pip3 install -r /project/requirements.txt

COPY lib/libnvrtc.so.10.0.130 /usr/local/cuda/lib64/libnvrtc.so.10.0

COPY ml_best_practice /project/ml_best_practice

RUN pip3 install -e /project/ml_best_practice/nsds

COPY stream_app /project/stream_app

COPY modules /project/modules

COPY run.py /project/

COPY docker-entrypoint.sh /project/

WORKDIR /project/modules/retinaface

RUN make

WORKDIR /project

EXPOSE $STREAM_PORT

EXPOSE $CAMERA_PORT

ENTRYPOINT ["bash", "docker-entrypoint.sh"]