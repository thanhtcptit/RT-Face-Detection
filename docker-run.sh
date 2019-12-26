#!/bin/bash
export STREAM_PORT=5005
export CAMERA_PORT=8080


nvidia-docker run --rm --name face_recogniton \
-e CURRENT_UID=$(id -u) \
-e STREAM_PORT=$STREAM_PORT \
-e CAMERA_PORT=$CAMERA_PORT \
-p $STREAM_PORT:$STREAM_PORT \
-p $CAMERA_PORT:$CAMERA_PORT \
--mount type=bind,source="$(pwd)"/data,target=/project/data \
--mount type=bind,source="$(pwd)"/config,target=/project/config \
--mount type=bind,source="$(pwd)"/notebooks,target=/project/notebooks \
face_recognition:dev