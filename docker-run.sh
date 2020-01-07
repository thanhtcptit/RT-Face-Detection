#!/bin/bash
export UI_PORT=5005
export CAMERA_PORT=8080
export APP=${1:-0}


nvidia-docker run --rm --name face_recogniton \
-e CURRENT_UID=$(id -u) \
-e UI_PORT=$UI_PORT \
-e CAMERA_PORT=$CAMERA_PORT \
-e APP=$APP \
-p $UI_PORT:$UI_PORT \
-p $CAMERA_PORT:$CAMERA_PORT \
--mount type=bind,source="$(pwd)"/data,target=/project/data \
--mount type=bind,source="$(pwd)"/config,target=/project/config \
--mount type=bind,source="$(pwd)"/notebooks,target=/project/notebooks \
face_recognition:dev