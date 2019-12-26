#!/bin/bash
adduser -u $CURRENT_UID host_user --no-create-home --disabled-password --gecos ""
export PYTHONPATH=$PWD
su host_user -c "python3 stream_app/app.py config/streaming.json"