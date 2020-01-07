#!/bin/bash
adduser -u $CURRENT_UID host_user --no-create-home --disabled-password --gecos ""
export PYTHONPATH=$PWD
if [[ $APP = "0" ]]
then
    su host_user -c "python3 stream_app/app.py config/streaming.json"
else
    su host_user -c "python3 demo_app/app.py config/demo.json"
fi