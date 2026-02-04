#!/usr/bin/env bash
apt-get update
apt-get install -y ffmpeg
uvicorn main:app --host 0.0.0.0 --port $PORT
