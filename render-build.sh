#!/usr/bin/env bash
# Update package lists and install ffmpeg
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar -xJ
mv ffmpeg-*-static/ffmpeg /usr/local/bin/
pip install -r requirements.txt
