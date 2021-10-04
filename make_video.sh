#!/usr/bin/env bash

echo -n "Give prefix: "
read -r prefix

ffmpeg -r 4 -f image2 -s 1920x1080 -i videos/${prefix}_%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/${prefix}.mp4
rm -r videos/${prefix}*.png
