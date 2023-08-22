#!/bin/bash

source .env

docker builder build --no-cache \
    --build-arg SD_MODEL_PATH=$SD_MODEL_PATH \
    --build-arg SD_MODEL_OPENPOSE=$SD_MODEL_OPENPOSE \
    --build-arg OPENPOSE_PORTRAIT=$OPENPOSE_PORTRAIT \
    -f ./_sd_portrait/Dockerfile .