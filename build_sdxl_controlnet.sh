#!/bin/bash

source .env

docker buildx build --no-cache \
    --build-arg "SDXL_MODEL_PATH=$SDXL_MODEL_PATH" \
    --build-arg "SDXL_REFINER_PATH=$SDXL_REFINER_PATH" \
    --build-arg "CANNY_CN_MODEL_PATH=$CANNY_CN_MODEL_PATH" \
    --build-arg "DEPTH_CN_MODEL_PATH=$DEPTH_CN_MODEL_PATH" \
    --build-arg "OPENPOSE_CN_MODEL_PATH=$OPENPOSE_CN_MODEL_PATH" \
    -f ./_sdxl_generic/Dockerfile .