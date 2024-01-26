#!/bin/bash

source .env

docker builder build --no-cache \
    --build-arg "SD_UPSCALE_PATH=$SD_UPSCALE_PATH" \
    --build-arg "TILE_CN_MODEL_PATH=$TILE_CN_MODEL_PATH" \
    -f ./_sd_upscale/Dockerfile .