#!/bin/bash

source .env

docker builder build --no-cache \
    --build-arg "SDXL_MODEL_PATH=$SDXL_MODEL_PATH" \
    -f ./_sdxl_upscale/Dockerfile .