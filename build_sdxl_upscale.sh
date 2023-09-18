#!/bin/bash

source .env

docker builder build --no-cache \
    --build-arg SDXL_MODEL_PATH=$SDXL_MODEL_PATH \
    --build-arg SDXL_REFINER_PATH=$SDXL_REFINER_PATH \
    -f ./_sdxl_upscale/Dockerfile .