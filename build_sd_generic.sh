#!/bin/bash

source .env

docker builder build --no-cache \
    --build-arg SD_MODEL_PATH=$SD_MODEL_PATH \
    -f ./_sd_generic/Dockerfile .