#!/bin/bash

source .env

docker buildx build --no-cache \
    --build-arg "FLUX_MODEL_PATH=$FLUX_MODEL_PATH" \
    -f ./_flux_generic/Dockerfile .