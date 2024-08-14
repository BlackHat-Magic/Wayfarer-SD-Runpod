#!/bin/bash

source .env

docker buildx build --no-cache \
    --build-arg "FLUX_MODEL_PATH=$FLUX_MODEL_PATH" \
    --build-arg "HUGGINGFACE_API_TOKEN=$HUGGINGFACE_API_TOKEN" \
    -f ./_flux_generic/Dockerfile .