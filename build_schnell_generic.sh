#!/bin/bash

source .env

docker buildx build --no-cache \
    --build-arg "FLUX_SCHNELL_MODEL_PATH=$FLUX_SCHNELL_MODEL_PATH" \
    -f ./_schnell_generic/Dockerfile .