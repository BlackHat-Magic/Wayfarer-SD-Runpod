#!/bin/bash

source .env

docker buildx build --no-cache \
    --build-arg "FLUX_GENERIC_MODEL_PATH=$FLUX_GENERIC_MODEL_PATH" \
    --build-arg "HUGGINGFACE_API_TOKEN=$HUGGINGFACE_API_TOKEN" \
    --build-arg "S3_ACCESS_KEY=$S3_ACCESS_KEY" \
    --build-arg "S3_SECRET_KEY=$S3_SECRET_KEY" \
    --build-arg "S3_BUCKET_NAME=$S3_BUCKET_NAME" \
    --build-arg "S3_REGION_NAME=$S3_REGION_NAME" \
    --build-arg "S3_ENDPOINT_URL=$S3_ENDPOINT_URL" \
    -f ./_flux_generic/Dockerfile .