#!/bin/bash

source .env

docker builder build --no-cache \
    --build-arg "SDXL_UPSCALE_TILE_CONTROLNET_MODEL_PATH=$SDXL_UPSCALE_TILE_CONTROLNET_MODEL_PATH" \
    --build-arg "SDXL_UPSCALE_SCHEDULER_MODEL_PATH=$SDXL_UPSCALE_SCHEDULER_MODEL_PATH" \
    --build-arg "SDXL_UPSCALE_VAE_MODEL_PATH=$SDXL_UPSCALE_VAE_MODEL_PATH" \
    --build-arg "SDXL_UPSCALE_BASE_MODEL_PATH=$SDXL_UPSCALE_BAE_MODEL_PATH" \
    --build-arg "S3_ACCESS_KEY=$S3_ACCESS_KEY" \
    --build-arg "S3_SECRET_KEY=$S3_SECRET_KEY" \
    --build-arg "S3_BUCKET_NAME=$S3_BUCKET_NAME" \
    --build-arg "S3_REGION_NAME=$S3_REGION_NAME" \
    --build-arg "S3_ENDPOINT_URL=$S3_ENDPOINT_URL" \
    -f ./_sdxl_upscale/Dockerfile .