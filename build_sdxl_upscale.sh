#!/bin/bash

source .env

docker builder build --no-cache \
    --build-arg "SDXL_UPSCALE_TILE_CONTROLNET_MODEL_PATH=$SDXL_UPSCALE_TILE_CONTROLNET_MODEL_PATH" \
    --build-arg "SDXL_UPSCALE_VAE_MODEL_PATH=$SDXL_UPSCALE_VAE_MODEL_PATH" \
    --build-arg "SDXL_UPSCALE_BASE_MODEL_PATH=$SDXL_UPSCALE_BAE_MODEL_PATH" \
    -f ./_sdxl_upscale/Dockerfile .