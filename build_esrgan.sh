#!/bin/bash

source .env

docker builder build --no-cache \
    -f ./_esrgan/Dockerfile .