FROM python:3.9-slim-bullseye

# setup
ARG SDXL_UPSCALE_TILE_CONTROLNET_MODEL_PATH
ARG SDXL_UPSCALE_SCHEDULER_MODEL_PATH
ARG SDXL_UPSCALE_VAE_MODEL_PATH
ARG SDXL_UPSCALE_BASE_MODEL_PATH
ARG S3_ACCESS_KEY
ARG S3_SECRET_KEY
ARG S3_BUCKET_NAME
ARG S3_REGION_NAME
ARG S3_ENDPOINT_URL


RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

WORKDIR /app

COPY ./.env .

# copy files
COPY ./requirements.txt .

COPY ./_sdxl_upscale/download.py .

COPY ./_sdxl_upscale/upscale.py .

RUN pip install --no-cache-dir -r requirements.txt

RUN python download.py

CMD [ "python", "-u", "upscale.py" ]
