FROM python:3.9-slim-bullseye

# setup
ARG SD_UPSCALE_PATH

ARG TILE_CN_MODEL_PATH


RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

WORKDIR /app

COPY ./.env .

# copy files
COPY ./requirements.txt .

COPY ./_sd_upscale/download.py .

COPY ./_sd_upscale/upscale.py .

RUN pip install --no-cache-dir -r requirements.txt

RUN python download.py

CMD [ "python", "-u", "upscale.py" ]
