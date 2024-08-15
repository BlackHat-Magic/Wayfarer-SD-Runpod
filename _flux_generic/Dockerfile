FROM python:3.8.19-slim-bullseye

# setup
ARG FLUX_GENERIC_MODEL_PATHNERIC_MODEL_PATH
ARG HUGGINGFACE_API_TOKENCE_API_TOKEN
ARG S3_ACCESS_KEY
ARG S3_SECRET_KEY
ARG S3_BUCKET_NAME
ARG S3_REGION_NAME
ARG S3_ENDPOINT_URL

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

WORKDIR /app

COPY ./.env .

# copy files
COPY ./requirements.txt .

COPY _flux_generic/download.py .

COPY _flux_generic/generic.py .

RUN pip install --no-cache-dir -r requirements.txt

RUN python download.py

CMD [ "python", "-u", "generic.py" ]