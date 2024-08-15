FROM python:3.8-slim-buster

# setup
ARG FLUX_SCHNELL_MODEL_PATH
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

COPY _schnell_generic/download.py .

COPY _schnell_generic/generic.py .

RUN pip install --no-cache-dir -r requirements.txt

RUN python download.py

CMD [ "python", "-u", "generic.py" ]