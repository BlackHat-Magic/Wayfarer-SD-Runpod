FROM python:3.8.19-slim-bullseye

# setup
ARG SDXL_MODEL_PATH

ARG SDXL_REFINER_PATH

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

WORKDIR /app

COPY ./.env .

# copy files
COPY ./requirements.txt .

COPY ./_sdxl_controlnet/download.py .

COPY ./_sdxl_controlnet/controlnet.py .

RUN pip install --no-cache-dir -r requirements.txt

RUN python download.py

CMD [ "python", "-u", "controlnet.py" ]
