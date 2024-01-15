FROM python:3.8-slim-buster

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

COPY ./_sdxl_upscale/download.py .

COPY ./_sdxl_upscale/upscale.py .

RUN pip install "git+https://github.com/BlackHat-Magic/BasicSR-Fixed-transforms#egg=basicsr"

RUN pip install --no-cache-dir -r requirements.txt

# RUN python download.py

RUN mkdir weights

RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights

CMD [ "python", "-u", "upscale.py" ]
