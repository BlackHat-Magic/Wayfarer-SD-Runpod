FROM wayfarer-sd-base

WORKDIR /app

COPY charsheet/*.py .

COPY download_controlnet_openpose.py .

COPY ./.env .

RUN download_controlnet_openpose.py

CMD [ "python", "-u", "/charsheet.py" ]