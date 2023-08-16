FROM wayfarer-sd-base

WORKDIR /app

COPY charsheet/*.py .

COPY download_controlnet_openpose.py .

COPY .env .

CMD [ "python", "-u", "portrait.py" ]