FROM python:3.8-slim-buster

WORKDIR /app

COPY ./.env .

COPY ./models .

COPY _sd_generic/generic.py .

CMD [ "python", "-u", "generic.py" ]