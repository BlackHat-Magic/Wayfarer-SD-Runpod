FROM python:3.8-slim-buster

WORKDIR /app

COPY ./.env .

COPY ./models .

COPY _sd_charsheet/charsheet.py .

CMD [ "python", "-u", "charsheet.py" ]