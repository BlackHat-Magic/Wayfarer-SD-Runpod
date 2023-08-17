FROM python:3.8-slim-buster

WORKDIR /app

COPY ./.env .

COPY ./models .

COPY ./requirements.txt .

COPY _sd_charsheet/charsheet.py .

RUN pip install -r --no-cache-dir requirements.txt

CMD [ "python", "-u", "charsheet.py" ]