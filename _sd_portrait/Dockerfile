FROM python:3.8-slim-buster

WORKDIR /app

COPY ./.env .

COPY ./models/ ./models/

COPY ./img/ ./img/

COPY ./requirements.txt .

COPY _sd_charsheet/charsheet.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "-u", "charsheet.py" ]