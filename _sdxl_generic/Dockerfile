FROM python:3.8-slim-buster

WORKDIR /app

COPY ./.env .

COPY ./models/ ./models/

COPY ./requirements.txt .

COPY _sdxl_generic/generic.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "-u", "generic.py" ]