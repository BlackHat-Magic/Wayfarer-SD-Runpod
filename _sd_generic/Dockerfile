FROM python:3.8-slim-buster

# setup
ARG SD_MODEL_PATH

WORKDIR /app

COPY ./.env .

# copy files
RUN mkdir ./models/

COPY ${SD_MODEL_PATH} ./models/

COPY ./requirements.txt .

COPY _sd_generic/generic.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "-u", "generic.py" ]