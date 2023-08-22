FROM python:3.8-slim-buster

# setup
ARG SD_MODEL_PATH

ARG SD_MODEL_OPENPOSE

ARG OPENPOSE_PORTRAIT

WORKDIR /app

COPY ./.env .

# copy files
RUN mkdir ./models/

COPY ${SD_MODEL_PATH} ./models/

COPY ${SD_MODEL_OPENPOSE} ./models/

RUN mkdir ./img/

COPY ${OPENPOSE_PORTRAIT} ./img/

COPY ./requirements.txt .

COPY _sd_charsheet/charsheet.py .

# install requirements
RUN pip install --no-cache-dir -r requirements.txt

# run server
CMD [ "python", "-u", "charsheet.py" ]