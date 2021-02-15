FROM python:3-slim

RUN apt update && apt install build-essential python3-numpy python3-scipy -y 

COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY app/ /app
WORKDIR /app
