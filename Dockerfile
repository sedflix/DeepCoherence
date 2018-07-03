FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install wget unzip
RUN pip install keras

WORKDIR  /src/
CMD bash
