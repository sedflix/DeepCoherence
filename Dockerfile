FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y wget unzip
RUN pip install keras

WORKDIR  /src/
CMD bash
