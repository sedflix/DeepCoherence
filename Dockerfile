FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install keras

WORKDIR  /src/
CMD bash
