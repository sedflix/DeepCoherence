FROM tensorflow/tensorflow:latest-gpu

RUN pip install keras

WORKDIR  /src/
CMD bash
