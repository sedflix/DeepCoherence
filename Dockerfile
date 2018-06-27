FROM tensorflow/tensorflow:latest-gpu

RUN pip3 install keras

WORKDIR  /src/
CMD bash
