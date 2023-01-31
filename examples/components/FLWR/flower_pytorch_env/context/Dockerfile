ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:22.09-py3
FROM ${PYTORCH_IMAGE}

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools

# Install dependencies missing in this container
ADD requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt
