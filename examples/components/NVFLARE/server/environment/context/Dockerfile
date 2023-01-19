ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:22.12-py3
FROM ${PYTORCH_IMAGE}

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install nvflare==2.2.3

WORKDIR /workspace/
RUN git clone https://github.com/NVIDIA/NVFlare.git

# Install dependencies missing in this container
ADD requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt
