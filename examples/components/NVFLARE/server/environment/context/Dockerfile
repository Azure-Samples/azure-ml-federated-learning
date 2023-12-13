ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:23.11-py3
FROM ${PYTORCH_IMAGE}

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools

RUN mkdir /tmp/nvflare && \
    cd /tmp/nvflare && \
    git clone https://github.com/vonodiripsa/NVFlare.git && \
    cd NVFlare && \
    git checkout redis-driver-rewrite && \
    python3 -m pip install . && \
    cd /tmp && \
    rm -rf /tmp/nvflare

# RUN python3 -m pip install nvflare==2.2.3

RUN python3 -m pip install azureml-mlflow==1.48.0
RUN python3 -m pip install redis==4.5.1
