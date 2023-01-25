ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:22.12-py3
FROM ${PYTORCH_IMAGE}

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install nvflare==2.2.3
RUN python3 -m pip install azureml-mlflow==1.48.0
