FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Install dependencies missing in this container
ADD requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt
