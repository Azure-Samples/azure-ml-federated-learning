FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20230412.v1

COPY vcpkg.json .
COPY SymmetricPSI /SymmetricPSI

# Update python and install dependencies
RUN conda update conda && conda install python=3.10.0
RUN pip install azureml-core==1.47.0 \
                azure-keyvault==4.2.0 \
                azureml-mlflow==1.48.0 \
                pandas==1.5.2 \
                redis==4.5.1 \
                numpy==1.24.2

# Install vcpkg and dependencies
RUN apt-get update && apt-get install -y zip pkg-config build-essential cmake
RUN git clone https://github.com/microsoft/vcpkg.git /vcpkg &&\
    chmod a+x /vcpkg/bootstrap-vcpkg.sh && /bin/bash /vcpkg/bootstrap-vcpkg.sh &&\
    /vcpkg/vcpkg install --triplet=x64-linux --x-buildtrees-root=/vcpkg/buildtrees --x-install-root=/vcpkg/installed --x-packages-root=/vcpkg/packages

# Install APSI
RUN git clone https://github.com/microsoft/APSI.git /APSI &&\
    cd /APSI && mkdir build &&\
    cd /APSI/build &&\
    cmake .. -DAPSI_USE_ASM=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake &&\
    make -j$(nproc) &&\
    make install

# Install SymmetricPSI and create Python bindings
RUN cd SymmetricPSI && mkdir build && cd build &&\
    cmake .. -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake -DAPSI_ROOT=/APSI/build -DVCPKG_TARGET_TRIPLET=x64-linux -DCMAKE_BUILD_TYPE=Release &&\
    make -j$(nproc)
