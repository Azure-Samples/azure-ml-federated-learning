# start from this base image because it already has the Azure CLI
FROM mcr.microsoft.com/azure-cli:latest
# install powershell as per https://docs.microsoft.com/en-us/powershell/scripting/install/install-alpine?view=powershell-7.2
RUN apk update \
    # Install the requirements
    && apk add --no-cache \
    ca-certificates \
    less \
    ncurses-terminfo-base \
    krb5-libs \
    libgcc \
    libintl \
    libssl1.1 \
    libstdc++ \
    tzdata \
    userspace-rcu \
    zlib \
    icu-libs \
    curl \
    && apk -X https://dl-cdn.alpinelinux.org/alpine/edge/main add --no-cache \
    lttng-ust \
    # Download the powershell '.tar.gz' archive
    && curl -L https://github.com/PowerShell/PowerShell/releases/download/v7.2.5/powershell-7.2.5-linux-alpine-x64.tar.gz -o /tmp/powershell.tar.gz \
    # Create the target folder where powershell will be placed
    && mkdir -p /opt/microsoft/powershell/7 \
    # Expand powershell to the target folder
    && tar zxf /tmp/powershell.tar.gz -C /opt/microsoft/powershell/7 \
    # Set execute permissions
    && chmod +x /opt/microsoft/powershell/7/pwsh \
    # Create the symbolic link that points to pwsh
    && ln -s /opt/microsoft/powershell/7/pwsh /usr/bin/pwsh
# Install helm3 as per https://helm.sh/docs/intro/install/#from-script
RUN curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 \
    && chmod 700 get_helm.sh \
    && ./get_helm.sh 
# Install connectedk8s, k8s-extension, and ml extensions 
RUN az extension add --name connectedk8s \
    && az extension add --name k8s-extension \
    && az extension add -n ml -y
# Install bicep CLI
RUN az bicep install
# Install azureml-core
RUN pip install azureml-core
# Copy all files in parent directory to the container
COPY . /mlops/external_silos
# Set working directory
WORKDIR /mlops/external_silos
# Start PowerShell
CMD pwsh