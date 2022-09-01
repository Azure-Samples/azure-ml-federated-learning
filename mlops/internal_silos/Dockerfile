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
# Install ml extension
RUN az extension add -n ml -y
# Install bicep CLI
RUN az bicep install
# Install extension for parsing YAML files (temporarily allow installing from PSGallery)
RUN pwsh -Command Set-PSRepository -Name "PSGallery" -InstallationPolicy Trusted
RUN pwsh -Command Install-Module -Name powershell-yaml
RUN pwsh -Command Set-PSRepository -Name "PSGallery" -InstallationPolicy Untrusted
# Copy all files in parent directory to the container
COPY . /mlops
# Set working directory
WORKDIR /mlops/internal_silos
# Start PowerShell
CMD pwsh