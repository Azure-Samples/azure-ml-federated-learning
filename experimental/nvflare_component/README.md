# EXPERIMENTAL: run an NVFlare job in AzureML

Current known limits:
- if job fails, you might have to cancel it manually

## Run a demo

1. Provision **using vnet peering**

```bash
# connect to azure
az login
az account set --name "your subscription name"

# create resource group
az group create -n nvflare-devbox-rg -l eastus

# deploy vnet peered sandbox
az deployment group create --template-file .\mlops\bicep\vnet_publicip_sandbox_setup.bicep --resource-group nvflare-devbox-rg --parameters demoBaseName="nvflaredev1" applyVNetPeering=true
```

2. Install az cli v2

    ```bash
    az extension add --name azure-cli-ml
    ```

3. Register the 2 required environments:

    ```bash
    # register nvflare-launcher environment
    az ml environment create --file ./environments/nvflare_launcher/env.yaml

    # register nvflare-pt environment
    az ml environment create --file ./environments/nvflare_pt/env.yaml
    ```

4. Obtain an example from NVFlare

    ```bash
    # clone NVFlare repo
    git clone https://github.com/NVIDIA/NVFlare.git

    # check version 2.2
    cd NVFlare
    git checkout -t 2.2
    ```

5. Check the file `nvflare_job.yaml` and modify the path to the sample app/ folder according to your local clone path.

    ```bash
    az ml job create --file ./nvflare_job.yaml
    ```

## What to expect

This will create a job in AzureML, this job will submit 4 other jobs within a pipeline, one in the orchestrator for the NVFlare server, and 3 in each silo for each of the NVFlare clients.

- `outputs/nvflare_host.log` : are the logs of the runtime that operates the nvflare host setup
- `user_logs/std_log.txt` : the regular logs of the job being run
