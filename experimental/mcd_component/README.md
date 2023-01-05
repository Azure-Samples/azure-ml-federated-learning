# EXPERIMENTAL: run a multi-compute distributed job in AzureML using Service Bus

Current known limits:
- if job fails, you might have to cancel it manually
- ManagedIdentity is consistently failing on the provisioned nodes.

## Provision a workspace with requirements

1. Provision using vnet peering

    ```bash
    # connect to azure
    az login
    az account set --name "your subscription name"

    # create resource group
    az group create -n nvflare-devbox-rg -l eastus

    # deploy vnet peered sandbox
    az deployment group create --template-file .\mlops\bicep\vnet_publicip_sandbox_setup.bicep --resource-group nvflare-devbox-rg --parameters demoBaseName="nvflaredev1" applyVNetPeering=true
    ```

2. Use Azure Portal to provision a Service Bus resource in the same resource group. Use Standard pricing tier. Name it `fldevsb` for instance.

3. In this service bus, create a Topic `mcd`.

WORK IN PROGRESS

## Run sample

1. Install az cli v2

    ```bash
    az extension add --name azure-cli-ml
    ```

2. Run the sample using az cli v2, check instructions about authentification method in the yaml file

    ```bash
    az ml job create --file ./mcd_job.yaml -w WORKSPACE -g GROUP
    ```

## What to expect

This will create a job in AzureML, this job will submit 4 other jobs, one in the orchestrator and 3 in each silo. Each job will be given environment variables containing the IP adressed of the other jobs.

Those jobs will have different logs:

- `outputs/mcd_runtime.log` : are the logs of the runtime that operates the multi-cloud distribution setup
- `user_logs/std_log.txt` : the regular logs of the job being run
