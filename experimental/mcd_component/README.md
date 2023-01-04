# EXPERIMENTAL: run a multi-compute distributed job in AzureML using Service Bus

Current known limits:
- if job fails, you might have to cancel it manually
- ManagedIdentity is consistently failing on the provisioned nodes.

## Provision using vnet peering

```bash
# connect to azure
az login
az account set --name "your subscription name"

# create resource group
az group create -n nvflare-devbox-rg -l eastus

# deploy vnet peered sandbox
az deployment group create --template-file .\mlops\bicep\vnet_publicip_sandbox_setup.bicep --resource-group nvflare-devbox-rg --parameters demoBaseName="nvflaredev1" applyVNetPeering=true
```

## Provision a Service Bus

1. Use Azure Portal to provision a Service Bus resource in the same resource group. Use Standard pricing tier. Name it `fldevsb` for instance.

2. Create a Topic `mcd`, set time to live to 1h.

## Run sample

1. Install az cli v2

    ```bash
    az extension add --name azure-cli-ml
    ```

2. Run the sample using az cli v2, check instructions about authentification method in the yaml file

    ```bash
    az ml job create --file ./pipeline.yaml -w WORKSPACE -g GROUP
    ```

3. To run script locally, install requirements `python -m pip install -r requirements.txt`.


## What to expect

This will create a pipeline in AzureML with 2 jobs in different locations. The first one is a head node in the orchestrator vnet (ip `1.0.0.*`). The second one is a worker on a silo vnet (ip `10.0.1.*`).

The first node online will wait for the other nodes to start. Once all nodes are online, the head node will send its IP address to all the workers.

Logs for both jobs are full of debug logs by design. Look out in the worker node (`sb_rank=2`) for a section like below where the job will display all the IP addresses of the worker node, and will show ping from the worker to the head node, proving network connectivity between those 2.

```logs
INFO : root : IFADDR: {2: [{'addr': '10.0.1.10', 'netmask': '255.255.255.0', 'broadcast': '10.0.1.255'}]}
INFO : root : Pinging head node
INFO : root : Head node address: 10.0.0.7
INFO : root : Reply from 10.0.0.7, 29 bytes in 67.06ms
INFO : root : Reply from 10.0.0.7, 29 bytes in 64.75ms
INFO : root : Reply from 10.0.0.7, 29 bytes in 64.71ms
INFO : root : Reply from 10.0.0.7, 29 bytes in 64.94ms
INFO : root : Reply from 10.0.0.7, 29 bytes in 65.01ms
```

The worker node then waits for the head node to complete. The head node enters into a sleeping mode for 10 minutes and completes, triggering all workers to complete as well.
