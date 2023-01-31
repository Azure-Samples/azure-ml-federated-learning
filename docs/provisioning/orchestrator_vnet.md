# Create an orchestrator behind a vnet with a new private storage

This tutorial applies in the case you want to create a **completely new storage** account for a federated learning orchestrator in a given region.

## Table of contents

- [Prerequisites](#prerequisites)
- [Create a compute and storage pair for the orchestrator](#create-a-compute-and-storage-pair-for-the-orchestrator)
  - [Using one click deployment](#using-one-click-deployment)
  - [Using az cli](#using-az-cli)

## Prerequisites

To run these deployment options, you first need:

- an existing Azure ML workspace (see [cookbook](README.md))
- an existing private DNS zone for storage, named `privatelink.blob.core.windows.net` (see below)
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- Optional: [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

> **To create a private DNS zone**  
> If you don't already have one, you will need to [manually create a private DNS zone](https://learn.microsoft.com/en-us/azure/dns/private-dns-privatednszone) for the storage account and compute of this pair.  
> To do that, go to the Azure portal, and in the resource group of your AzureML workspace, create a new private DNS zone named `privatelink.blob.core.windows.net`.  
> You only need one unique zone for all the pairs you create (both orchestrator and silos). All private DNS entries will be written in that single zone.

## Create a compute and storage pair for the orchestrator

> Note: both orchestrator and [silo](./silo_vnet_newstorage.md) can be deployed using the same arm/bicep script, changing **Pair Base Name** and `storagePublicNetworkAccess` accordingly.

:important: make sure the subnet address space is not overlapping with any other subnet in your vnet, in particular that it is unique accross all your silos and orchestrator. For instance you can use `10.0.0.0/24` for the orchestrator, then `10.0.N.0/24` for each silo, with a distinct N value.

### Using one click deployment

1. Click on [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fvnet_compute_storage_pair.json)

2. Adjust parameters, in particular:

    - Region: this will be set by Azure to the region of your resource group.
    - Machine Learning Name: need to match the name of the AzureML workspace in the resource group.
    - Machine Learning Region: the region in which the AzureML workspace was deployed (default: same as resource group).
    - Pair Region: the region where the compute and storage will be deployed (default: same as resource group).
    - Pair Base Name: a unique name for the **orchestrator**, example `orch`. This will be used to create all other resources (storage name, compute name, etc.).
    - Storage Public Network Access: turn this to **Enabled** to allow connections to the orchestrator via its public IP (default: **Disabled**). If you pick Disabled here, you will have to setup your private DNS zone manually during the addition of the silos.

### Using az cli

In the resource group of your AzureML workspace, use the following command with parameters corresponding to your setup:

```bash
az deployment group create --template-file ./mlops/bicep/modules/fl_pairs/vnet_compute_storage_pair.bicep --resource-group <resource group name> --parameters pairBaseName="orch" pairRegion="eastus" machineLearningName="aml-fldemo" machineLearningRegion="eastus" subnetPrefix="10.0.0.0/24" storagePublicNetworkAccess=Enabled
```
