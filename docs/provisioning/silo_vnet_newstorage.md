# Create a silo behind a vnet with a new private storage

This tutorial applies in the case you want to create a **completely new storage** account in a federated learning silo in a given region.

## Table of contents

- [Prerequisites](#prerequisites)
- [Important: understand the design](#important-understand-the-design)
- [Create a compute and storage pair for the silo](#need-support)
  - [Option 1: one click deployment](#option-1-one-click-deployment)
  - [Option 2: using az cli](#option-2-using-az-cli)
- [Set permissions for the silo's compute to R/W from/to the orchestrator](#set-permissions-for-the-silos-compute-to-rw-fromto-the-orchestrator)

## Prerequisites

To run these deployment options, you first need:
- an existing Azure ML workspace (see [cookbook](README.md))
- an existing private DNS zone for storage, named `privatelink.blob.core.windows.net` (see below)

> **To create a private DNS zone**  
> If you don't already have one, you will need to manually create a private DNS zone for the storage account and compute of this pair.  
> To do that, go to the Azure portal, and in the resource group of your AzureML workspace, create a new private DNS zone named `privatelink.blob.core.windows.net`.  
> You only need one unique zone for all the pairs you create (both orchestrator and silos). All private DNS entries will be written in that single zone.

## Important: understand the design

In this tutorial, we're provisioning the resources according to the following schema. In this design, the silo is located in a region that is distinct from the Azure ML workspace and the orchestrator storage account.

If any of that design doesn't fit your requirements, feel free to check out the bicep provisioning scripts and adapt them to your needs. Also please give us feedback by filing an issue in this repo.

![](../pics/vnet_silo_provisioning.png)

The provisioning script will:
- create a new [vnet and subnet](https://learn.microsoft.com/en-us/azure/virtual-network/virtual-networks-overview), with a [network security group](https://learn.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview),
- create a new [managed identity](https://learn.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/overview) (User Assigned) to manage permissions of the compute,
- create a new [storage account](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-overview) in a given region, with a [private endpoint](https://learn.microsoft.com/en-us/azure/storage/common/storage-private-endpoints) inside the vnet,
- register this storage account as a datastore in the Azure ML workspace,
- create a new Azure ML [compute](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) in the same region as the storage, inside the vnet, with a public IP enabled to interact with the Azure ML workspace (with an NSG to restrict access to the orchestrator IP address).

In this scenario, the silo's blob storage account's networking settings are such that **public network access is disabled** (`publicNetworkAccess:false`). The compute can access it thanks to the private endpoint, and the UAI being given R/W permissions to the storage.

In addition, the compute can interact with the orchestrator storage account, through public network, the UAI being also given R/W permissions to the orchestrator storage.

## Create a compute and storage pair for the orchestrator

> Note: both orchestrator and [silo](./silo_vnet.md) can be deployed using the same arm/bicep script, changing **Pair Base Name** and `storagePublicNetworkAccess` accordingly.

### Option 1: one click deployment

1. Click on [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fjfomhover%2Fprovioningscenarios%2Fmlops%2Farm%2Fopen_compute_storage_pair.json)

2. Adjust parameters, in particular:

    - Region: this will be set by Azure to the region of your resource group.
    - Machine Learning Name: need to match the name of the AzureML workspace in the resource group.
    - Machine Learning Region: the region in which the AzureML workspace was deployed (default: same as resource group).
    - Pair Region: the region where the compute and storage will be deployed (default: same as resource group).
    - Pair Base Name: a unique name for the **orchestrator**, example `orch`. This will be used to create all other resources (storage name, compute name, etc.).

### Option 2: using az cli

In the resource group of your AzureML workspace, use the following command with parameters corresponding to your setup:

```bash
az deployment group create --template-file ./mlops/bicep/modules/resources/open_compute_storage_pair.bicep --resource-group <resource group name> --parameters pairBaseName="orch" pairRegion="eastus" machineLearningName="aml-fldemo" machineLearningRegion="eastus"
```

## Set permissions for the silo's compute to R/W from/to the orchestrator

1. Navigate the Azure portal to find your resource group.

2. Look for a resource of type **Managed Identity** in the region of the silo named like `uai-<pairBaseName>`. It should have been created by the instructions above.

3. Open this identity and click on **Azure role assignments**. You should see the list of assignments for this identity.

    It should contain 3 roles towards the storage account of the silo itself:
    - **Storage Blob Data Contributor**
    - **Reader and Data Access**
    - **Storage Account Key Operator Service Role**

4. Click on **Add role assignment** and add each of these same role towards the storage account of your orchestrator.
