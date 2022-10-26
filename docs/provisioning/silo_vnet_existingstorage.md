# Create a silo with compute behind a vnet accessing your existing storage account

This tutorial applies in the case you want to create a compute to process data from an existing storage account located in the same tenant (different subscription or resource group).

## Table of contents

- [Prerequisites](#prerequisites)
- [Important: understand the design](#important-understand-the-design)
- [Create a compute and storage pair for the silo](#need-support)
  - [Option 1: one click deployment](#option-1-one-click-deployment)
  - [Option 2: using az cli](#option-2-using-az-cli)
- [Set required permissions](#set-required-permissions)

## Prerequisites

To run these deployment options, you first need:
- an existing Azure ML workspace (see [cookbook](README.md))
- an existing private DNS zone for storage, named `privatelink.blob.core.windows.net` (see below)
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- Optional: [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).


> **To create a private DNS zone**  
> If you don't already have one, you will need to manually create a private DNS zone for the storage account and compute of this pair.  
> To do that, go to the Azure portal, and in the resource group of your AzureML workspace, create a new private DNS zone named `privatelink.blob.core.windows.net`.  
> You only need one unique zone for all the pairs you create (both orchestrator and silos). All private DNS entries will be written in that single zone.

## Important: understand the design

The design we used for this tutorial is identical to [a silo provisioned with a new storage](./silo_vnet_newstorage.md#important-understand-the-design). What is different in this case is that we do not provision the storage, so we rely on the previously configured storage account.

It is important in this case that you set the following on your existing storage account:
- make sure the storage account is in the same tenant as the AzureML workspace


## Create a compute pair for the silo, attach storage as datastore

:important: make sure the subnet address space is not overlapping with any other subnet in your vnet, in particular that it is unique accross all your silos and orchestrator. For instance you can use `10.0.0.0/24` for the orchestrator, then `10.0.N.0/24` for each silo, with a distinct N value.

### Option 1: one click deployment

1. Click on [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Frelease-sdkv2-iteration-03%2Fmlops%2Farm%2Fvnet_compute_existing_storage.json)

2. Adjust parameters, in particular:

    - Region: this will be set by Azure to the region of your resource group.
    - Machine Learning Name: need to match the name of the AzureML workspace in the resource group.
    - Machine Learning Region: the region in which the AzureML workspace was deployed (default: same as resource group).
    - Pair Region: the region where the compute and storage will be deployed (default: same as resource group).
    - Pair Base Name: a unique name for the **silo**, example `silo1-westus`. This will be used to create all other resources (storage name, compute name, etc.).

### Option 2: using az cli

In the resource group of your AzureML workspace, use the following command with parameters corresponding to your setup:

```bash
az deployment group create --template-file ./mlops/bicep/modules/fl_pairs/vnet_compute_existing_storage.bicep --resource-group <resource group name> --parameters pairBaseName="silo1-westus" pairRegion="westus" machineLearningName="aml-fldemo" machineLearningRegion="eastus" subnetPrefix="10.0.1.0/24"
```

## Set required permissions

### 1. Internally between the silo's compute and the silo's existing storage account

In the 
1. Navigate the Azure portal to find your resource group.

2. Look for a resource of type **Managed Identity** in the region of the silo named like `uai-<pairBaseName>`. It should have been created by the instructions above.

3. Open this identity and click on **Azure role assignments**. You should see the list of assignments for this identity.

    It should contain 3 roles towards the storage account of the silo itself:
    - **Storage Blob Data Contributor**
    - **Reader and Data Access**
    - **Storage Account Key Operator Service Role**

4. Click on **Add role assignment** and add each of these same role towards the storage account of your orchestrator.


### 2. Externally between the silo's compute to R/W from/to the orchestrator storage

1. Navigate the Azure portal to find your resource group.

2. Look for a resource of type **Managed Identity** in the region of the silo named like `uai-<pairBaseName>`. It should have been created by the instructions above.

3. Open this identity and click on **Azure role assignments**. You should see the list of assignments for this identity.

    It should contain 3 roles towards the storage account of the silo itself:
    - **Storage Blob Data Contributor**
    - **Reader and Data Access**
    - **Storage Account Key Operator Service Role**

4. Click on **Add role assignment** and add each of these same role towards the storage account of your orchestrator.
