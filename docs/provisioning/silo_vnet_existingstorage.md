# Create a silo with compute behind a vnet accessing your existing storage account

This tutorial applies in the case you want to create a compute to process data from an existing storage account located **in the same tenant** (different subscription or resource group).

Typically, this would happen when using internal silos corresponding to various regions, with storages not managed by a single entity but scattered accross multiple subscriptions in a single company.

## Table of contents

- [Prerequisites](#prerequisites)
- [Important: understand the design](#important-understand-the-design)
- [Create a compute pair for the silo, attach storage as datastore](#create-a-compute-pair-for-the-silo-attach-storage-as-datastore)
  - [Using one click deployment](#using-one-click-deployment)
  - [Using az cli](#using-az-cli)
- [Set up interactions within the silo](#set-up-interactions-within-the-silo)
- [Set up interactions with the orchestrator](#set-up-interactions-with-the-orchestrator)

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

## Important: understand the design

The design we used for this tutorial is identical to [a silo provisioned with a new storage](./silo_vnet_newstorage.md#important-understand-the-design). What is different in this case is that we do not provision the storage, so we rely on the previously configured storage account.

It is important in this case that you set the following on your existing storage account:

- make sure the storage account is **in the same tenant** as the AzureML workspace
- in the networking settings of the storage, it is recommended to set **Public network access** to "Disabled" (access will be allowed only via a private endpoint)
- create a container in this storage account for your fl data

## Create a compute pair for the silo, attach storage as datastore

:important: make sure the subnet address space is not overlapping with any other subnet in your vnet, in particular that it is unique accross all your silos and orchestrator. For instance you can use `10.0.0.0/24` for the orchestrator, then `10.0.N.0/24` for each silo, with a distinct N value.

### Using one click deployment

1. Click on [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fvnet_compute_existing_storage.json)

2. Adjust parameters, in particular:

    - Region: this will be set by Azure to the region of your resource group.
    - Machine Learning Name: need to match the name of the AzureML workspace in the resource group.
    - Machine Learning Region: the region in which the AzureML workspace was deployed (default: same as resource group).
    - Pair Region: the region where the compute and storage will be deployed (default: same as resource group), make sure this matches with the location of your storage account.
    - Pair Base Name: a unique name for the **silo**, example `silo1-westus`. This will be used to create all other resources (storage name, compute name, etc.).
    - Existing Storage Account Name: name of the storage account to attach to this silo.
    - Existing Storage Account Resource Group: name of the resource group in which the storage is provisioned.
    - Existing Storage Account Subscription Id: id of the subscription in which the storage is provisioned.
    - Existing Storage Container Name: name of container where the data will be located.

### Using az cli

In the resource group of your AzureML workspace, use the following command with parameters corresponding to your setup:

```bash
az deployment group create --template-file ./mlops/bicep/modules/fl_pairs/vnet_compute_existing_storage.bicep --resource-group <resource group name> --parameters pairBaseName="silo1-westus" pairRegion="westus" machineLearningName="aml-fldemo" machineLearningRegion="eastus" subnetPrefix="10.0.1.0/24" existingStorageAccountName="..." existingStorageAccountResourceGroup="..." existingStorageAccountSubscriptionId="..."
```

Make sure `pairRegion` matches with the region of your storage account.

## Set up interactions within the silo

Let's set required permissions between the silo's compute and the silo's existing storage account.

1. Navigate the Azure portal to find your resource group.

2. Look for a resource of type **Managed Identity** in the region of the silo named like `uai-<pairBaseName>`. It should have been created by the instructions above.

3. Open this identity and click on **Azure role assignments**. You should see the list of assignments for this identity.

    It should contain 3 roles towards the storage account of the silo itself:
    - **Storage Blob Data Contributor**
    - **Reader and Data Access**
    - **Storage Account Key Operator Service Role**

4. Click on **Add role assignment** and add each of these same role towards the storage account of your orchestrator.

## Set up interactions with the orchestrator

### Option 1: public storage account

All you'll have to set are permissions for the silo's compute to R/W from/to the orchestrator.

1. Navigate the Azure portal to find your resource group.

2. Look for a resource of type **Managed Identity** in the region of the silo named like `uai-<pairBaseName>`. It should have been created by the instructions above.

3. Open this identity and click on **Azure role assignments**. You should see the list of assignments for this identity.

    It should contain 3 roles towards the storage account of the silo itself:
    - **Storage Blob Data Contributor**
    - **Reader and Data Access**
    - **Storage Account Key Operator Service Role**

4. Click on **Add role assignment** and add each of these same role towards the storage account of your orchestrator.

### Option 2: private storage with endpoints

:construction: work in progress :construction:
