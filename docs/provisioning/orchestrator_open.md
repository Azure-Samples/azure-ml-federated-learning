# Create an open sandbox orchestrator

:warning: This should be used for **development purpose only**.

## Prerequisites

To run these deployment options, you first need:

- an existing Azure ML workspace (see [cookbook](README.md#create-an-azure-ml-workspace))
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- Optional: [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

## Create a compute and storage pair for the orchestrator

> Note: both orchestrator and [silo](./silo_open.md) can be deployed using the same arm/bicep script, changing **Pair Base Name** accordingly.

### Option 1 : one click deployment

1. Click on [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fopen_compute_storage_pair.json)

2. Adjust parameters, in particular:

    - Region: this will be set by Azure to the region of your resource group.
    - Machine Learning Name: need to match the name of the AzureML workspace in the resource group.
    - Machine Learning Region: the region in which the AzureML workspace was deployed (default: same as resource group).
    - Pair Region: the region where the compute and storage will be deployed (default: same as resource group).
    - Pair Base Name: a unique name for the **orchestrator**, example `orch`. This will be used to create all other resources (storage name, compute name, etc.).

### Option 2 : deployment using az cli

In the resource group of your AzureML workspace, use the following command with parameters corresponding to your setup:

```bash
az deployment group create --template-file ./mlops/bicep/modules/fl_pairs/open_compute_storage_pair.bicep --resource-group <resource group name> --parameters pairBaseName="orch" pairRegion="eastus" machineLearningName="aml-fldemo" machineLearningRegion="eastus"
```
