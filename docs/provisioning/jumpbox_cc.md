# Create a confidential compute jumpbox VM inside a vnet

This tutorial will let you create a jumpbox VM inside a vnet, optionally by using Azure Bastion to connect via HTTPS.

:warning: This should be used for **development purpose only**.

## Prerequisites

To enjoy these quickstart, you will need to:

- have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

## Deploy a confidential compute VM inside a vNet

> Check availability of [confidential compute VMS in your region.](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=virtual-machines&regions=all).

### Option 1 : one click deployment

1. Click on [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fjumpbox_cc.json)

2. Adjust parameters, in particular:

    - vnetName: name of the vNet to join.
    - subnetName: name of the subnet inside the vNet.
    - nsgName: name of the existing security group applying to the VM.

### Option 2 : deployment using az cli

In the resource group of your AzureML workspace, use the following command with parameters corresponding to your setup:

```bash
az deployment group create --template-file ./mlops/bicep/modules/resources/jumpbox_cc.bicep --resource-group <resource group name> --parameters vnetName="..." subnetName="..." nsgName="..." jumpboxOs="linux"
```
