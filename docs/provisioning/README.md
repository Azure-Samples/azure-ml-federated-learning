# AzureML FL provisioning cookbook

This page provides a "pick and choose" guide to provisioning a new FL environment. Each section provides multiple alternatives depending on your setup. Some sections are optional and can be skipped if you already have the required infrastructure in place.

A lot of those steps are still marked :construction:, we are actively working to provide them im the coming iterations.

If you are looking for a ready-to-use sandbox environment, please check out our [sandboxes](./sandboxes.md).

## Prerequisites

To enjoy these quickstart, you will need to:

- have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

## Create an Azure ML Workspace

Creating an Azure ML workspace is the starting point to create your full Federated Learning environment. Your workspace will be the one portal to:

- connect all your resources (computes, datastores),
- coordinate the jobs between the orchestrator (aggregation) and the silos (processing, training),
- run your experiments at scale,
- collect and analyze your experiment results, register your model candidates,
- deploy your models for production.

Below are two options you could use, drawing from the existing documentation. We invite you to check the existing Azure ML documentation for more options.

| Tutorial | Description |
| :-- | :-- |
| [Docs](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources) | Create a workspace and then add compute resources to the workspace. You'll then have everything you need to get started with Azure Machine Learning. |
| [Docs](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-create-secure-workspace) | Learn how to create and connect to a secure Azure Machine Learning workspace. A secure workspace uses Azure Virtual Network to create a security boundary around resources used by Azure Machine Learning. |

## Create an orchestrator in Azure

The orchestrator is the central server of the Federated Learning pipeline. It is responsible for managing the training process and the communication between the silos.

| Manual | Description |
| :-- | :-- |
| [Tutorial](./orchestrator_open.md) | Create a sandbox open pair of compute and storage for the orchestrator. |
| [Tutorial](./orchestrator_vnet.md) | Create a vnet with a compute and private endpoints to a **new** blob storage created in the same resource group. |

## Create internal silos

These tutorials will let you create silos as a pair of compute and storage, optionally behind a vnet with private endpoints. Use the button for convenience, but check the manual for more details.

| Manual | Description |
| :-- | :-- |
| [Tutorial](./silo_open.md) | Create a simple, open pair of compute and storage for the silos in a given region (for sandbox/dev use only). |
| [Tutorial](./silo_open_aks_with_cc.md) | Create an open pair (for sandbox/dev use only) using an AKS cluster with confidential compute. |
| [Tutorial](./silo_vnet_newstorage.md) | Create a vnet with a compute and private endpoints to a **new** blob storage created in the same resource group. |
| [Tutorial](./silo_vnet_existingstorage.md) | Create a vnet with a compute and private endpoint to an **existing** blob storage, in the **same tenant**. |
| :construction: | Create a vnet with an AKS cluster using **confidential computes** and private endpoint to an **existing** blob storage. |

## Create external silos

These tutorials will let you create an _external_ silo by creating a storage account and linking it to an existing kubernetes cluster, optionally behind a vnet with private endpoints.

| Manual | Description |
| :-- | :-- |
| [Tutorial](./external-silos.md) | Create a simple, external silo based on an existing k8s cluster (for sandbox/dev use only). |
| :construction:| Create an open pair (for sandbox/dev use only) using an AKS cluster with confidential compute. |

## Set permission model between orchestrator and silos

:construction:

## Utilities

| Manual | Description |
| :-- | :-- |
| [Tutorial](./jumpbox_cc.md) | Provision a virtual machine inside a vnet to operate private resources (ex: azureml workspace), optionally by using Azure Bastion. |
