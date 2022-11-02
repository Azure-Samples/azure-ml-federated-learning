# Pneumonia detection from chest radiographs

## Background
In this example, we train a model to detect pneumonia from chest radiographs. The model is trained on a dataset of chest radiographs from the [NIH Chest X-ray dataset](https://www.kaggle.com/nih-chest-xrays/data). This example is adapted from [that solution](https://github.com/Azure/medical-imaging/tree/main/federated-learning) by Harmke Alkemade _et al._

We will mimic a real-world scenario where 3 hospitals want to collaborate on training a model to detect pneumonia from chest radiographs. The hospitals have their own data, and they want to train a model on their data without sharing it with each other. The model will be trained in a federated manner, where each hospital will train a model on its own data, and the models will be aggregated to produce a final model.

> For the sake of simplicity, we will only provision an _open_ setup. Do not upload sensitive data to it! 

## Prerequisites
To enjoy this tutorial, you will need to:
- have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli),
- have a way to run bash scripts.

## Procedure
The procedure to run this example  has three phases:
- provision the Azure resources;
- prepare the data;
- run the job.

Each phase is described in the subsections below.

### Provisioning
This phase is easy. Just follow the instructions in the [quickstart](../quickstart.md) to provision an open sandbox.

### Data preparation

### Run the FL job