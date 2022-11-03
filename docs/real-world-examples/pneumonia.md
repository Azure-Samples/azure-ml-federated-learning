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
- have a way to run bash scripts;
- **fork** this repository (fork as opposed to clone because you will need to create GitHub secrets and run GitHub actions to prepare the data).

## Procedure
The procedure to run this example  has three phases:
- provision the Azure resources;
- prepare the data;
- run the job.

Each phase is described in the subsections below.

### Provisioning
This phase is easy. Just follow the instructions in the [quickstart](../quickstart.md) to provision an open sandbox. Make note of the name of the resource group you provisioned, as well as the name of the workspace.

### Data preparation
The goal of this phase is to upload to Azure ML 3 distinct datasets (one per silo), each subdivided into train, test, and validation. This is achieved by running a GitHub action that will download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), split it evenly between the silos, and upload the 3 data assets. We will need to first create an Azure Service Principal with enough permissions to create the data assets in Azure ML. The whole procedure is explained below, and closely follows [the one](https://github.com/Azure/medical-imaging/blob/main/federated-learning/README.md#1-prepare-your-experiment) developed by Harmke Alkemade _et al_ .

1. Create a service principal with contributor access to your resource group by running the following Azure CLI command.
```bash
  az ad sp create-for-rbac --name "<service-principal-name>" --role contributor --scopes /subscriptions/<subscription-id>/resourceGroups/<your-resource-group-name> --sdk-auth
```
2. Create a GitHub secret in your _forked_ repository. Name it AZURE_CREDENTIALS_DATAPREP and copy and paste the output of the above command to the Value field of the secret.
3. Add the following secrets to your repository with corresponding values:
  - KAGGLE_USERNAME: your Kaggle username.
  - KAGGLE_PASSWORD: your Kaggle API key. More info [here](https://www.kaggle.com/docs/api).
4. Navigate to the GitHub Actions tab, and run the workflow with the name *FL data preparation - pneumonia example*. Provide the names of your workspace and resource group. This will register the datasets required for your experiment in your workspace.


### Run the FL job