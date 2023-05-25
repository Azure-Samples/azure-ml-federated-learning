# Federated Learning with AzureML and Confidential Computes - an end-to-end tutorial

## Objectives
The goal of this tutorial is two-fold: introduce the resources in our [accelerator repository](https://github.com/Azure-Samples/azure-ml-federated-learning/tree/main), and guide you through your first implementation of Federated Learning on Confidential Compute.

## Scenario
The scenario we will be addressing is hand-written digit recognition, based on the classic MNIST dataset. We will train the model using Federated Learning.

The setup will be made of 1 orchestrator and 3 silos in the same Azure subscription/tenant (cross-tenant silos are possible, but harder to set up). All entities will be using Confidential Computes ([CVM's from dcasv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/dcasv5-dcadsv5-series)).

### Disclaimer
For the sake of time, we will NOT be uploading non-intersecting datasets to the 3 silos. We will just use the same [MNIST dataset](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-mnist?tabs=azureml-opendatasets) (publicly available in Azure Open Datasets) in all 3 silos.

## Prerequisites
To enjoy this quickstart, you will need to:

* have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
* have permissions to create resources, **set permissions**, and create identities in this subscription (or at least in one resource group),
  - :warning: **Please note that typically, the standard Contributor role in Azure does NOT allow you to set permissions**. You will need to be **Owner** of the resource group to be able to set permissions. Creating the resource group yourself will NOT automatically give you the Owner role. If you are not Owner, please ask your subscription Admin to grant you the Owner role on the resource group you will be using for this tutorial.
* [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli),
* install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/),
* clone the [current repository](https://github.com/Azure-Samples/azure-ml-federated-learning).


## Provision the setup
To make sure you have all the resources you will need, it is strongly recommended to provision a [confidential sandbox](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/docs/provisioning/sandboxes.md#confidential-sandboxes). Just follow the instructions at the previous link. You will need to provide the name of the resource group where you can adjust permissions. We also suggest you provide a custom Demo Base Name to avoid resource names conflicts. Keep that name short. 

The sandbox provisioning should take about 15-20 minutes to complete. When that is done, you will need to follow [those instructions](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/docs/provisioning/silo_open_aks_with_cc.md#create-instancetype), which will guide you through the process of creating an InstanceType. If you don't understand the meaning of the last instruction in the linked page, do not worry. You can just skip it for now. The meaning should become clear in the next phase, and you can do that last part then.

## Run the demo job
To run the demo job, the best way is to follow the instructions [here](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/docs/quickstart.md#launch-the-demo-experiment) in our quickstart document. If you kept the default values when provisioning the setup, you should not have anything to change in the [yaml config file](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/examples/pipelines/fl_cross_silo_literal/config.yaml), besides adding the instance type (the last instruction that you might have skipped before).

## Potential follow-ups

### Enable encryption at rest
To further protect your data, you can enable encryption at rest. This will encrypt the data in the silos storages. Our [credit card fraud detection example](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/docs/real-world-examples/ccfraud-horizontal.md) covers that. Please note that for this example, you will need a Kaggle account (since the data are downloaded from Kaggle).
