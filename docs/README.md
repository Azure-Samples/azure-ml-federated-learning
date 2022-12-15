# Federated Learning in Azure ML

:warning: Running a full federated learning pipeline raises **security questions that you need to address** before using this repository for production purposes. Please consider this repository as a sample only.


## Table of contents

- [Motivation](#motivation)
- [Getting Started](#getting-started)
- [Why Federated Learning?](#why-should-you-consider-federated-learning)
- [What this repo as to offer?](#what-this-repo-as-to-offer)
- [Tutorial on how to adapt the "literal" and the "factory" code](#tutorial-on-how-to-adapt-the-literal-and-the-factory-code)
- [Real-world examples](#real-world-examples)
- [Glossary](#glossary)

## Motivation

Local privacy regulations impose constraints on the movement of data out of a given region, or out of government agencies. Also, institutions or companies working together to leverage their respective data might require or desire to limit circulation of this data, and impose trust boundaries.

In those contexts, the data cannot be gathered in a central location, as is usual practice for training Machine Learning (ML) models. A technique called Federated Learning (FL) allows for training models in this highly constrained environment. It enables companies and institutions to comply with regulations related to data location and data access while allowing for innovation and achieving better quality models.

## Getting Started

No time to read? Get directly to the [**quickstart**](./quickstart.md) to provision a demo within minutes in your own subscription.

To know more about the resource provisioning alternatives, please go to the provisioning [**cookbook**](./provisioning/README.md). 

A step-by-step guide for performing a Federated Learning experiment can be found [**here**](./guide.md).

## Why should you consider Federated Learning?

Let's take the example of a data scientist working in a hospital to classify medical images to detect a specific patient condition. The team at the hospital _already_ has a deep learning model trained in a centralized fashion with their own patient data. The model achieved reasonable performance. Now the hospital wants to further improve the model's performance by partnering with other hospitals. Federated Learning will enable them to collaborate on the model training while keeping control of the hospital's own data, complying with their local regulations and privacy obligations, while enabling better quality models for the benefit of their patients.

Federated Learning (FL) is a framework where one trains a single ML model on distinct datasets that cannot be gathered in a single central location. The basic idea of FL is to train a model by aggregating the results of N isolated training jobs, each running on separated computes with restricted access to given data storages.

The training is orchestrated between a central server (_a.k.a._ orchestrator) and multiple clients (_a.k.a._ silos or embassies). The actual model training happens locally inside the silos/clients on their respective data, without the data ever leaving their respective trust boundaries. Only the local models are sent back to the central server/orchestrator for aggregation.

When the computes and data are in the cloud, we say they live in silos, and cross-silo federated learning consists in orchestrating the training and aggregation jobs against the cloud provider. The following figure illustrates what a federated learning solution looks like.

<br/><br/>
<img src="./pics/fl_fig.png" alt="Federated Learning Solution Figure" width="300">

Creating such a graph of jobs can be complex. This repository provides a recipe to help.

## What this repo as to offer?

This repo provides some code samples for running a federated learning pipeline in the Azure Machine Learning platform.

| Folder | Description |
| :--- | :--- |
| [examples](../examples) | Scripts and pipelines to run FL sample experiments. |
| [mlops](../mlops) | Provisioning scripts. See instructions [here](./provisioning/README.md). |


## Tutorial on how to adapt the "literal" and the "factory" code

The complete tutorial can be found [**here**](./literal-factory-tutorial.md)

## Real-world examples

In addition to the [literal](../examples/pipelines/fl_cross_silo_literal/) and [factory](../examples/pipelines/fl_cross_silo_factory/) sample experiments, we also provide examples based on real-world applications.

> Note: The `upload-data` scripts are only included in the examples for the convenience of executing the FL examples. Please ignore this section if you are performing an actual FL experiment for your scenario.

| Medical Imaging | Named Entity Recognition | Fraud Detection |
| :-: | :-: | :-: |
| [![](./pics/industry-medical-imaging.png)](./real-world-examples/pneumonia.md) | [![](./pics/industry-ner.png)](./real-world-examples/ner.md) | [![](./pics/industry-fraud-detection.png)](./real-world-examples/ccfraud.md) |
| [pneumonia.md](./real-world-examples/pneumonia.md) | [ner.md](./real-world-examples/ner.md) | [ccfraud.md](./real-world-examples/ccfraud.md) |

### Pneumonia detection from chest radiographs
In this example, we train a model to detect pneumonia from chest radiographs. The model is trained on the [Chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)from Kaggle. This example is adapted from [that solution](https://github.com/Azure/medical-imaging/tree/main/federated-learning) by Harmke Alkemade _et al._ See [here](./real-world-examples/pneumonia.md) for detailed instructions on how to run this example.

### Named Entity Recognition using MultiNERD dataset
This example shows how to train a federated model for the Named Entity Recognition task. This tutorial uses the [MultiNERD](https://github.com/Babelscape/multinerd/blob/master/README.md) dataset. See [here](./real-world-examples/ner.md) for detailed instructions on how to run this example.

### Credit card fraud detection using synthetic transactional data
This example shows how to train a federated model for credit card fraud detection using synthetically generated dataset [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection). The techniques used include **Dense DNN**, **LSTM**, **LSTM based VAE**. See [here](./real-world-examples/ccfraud.md) for detailed instructions on how to run this example.

## Targeted tutorials
Besides the full end-to-end real-world examples, we also provide targeted tutorials for specific scenarios.

### Read local data in an on-premises Kubernetes silo
This tutorial will show you how to access, within an Azure ML job running on an on-premises Kubernetes (k8s) cluster, some data in the local file system. It will guide you through the entire process: k8s cluster creation and configuration, attachment to Azure ML as an external silo, and finally test job. See [here](./targeted-tutorials/read-local-data-in-k8s-silo.md) for detailed instructions.


## Troubleshooting guide

If you experience an issue using this repository, please check the [**troubleshooting guide**](./tsg.md) for possible solutions. If you are unable to find a solution, please open an issue in this repository.

## Glossary

The complete glossary list can be seen [**here**](./glossary.md).