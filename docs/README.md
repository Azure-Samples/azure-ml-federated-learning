# Federated Learning in Azure ML

:warning: Running a full federated learning pipeline raises **security questions that you need to address** before using this repository for production purpose. Please consider this repository as a sample only.


## Table of contents

- [Motivation](#motivation)
- [Getting Started](#getting-started)
- [Why Federated Learning?](#why-should-you-consider-federated-learning)
- [What this repo as to offer?](#what-this-repo-as-to-offer)
- [Tutorial on how to adapt the Factory code](#tutorial-on-how-to-adapt-thefactory-code)
- [Glossary](#glossary)

### Motivation

Due to privacy regulations and limitations created due to trust boundaries, not all data used for training Machine Learning (ML) models can be gathered in a central location to train a model on all available data (_e.g._, some highly sensitive user data located in the DataProvider embassies). This constraint makes it hard to leverage the data for training models in classic training techniques. 

An ML technique called Federated Learning (FL) does just that: train the various models locally on the data available in the clients (_a.k.a._ the silos, and in our case _embassies_), then combine the different models on a central server (_a.k.a._ the _orchestrator_) without data ever leaving the embassy. 

### Getting Started

No time to read? Get directly to the [**quickstart**](./quickstart.md) to provision a demo within minutes in your own subscription.

A step-by-step guide for performing a Federated Learning experiment can be found [**here**](./guide.md).

### Why should you consider Federated Learning?

Federated Learning (FL) is a framework where one trains a single ML model on distinct datasets that cannot be gathered in a single central location. This enables companies and institutions to comply with regulations related to data location and data access while allowing for innovation and personalization.

The basic idea of FL is to train a model by aggregating the results of N isolated training jobs, each running on separated computes with restricted access to given data storages. 

When the computes and data are in the cloud, we say they live in silos, and cross-silo federated learning consists in orchestrating the training and aggregation jobs against the cloud provider. The following figure illustrates what a federated learning solution looks like.

<br/><br/>
<img src="./pics/fl_fig.png" alt="Federated Learning Solution Figure" width="300">

One example of a hospital (i.e., the customer) is described below. The hospital’s task is to classify a medical image into positive or negative of a specific disease. The data scientists at the hospital _already_ have a neural network model trained in a centralized fashion with their own patient data. The model achieved reasonable performance. Now the hospital wants to further improve the model's performance as more hospitals would like to participate in a federated learning way without their data leaving their clusters. 

### What this repo as to offer?

This repo provides some code samples for running a federated learning pipeline in the Azure Machine Learning platform.

| Folder | Description |
| :--- | :--- |
| [examples](../examples) | Scripts and pipelines to run FL sample experiments |
| [mlops](../mlops) | Provisioning scripts |


### Tutorial on how to adapt the Factory code

⚠️ **Work In Progress**

Before proceeding, please read the following points to have a better understanding of the factory code:
1. It has a `set_orchestrator` method that you can leverage to add an orchestrator to your pipeline.
2. The `add_silo` method lets you add `n` number of silos to the pipeline and you don't have to worry about the configuration. It is being taken care of.
3. It has a soft validation component that ensures that the appropriate permissions are granted for your assets. That being said, a dataset `b` should not have access by a compute `a` and so on.
4. You can bypass the validation if you have your own custom rules.
5. It makes sure that no data is being saved and only model weights are kept on the datastore by enabling type-check.

⚠️ **Work In Progress**

### Glossary

The complete glossary list can be seen [**here**](./glossary.md).