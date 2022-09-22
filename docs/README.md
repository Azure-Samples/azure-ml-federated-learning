# Federated Learning in Azure ML

:warning: Running a full federated learning pipeline raises **security questions that you need to address** before using this repository for production purpose. Please consider this repository as a sample only.


## Table of contents

- [Getting Started](#getting-started)
- [Why Federated Learning?](#why-should-you-consider-federated-learning)
- [What this repo as to offer?](#what-this-repo-as-to-offer)
- [Glossary](#glossary)

### Getting Started

No time to read? Get directly to the [**quickstart**](./quickstart.md) to provision a demo within minutes in your own subscription.

A step-by-step guide for performing a Federated Learning experiment can be found [**here**](./guide.md).

### Why should you consider Federated Learning?

Federated Learning (FL) is a framework where one trains a single ML model on distinct datasets that cannot be gathered in a single central location. This enables companies and institutions to comply with regulations related to data location and data access while allowing for innovation and personalization.

The basic idea of FL is to train a model by aggregating the results of N isolated training jobs, each running on separated computes with restricted access to given data storages. 

When the computes and data are in the cloud, we say they live in silos, and cross-silo federated learning consists in orchestrating the training and aggregation jobs against the cloud provider. The following figure illustrates what a federated learning solution looks like.

<br/><br/>
<img src="./pics/fl_fig.png" alt="Federated Learning Solution Figure" width="300">

### What this repo as to offer?

This repo provides some code samples for running a federated learning pipeline in the Azure Machine Learning platform.

| Folder | Description |
| :--- | :--- |
| [examples](../examples) | Scripts and pipelines to run FL sample experiments |
| [mlops](../mlops) | Provisioning scripts |

### Glossary

The complete glossary list can be seen [**here**](./glossary.md).