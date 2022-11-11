# Finance example for Federated Learning in Azure ML

Federated Learning (FL) is an approach where we train a single ML models on a distinct datasets, usually in distinct location which cannot be gathered in one central location for reasons like compliance, security or many others. This is more concern in financial field than most probably in any other as this data contain highly sensitive information about either people or institutions.

Therefore, we came up with and example how an institutions like a banks or credit card companies can train a model for fraud detection in federating learning fashion.

:warning: Please familiarize yourself with basics and remarks mentioned in the [**documentation**](../../../README.md) file in the root of the repository.

:warning: This is an example for demonstration purposes only and the authors of this repository do not take any responsibility for loss or harm made by its usage

# Data

As for the source of the data for this example we have used Kaggle dataset, [**Credit Card Transactions Fraud Detection Dataset**](https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount&types=competitions) generated using simulation that contains both genuine and fraudulent transactions. Compared to other dataset this one does not only contain an abstraction of underlying features but rather all features in their raw format. This allows us to split data into distinct geo-location and also let us show how to do a simple **preprocessing** on the **tabular data**.

# Getting started

## Prerequisites
- Properly set up AzureML environment in accordance to the tutorial in the [**documentation**](../../../README.md) in the root of the repository

## Running example
- Update [**config**](./config.yaml) to reflect your AML FL orchestration setup
- Update configuration names in the `preprocessing/config` folder to reflect names of the computer in your own AML FL orchestration setup
- Run the example by running `submit.py` file using python environment from the `pipelines` folder

