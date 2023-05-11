# Federated cross-geo credit card fraud detection

**Scenario** - This is a short example where we showcase possibilities of using Azure Machine Learning(AML) for training a model for credit card fraud detection in **vertical** federating learning fashion, while transferring the intermediate outputs only once. The example is inspired by work from [Wu et. al. https://arxiv.org/abs/2208.10278](https://arxiv.org/abs/2208.10278). We have simulated a FL scenario by splitting the features for each sample into **distinct geo-location**. The example utilizes **unsupervised learning** to create latent representation of samples for each contributor. This allows for **higher utilization of available data** as the samples used for unsupervised training do not need to overlap across parties.

**Dataset** - This example is trained using the Kaggle dataset [**Credit Card Transactions Fraud Detection Dataset**](https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount&types=competitions). This dataset is generated using a simulation that contains both genuine and fraudulent transactions.

## Table of contents

- [Federated cross-geo credit card fraud detection](#federated-cross-geo-credit-card-fraud-detection)
  - [Table of contents](#table-of-contents)
  - [Install the required dependencies](#install-the-required-dependencies)
  - [Provision an FL sandbox (with your kaggle credentials!)](#provision-an-fl-sandbox-with-your-kaggle-credentials)
  - [Run a job to download and store the dataset in each silo](#run-a-job-to-download-and-store-the-dataset-in-each-silo)
  - [Run the demo experiment](#run-the-demo-experiment)
  - [Beyond the experiment](#beyond-the-experiment)

## Install the required dependencies

You'll need python to submit experiments to AzureML. You can install the required dependencies by running:

```bash
conda env create --file ./examples/pipelines/environment.yml
conda activate fl_experiment_conda_env
```

Alternatively, you can just install the required dependencies:

```bash
python -m pip install -r ./examples/pipelines/requirements.txt
```

## Provision an FL sandbox (with your kaggle credentials!)

To run this example, you will need to provision one of our [sandboxes](../provisioning/sandboxes.md). Any sandbox should work with this tutorial below (if not, please [reach out](https://github.com/Azure-Samples/azure-ml-federated-learning/issues)). We will use the same names for the computes and datastores created by default in those sandboxes.

If you have already provisioned a sandbox during the [quickstart](../quickstart.md) you can reuse it, but you need to **make sure you have added your [Kaggle](https://www.kaggle.com/) credentials** so we can upload the required dataset in each silo. If not, please refer to our [tutorial on how to add your kaggle credentials](../tutorials/add-kaggle-credentials.md) to the workspace secret store before running the following sections.

You can also [re-provision another sandbox](../provisioning/sandboxes.md) with a different base name using the [deploy buttons on the sandbox page](../provisioning/sandboxes.md), and provide your **kaggleUsername** and **kaggleKey** as parameters, so they will be injected securely in your workspace secret store.

:notebook: take note of your workspace name, resource group and subscription id. You will need them to submit the experiment.

## Run a job to download and store the dataset in each silo

This can all be performed with ease using a data provisioning pipeline. To run it follow these steps:

1. If you are **not** using the sandbox default setup, adjust the config file  `config.yaml` in `examples/pipelines/utils/upload_data/` to match your setup.
2. Submit the experiment by running:

   ```bash
   python ./examples/pipelines/utils/upload_data/submit.py --example CCFRAUD_VERTICAL_FEDONCE --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

:warning: Proceed to the next step only once the pipeline completes. This pipeline will create data in 3 distinct locations.

## Run the demo experiment

1. If you are **not** using the sandbox default setup, adjust the config file  `config.yaml` in `examples/pipelines/ccfraud_vertical_fedonce/` to match your setup.
2. Submit the FL experiment by running:

   ```bash
   python ./examples/pipelines/ccfraud_vertical_fedonce/submit.py --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

## Beyond the experiment
The example above describe split-learning vertical federated learning scenario. However, we also provide an example for [**Vertical Federated Learning with Split Learning**](./ccfraud-vertical.md).