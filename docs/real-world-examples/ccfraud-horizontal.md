# Federated cross-geo credit card fraud detection

**Scenario** - This is a short example where we showcase possibilities of using Azure Machine Learning(AML) for training a model for credit card fraud detection in federating learning fashion. The example utilizes multitude of model architectures to demonstrate versatility of the proposed solution on a typical use case for the finance indutry. We have simulated a FL scenario by splitting the data into **distinct geo-location**. The sample provides a simple implementation  for **preprocessing** on **tabular data**.

**Dataset** - This example is trained using the Kaggle dataset [**Credit Card Transactions Fraud Detection Dataset**](https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount&types=competitions). This dataset is generated using a simulation that contains both genuine and fraudulent transactions.

## Table of contents

- [Install the required dependencies](#install-the-required-dependencies)
- [Provision an FL sandbox workspace](#provision-an-fl-sandbox-with-your-kaggle-credentials)
- [Add your Kaggle credentials to the workspace key vault](#add-your-kaggle-credentials-to-the-workspace-key-vault)
- [Run a job to download and store the dataset in each silo](#run-a-job-to-download-and-store-the-dataset-in-each-silo)
- [Run the demo experiment](#run-the-demo-experiment)
- [Beyond the experiment](#beyond-the-experiment)
  - [Test model variants](#test-model-variants)
  - [Scale up using distributed training](#scale-up-using-distributed-training)
  - [Enable confidentiality with encryption at rest](#enable-confidentiality-with-encryption-at-rest)
  - [Vertical Federated Learning] (#vertical-federated-learning)

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

    :closed_lock_with_key: Optional: this job can support [encryption at rest](../concepts/confidentiality.md), during the upload the data can be encrypted with a custom key located in an Azure Key Vault. This is a good practice to ensure that your data is encrypted with a key yourself or your team or the cloud provider doesn't have access to. To turn this on:

    - check out the `confidentiality` section in the config file in `examples/pipelines/utils/upload_data/config.yaml` and set `enable:true`
    - modify your key vault name to match with the base name you used during provisioning (ex: `kv-{basename}`).

2. Submit the experiment by running:

   ```bash
   python ./examples/pipelines/utils/upload_data/submit.py --example CCFRAUD --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

:warning: Proceed to the next step only once the pipeline completes. This pipeline will create data in 3 distinct locations.

## Run the demo experiment

1. If you are **not** using the sandbox default setup, adjust the config file  `config.yaml` in `examples/pipelines/ccfraud/` to match your setup.

    :closed_lock_with_key: Optional: this demo can support [encryption at rest](../concepts/confidentiality.md), the preprocessing can consume and generate encrypted data (you need to have enabled confidentiality in the previous step). To turn this on:

    - check out the `confidentiality` section in the config file in `examples/pipelines/ccfraud/config.yaml` and set `enable:true`
    - modify your key vault name to match with the base name you used during provisioning (ex: `kv-{basename}`).

2. Submit the FL experiment by running:

   ```bash
   python ./examples/pipelines/ccfraud/submit.py --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

## Beyond the experiment

### Test model variants

This sample experiment provides multiple models you can try:

- **SimpleLinear** : a model fully composed of `torch.Linear` layers with `ReLU` activations, takes data as-is sample-by-sample
- **SimpleLSTM** : a model composed by 4 LSTM layers connected to linear head with architecture similar to **SimpleLinear**, takes data ordered by time in sequences that overlap each other
- **SimpleVAE** : a model composed of 2 encoder LSTM layers and 2 decoder LSTM layers that tries to recreate consumed sequence of transactions, the latent space created by encoder is consumed by a linear layer to perform prediction, takes data ordered by time in sequences that overlap each other

To switch between models, please update the `config.yaml` file in `examples/pipelines/ccfraud/`. Look for the field `model_name` in the `training_parameters` section (use `SimpleLinear`, `SimpleLSTM`, or `SimpleVAE`).

### Scale-up using distributed training

This sample can be ran in distributed fashion, using [PyTorch Data Distributed Parallel (DDP) with NCCL backend](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). However, this requires us to provision new cluster with >1 CUDA enabled GPUs and allow them to access datastorages in corresponding regions. In order to do so follow these steps for every region you want to run distributed training in:

- Provision GPU cluster with 1+ NVIDIA GPU alongside with storage according to tutorial [here](../provisioning/README.md)
- Adjust the config file  `config.yaml` in `examples/pipelines/ccfraud/` to use the newly created compute
- The pipeline automatically detects number of GPUs on a given compute and scales the job accordingly
- If your cluster can be scaled to more than 1 machine, you can modify `config.yaml` in `examples/pipelines/ccfraud/` by adding `instance_count` to your silo config with number of nodes you would like to run the training on

After performing all these steps you can rerun the experiment and it will run using DDP.

### Enable confidentiality with encryption at rest

As mentioned in the tutorial, this demo supports encryption at rest in the preprocessing phase. This means that the data is encrypted with a custom key that is stored in an Azure Key Vault. This is a good practice to ensure that your data is encrypted with a key that the data science team or the cloud provider doesn't have access to, but can be used by the computes automatically.

In this demo experiment, the data upload encrypts the data with a custom key and stores it in the silo. The preprocessing phase then decrypts the data and uses it to train the model.

Note that we have left the model unencrypted before aggregation, but you could apply this same patter in every single step of your pipeline.

To understand how this works, check `confidential_io.py` in `examples/components/CCFRAUD/upload_data/` and `examples/components/CCFRAUD/preprocessing/`. These files contain the logic to encrypt and decrypt data with a few helper functions.


### Vertical Federated Learning

By default the example above describe horizontal federated learning scenario. In case you are interested in running vertical scenario, or find out more about this topic, please refer to following docs:
- [**Vertical Federated Learning with Split Learning**](../concepts/vertical-fl.md#ccfraud).
- [**Vertical Federated Learning with VAE and Single Shot Communication**](../concepts/vertical-fl.md#ccfraud-with-fedonce).