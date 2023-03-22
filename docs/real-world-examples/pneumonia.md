# Federated medical imaging : radiographs classification

**Scenario** - In this example, we train a federated learning model to detect pneumonia from chest radiographs. We mimic a real-world FL scenario where 3 hospitals in 3 different regions want to collaborate on training a model to detect pneumonia from chest radiographs. The hospitals have their own data, and they want to train a model on all data without directly sharing data with each other, or with a central entity.  
The model will be trained in a federated manner, where each entity will train a model on its own data, and the models will be aggregated to produce a final model.

**Dataset** - The model is trained on the [Chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle. This example is adapted from [another FL solution](https://github.com/Azure/medical-imaging/tree/main/federated-learning) by Harmke Alkemade _et al._.

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
   python ./examples/pipelines/utils/upload_data/submit.py --example PNEUMONIA --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

:warning: Proceed to the next step only once the pipeline completes. This pipeline will create data in 3 distinct locations.

## Run the demo experiment

1. If you are **not** using the sandbox default setup, adjust the config file  `config.yaml` in `examples/pipelines/pneumonia/` to match your setup.

2. Submit the FL experiment by running:

   ```bash
   python ./examples/pipelines/pneumonia/submit.py --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.
