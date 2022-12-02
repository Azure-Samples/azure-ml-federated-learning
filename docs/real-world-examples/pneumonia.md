# Pneumonia detection from chest radiographs

## Background
In this example, we train a model to detect pneumonia from chest radiographs. The model is trained on a dataset of chest radiographs from the [NIH Chest X-ray dataset](https://www.kaggle.com/nih-chest-xrays/data). This example is adapted from [that solution](https://github.com/Azure/medical-imaging/tree/main/federated-learning) by Harmke Alkemade _et al._, that is relying on NVFlare

We mimic a real-world scenario where 3 hospitals in 3 different regions want to collaborate on training a model to detect pneumonia from chest radiographs. The hospitals have their own data, and they want to train a model on all data without directly sharing data with each other, or with a central entity. The model will be trained in a federated manner, where each hospital will train a model on its own data, and the models will be aggregated to produce a final model.

> For the sake of simplicity, we will only provision an _open_ setup. Do not upload sensitive data to it! 

## Prerequisites
To enjoy this tutorial, you will need to:
- have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli),
- have a way to run bash scripts;
- **fork** this repository (_fork_ as opposed to _clone_ because you will need to create GitHub secrets and run GitHub actions to prepare the data).

## Procedure
The procedure to run this example  has three phases:
- provision the Azure resources;
- prepare the data;
- run the job.

Each phase is described in the subsections below.

### Provisioning
This phase is easy. Just follow the instructions in the [quickstart](../quickstart.md) to provision an open sandbox with 3 silos. Make note of the name of the resource group you provisioned, as well as the name of the workspace.

### Data preparation
The goal of this phase is to upload to the 3 silos' storages 3 distinct datasets (one per silo), each subdivided into train, test, and validation. This is achieved by running a simple pipeline job in Azure ML. This simple job is made up of 3 independent jobs, which will each run in a given silo compute. Each job will download the full [pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle, partition it, and write one partition to the silo's storage.

1. First, you'll need to store your Kaggle user name and API key in a key vault, so the Azure ML job can use these secrets to authenticate to the Kaggle API.
    - Start by locating the key vault associated with your Azure ML workspace in the [Azure portal](https://portal.azure.com). It should be named like your Azure ML workspace, but starting with '`kv-`' instead of '`aml-`'.
    - Once you have located the key vault in the Azure portal, open the "Access Policies" tab. You're going to create a new access policy to give yourself the permissions to manage secrets. To do so, click on "Create", select all "Secret Management Operations", and click "Next". Select your user id, click "Next" twice, then "Create".
    - You should now be able to create secrets. Navigate to the "Secrets" tab and create the 2 following secrets, with these exact names:
      - _kaggleusername_: your Kaggle username.
      - _kagglekey_: your Kaggle API key. More info [here](https://www.kaggle.com/docs/api).
2. Then, you will need to run the `upload_data` pipeline. First, make sure the compute and datastore names in the `./examples/pipelines/utils/upload_data/config.yaml` and `./examples/pipelines/pneumonia/config.yaml` files are the same (they should match those created by the [quickstart](../quickstart.md) provisioning). Then, adjust the workspace name, resource group, and subscription Id in `./examples/pipelines/utils/upload_data/config.yaml`. Finally, run the following commands to create a conda environment, activate it, and submit the pipeline.
    ```bash
    conda env create --file ./examples/pipelines/environment.yml
    conda activate fl_experiment_conda_env
    python ./examples/pipelines/utils/upload_data/submit.py --example PNEUMONIA --submit
    ```
3. Once the pipeline has completed, you can move on to the next phase. 

### Run the FL job

1. Adjust the config file (if you kept everything default you'll only have to adjust subscription id, resource group, and workspace name)
2. Submit the experiment (from the `fl_experiment_conda_env` conda environment like in the previous phase).
   ```bash
   python ./examples/pipelines/pneumonia/pneumonia_submit.py --submit
   ```