# Pneumonia detection from chest radiographs

**Scenario** - In this example, we train a federated learning model to detect pneumonia from chest radiographs. We mimic a real-world FL scenario where 3 hospitals in 3 different regions want to collaborate on training a model to detect pneumonia from chest radiographs. The hospitals have their own data, and they want to train a model on all data without directly sharing data with each other, or with a central entity.  
The model will be trained in a federated manner, where each entity will train a model on its own data, and the models will be aggregated to produce a final model.

**Dataset** - The model is trained the [NIH Chest X-ray dataset](https://www.kaggle.com/nih-chest-xrays/data). This example is adapted from [another FL solution](https://github.com/Azure/medical-imaging/tree/main/federated-learning) by Harmke Alkemade _et al._.


## Install the required dependencies

You'll need python to submit experiments to AzureML. You can install the required dependencies by running:

```bash
conda env create --file ./examples/pipelines/environment.yml
conda activate fl_experiment_conda_env
```

## Provision an FL sandbox workspace

To run this example, you will need to provision an AzureML workspace ready for Federated Learning. We strongly recommend you use the setup provided in the repository [quickstart](../quickstart.md). We will use the same names for the computes and datastores created by default during this quickstart.

## Add your kaggle credentials to the workspace keyvault

We propose to run a job in the AzureML workspace that will unpack this demo dataset into each of your silos.

Since the dataset is from Kaggle, we will first store your username/password in the workspace keyvault. This will allow the job to download the dataset from Kaggle directly.

1. In your workspace resource group (provisioned in the previous step), open "Access Policies" tab in the newly keyvault.

2. Select "Select all" right under "Secret Management Operations" and press "Next".

3. Click "Create" button in the top. Lookup currently logged in user (using user id or an email), select it and press "Next". 

4. Press "Next" and "Create" in the next screens.

    We are now able to create a secret in the keyvault.

5. Open the "Secrets" tab. Create two plain text secrets:
    
    - **kaggleusername** - specifies your kaggle user name
    - **kagglekey** - this is API key that can be obtained from your profile on the kaggle

## Run a job to download and store the dataset in each silo

This can all be performed with ease using a data provisioning pipeline. To run it follow these steps:

1. In this repository, navigate in the folder `examples/pipelines/utils/upload_data`

2. If you are not using the quickstart setup, check the config file under ...

3. Submit the experiment by running:

   ```bash
   python ./examples/pipelines/utils/upload_data/submit.py --submit --example PNEUMONIA --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

:warning: Proceed to the next step only once the pipeline completes. This pipeline will create data in 3 distinct locations.

TODO

## Run the demo experiment

1. If you are not using the quickstart setup, adjust the config file under ...

2. Submit the FL experiment by running:

   ```bash
   python ./examples/pipelines/pneumonia/submit.py --submit
   ```
