# Federated cross-geo credit card fraud detection

**Scenario** - This is a short example where we showcase possibilities of using Azure Machine Learning(AML) for training a model for credit card fraud detection in federating learning fashion. The example utilizes multitude of model architectures to demonstrate versatility of the proposed solution on a typical use case for the finance indutry. We have simulated a FL scenario by splitting the data into **distinct geo-location**. The sample provides a simple implementation  for **preprocessing** on **tabular data**.

**Dataset** - This example is trained using the Kaggle dataset [**Credit Card Transactions Fraud Detection Dataset**](https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount&types=competitions). This dataset is generated using a simulation that contains both genuine and fraudulent transactions. 

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
   python ./examples/pipelines/utils/upload_data/submit.py --submit --example CCFRAUD --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

:warning: Proceed to the next step only once the pipeline completes. This pipeline will create data in 3 distinct locations.

TODO

## Run the demo experiment

1. If you are not using the quickstart setup, adjust the config file under ...

2. Submit the FL experiment by running:

   ```bash
   python ./examples/pipelines/ccfraud/submit.py --submit
   ```

## Beyond the experiment

This sample experiment provides multiple models you can try.

Please update [**config**](../../examples/pipelines/ccfraud/config.yaml), field `model_name` in the `training_parameters` section, to reflect desired model to be trained, options include: SimpleLinear, SimpleLSTM, SimpleVAE

- **SimpleLinear** - model fully composed of `torch.Linear` layers with `ReLU` activations, takes data as-is sample-by-sample
- **SimpleLSTM** - model composed by 4 LSTM layers connected to linear head with architecture similar to **SimpleLinear**, takes data ordered by time in sequences that overlap each other
- **SimpleVAE** - model composed of 2 encoder LSTM layers and 2 decoder LSTM layers that tries to recreate consumed sequence of transactions, the latent space created by encoder is consumed by a linear layer to perform prediction, takes data ordered by time in sequences that overlap each other
