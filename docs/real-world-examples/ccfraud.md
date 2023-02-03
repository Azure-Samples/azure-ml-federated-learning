# Federated cross-geo credit card fraud detection

**Scenario** - This is a short example where we showcase possibilities of using Azure Machine Learning(AML) for training a model for credit card fraud detection in federating learning fashion. The example utilizes multitude of model architectures to demonstrate versatility of the proposed solution on a typical use case for the finance indutry. We have simulated a FL scenario by splitting the data into **distinct geo-location**. The sample provides a simple implementation  for **preprocessing** on **tabular data**.

**Dataset** - This example is trained using the Kaggle dataset [**Credit Card Transactions Fraud Detection Dataset**](https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount&types=competitions). This dataset is generated using a simulation that contains both genuine and fraudulent transactions.

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

## Provision an FL sandbox workspace

To run this example, you will need to provision an AzureML workspace ready for Federated Learning. We strongly recommend you use the setup provided in the repository [quickstart](../quickstart.md). We will use the same names for the computes and datastores created by default during this quickstart.

:notebook: take note of your workspace name, resource group and subscription id. You will need them to submit the experiment.

## Add your Kaggle credentials to the workspace key vault

In the next section, we will run a job in the AzureML workspace that will unpack the demo dataset from Kaggle into each of your silos.

Kaggle requires a username and an [API key](https://github.com/Kaggle/kaggle-api#api-credentials), so we will first store safely those credentials in the workspace key vault.

### Option 1: using Azure CLI

1. Let's first obtain your AAD identifier (object id) by running the following command. We'll use it in the next step.

    ```bash
    az ad signed-in-user show --query id
    ```

2. Create a new key vault policy for yourself, and grant permissions to list, set & delete secrets.

    ```bash
    az keyvault set-policy -n <key-vault-name> --secret-permissions list set delete --object-id <object-id>
    ```

    > Note: The AML workspace you created with the aforementioned script contains the name of the key vault. Default is `kv-fldemo`.

3. With your newly created permissions, you can now create a secret to store the `kaggleusername`.

    ```bash
    az keyvault secret set --name kaggleusername --vault-name <key-vault-name> --value <kaggle-username>
    ```

    > Make sure to provide your *Kaggle Username*.

4. Create a secret to store the `kagglekey`.

    ```bash
    az keyvault secret set --name kagglekey --vault-name <key-vault-name> --value <kaggle-api-token>
    ```

    > Make sure to provide the *[Kaggle API Token]((<https://github.com/Kaggle/kaggle-api#api-credentials>))*.

### Option 2: using Azure UI

1. In your resource group (provisioned in the previous step), open "Access Policies" tab in the newly created key vault and click "Create".

2. Select *List, Set & Delete* right under "Secret Management Operations" and press "Next".

3. Lookup currently logged in user (using user id or an email), select it and press "Next".

4. Press "Next" and "Create" in the next screens.

    We are now able to create a secret in the key vault.

5. Open the "Secrets" tab. Create two plain text secrets:

    - **kaggleusername** - specifies your Kaggle user name
    - **kagglekey** - this is the API key that can be obtained from your account page on the Kaggle website.

## Run a job to download and store the dataset in each silo

This can all be performed with ease using a data provisioning pipeline. To run it follow these steps:

1. If you are not using the quickstart setup, adjust the config file  `config.yaml` in `examples/pipelines/utils/upload_data/` to match your setup.

2. Submit the experiment by running:

   ```bash
   python ./examples/pipelines/utils/upload_data/submit.py --example CCFRAUD --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

:warning: Proceed to the next step only once the pipeline completes. This pipeline will create data in 3 distinct locations.

## Run the demo experiment

1. If you are not using the quickstart setup, adjust the config file  `config.yaml` in `examples/pipelines/ccfraud/` to match your setup.

2. Submit the FL experiment by running:

   ```bash
   python ./examples/pipelines/ccfraud/submit.py --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

## Beyond the experiment

This sample experiment provides multiple models you can try:

- **SimpleLinear** : a model fully composed of `torch.Linear` layers with `ReLU` activations, takes data as-is sample-by-sample
- **SimpleLSTM** : a model composed by 4 LSTM layers connected to linear head with architecture similar to **SimpleLinear**, takes data ordered by time in sequences that overlap each other
- **SimpleVAE** : a model composed of 2 encoder LSTM layers and 2 decoder LSTM layers that tries to recreate consumed sequence of transactions, the latent space created by encoder is consumed by a linear layer to perform prediction, takes data ordered by time in sequences that overlap each other

To switch between models, please update the `config.yaml` file in `examples/pipelines/ccfraud/`. Look for the field `model_name` in the `training_parameters` section (use `SimpleLinear`, `SimpleLSTM`, or `SimpleVAE`).

## Distributed training

This sample can be ran in distributed fashion, using [PyTorch Data Distributed Parallel (DDP) with NCCL backend](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). However, this requires us to provision new cluster with >1 CUDA enabled GPUs and allow them to access datastorages in corresponding regions. In order to do so follow these steps for every region you want to run distributed training in:

- Provision GPU cluster with 1+ NVIDIA GPU alongside with storage according to tutorial [here](../provisioning/README.md)
- Adjust the config file  `config.yaml` in `examples/pipelines/ccfraud/` to use the newly created compute
- The pipeline automatically detects number of GPUs on a given compute and scales the job accordingly
- If your cluster can be scaled to more than 1 machine, you can modify `config.yaml` in `examples/pipelines/ccfraud/` by adding `instance_count` to your silo config with number of nodes you would like to run the training on

After performing all these steps you can rerun the experiment and it will run using DDP.
