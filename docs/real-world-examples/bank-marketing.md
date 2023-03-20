# Vertically federated cross-geo bank marketing campaign prediction

**Scenario** - This is a short example where we showcase possibilities of using Azure Machine Learning(AML) for training a model for bank marketing campaign in vertical federating learning fashion. The example showcases how to setup training in the case, where part of the data, including the feature space, is owned by the host. We have simulated a FL scenario by splitting the data into **distinct geo-location**. The sample utilizes **tabular data**.

**Dataset** - This example is trained using the Kaggle dataset [**Bank marketing campaigns dataset | Opening Deposit**](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset). This dataset is describing Portugal bank marketing campaign results, where the campaigns were conducted by calling clients and offering them to place a term deposit.

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

To run this example, you will need to provision an AzureML workspace ready for Federated Learning. We strongly recommend you use the setup provided in the repository [quickstart](../quickstart.md), make sure you use vnet option as this is recommended for all vertical FL examples. We will use the same names for the computes and datastores created by default during this quickstart.

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

Follow the steps in the [Vertical Federated Learning tutorial from Data Provisioning Step](../tutorials/vertical-fl.md#data-provisioning)