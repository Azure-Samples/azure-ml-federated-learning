# Run a Federated Learning Demo in 5 mins

In this tutorial, you run your first Python script in the cloud with Azure Machine Learning. This tutorial is *part 1 of a three-part tutorial series*.

This tutorial avoids the complexity of training a machine learning model. You will run a "Hello World" Python script in the cloud. You will learn how a control script is used to configure and create a run in Azure Machine Learning.

In this tutorial, you will:
* Provision a fully functional environment in your own Azure subscription
* Run a sample federated learning pipeline in Azure ML

## Prerequisites

To enjoy this quickstart, you will need:
- [ ] to have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purpose,
- [ ] to have permissions to create resources, set permissions and create identities in this subscription,
- [ ] to [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

## Deploy demo resources in Azure

In this section, we will use a [bicep](https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/overview) script to automatically provision a minimal set of resources for a FL sandbox demo.

We will provision:
- 1 Azure ML workspace
- 1 CPU cluster and 1 blob storage account for the [orchestrator](concepts.md)
- 3 [internal silos](concepts.md) in 3 different regions (`eastus2`, `westus`, `westus2`) with their respective compute cluster and storage account
- 4 user assigned identifies (1 for orchestration, 1 for each silo) to restrict permission access to the silos storage accounts.

1. Using the [`az` cli](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli), log into your Azure subscription:

    ```bash
    az login
    az account set --name <subscription name>
    ```

2. Run the bicep deployment script:

    ```bash
    az deployment sub create --template-file ./mlops/bicep/vanilla_demo_setup.bicep --location eastus --parameters demoBaseName="fldemo"
    ```

    > NOTE: if someone already provisioned a demo with the same name in your subscription, change `demoBaseName` parameter to a unique value.

## Launch the demo experiment

In this section, we'll use a sample python script to submit a federated learning experiment to Azure ML. The script will need to connect to your newly created Azure ML workspace first.

1. Install the python dependencies
    
    ```bash
    python -m pip install -r ./examples/pipelines/requirements.txt
    ```

2. To connect to your newly created Azure ML workspace, we'll need to create a `config.json` file at the root of this repo. Follow the instructions on how to get this from the [Azure ML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace).

    ```json
    {
        "subscription_id": "<subscription-id>",
        "resource_group": "<resource-group>",
        "workspace_name": "<workspace-name>"
    }
    ```

    > NOTE: `config.json` is in our `.gitignore` to avoid pushing it to git.

3. Run a sample python script:

    ```bash
    python ./examples/pipelines/fl_cross_silo_basic/submit.py --example MNIST --submit
    ```

The script will submit the experiment to Azure ML. **It should open a direct link to the experiment** in the Azure ML UI.

If not, the script will print the url to use in clear:

```log
Submitting the pipeline job to your AzureML workspace...
Uploading preprocessing (0.01 MBs): 100%|#######################################| 7282/7282 [00:00<00:00, 23820.31it/s]
Uploading traininsilo (0.01 MBs): 100%|#########################################| 9953/9953 [00:00<00:00, 32014.81it/s]
Uploading aggregatemodelweights (0.01 MBs): 100%|###############################| 5514/5514 [00:00<00:00, 14065.83it/s]

The url to see your live job running is returned by the sdk:
https://ml.azure.com/runs/.....
```
