# Run a Federated Learning Demo in 5 mins

In this tutorial, you will:
* Provision a fully functional environment in your own Azure subscription
* Run a sample federated learning pipeline in Azure ML

:warning: **IMPORTANT** :warning: This setup is intended only for demo purposes.
The data is still accessible by the a user of your subscription when opening the storage accounts,
and data exfiltration is easy.

## Prerequisites

To enjoy this quickstart, you will need to:
- have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

## Deploy demo resources in Azure

In this section, we will use a [bicep](https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/overview) script to automatically provision a minimal set of resources for an FL sandbox demo.

This will help you provision a Federated Learning setup with [_internal silos_](./glossary.md), _i.e._ silos that are in the same Azure tenant as the orchestrator. You will be able to use this setup to run the examples in the `./examples/pipelines` directory.

For this iteration, we are only provisioning a _vanilla_ setup, _i.e._ a setup with no security or governance features, where the silos are NOT locked down. We will add these features in future iterations. In the mean time, you should NOT be uploading sensitive data to this setup. However, you CAN use this setup for refining your training pipelines and algorithms. The only elements that will need to change when working on a _real_ secure setup will just be the orchestrator and silos names in a config file - you will be able to re-use all your code as-is.

We will provision:
- 1 Azure ML workspace
- 1 CPU cluster and 1 blob storage account for the [orchestrator](./glossary.md)
- 3 [internal silos](./glossary.md) in 3 different regions (`westus`, `francecentral`, `brazilsouth`) with their respective compute cluster and storage account
- 4 user assigned identifies (1 for orchestration, 1 for each silo) to restrict access to the silo's storage accounts.

1. Using the [`az` cli](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli), log into your Azure subscription:

    ```bash
    az login
    az account set --name <subscription name>
    ```

2. Run the bicep deployment script:

    ```bash
    # create a resource group for the resources
    az group create --name <resource group name> --location <region>

    # deploy the demo resources in your resource group
    az deployment group create --template-file ./mlops/bicep/open_sandbox_setup.bicep --resource-group <resource group name> --parameters demoBaseName="fldemo"
    ```

    > NOTE: if someone already provisioned a demo with the same name in your subscription, change `demoBaseName` parameter to a unique value.

## Launch the demo experiment

In this section, we'll use a sample python script to submit a federated learning experiment to Azure ML. The script will need to connect to your newly created Azure ML workspace first.

1. Install the python dependencies
    
    ```bash
    python -m pip install -r ./examples/pipelines/fl_cross_silo_native/requirements.txt
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
    python ./examples/pipelines/fl_cross_silo_native/submit.py --example MNIST --submit
    ```

The script will submit the experiment to Azure ML. **It should open a direct link to the experiment** in the Azure ML UI.

If not, the script will print the URL to use in clear:

```log
Submitting the pipeline job to your AzureML workspace...
Uploading preprocessing (0.01 MBs): 100%|#######################################| 7282/7282 [00:00<00:00, 23820.31it/s]
Uploading traininsilo (0.01 MBs): 100%|#########################################| 9953/9953 [00:00<00:00, 32014.81it/s]
Uploading aggregatemodelweights (0.01 MBs): 100%|###############################| 5514/5514 [00:00<00:00, 14065.83it/s]

The url to see your live job running is returned by the sdk:
https://ml.azure.com/runs/.....
```
### Look at the pipeline in the AML UI

Go to the above URL and your pipeline would look similar to this: 

<br/><br/>
<img src="./pics/pipeline-aml.PNG" alt="Federated Learning AML pipeline Figure" width="400">

If you want to look at the pipeline metrics, go to the "Job overview" (top-right corner) and then click on the "Metrics(preview)". The following screenshot shows what that would look like.

<br/><br/>
<img src="./pics/metrics.PNG" alt="Federated Learning AML pipeline Figure" width="400">

You can also create your own custom graph by clicking on the "Create custom chart" icon. Here is a sample custom chart showing the "Training Loss" of multiple silos in one graph. 

<br/><br/>
<img src="./pics/combined-losses-silos.PNG" alt="Federated Learning muliple silos Training Loss Figure" width="400">
