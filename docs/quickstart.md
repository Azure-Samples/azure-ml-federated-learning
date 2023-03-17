# Run a Federated Learning Demo in 5 mins

In this tutorial, you will:

* Provision a fully functional environment in your own Azure subscription
* Run a sample federated learning pipeline in Azure ML

## Prerequisites

To enjoy this quickstart, you will need to:

* have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
* have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  * Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
* [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

## Deploy demo resources in Azure

### Option 1: One click ARM deployment

Click on the buttons below depending on your goal. It will open in Azure Portal a page to deploy the resources in your subscription.

| Button | Description |
| :-- | :-- |
| [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fopen_sandbox_setup.json) | Deploy a completely open sandbox to allow you to try things out in an eyes-on environment. This setup is intended only for demo purposes. The data is still accessible by the users of your subscription when opening the storage accounts, and data exfiltration is possible. |
| [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fvnet_publicip_sandbox_setup.json) | Deploy a sandbox where the silos storages are kept eyes-off by a private service endpoint, accessible only by the silo compute through a vnet. |

> Notes:
>
> * If someone already provisioned a demo with the same name in your subscription, change **Demo Base Name** parameter to a unique value.
> * If you need to provision GPU's instead of CPU's, you can just use a GPU SKU value for the "Compute SKU" parameter, `Standard_NC12s_v3` for instance. An overview of the GPU SKU's available in Azure can be found [here](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu). Beware though, SKU availability may vary depending on the region you choose, so you may have to use different Azure regions instead of the default ones.

### Option 2: Step by step tutorial

In this section, we will use [bicep](https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/overview) scripts to automatically provision a minimal set of resources for an FL sandbox demo.

This will help you provision a Federated Learning setup with [_internal silos_](./concepts/glossary.md), _i.e._ silos that are in the same Azure tenant as the orchestrator. You will be able to use this setup to run the examples in the `./examples/pipelines` directory.

In this setup, the communications between the silos and the orchestrator are secure, and the silos will not have any access to the other silos' data.

We will provision:

* 1 Azure ML workspace
* 1 CPU cluster and 1 blob storage account for the [orchestrator](./concepts/glossary.md)
* 3 [internal silos](./concepts/glossary.md) in 3 different regions (`westus`, `francecentral`, `brazilsouth`) with their respective compute cluster and storage account
* 4 user assigned identifies (1 for orchestration, 1 for each silo) to restrict access to the silo's storage accounts.

1. Using the [`az` cli](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli), log into your Azure subscription:

    ```bash
    az login
    az account set --name <subscription name>
    ```

2. Optional: Create a new resource group for the demo resources. Having a new group would make it easier to delete the resources afterwards (deleting this RG will delete all resources within).

    ```bash
    # create a resource group for the resources
    az group create --name <resource group name> --location <region>
    ```

    > Notes:
    >
    > * If you have _Owner_ role only in a given resource group (as opposed to in the whole subscription), just use that resource group instead of creating a new one.

3. Run the bicep deployment script in a resource group you own:

    ```bash
    # deploy the demo resources in your resource group
    az deployment group create --template-file ./mlops/bicep/open_sandbox_setup.bicep --resource-group <resource group name> --parameters demoBaseName="fldemo"
    ```

    > Notes:
      > * If someone already provisioned a demo with the same name in your subscription, change `demoBaseName` parameter to a unique value.
      > * :warning: **IMPORTANT** :warning: This setup is intended only for demo purposes. The data is still accessible by the users of your subscription when opening the storage accounts, and data exfiltration is possible.
      > * Alternatively, you can try provisioning a sandbox where the silos storages are kept eyes-off by a private service endpoint, accessible only by the silo compute through a vnet. To try it out, use `--template-file ./mlops/bicep/vnet_publicip_sandbox_setup.bicep` instead. Please check the header of that bicep file to understand its capabilities and limitations. To enable VNet Peering, set the `applyVNetPeering` parameter to `true`.
      > * By default, only one CPU compute is created for each silo. Please set the `compute2` parameter to `true` if you wish to create both CPU & GPU computes for each silo.
      > * Some regions don't have enough quota to provision GPU computes. Please look at the headers of the `bicep` script to change the `region`/`computeSKU`.

## Launch the demo experiment

In this section, we'll use a sample python script to submit a federated learning experiment to Azure ML. The script will need to connect to your newly created Azure ML workspace first.

1. Create a conda environment with all the python dependencies, then activate it.

    ```bash
    conda env create --file ./examples/pipelines/environment.yml
    conda activate fl_experiment_conda_env
    ```

    Alternatively, you can install the dependencies directly:

    ```bash
    python -m pip install -r ./examples/pipelines/requirements.txt
    ```

2. To connect to your newly created Azure ML workspace, you'll need to provide the following info in the sample python script as CLI arguments.

    ```bash
    python ./examples/pipelines/fl_cross_silo_literal/submit.py --subscription_id <subscription_id> --resource_group <resource_group> --workspace_name <workspace_name> --example MNIST
    ```
    > Notes: 
        > * You can use --offline flag when running the job to just build and validate pipeline without submitting it.
        > * Differential privacy is disabled by default, but you can quickly turn it on by setting the `config.yaml` file's `dp` parameter to `true`.
    
    Note: you can also create a `config.json` file at the root of this repo to provide the above information. Follow the instructions on how to get this from the [Azure ML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace).

    ```json
    {
        "subscription_id": "<subscription-id>",
        "resource_group": "<resource-group>",
        "workspace_name": "<workspace-name>"
    }
    ```

    >Note: The `config.json` is in our `.gitignore` to avoid pushing it to git.

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
