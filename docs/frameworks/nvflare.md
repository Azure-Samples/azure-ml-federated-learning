# FL using NVFlare in AzureML (Experimental)

**Scenario** - This tutorial is an extension of the [pneumonia / radiography classification tutorial](../real-world-examples/pneumonia.md) using the NVFlare framework instead.  
We run an NVFlare federated learning job to detect pneumonia from chest radiographs. We mimic a real-world FL scenario where 3 hospitals in 3 different regions want to collaborate on training a model to detect pneumonia from chest radiographs. The hospitals have their own data, and they want to train a model on all data without directly sharing data with each other, or with a central entity.  

:warning: Experimental :warning: This tutorial is relying on some experimental code. It will work on our quickstart sandbox, but running it outside of this controlled setup might not work. We're actively working to release a more robust and generic solution.

**Dataset** - The model is trained on the [Chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle. This example is adapted from [another FL solution](https://github.com/Azure/medical-imaging/tree/main/federated-learning) by Harmke Alkemade _et al._. The code has been modified only to allow for the ingestion of training data from a mounted AzureML datastore.

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

## Provision an FL sandbox workspace using vnet peering

To run this example, you will need to provision an AzureML workspace ready for Federated Learning.

**IMPORTANT**: Provision a [quickstart eyes-off/vnet-based setup](../quickstart.md) and select `applyVNetPeering="true"` to apply peering to the orchestrator and silos vnet. Without this setting, this demo will NOT work.

In the following tutorial, we will use the same names for the computes and datastores created by default during this quickstart.

:notebook: take note of your workspace name, resource group and subscription id. You will need them to submit the experiment.

## Create the datasets

Follow the instructions in the [pneumonia real-world-example](../real-world-examples/pneumonia.md) to unpack the dataset: 1) add your kaggle credentials, 2) run a job to download and store the dataset in each silo.

## Run the demo experiment

1. If you are not using the quickstart setup, adjust the config file `project.yaml` in `examples/pipelines/pneumonia_nvflare/` to match your setup.

2. Submit the NVFlare+AzureML experiment by running:

   ```bash
   python ./examples/pipelines/pneumonia_nvflare/submit.py --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

## What to expect

This will create a job in AzureML, this job will submit 4 other jobs within a pipeline, one in the orchestrator for the NVFlare server, and 3 in each silo for each of the NVFlare clients.

Those jobs connect to one another: each silo client will directly connect to the server job using its private IP through the vnet peering, using the NVFlare protocol. This works thanks to a trick using mlflow: the server job will report its private IP address as an mlflow tag, and the client jobs will fetch it from there. This currently works only through vnets and private IPs.

During the execution, you can check out the logs in each of both the server and the clients jobs::
- `outputs/nvflare_host.log` : the logs of the "runtime" that wraps the NVFlare server/clients as an AzureML job
- `user_logs/std_log.txt` : the logs of the NVFlare application

Also, if you click on the server job, you will see the MLFlow metrics reporting there.

## How to adapt to your own scenario

The example is designed to be as simple as possible, and to be easily adapted to your own scenario.

1. Change the arguments of `submit.py`, in particular:

    - `--project_config` should point to your NVFlare project/provision config file
    - `--nvflare_app` should point to the NVFlare application folder containing your training code

2. Change the provisioning config `project.yaml` to match your setup, in particular:

    - leave admin as is (`name: "admin@azure.ml"`)
    - only one server is supported
    - add any worker but modify the `azureml` section in each worker to specify the data, compute, datastore.

3. If you need to change the environment for your NVFlare training code, check the container definitions in `/examples/components/NVFLARE/client/environment/` and `/examples/components/NVFLARE/server/environment/`.
