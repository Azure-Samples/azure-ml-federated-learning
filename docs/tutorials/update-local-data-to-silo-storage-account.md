# Upload local data to silo storage account

## Background
When taking their first steps with Federated Learning on Azure ML, data scientists might want to upload local data to the silo storage account. Assuming the setup has been provisioned _via_ the templates in this repository, the data scientist will NOT have access to the silo storage account. This makes it impossible to manually upload the local data.

## Objective and Contents
The goal of this tutorial is to show how to upload the contents of a local folder to a silo storage account. We will be using a CLI job to do the upload. The job will run on the silo compute which, unlike the data scientist, does have access to the silo storage account.

## Prerequisites
To enjoy this tutorial, you will need:
- a Federated Learning setup (provisioned using the templates in our [quickstart](../quickstart.md), for instance);
- the [Azure ML CLI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public) installed on your machine;
- some local folder with data to upload.

## Procedure
We will be using the Azure ML CLI to submit a job to the silo compute. The job will upload the local data to the silo storage account. The job is defined in a YAML file that we will call the "job YAML" and that you mill need to adjust to your silo. Detailed steps are as follows.

### 0. Clone the repository
First of all, you will want to clone the current repository to your machine, if you haven't already done so.

### 1. Adjust the job YAML
A sample job YAML is given [here](../../examples/cli-jobs/upload-local-data-to-silo-storage.yml) in our repository, in the `/examples/cli-jobs/upload-local-data-to-silo-storage.yml` file. The contents are shown below.
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json

command: |
  cp -r ${{inputs.local_data_folder}}/* ${{outputs.destination_folder}}
inputs:
  local_data_folder:
    type: uri_folder
    path: /path/to/local/data/folder # replace '/path/to/local/data/folder' by the actual path to the folder whose contents you want to upload

outputs:
  destination_folder:
    type: uri_folder
    mode: upload
    path: azureml://datastores/<your-silo-datastore-name>/paths/<your-custom-local-path>/ # replace '<your-silo-datastore-name>' by the actual datastore name for your silo, and <your-custom-local-path> by the path you want to use in the silo storage account

environment: azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest

compute: azureml:<your-silo-compute-name> # replace '<your-silo-compute-name>' by the actual compute name for your silo
```

 You will need to adjust the following parameters.
- `inputs.local_data_folder.path`: the path to the local folder whose contents you want to upload (no '/' character at the end).
- `outputs.destination_folder.path`: the path in the silo storage account where you want to upload the local data (with a '/' character at the end). **The key part to adjust is the name of the datastore.** You also have the ability to adjust the detailed path in the storage account, if you so wish.
- `compute`: the name of the silo compute. This is the compute that will run the job. It needs to match the datastore mentioned above.

### 2. Run the upload job
Once you have adjusted the job YAML to your needs, you can submit the job using the following command. You will need to replace the `<placeholders>` with the actual values for your workspace.
```
az ml job create --file ./examples/cli-jobs/upload-local-data-to-silo-storage.yml --resource-group <your-workspace-resource-group> --workspace-name <your-workspace-name> --subscription <your-subscription-id>
```
As long as you have provided the proper datastore and compute names corresponding to your silo, the job should succeed.
