# Named Entity Recognition using MultiNERD dataset

## Background
This example shows how to train a federated model for the Named Entity Recognition task. This tutorial uses the [MutliNERD](https://github.com/Babelscape/multinerd/blob/master/README.md) dataset.  
The steps listed below describe a step-by-step process for carrying out this experiment, including provisioning resources, uploading data, creating training scripts, etc. Any NLP work can be carried out in a federated fashion using this example as a template.

> For the sake of simplicity, we will only provision an _open_ setup. Do not upload sensitive data to it! 
## Prerequisites
To enjoy this tutorial, you will need to:
- have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

## Procedure
Please follow the below instructions to provision resources, upload data, and then run the experiment:

### Provision Resources
The instructions are provided in the [quickstart](../quickstart.md) to provision an open sandbox. Make note of the name of the resource group you provisioned, as well as the name of the workspace.

### Upload Data
The steps to upload data to various datastores are as follows:

Note: This is not required if you've already uploaded the data into their respective datastores.

1. Make sure the train and test paths in the `./examples/components/NER/upload_data_to_silos/spec.yaml` and `./examples/pipelines/ner/config.yaml` files are the same. 

2. Run the below command to create a job in Azure ML that uploads data to datastores. (Note: This may take a few minutes to finish.)
   ```bash
   az ml job create --file ./examples/components/NER/upload_data_to_silos/spec.yaml --workspace-name <workspace-name> --resource-group <resource-group-name>
   ```

3. Verify if the data is successfully uploaded. (Go to AML Studio ->  Data -> DataStores -> (datastore-name) -> Browse)


### Run the FL job

1. Create a conda environment for _submitting_ the job, and activate it.
   ```bash
   conda env create --file ./examples/pipelines/ner/environment.yml
   conda activate fl_ner_env
   ```

2. Adjust the `./examples/pipelines/ner/config.yaml` file (if you kept everything default you'll only have to adjust subscription id, resource group, and workspace name)

3. Submit the experiment.
   ```bash
   python ./examples/pipelines/ner/submit.py --submit
   ```

#### References

- [HuggingFace Token Classification](https://huggingface.co/course/chapter7/2?fw=pt#token-classification)