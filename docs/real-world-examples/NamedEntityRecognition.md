# Named Entity Recognition using MultiNERD dataset

## Background
This examples shows how to train a federated model for Named Entity Recognition task. The model is trained on the [MutliNERD](https://github.com/Babelscape/multinerd/blob/master/README.md) dataset.  
The below instructions cover an end-to-end procedure that includes provisioning resources, training scripts, etc to perform this experiment. This example can be used as a template to perform any NLP task in a federated manner. 

> For the sake of simplicity, we will only provision an _open_ setup. Do not upload sensitive data to it! 
## Prerequisites
To enjoy this tutorial, you will need to:
- have an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli),
- have a way to run bash scripts;
- **fork** this repository (_fork_ as opposed to _clone_ because you will need to create GitHub secrets and run GitHub actions to prepare the data).

## Procedure
Please follow the below instructions to provision resources and then run the experiment:

### Provisioning
Follow the instructions in the [quickstart](../quickstart.md) to provision an open sandbox. Make note of the name of the resource group you provisioned, as well as the name of the workspace.

### Run the FL job

1. Create a conda environment for _submitting_ the job, and activate it.
   ```bash
   conda env create --file ./examples/pipelines/namedentityrecognition/environment.yml
   conda activate fl_ner_env
   ```
2. Adjust config file (if you kept everything default you'll only have to adjust subscription id, resource group, and workspace name)
3. Submit the experiment.
   ```bash
   python ./examples/pipelines/namedentityrecognition/submit.py --submit
   ```