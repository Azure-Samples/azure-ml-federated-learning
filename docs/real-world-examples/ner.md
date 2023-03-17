# Federated Named Entity Recognition

**Scenario** - This example shows how to train a federated model for the Named Entity Recognition task. We mimic a real-world FL scenario where multiple institutions share labelled data for NER, but do not want to share the data with each other or with a central entity.  
The model will be trained in a federated manner, where each entity will train a model on its own data, and the models will be aggregated to produce a final model.

**Dataset** - This tutorial uses the [MutliNERD](https://github.com/Babelscape/multinerd/blob/master/README.md) dataset. To simulate an FL scenario, we split the dataset randomly into distinct parts, each on a distinct silo.

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

:warning: **Please provision GPU computes as the example will not work with the `STANDARD_DS3_V2` CPU compute. To do so, simply adjust (either via the command line, or in the one-click deployment UI) the `computeSKU` parameter in the quickstart to a GPU SKU, for instance `Standard_NC12s_v3`.** An overview of the GPU SKU's available in Azure can be found [here](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu).

:notebook: Take note of your workspace name, resource group and subscription id. You will need them to submit the experiment.

## Run a job to download and store the dataset in each silo

This can all be performed with ease using a data provisioning pipeline. To run it follow these steps:

1. If you are not using the quickstart setup, adjust the config file  `config.yaml` in `examples/pipelines/utils/upload_data/` to match your setup. You might need to change the computes and datastores names to those of your GPU's.

2. Submit the experiment by running:

   ```bash
   python ./examples/pipelines/utils/upload_data/submit.py --example NER --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

:warning: Proceed to the next step only once the pipeline completes. This pipeline will create data in 3 distinct locations.

## Run the demo experiment

1. If you are not using the quickstart setup, adjust the config file  `config.yaml` in `examples/pipelines/ner/` to match your setup. You might need to change the compute and datastore names to those of your GPU's, and also to modify the data paths accordingly.

2. Submit the FL experiment by running:

   ```bash
   python ./examples/pipelines/ner/submit.py --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

## References

- [HuggingFace Token Classification](https://huggingface.co/course/chapter7/2?fw=pt#token-classification)
