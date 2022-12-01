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

## Provision an FL sandbox workspace

To run this example, you will need to provision an AzureML workspace ready for Federated Learning. We strongly recommend you use the setup provided in the repository [quickstart](../quickstart.md). We will use the same names for the computes and datastores created by default during this quickstart.

## Run a job to download and store the dataset in each silo

This can all be performed with ease using a data provisioning pipeline. To run it follow these steps:

1. In this repository, navigate in the folder `examples/pipelines/utils/upload_data`

2. If you are not using the quickstart setup, check the config file under ...

3. Submit the experiment by running:

   ```bash
   python ./examples/pipelines/utils/upload_data/submit.py --submit --example NER --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

:warning: Proceed to the next step only once the pipeline completes. This pipeline will create data in 3 distinct locations.

TODO

## Run the demo experiment

1. If you are not using the quickstart setup, adjust the config file under ...

2. Submit the FL experiment by running:

   ```bash
   python ./examples/pipelines/ner/submit.py --submit
   ```

#### References

- [HuggingFace Token Classification](https://huggingface.co/course/chapter7/2?fw=pt#token-classification)
