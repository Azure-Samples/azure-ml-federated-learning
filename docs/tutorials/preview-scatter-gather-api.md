# Scatter-Gather API for Federated Learning

**These instructions were copied (and slightly adapted) from the [instructions](https://github.com/Azure/azureml-insiders/blob/main/previews/federated_learning_sdk/README.md) in the private _azureml-insiders_ repository.**

**Please note that in the current branch, the aggregation component in `examples/components/utils/aggregatemodelweights` is slightly different from the one in the main branch, to accommodate for the scatter-gather API.**

## Overview
When authoring Federated Learning (FL) experiments, one will always end up writing code to define graphs with similar shapes (loop over all silos, then outer loop for total number of iterations). Furthermore, one will also need to _anchor_ the components to their respective computes and associated datastores (silo-specific). This is repetitive, and error-prone. The Scatter-Gather API aims to simplify this process by providing a set of primitives that can be used to define a graph without having to use boilerplate code, or to manually anchor components.

If you want to learn more about how to implement Federated Learning on AzureML, please check out our [FL accelerator repository](https://github.com/Azure-Samples/azure-ml-federated-learning). For a success story of Federated Learning on AzureML being used in production, see [this blog post](https://customers.microsoft.com/en-us/story/1587521717158304168-microsoft-partner-professional-services-azure).

## Installation

### Prerequisites
To enjoy this Private Preview, you will need the following:
- an Azure subscription;
- the infrastructure for doing FL on AzureML (we highly recommend you just use one of the [sandboxes](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/docs/provisioning/sandboxes.md) from our FL Accelerator repository - this way, you will not have to adjust silos/orchestrator names in config files).
  - For instance, you can use the _Minimal Sanbox_ and deploy it by clicking the button below. If you choose this sandbox though, please remember that it is intended **only for demo** purposes. The data is still accessible by the users of your subscription when opening the storage accounts, and data exfiltration is possible. This supports only Horizontal FL scenarios.
  
    [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fsandbox_minimal.json)

### Set up your environment
- Clone the current repository and set the root as your working directory.
- Create a conda environment from the provided yml file and activate it with the following commands.
  ```bash
  conda env create -f ./examples/pipelines/environment_scatter_gather_preview.yml
  conda activate fl_sdk_preview_env
  ```

## Usage
To demonstrate how to use the Scatter-Gather API, we will perform the classic hand-written digit recognition task on the MNIST dataset. We will build the graph and submit the job using [this notebook](../../examples/pipelines/fl_scatter_gather_preview/sample.ipynb).

### 1. Point to your AzureML workspace
First, you will want to open the [examples/pipelines/fl_scatter_gather_preview/config.yaml](../../examples/pipelines/fl_scatter_gather_preview/config.yaml) file. Provide your subscription Id, resource group name, and workspace name in the `aml` section (you will also need to uncomment these lines).

Then you can double check the names of the silos' and orchestrator's compute and datastore. If you used one of our sandboxes for provisioning your FL infrastructure, you should not need to modify anything (as we mentioned earlier).

### 2. Follow instructions in the notebook
At this point, you can just run the notebook and follow the instructions. In this document, we will just highlight the key parts that are different from what you should be used to if you are familiar with our [accelerator repository](https://github.com/Azure-Samples/azure-ml-federated-learning).

- Make sure you set the `"AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED"` environment variable to `"True"` **before** importing the AzureML SDK (azure.ai.ml).
- Create the list containing your silos' information. This is done with the following code snippet. See how we just read the information from the [config.yaml](../../examples/pipelines/fl_scatter_gather_preview/config.yaml) config file.
  ```python
  silo_list = [
      FederatedLearningSilo(
          compute=silo_config["computes"][0],
          datastore=silo_config["datastore"],
          inputs= {
              "silo_name": silo_config["name"],
              "raw_train_data": Input(**dict(silo_config["inputs"])["training_data"]),
              "raw_test_data": Input(**dict(silo_config["inputs"])["testing_data"]),
          },
      )
      for silo_config in YAML_CONFIG.strategy.horizontal
  ]
  ```
  - Note that using a config file is NOT mandatory. We usually find it more convenient to put all parameters in one file, but if you prefer you can also just create the list of silos directly in the notebook.
- Create mappings for arguments and inputs - you should not have to modify anything in those cells.
- Build your silo subgraph as usual - nothing different from what you should be already used to.
- Build the FL pipeline with the code snippet below. This is where the magic happens. We just use the `fl_scatter_gather` API, which will build the whole FL graph for us, and anchor the silos' components appropriately. See how we use the list of silos and subgraph we created earlier, along with the mappings, and some additional arguments from the config file.
  ```python
  fl_node = fl.fl_scatter_gather(
      silo_configs=silo_list,
      silo_component=silo_scatter_subgraph,
      aggregation_component=aggregate_component,
      aggregation_compute=YAML_CONFIG.orchestrator.compute,
      aggregation_datastore=YAML_CONFIG.orchestrator.datastore,
      shared_silo_kwargs=silo_kwargs,
      aggregation_kwargs=agg_kwargs,
      silo_to_aggregation_argument_map=silo_to_aggregation_argument_map,
      aggregation_to_silo_argument_map=aggregation_to_silo_argument_map,
      max_iterations=YAML_CONFIG.general_training_parameters.num_of_iterations,
  )
  ```
- Submit your job as usual.

## Feedback
Thank you for participating in our **private preview** of Azure Machine Learning's new **Scatter-Gather API for Federated Learning**.

We would love to learn from your experience using this new functionality as your feedback is critical to helping us improve our product. 

This [short survey](https://forms.office.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbRwK9MyoT9PJEv-TmHpMxJuNUNlMyV0k0NlhNVTVPUkxDQVhMVFpWNFVDOC4u) should take approximately 5 minutes to complete. We’re looking forward to what you have to say, so please give us your honest opinions.  

