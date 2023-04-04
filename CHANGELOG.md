# FL Accelerator Changelog

<!--
TEMPLATE FOR MONTHLY UPDATES

## <Month> <Year>
<Short paragraph highlighting the 2-3 most important changes>

### FL Experience
<Include all changes that are worth mentioning in this or the following sections. Not everything will warrant a mention (e.g. a very minor doc change, some change to our CI/CD pipelines that will not matter to end users, etc...). Include links where relevant>

### Provisioning

### Documentation

### Repository structure
-->

## March 2023

We are changing the release process of our [FL Accelerator repository](https://github.com/Azure-Samples/azure-ml-federated-learning). We are moving from monthly releases to continuous releases. We will however keep updating this changelog once a month to highlight the most significant changes.

Our major updates for this month are the addition of resources to provision various flavors of FL sandboxes, a new example for Vertical FL, and better support for encryption (data at rest, communications in Vertical FL).

### FL Experience
- Added support for data encryption at rest to the CCFRAUD example (instructions [here](./docs/real-world-examples/ccfraud.md#enable-confidentiality-with-encryption-at-rest)).
- Introduced new Vertical FL example: [Bank Marketing Campaign Prediction](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/docs/real-world-examples/bank-marketing.md).
- Improved the communications in Vertical FL jobs (support for redis streams and encrypted communications, better logging to measure communications overhead).

### Provisioning
- Released a [new bicep script](./mlops/bicep/modules/fl_pairs/open_aks_with_confcomp_storage_pair.bicep) to deploy silos with AKS clusters using confidential computes and set up open orchestrator.
- Added support for including Kaggle credentials during workspace provisioning (useful for downloading Kaggle data to run our examples); see instructions included in our [real-world examples documentation](./docs/real-world-examples/), or standalone instructions [here](./docs/tutorials/add-kaggle-credentials.md).

### Documentation
- Added some [documentation](./docs/provisioning/sandboxes.md) and the associated bicep scripts to easily deploy sandboxes for exploring different variants of FL setups.
- Added [instructions](./docs/concepts/mlops_for_fl.md) on how to leverage MLOps to restrict FL to peer-reviewed code.
- Added [generic instructions](./docs/tutorials/update-local-data-to-silo-storage-account.md) for uploading encrypted data.

### Repository structure
- Added more CI/CD tests for the examples introduced last month.
- Added unit tests for communication and encryption components.


##  February 2023 release

We are excited to announce the release of the February iteration of our [FL Accelerator repository](https://github.com/Azure-Samples/azure-ml-federated-learning).

Some of the major updates we have made include the launch of a vertical federated learning feature, an FL pipeline that offers a native AML FL experience integrated with the factory engine, and benchmark results that reveal a comprehensive comparison between FL and non-FL experiments.

### FL Experience
- Implemented _Vertical Federated Learning_ and offered a [tutorial](./docs/tutorials/vertical-fl.md) to run MNIST or CCFRAUD examples.
- Introduced a [scatter-gather](./docs/tutorials/literal-scatter-gather-tutorial.md) pipeline that delivers a real AML FL native experience.
- Conducted a comprehensive comparison b/w FL and non-FL experiments and the benchmark report can be accessed [here](./docs/concepts/benchmarking.md).

### Provisioning
- Provided [instructions](./docs/tutorials/update-local-data-to-silo-storage-account.md) and a script to facilitate the upload of local data to a silo storage.
- Incremental improvements:
  - Enhanced the network security rules and minimized the workspace dependencies for provisioning resources.
<!-- ### Documentation -->

<!-- ### Repository structure
-->

To get started, go [here](./docs/quickstart.md)!

If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Azure-Samples/azure-ml-federated-learning/issues).

##  January 2023 release

We are excited to announce the release of the January iteration of our [FL Accelerator repository](https://github.com/Azure-Samples/azure-ml-federated-learning).

Here below we list all the new features. The most significant changes are the introduction of a guide to help onboard people to FL on Azure ML, the implementation of _Differential Privacy_ in all 3 industry-relevant examples, the support of third-party FL frameworks (_NVFlare_ and _Flower_), and the support of distributed training.

### FL Experience
- Implemented _Differential Privacy_ in all 3 industry-relevant examples, and added a [tutorial](./docs/tutorials/dp-for-cross-silo-horizontal-fl.md) that explains the process.
- Introduced experimental support for third-party FL frameworks. The [pneumonia example](./docs/real-world-examples/pneumonia.md) can now be run using _NVFlare_ (see [tutorial](./docs/frameworks/nvflare.md)) or _Flower_ (see [tutorial](./docs/frameworks/flower.md)). 
- Implemented distributed training (in each silo individually) in all 3 industry-relevant examples for scalable training.
- Introduced support for multiple computes per silo (for instance: to use CPU's for pre-processing and GPU's for training).
- Introduced resources for Exploratory Data Analysis on the [credit card fraud example](./docs/real-world-examples/ccfraud.md).
- Incremental improvements:
  - Made sure component scripts can run locally to facilitate authoring and debugging.  
  - Fixed a bug about data loading for the MNIST example.

### Provisioning
- Added [instructions on how to properly configure Confidential Computes](./docs/provisioning/silo_open_aks_with_cc.md) so all of the resources can be properly utilized.
- Updated the bicep templates to also allow for GPU's provisioning (not restricted to CPU's anymore).

### Documentation
- Introduced a [guide to help people plan their onboarding to FL on Azure ML](./docs/concepts/plan-your-fl-project.md).

<!-- ### Repository structure
-->

To get started, go [here](./docs/quickstart.md)!

If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Azure-Samples/azure-ml-federated-learning/issues).

##  November 2022 release

We are excited to announce the release of the November iteration of our [FL Accelerator repository](https://github.com/Azure-Samples/azure-ml-federated-learning).

Here below we list all the new features. The most significant ones are the addition of 3 industry-relevant examples ([credit card fraud detection](./docs/real-world-examples/ccfraud.md), [named entity recognition](./docs/real-world-examples/ner.md), [pneumonia detection](./docs/real-world-examples/pneumonia.md)), and the introduction of a [Troubleshooting Guide](./docs/tsg.md).

<!-- ### Provisioning
-  -->

### FL Experience
- Added 3 industry-relevant examples. The examples include 2 jobs. The first one (generic) downloads some public data, partitions them, and uploads them to the silos. The second job (example-specific) trains a model using FL. The 3 examples address the following topics:
  - [credit card fraud detection](./docs/real-world-examples/ccfraud.md) (finance example);
  - [named entity recognition](./docs/real-world-examples/ner.md) (NLP example);
  - [pneumonia detection](./docs/real-world-examples/pneumonia.md) (medical imaging example). 
- Fixed a bug causing the contents of the `config.json` file to take precedence over the CLI arguments (subscription id, resource group, workspace name) when submitting a job. Now, the CLI arguments takes precedence over the `config.yaml` file, which itself takes precedence over the `config.json` file.

### Documentation
- Introduced a new [Troubleshooting Guide](./docs/tsg.md) to help you troubleshoot common issues. We will keep adding to it as we become aware of more common issues.

### Repository structure
- Improved the [documentation home page](./docs/README.md) by adding some pictures and introducing the industry-relevant examples.
- Revisited our CI/CD processes for better agility (enable concurrent jobs, accommodate token expiration issue).


##  October 2022 release

We are excited to announce the release of the October iteration of our [FL Accelerator repository](https://github.com/Azure-Samples/azure-ml-federated-learning).

Here are the new features.

### Provisioning
- A new provisioning [cookbook](./docs/provisioning/README.md) so you can pick and choose the setup that best suits your needs, based on the provided templates (orchestrator behind vnet, silos behind vnets, silos using an AKS cluster with confidential compute, silos using an existing storage account...).
- All bicep scripts can now be used as standalone to adapt to your specific infrastructure needs.

### Documentation
- Instructions on how to provision [external silos](./docs/provisioning/external-silos.md).
- Tutorial on [how to adapt literal and factory code to your scenarios](./docs/literal-factory-tutorial.md).
- Introduced "one-click deployments" for all bicep templates.

### FL Experience
- Using AzureML SDK subgraphs (preview) as a basis for the factory code sample, so that writing an FL experiment is easier and more flexible, and the experiment graph is shown better in the Azure ML UI.
- Added Subscription ID, Resource Group, and Workspace Name as CLI arguments for job submission.

### Repository structure
- Enabled CI/CD on the repo to ensure nothing breaks.

##  September 2022 release

We are excited to announce the release of the September iteration of our [FL Accelerator repository](https://github.com/Azure-Samples/azure-ml-federated-learning).

Here are the new features.


### Repository structure
- Cleaned up repository - removed redundant directories/files.

### Provisioning
- Auto-provisioning scripts to create a complete sandbox using bicep templates (by default an open sandbox on 3 regions). An experimental script with VNets and private endpoints is also available.
- A quickstart to run an FL demo in minutes, and easily run FL jobs on your own data!

### Documentation
- [Single location](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/release-sdkv2-iteration-02/README.md) for all documentation.

### FL Experience
- Easy navigation using Azure Storage Explorer to discover models of various iterations, here is a more intuitive path (\<experiment-name\>/\<iteration-num\>/\<model-name\>)
- Introduction of "factory" code to simplify the creation of FL experiments.
  - Decoupled components for more readability.
  - Soft validation to check the correct permissions of your assets.
- Metrics:
  - The combined losses and model performances of several silos can now be seen at the pipeline level.
