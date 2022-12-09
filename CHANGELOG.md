# FL Accelerator Changelog

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

To get started, go [here](./docs/quickstart.md)!

If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Azure-Samples/azure-ml-federated-learning/issues).


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
