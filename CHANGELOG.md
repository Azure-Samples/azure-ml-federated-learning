# FL Accelerator Changelog

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

To get started, go [here](./docs/quickstart.md)!

If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Azure-Samples/azure-ml-federated-learning/issues).
