# Adapt our sample "literal" code to your needs

IMPORTANT: the "literal" code available in this repo has been intentionally designed to:

- provide an effortless setup to get started.
- rely only on features that are currently generally available in AzureML SDK v2.

This tutorial addresses the following scenarios:

- To add/remove a silo:
  - You just need to make the changes in the "`federated_learning/silos`" section of the `examples/pipelines/fl_cross_silo_literal/config.yaml` file.

- To change training hyper-parameters:
  - Adjust the parameters in the "`training_parameters`" section of the `examples/pipelines/fl_cross_silo_literal/config.yaml` file.

## Tutorial on how to adapt the "scatter-gather" code

Please read the following points to have a better understanding of the "scatter-gather" code:

- It has a `set_orchestrator` method that you can leverage to add an orchestrator to your pipeline.
- The `add_silo` method lets you add `n` number of silos to the pipeline and you don't have to worry about the configuration.
- It has a soft validation component that ensures that the appropriate permissions are granted for your assets. That being said, a compute `a` should not have access to dataset `b` and so on.
- You can bypass the validation if you have your own custom rules.
- Enabling type-check, ensures that no data is being saved and only model weights are kept in the datastore.

This tutorial addresses the following scenarios:

- To add/remove a silo:
  - You just need to make the changes in the "`strategy/horizontal`" section of the `examples/pipelines/fl_cross_silo_scatter_gather/config.yaml` file.

- To change the training hyper-parameters:
  - Adjust the parameters in the "`inputs`" section of the `examples/pipelines/fl_cross_silo_scatter_gather/config.yaml` file.

- To edit the flow of your training pipeline:
  - Pass your custom subgraph as a parameter to the `scatter_gather` method in the `examples/pipelines/fl_cross_silo_scatter_gather/submit.py` file.

- To bypass the soft validation:
  - Use `--ignore_validation` argument while executing the `examples/pipelines/fl_cross_silo_scatter_gather/submit.py` file.

- To enable multiple computes(CPU for preprocessing & GPU for training):
  - Set the `compute2` parameter to `true` while [provisioning](../quickstart.md#deploy-demo-resources-in-azure) the resources.(No further changes are required)
