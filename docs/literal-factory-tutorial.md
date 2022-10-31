## Tutorial on how to adapt the "literal" code

Before proceeding, please read the following points to have a better understanding of the "literal" code:
1. It has an effortless setup to get started.
2. It does not have complex/advanced features such as subgraphs, assest permissions validation, etc. 

Scenarios:
1. To add/remove a silo:
    - You just need to make the changes in the "`federated_learning/silos`" section of the `examples/pipelines/fl_cross_silo_literal/config.yaml` file.

2. To change training hyper-parameters:
    - Adjust the parameters in the "`training_parameters`" section of the `examples/pipelines/fl_cross_silo_literal/config.yaml` file.


## Tutorial on how to adapt the "factory" code

Before proceeding, please read the following points to have a better understanding of the "factory" code:
1. It has a `set_orchestrator` method that you can leverage to add an orchestrator to your pipeline.
2. The `add_silo` method lets you add `n` number of silos to the pipeline and you don't have to worry about the configuration.
3. It has a soft validation component that ensures that the appropriate permissions are granted for your assets. That being said, a compute `a` should not have access to dataset `b` and so on.
4. You can bypass the validation if you have your own custom rules.
5. Enabling type-check, ensures that no data is being saved and only model weights are kept in the datastore.

Scenarios:
1. To add/remove a silo:
    - You just need to make the changes in the "`federated_learning/silos`" section of the `examples/pipelines/fl_cross_silo_factory/config.yaml` file.

2. To change the training hyper-parameters:
    - Adjust the parameters in the "`training_parameters`" section of the `examples/pipelines/fl_cross_silo_factory/config.yaml` file.

3. To edit the flow of your training pipeline:
    - Pass your custom subgraph as a parameter to the `build_flexible_fl_pipeline` method in the `examples/pipelines/fl_cross_silo_factory/submit.py` file.

4. To bypass the soft validation:
    - Use `--ignore_validation` argument while executing the `examples/pipelines/fl_cross_silo_factory/submit.py` file. 



