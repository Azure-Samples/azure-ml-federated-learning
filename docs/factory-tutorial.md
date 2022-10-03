## Tutorial on how to adapt the Factory code

Before proceeding, please read the following points to have a better understanding of the factory code:
1. It has a `set_orchestrator` method that you can leverage to add an orchestrator to your pipeline.
2. The `add_silo` method lets you add `n` number of silos to the pipeline and you don't have to worry about the configuration. It is being taken care of.
3. It has a soft validation component that ensures that the appropriate permissions are granted for your assets. That being said, a dataset `b` should not have access by a compute `a` and so on.
4. You can bypass the validation if you have your own custom rules.
5. It makes sure that no data is being saved and only model weights are kept on the datastore by enabling type-check.
