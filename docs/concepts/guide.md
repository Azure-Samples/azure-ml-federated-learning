# Step-by-Step guide to perform a Federated Learning experiment

The following are some key milestones that can help to perform an experiment in an efficient manner:

- A machine learning model. A working model or model architecture with a learning algorithm in a non-federated scheme.
- Resource provisioning. Get the orchestrator and silos ready for FL.
- Validate FL job. Run a synthetic test for the FL framework and understand the effect of each FL hyperparameter.
- Submit an actual FL job.
- Evaluate the performance of a federated model.  
- Deploy a federated model.

We describe the key milestones in detail below.

## Machine learning model

The pre-requisite of a _federated_ learning job is a _machine_ learning model, which can be trained in a non-federated scheme. Such a model can be  

- a working model trained with some real yet small data, or  
- a model architecture that is proven effective via synthetic training data.

## Resource provisioning

If all the data live in __one__ AAD (Azure Active Directory) tenant, we could simply create “vanilla” AML computes and easily use managed identity to enforce silos (via compute --> storage access). Briefly speaking, per-silo storage accounts should be locked down with only RBAC for access and don’t give anyone access keys, and per-silo compute should have a managed identity, giving just that managed identity the "blob storage contributor" roles to access the data. For further details, please check out this [public document](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-managed-identities?tabs=python) on managed identity.

To quickly provision resources for your Federated Learning experiment setup, follow the steps given [here](../quickstart.md/#deploy-demo-resources-in-azure).

Please visit the provisioning [__cookbook__](../provisioning/README.md) to learn more about the resource provisioning alternatives that are available.

## How to validate an FL job

1. Upload the provided data to each silo’s storage.
2. Download the provided FL script and model to the machine that can connect to the orchestrator workspace.
3. Define one or several metric(s) of interest (they should be consistent with your actual FL job).
4. Select the setting according to the scenario of your actual FL job. For example, differential privacy may be added to improve privacy preservation.
5. Submit the test job to AML and compare the results to our results.
6. Play around with the FL hyperparameters (e.g., parameters in differential privacy, weights for aggregation) to see their impact on the metrics of interest.  

## Submit an FL job

1. Ready the baseline model (or base query in the case of Federated _Analysis_) and the resources that can be used to train the model.
2. Confirm there are data in various silos. Access to these data should be restricted.
3. Submit your actual FL job using the FL pipeline with proper hyperparameters.

## Evaluate a federated model

1. You can find the Mlflow metrics or artifacts in the pipeline metrics section.
2. Look at the loss/evaluation metrics to adjust the hyper-parameters such as learning rate, optimizer, etc.  
3. Keep iterating until you get the desired results.

## Deploy a federated model

After training and evaluating your federated model, the last step is to deploy it! That means using the
model in your product, whether that involves batch (offline) inference or hosting it at a production
endpoint. Since the federated model is a single model, deployment can happen just like with your
non-federated models.
