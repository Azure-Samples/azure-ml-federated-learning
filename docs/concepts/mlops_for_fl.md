# Integrate and secure FL experiments with MLOps

This page tries to answer the following question: What can we do to avoid users running a job intentionally or unintentionally leaking sensitive data?

Among the various solutions to this large question, here we look at it from the perspective of **securing the code to run FL** experiments to avoid (un)intentional bad practices and security breaches.

The solution we're proposing is to **leverage Azure ML integration with MLOps to allow only for peer-reviewed code to run** in a production FL environment.

## Azure ML integration with MLOps

Azure ML provides a [MLOps](https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/manage/mlops-machine-learning) integration, meaning that you can automate the entire lifecycle of your machine learning models, from experimentation to deployment.

> MLOps is based on DevOps principles and practices that increase the efficiency of workflows. Examples include continuous integration, delivery, and deployment. MLOps applies these principles to the machine learning process, with the goal of:
>
> - Faster experimentation and development of models.
> - Faster deployment of models into production.
> - Quality assurance and end-to-end lineage tracking.

What's particularly relevant for FL is that you can use MLOps capabilities to submit the FL experiments for your users, enabling to restrict those submissions only when they match specific approval requirements.

## Restricting production FL experiments to MLOps-only

One strategy to secure the code running FL experiments in your Azure ML workspace is to leverage both RBAC and MLOps integration as follows.

In order to allow for agility, we suggest to create two Azure ML workspaces:
- one for "development" which will be similar to our current [sandboxes](../provisioning/sandboxes.md), in which all users can run anything but on synthetic or non-sensitive (eyes-on) data,
- one for "production" which will be tightly restricted to allow only for peer-reviewed code to run.

For the production workspace, we suggest the following setup.

1. Set a specific repository to host your production code, either in Azure DevOps or GitHub.

2. Create strong **branch policies** such as all changes to the main branch, or certain project branches, imperatively need to be peer-reviewed before being merged.

    This can be quite flexible and can be set up to allow for different levels of peer-review depending on the branch. You can also require specific reviewers for specific branches. This is all configurable in Azure DevOps or GitHub.

3. **Restrict RBAC role of your users to _Reader_** of the Azure ML workspace. They won't be able to run a job, but they will be able to see the code and the results of the experiments.

4. **Create a service principal (SP) with the _Contributor_ role** on the Azure ML workspace (see [How-to](#how-to-implement-mlops-with-azure-ml) below).

    This service principal will be used by the MLOps pipeline to run jobs in the production workspace.

5. Implement an MLOps pipeline either in Azure DevOps or GitHub actions leveraging this SP to run jobs coming from the restricted branches.

    You can leverage one of our sample jobs using the Azure ML SDK to submit an experiment with the references to the workspace coming from the pipeline variables.

    Also, check out how we do that to ensure our own CI/CD in [our own GitHub Actions](../../.github/workflows/pipeline-e2e-test.yaml).

The result is a production Azure ML workspace where only peer-reviewed code from your main branch can be submitted and run on the silos and orchestrators.

## How to implement MLOps with Azure ML

The process described above is documented in the Azure ML product docs on how to set up MLOps with Azure DevOps or GitHub actions. Here's the key relevant links below.

From **Azure DevOps**:

- [Set up MLOps with Azure DevOps](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-mlops-azureml)
- [Use Azure Pipelines with Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-devops-machine-learning)

From **GitHub actions**:

- [Set up MLOps with GitHub](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-mlops-github-azure-ml)
- [Use GitHub Actions with Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-github-actions-machine-learning)

You can also leverage the MLOps accelerator repository:

- [Azure MLOps (v2) Solution Accelerator](https://github.com/Azure/mlops-v2)
- [Deploy MLOps on Azure in Less Than an Hour (MLOps V2 Accelerator)](https://www.youtube.com/watch?v=5yPDkWCMmtk)
