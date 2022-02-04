# Milestones to Cross-silo Federated Learning Using Azure ML 

## Summary
This document is for third party customers who are interested in cross-silo Federated Learning (FL) on Azure ML. Specifically, we discuss milestones and main steps to getting a FL project running. 

:exclamation: Note that this document is still bound to evolve. 

## Vocabulary

__Data__. Any file or collection of files. Data will be described in terms of classification. 

Only three classifications are required for the context of this document. "Sensitive" (cannot be moved or even looked at), "intermediate" (can be moved around, but looser restrictions on visibility), and "eyes-on" (can be moved freely and seen by everyone participating in the federated training). 

__Storage__. Wherever data is stored. In this file, storage is assumed to live in Azure. It may exist in locked-down virtual networks. 

__Compute__. Anything that can run "code" (deliberately vague). In this file, compute is assumed to live in Azure. 

__Job__. Execute code (a collection of files) in an environment (a Docker image) against data (from storage). A job can consume data from multiple storage instances and write back to multiple instances. 

__Approval__. REST endpoint which the platform "asks permission" before running any job. The platform sends the approval endpoint information including: 

1. Input and output storage 
2. Which compute the job wishes to run in 
3. The author of the code the job is running 
4. Whether or not the job has been code-signed by the configured policies 

The approval endpoint can either approve / reject the job based on checked-in configuration (e.g., of which storage accounts are associated with which silo) or pass this information on for manual approval. 

:exclamation: Note that the approval endpoints do not support 3P-facing AML yet. 

__Silo__. Isolated collection of storage and compute. Here, "isolated" means that the platform guarantees: 

- Only compute within the silo can "touch" storage within the silo. 
- Only data of intermediate or eyes-on classification can be moved outside the silo. 
- Only "approved" jobs can change the classification of data or move it outside the silo. 

Silos are expected to be reliable (i.e., no concerns around network connectivity or uptime). 

:exclamation:  Note that we assume a hard cap of ≤ 100 silos at current stage. 

__Orchestrator__. Collection of storage and compute. The storage is for model parameters, rather than the actual data. A task orchestrator broadcasts the FL task, sends the current model to each silo, and aggregates the gradients from the silos. In this file, orchestrator is assumed to live in an AML workspace. 


## Key Milestones

- A machine learning model. A working model or model architecture with learning algorithm in a non-federated scheme. 
- Resource provision. Get the orchestrator and silos ready for FL. 
- A test FL job. Run a synthetic test for the FL framework and understand the effect of each FL hyperparameters. 
- An actual FL job. Set the FL hyperparameters and execute the actual job. 

We describe the key milestones in detail below. 


## Machine learning model 

The prerequisite of a _federated_ learning job is a _machine_ learning model, which can be trained in a non-federated scheme. Such a model can be  
- a working model trained with some real yet small data, or  
- a model architecture that is proven effective via synthetic training data. 

One example of a hospital (i.e., the customer) is described below. The hospital’s task is to classify a medical image into positive or negative of a specific disease. The data scientists at the hospital already have a neural network model trained in a centralized fashion with their own patient data. The model achieved reasonable performance. Now the hospital wants to further improve the model's performance as more hospitals would like to participate in a federated learning way without their data leaving their cluster. 

## Resource provision 

If all the data live in __one__ AAD (Azure Active Directory) tenant, we could simply create “vanilla” AML computes and easily use managed identity to enforce silos (via compute --> storage access). Briefly speaking, per-silo storage accounts should be locked down with only RBAC for access and don’t give anyone access keys, and per-silo compute should have a managed identity, give just that managed identity the "blob storage contributor" roles to access the data. For further details, please check out this [public document](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-managed-identities?tabs=python) on managed identity.

If __multiple__ AAD tenants are involved, we provide the following resource provision approach. Specifically, arc-enabled k8s compute clusters are used, since managed identity won’t work for cross-tenants scenarios. Please refer to [this file](https://github.com/Azure-Samples/azure-ml-federated-learning/tree/main/automated_provisioning) for more information. 

1. There should be computes in each of the silos. These computes should all be in their own K8s cluster. If not, we provide [this script](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/automated_provisioning/ps/CreateK8sCluster.ps1) to create a K8s cluster. 
2. Provision an orchestrator workspace, and attach all the silo computes/k8s clusters via Azure Arc. We provide [this script](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/automated_provisioning/ps/ConnectSiloToOrchestrator.ps1) to connect the silos to the orchestrator here.  
3. We also provide a mock [script](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/automated_provisioning/sample_job/RunSampleJob.ps1) here that does the fan-out/fan-in and uses the silos to ensure the orchestrator can access the k8s in each silo. 


## Test FL job 

:exclamation: Note that this part is currently _ongoing_. Thr goal is to provide the same job with our resources as a reference for the customers.  

4. Upload the provided data to each silo’s storage. 
5. Download the provided FL script and model to the machine that can connect to the orchestrator workspace. 
6. Define one or several metric(s) of interest (they should be consistent with your actual FL job). 
7. Select the setting according to the scenario of your actual FL job. For example, differentiable privacy may be added to improve privacy preservation. 
8. Submit the test job to AML and compare the results to our results. 
9. Play around the FL hyperparameters (e.g., the parameters in differentiable privacy, the weights for aggregation) to see their impact on the metrics of interest.  

## Actual FL job 

10. Ready the baseline model (or base query in the case of Federated _Analysis_) and the resources that can be used to train the model. 
11. Confirm there are data in various silos. Access to these data should be restricted. 
12. Submit your actual FL job using the FL pipeline with proper hyperparameters