# Road Map for a Federated Learning Proof of Concept on Azure

## Contents

This document aims at laying down the concrete steps to be followed to get a Federated Learning (FL) proof of concept (PoC) running on Azure. Its goal is to bring clarity about these steps, and to help assess the cost in terms of time and people. It means to provide more detailed guidance than that provided in our [milestones](https://github.com/Azure-Samples/azure-ml-federated-learning/blob/main/plan/milestones.md) document.

After listing the prerequisites, the document is structured as follows.
1. Create all the basic building blocks, and run a mock FL job to ensure all the pieces are working. At this stage we won't have the security features yet, so we will not be working with any sensitive data.
2. Add all the security features to lock down access to data, enforce code signing, etc... When all the features are in place, run a _real_ FL job on _real_ data.
3. If some specific features are required, then will be time to implement. Specific features of a PoC could be, for instance, some tooling to help spinning up new silos, a detailed comparison of performance against a baseline model, etc...

This is the very first version of this document, and it is still preliminary. As usual, we expect to refine it as we make progress along the journey towards an FL PoC. Any feedback is appreciated.

## Prerequisites

### PoC specification

Before embarking on the journey towards a PoC, **there must be agreement between the customer and the Constrained ML team about the scope of the PoC**. To help reach clarity and agreement, we highly recommend filling out the PoC Specification template, that we shared privately.

### Access to an Azure subscription

Another fundamental prerequisite to doing a FL PoC on Azure is, well, to have access to Azure. **The customer needs to have access to an Azure subscription, and to have enough permissions to create resources.** Resources include (but are not limited to) Azure Machine Learning (AML) workspaces, compute clusters, datastores, Arc clusters, virtual networks, policies...

## 1. First part: Start with the basic blocs

In this first part, we create the basic blocks that are required for doing FL: an orchestrator and some silos. Please note that for the time being, we will not address security features. This implies we should not be bringing in any sensitive data quite yet.

### 1.1 Orchestrator

For all customers, the orchestrator will be an AML workspace. Even if you already have an AML workspace, we recommend you create a specific one for ths PoC. You can find more instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace).

### 1.2 Silos

AML can accommodate silos in the same Azure Active Directory (AAD) tenant as the orchestrator AML workspace. As long as the silos's compute can be packaged in a Kubernetes container, AML will also accommodate silos in a different AAD tenant, in a different cloud provider, or even in custom data centers. Depending on the customer's needs, this is where the instructions will diverge. 

#### 1.2.1 Case 1: silos in the same AAD tenant as the orchestrator

The easiest case is when the silos will be in the same AAD tenant as the orchestrator. This is what would happen if, for instance, the silos owners (_a.k.a._ data providers) gave the raw data files and allowed the customer to store them in their own storage account. In this case, the customer will need to create one [compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python) and one [datastore](https://docs.microsoft.com/en-us/azure/machine-learning/concept-datastore) _per silo_ - so, a minimum of two of each. An [Azure Blob datastore](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-datastore?tabs=cli-identity-based-access%2Ccli-adls-identity-based-access%2Ccli-azfiles-account-key%2Ccli-adlsgen1-identity-based-access#create-an-azure-blob-datastore) should be the appropriate type.

#### 1.2.2 Case 2: silos in a different AAD tenant (or a different cloud)

All other types of silos (silos in a different AAD tenant, in a different cloud, or on-premises) share a commonality: they will be connected to the orchestrator _via_ an [Azure Arc](https://azure.microsoft.com/en-us/services/azure-arc/) cluster. 

This means that for a PoC, all these types of silos can be mimicked in the same way: by introducing an Arc cluster, connected on the one hand to the orchestrator workspace, and on the other hand to a Kubernetes compute cluster mimicking a silo.

For creating the Arc and Kubernetes clusters, please follow the instructions [here](https://github.com/Azure-Samples/azure-ml-federated-learning/tree/main/automated_provisioning). As indicated in the instructions, please make sure you can actually run a sample (non-ML) job.

Once the silos' computes have been created, please create a datastore for each silo - refer to the instructions provided above for case 1.

### 1.3 :checkered_flag: Exit criterion: Run mock FL job

Once an orchestrator and a couple of silos are available, we can start running a mock FL job. This is a very simple job, and it will be used to ensure that the basic infrastructure is working. We recommend simply adapting [this example job](https://github.com/Azure-Samples/azure-ml-federated-learning/tree/main/fl_arc_k8s) to point at your own orchestrator and silos. 

Please note that for this job to run, you will need to upload some sample data. Detailed instructions can be found at: :construction: **Work in Progress** :construction:.

## 2. Second part: Add the security mechanisms

### 2.1 Access to the orchestrator

First you will want to control the access to the orchestrator. [This page](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles) will tell you all the details you need to know, but the gist of it is that people who will work on _building the PoC_ will need the _Contributor_ role, while people who will only _run FL experiments_ will only need the _AzureML Data Scientist_ role.

### 2.2 Lock down the silos

By "locking down the silos", we mean 2 things:
- "lock the data" so that only the silo's compute can connect to the silo's datastore;
- "lock the compute" so that only the orchestrator can trigger jobs on the silo's compute.

#### 2.2.1 Data

Securing the data will be achieved by leveraging Azure's Managed Identities, and Role-based Access Control (RBAC), as explained [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-managed-identities?tabs=python). Basically each silo's datastore will have to be locked by RBAC, each silo's compute (either directly the compute cluster in case 1, or the Arc cluster in case 2) will need to be assigned a Managed Identity, and this Managed Identity will be given the Blob storage Contributor role.

#### 2.2.2 Compute

:construction: **Work in Progress** :construction: 

Need to first determine, then to explain how to ensure only the orchestrator can trigger jobs on the silos.

### 2.3 Communication between silos and orchestrator

Communications between each silo and the orchestrator will involve model weights or other system metadata such as events for computing metrics. Those will be written to purpose-specific blob storage datastores, one for each silo.

You will need to create these datastores following the instructions above. They should only be readable and writable by the orchestrator and the associated silo; use the same RBAC mechanism as described previously.

### 2.4 Virtual networks

:construction: **Work in Progress** :construction: 

Explain how [virtual networks](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-network-security-overview) (VNets) can be used to isolate the silos from the outside world.

### 2.5 Policies

:construction: **Work in Progress** :construction: 

Explain how [Azure Policy](https://azure.microsoft.com/en-us/services/azure-policy/?OCID=AID2200277_SEM_839f1a5fe93116320e5678c287be7087:G:s&ef_id=839f1a5fe93116320e5678c287be7087:G:s&msclkid=839f1a5fe93116320e5678c287be7087#overview) can be used to enforce various security measures such as "only signed code can run" or "only some type of data (model weights or system metadata) can leave the silo".

### 2.6 :checkered_flag: Exit criterion: Run real FL job

Once all the security features above have been implemented, it will be time to run a _real_ FL job, on _real_ data. A preliminary step will be to re-run the mock FL job, to make sure all the components are still functional after the introduction of the security features.

Next, one will need to upload the actual data to the silos. After that, the training scripts will need to be prepared. Finally, the _Federated Pipeline_ will need to be written. We recommend using _shrike_ for that, as shown in the [example job](https://github.com/Azure-Samples/azure-ml-federated-learning/tree/main/fl_arc_k8s) you used for validating Part 1.

When all of that is done, the first FL attempts are ready to be made. As with any general ML task, the various (hyper-) parameters will need to be tuned appropriately.

> After this step, you will have a **functional, secure FL training job on Azure**. This is a big accomplishment, you should pat yourself on the back! It might not be the end of the work on your PoC, but that means you are almost there.

## 3. Third part: Enrich the PoC with customer-specific features

### 3.1 Guide/tooling for easily adding another client

For some customers, the PoC is less about demonstrating improved model performance when training _via_ FL, and more about getting a feel of the experience of doing FL in Azure. A key part of the experience will be adding new silos (compute + data). This is expected to happen periodically, and the ease and celerity of the procedure will be tantamount.

:construction: **Work in Progress** :construction: 

We will provide more details about this part once we have one working, properly secured silo. Most likely, we will just generalize what we have done manually, and deliver some instructions and some automation tools.

### 3.2 Privacy-preserving algorithms

Many customers will want to incorporate some privacy-preserving algorithms into their PoC, most likely Secure Aggregation (SA) and Differentially Private Stochastic Gradient Descent (DP-SGD).

:construction: **Work in Progress** :construction: 

- Explain the broad lines of SA and DP-SGD, and what they help protect against.
- Provide more details on how to implement them (tooling in shrike for SA, documentation and code snippets for DP-SGD...).

### 3.3 Comparison to existing model

Some customers will want to demonstrate, at the PoC stage, some improvement (or at least parity) with respect to some baseline model. We will provide tooling to easily compare 2 models.

:construction: **Work in Progress** :construction: 

Need to actually build the tooling (a sample experiment taking as inputs a dataset, 2 different models, and some metric(s) definition(s)), or more likely to adapt some already existing tooling. Once the tooling is built, provide the usage instructions here.




