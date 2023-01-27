# Plan your journey towards Federated Learning on Azure ML

Adopting a Federated Learning strategy can be complex because it requires both machine learning skills on one side, and infrastructure and security skills on the other. This page provides a general ramp-up plan for a team that is new to Federated Learning (FL), and wants to leverage this technology at scale using Azure ML.

The plan follows the Crawl-Walk-Run<!-- -Fly--> framework, where we suggest to start with a simple example, and then progressively add more complexity.

Multiple actors will ne needed to successfully implement a production-ready FL stack:
- For **Team leads**, this whole guide provides a holistic project structure and investment areas for onboarding FL on Azure ML.
- For **Data Scientists**, the **Crawl** and **Walk** phases of this guide will show how to tackle FL from an ML perspective.
- For **Data/ML Engineers**, the **Walk** and **Run** phases will explain the areas of investment in the infrastructure.

## Table of contents

- [:turtle: **Crawl** - Kick the tires and get a feel for the technology](#i-turtle-crawl---kick-the-tires-and-get-a-feel-for-the-technology)
- [:walking: **Walk** - Provision a custom setup, connect non-sensitive data, train first FL model](#ii-walking-walk---provision-a-custom-setup-connect-non-sensitive-data-train-first-fl-model)
- [:running: **Run** - Provision a custom and _secure_ setup, connect to sensitive data, train real FL model](#iii-running-run---provision-a-custom-and-secure-setup-connect-to-sensitive-data-train-real-fl-model)

## I. :turtle: **Crawl** - Kick the tires and get a feel for the technology

- Prerequisites: none, this is done even before starting a project
- Who: data scientists, ML engineers
- Investment: 1-2 days

The **goals** of this phase are to:
- get a feel for the technology and the Azure ML platform,
- identify the key areas of your future work plan and investments,
- understand the different components of the FL training process, and how to customize them.

Any organization that hasn't done any FL training before will have to learn about multiple aspects of the stack: machine (federated) learning itself, but also infrastructure setup and security, as well as Azure ML as a cloud provider. A reasonable first goal for you or your team is to provision a sandbox environment you can use to learn about FL, then run some hands-on examples and why not adapt them for a first prototype.

During this phase, we recommend you to get used to both the scientific and the infrastructure aspects of FL.

We recommend you to go through the following hands-on content:
- Run our [quickstart](./quickstart.md), it takes 5-10 minutes, and shows off the entire stack.
- Run our [industry-relevant examples](./README.md#real-world-examples), they take 30-60 minutes, and show how to train FL models on real-world data, on samples.

While doing so, there are key concepts that you'll want to understand:
- the notion of orchestrator and silo,
- the notion of an Azure ML pipeline and how steps are running some in the silos, some in the orchestrator, to support FL,
- how FL ensures the privacy and co-location of the data in different regions,
- the different kinds of silos and how they are connected to the orchestrator (see our [provisioning guide](../provisioning/README.md) as a starting point),
- how the FL machine learning code is different from centralized machine learning code (for instance the use of aggregation).

Once you've run a few examples, we recommend using the sandbox Azure ML workspace as a way to try things out and explore how you could transpose the sample code for your use case. In particular, here's a few questions you will want to answer before moving to the next phase:
- In your use case, what kind of silos would you need? internal or external? same tenant or different tenants?
- What kind of constraints will impact your own FL project? (data location constraints? privacy constraints? legal IP/agreement constraints?)
- Is your FL workflow horizontal or vertical?

**:checkered_flag: Checkpoint - you can move on to the next phase once you have:**
- Run some sample FL training jobs,
- Identified which kind of federation you will work with.


## II. :walking: **Walk** - Provision a custom setup, connect non-sensitive data, train first FL model

- Who: data scientists, ML engineers
- Investment: 1-2 weeks

The **goals** of this phase are to:
- ML: implement a proof of concept FL pipeline,
- Infra: provision a custom sandbox environment,
- Project: identify a path to production.

This will likely apply to any organization that has minimal FL experience on Azure ML and that is ready to build its first prototype.

During this phase, we recommend you start building your FL infrastructure (the _Infrastructure_ track) and you train a FL model to solve your business problem, working with non-sensitive data at first (the _Science_ track). This will help you understand the different components of the FL training process, and how to customize them. And should you make a mistake with the security settings or inadvertently expose some data, that is okay because it is non-sensitive data.

### Infrastructure track

First, you will want to provision a FL setup relevant to your use case (internal geographically-distributed silos, external silos, etc...). Start with our [provisioning README](./provisioning/README.md) to identify which ingredients you will need, and provision your custom setup using our Bicep templates. Note that since you will not be working with sensitive data, you don't have to use the templates using VNets` and Private Endpoints. If you do though, it will make the **Run** phase a bit easier, since your setup will already have these safety measures applied - up to you.

Next, you will want to connect your non-sensitive data to your FL setup, by creating a [data asset](https://learn.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?tabs=cli#data). There are various ways to do so. For instance, you can manually upload data to your Azure ML workspace via the UI, or [use the CLI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?tabs=cli).

Now would be the appropriate time to initiate a discussion with your security team to understand what security measures you need to put in place to train a FL model on sensitive data. This will help you understand the constraints you will have to work with, and will help set you up for success in the next phase.

### Science track

After that, you are ready to start developing the training procedure for your business problem. Our repository proposes some [industry-relevant examples](./README.md#real-world-examples). If your use case is not too different from one of those examples, that would be a good starting point! If you would like to see a new industry-relevant example similar to your use case added to our repository, please [open a Feature Request](https://github.com/Azure-Samples/azure-ml-federated-learning/issues/new?assignees=&labels=&template=feature_request.md&title=) and we will consider enriching our corpus of examples.

Once you have trained your FL model, we recommend comparing it to a baseline model trained on the central data (or whatever other data allow for a sensible comparison). This will help you understand the impact of FL on your business problem, and assess whether you want to move forward.

>**:checkered_flag: :walking: Checkpoint:** You have trained your first FL model relevant to your business goals and have demonstrated the feasibility of the approach and the value for your organization.

## III. :running: **Run** - Provision a custom and _secure_ setup, connect to sensitive data, train real FL model

This will likely apply to an organization that has validated its first FL prototype on synthetic/non-sensitive data and is ready to train a real FL model on production data. It is assumed that the Science part (how to train the model) has been mostly figured out at this point. 

During this phase, you will be training a model on real, sensitive data, in a _secure_ setup. The focus will be on infrastructure.

First, if you haven't already, now is time to create a setup leveraging VNets and private endpoints for added security. Again, please refer to our [provisioning README](./provisioning/README.md) to identify which ingredients you will need. Note that it is possible to create "hybrid" setups, with both internal and external silos. Also note that VNets and private endpoints are a step in the right direction, but they are NOT silver bullets that guarantee 100% security. For instance, you will want to triple check your code to make sure data are not exposed by accident, or implement privacy-preserving algorithms such as Secure Aggregation or Differential Privacy. We will add instructions about these to our repository, so stay tuned!

After that, you should connect your _sensitive_ data to your workspace. Depending on your use case, you have several options.
1. Internal silos
    - If your data already live in an Azure storage account and you haven't created your silo yet, consider following our [instructions for creating a silo using an already-existing storage account](./provisioning/silo_vnet_existingstorage.md). The compute in the silo will have R/W access to the storage account (but the orchestrator compute in the central Azure ML workspace will not).
    - If your data already live in an Azure storage account and you have already created your silo but didn't link it to the existing storage account, you can always do that at a later stage. Navigate to the Azure portal to find your resource group, find the **Managed Identity** corresponding to your silo compute, and give it the following roles (towards the storage account): "Storage Blob Data Contributor", "Reader and Data Access", "Storage Account Key Operator Service Role".
2. External silos on-premises
    - If your data live on the same machine hosting the kubernetes cluster making up your external silo, you can expose the data to the Azure ML job following [these instructions](./targeted-tutorials/read-local-data-in-k8s-silo.md).
    - If your data live in an on-premises data server, work with your local IT admin to make sure the data can be pulled from the Azure ML job (no Azure constructs should be required for this). 
3. External silos in a different Azure tenant
    - :construction: **Work In Progress** :construction:If your data are located in a storage account corresponding to a different tenant, you will need to...

After all that, you're ready to cook with gas! Iterate on your training algorithm until you are satisfied with the outcome.

_Please note that if you want to upload data to the storage account later on, you will NOT be able to do so through the workspace UI, because the AML workspace itself should NOT have permissions to the storage account. You will need to upload your data directly to the storage account, from the Azure portal._

>**:checkered_flag: :running: Checkpoint:** You have trained your FL model on sensitive data, and it performs well. You have performed some parity check between your FL model and a relevant baseline model. You are now ready to use your FL model in production.

<!-- Here below is a stub for the Fly phase. Not sure if we want to tackle this now, or wait till later and we get a better view into what mature organizations will need.-->
<!--
## :airplane: **Fly** - Scale up to production, introduce MLOPS

>**:checkered_flag: :airplane: Checkpoint:** You have one or several FL models deployed in production, and you have introduced MLOPS practices such as automated model evaluation/retraining, process for auditing, etc... 
-->