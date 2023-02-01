# Plan your journey towards Federated Learning on Azure ML

Adopting a Federated Learning strategy can be complex because it requires both machine learning skills on one side, and infrastructure and security skills on the other. This page provides a general ramp-up plan for a team that is new to Federated Learning (FL), and wants to leverage this technique at scale using Azure ML.

This guide is intended to support the various actors that will have to work together to implement a production-ready FL stack:

- For **Team Leads**, this whole guide provides a holistic project structure and investment areas for onboarding FL on Azure ML.
- For **Data Scientists**, the early phases of this onboarding process will show how to tackle FL from an ML perspective.
- For **Data/ML Engineers**, the later phases will explain the areas of investment in the infrastructure.

## Table of contents

- [I. Hands-on introduction to the terms of the FL problem](#i-hands-on-introduction-to-the-terms-of-the-fl-problem)
- [II. Build a prototype to demonstrate the value of FL](#ii-build-a-prototype-to-demonstrate-the-value-of-fl)
- [III. Do FL on sensitive data in a Production setting](#iii-do-fl-on-sensitive-data-in-a-production-setting)

## I. Hands-on introduction to the terms of the FL problem

- Who: Data Scientists, ML Engineers
- Investment: 1-2 days

### Goals

The **goals** of this phase are to:

- get a feel for the technology and the Azure ML platform;
- understand the different components of the FL training process;
- identify the key areas of your future work plan and investments.

### Guidelines

Any organization that hasn't done any federated training before will have to learn many aspects of the FL stack beyond ML itself: how to run a training in a federated fashion, how to run a training in alignment with legal or contractual obligations, how to properly and securely setup infrastructure, how to design an FL experiment, as well as how to use a cloud provider like Azure ML.

On this journey, a reasonable first goal is to understand the terms of the problem. A first step is for you or your team to provision a sandbox environment you can use to learn about FL, then run some hands-on examples and get a feel for the technology. During this phase, we recommend you to get used to both the scientific and the infrastructure aspects of FL.

As a starter, we recommend you to go through our hands-on content:

- run our [quickstart](./quickstart.md), it takes 5-10 minutes, and shows off the entire stack;
- run our [industry-relevant examples](./README.md#real-world-examples), they take 30-60 minutes, and show how to train FL models on real-world data, on samples.

While doing so, there are key concepts that you'll want to observe at work during training:

- the notion of orchestrator and silo, and in particular which cloud resources are provisioned under what we call a "silo" (ex: see [the vnet silo](../provisioning/silo_vnet_newstorage.md)),
- the notion of an Azure ML pipeline made on components, how the Azure ML SDK can be leveraged (ex: see [submit.py](../../examples/pipelines/fl_cross_silo_literal/submit.py)) to set up each component to run independently (some in silos, some in orchestrator),
- the different kinds of silos and how they are connected or not to the orchestrator (see our [provisioning guide](../provisioning/README.md) as a starting point), and how each of those is just interchangeable within an Azure ML workspace.

### :checkered_flag: Checkpoint

As a checkpoint for this learning phase, there are a couple key questions you will want to answer for yourself and your organization.

- In your use case, what kind of silos would you need? internal or external? same tenant or different tenants or on-prem?
- Where will the data be located? in the cloud? on-prem?
- What kind of constraints will impact your own FL project? (data location constraints? privacy constraints? legal IP/agreement constraints?)
- Is your FL workflow horizontal or vertical?

## II. Build a prototype to demonstrate the value of FL

- Who: Data Scientists, ML Engineers
- Investment: 1-2 weeks

### Goals
The **goals** of this phase are to:

- implement an FL pipeline on some non-sensitive dataset;
- provision a custom sandbox environment;
- identify a path to production.

### Guidelines
During this phase, we recommend you start building your FL infrastructure (the _Infrastructure_ track) and you train a FL model to solve your business problem, working with non-sensitive data at first (the _Science_ track). This will help you understand the different components of the FL training process, and how to customize them. And should you make a mistake with the security settings or inadvertently expose some data, that is okay because it is non-sensitive data.

_Infrastructure track_

First, you will want to provision a FL setup relevant to your use case (internal geographically-distributed silos, external silos, etc...). Start with our [provisioning README](./provisioning/README.md) to identify which ingredients you will need, and provision your custom setup using our Bicep templates. Note that since you will not be working with sensitive data, you don't have to use the templates using VNets and Private Endpoints. If you do though, it will make phase III below a bit easier, since your setup will already have these safety measures applied - up to you.

Next, you will want to connect your non-sensitive data to your FL setup, by creating a [data asset](https://learn.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?tabs=cli#data). There are various ways to do so. For instance, you can manually upload data to your Azure ML workspace via the UI, or [use the CLI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?tabs=cli).

Now would be the appropriate time to initiate a discussion with your security team to understand what security measures you need to put in place to train a FL model on sensitive data. This will help you understand the constraints you will have to work with, and will help set you up for success in the next phase.

_Science track_

After that, you are ready to start developing the training procedure for your business problem. Our repository proposes some [industry-relevant examples](./README.md#real-world-examples). If your use case is not too different from one of those examples, that would be a good starting point! If you would like to see a new industry-relevant example similar to your use case added to our repository, please [open a Feature Request](https://github.com/Azure-Samples/azure-ml-federated-learning/issues/new?assignees=&labels=&template=feature_request.md&title=) and we will consider enriching our corpus of examples.

Once you have trained your FL model, we recommend comparing it to a baseline model trained on the central data (or whatever other data allow for a sensible comparison). This will help you understand the impact of FL on your business problem, and assess whether you want to move forward.

### :checkered_flag: Checkpoint
 You have trained your first FL model relevant to your business goals and have demonstrated the feasibility of the approach and the value for your organization.

## III. Do FL on sensitive data in a Production setting

- Who: ML Engineers
- Investment: several weeks at least, depending on security review.

### Goals
The **goals** of this phase are to:

- provision a secure, Production-ready FL setup;
- implement an FL pipeline on sensitive data.

### Guidelines
First, if you haven't already, now is time to create a setup leveraging VNets and private endpoints for added security. Again, please refer to our [provisioning README](./provisioning/README.md) to identify which ingredients you will need. Note that it is possible to create "hybrid" setups, with both internal and external silos. Also note that VNets and private endpoints are a step in the right direction, but they are NOT silver bullets that guarantee 100% security. For instance, you will want to triple check your code to make sure data are not exposed by accident, or implement privacy-preserving algorithms such as Secure Aggregation or Differential Privacy. We will add instructions about these to our repository, so stay tuned!

After that, you should connect your _sensitive_ data to your workspace. Depending on your use case, you have several options.

- Internal silos
    - If your data already live in an Azure storage account and you haven't created your silo yet, consider following our [instructions for creating a silo using an already-existing storage account](./provisioning/silo_vnet_existingstorage.md). The compute in the silo will have R/W access to the storage account (but the orchestrator compute in the central Azure ML workspace will not).
    - If your data already live in an Azure storage account and you have already created your silo but didn't link it to the existing storage account, you can always do that at a later stage. Navigate to the Azure portal to find your resource group, find the **Managed Identity** corresponding to your silo compute, and give it the following roles (towards the storage account): "Storage Blob Data Contributor", "Reader and Data Access", "Storage Account Key Operator Service Role".
- External silos on-premises
    - If your data live on the same machine hosting the kubernetes cluster making up your external silo, you can expose the data to the Azure ML job following [these instructions](./targeted-tutorials/read-local-data-in-k8s-silo.md).
    - If your data live in an on-premises data server, work with your local IT admin to make sure the data can be pulled from the Azure ML job (no Azure constructs should be required for this).
- External silos in a different Azure tenant
    - :construction: **Work In Progress** :construction:If your data are located in a storage account corresponding to a different tenant, you will need to...

After all that, you're ready to cook with gas! Iterate on your training algorithm until you are satisfied with the outcome.

_Please note that if you want to upload data to the storage account later on, you will NOT be able to do so through the workspace UI, because the AML workspace itself should NOT have permissions to the storage account. You will need to upload your data directly to the storage account, from the Azure portal._

### :checkered_flag: Checkpoint
You have trained your FL model on sensitive data, and it performs well. You have performed some parity check between your FL model and a relevant baseline model. You are now ready to use your FL model in production.
