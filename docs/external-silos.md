# Advanced - Provisioning a setup with _external_ silos
**:construction: :warning: This is a work in progress, to be treated as an RFC for the time being. :warning: :construction:**

## Contents
Many real-world Federated Learning (FL) applications will rely on silos that are not in the same Azure tenant as the orchestrator. This is the case for example when the silos are owned by different companies. Furthermore, those silos might not even be in Azure at all - they might be on different cloud platforms, or on-premises.

We refer to those types of silos as _external_ silos. The goal of this document is to **provide guidance on how to provision a FL setup with such _external_ silos.**

## Nomenclature
- FL Admin: ...
- Silo Admin: ...

## Prerequisites
- One orchestrator workspace in Azure ML
  - Link to explanations on how to create the orchestrator only (no silos)
- Some Kubernetes clusters (at least one) with version <= 1.24.0, either on-premises, or in Azure (in a different tenant from that of the orchestrator)
  - Link to explanations/resources on how to create the k8s clusters, for either case (Azure or on-prem).
- ...

## Procedure
1. Connect the existing cluster to Azure Arc (instructions [here](https://learn.microsoft.com/en-us/azure/azure-arc/kubernetes/quickstart-connect-cluster?tabs=azure-cli)).
    - Provide shorter, targeted instructions.
2. Deploy the Azure ML extension on the k8s cluster (instructions [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-kubernetes-extension?tabs=deploy-extension-with-cli)).
    - Provide shorter, targeted instructions.
3. Attach the k8s cluster to the Azure ML workspace (instructions [there](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-attach-kubernetes-to-workspace?tabs=cli)).
    - Provide shorter, targeted instructions.
4. Run a test job.
    - Provide shorter, targeted instructions.

## Add more clusters 
Well, that's easy - just rince and repeat.