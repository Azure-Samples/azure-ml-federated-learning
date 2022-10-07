# Advanced - Provisioning a setup with _external_ silos
**:construction: :warning: This is still a work in progress. :warning: :construction:**


## Contents
Many real-world Federated Learning (FL) applications will rely on silos that are not in the same Azure tenant as the orchestrator. This is the case for example when the silos are owned by different companies. Furthermore, those silos might not even be in Azure at all - they might be on different cloud platforms, or on-premises.

We refer to those types of silos as _external_ silos. The goal of this document is to **provide guidance on how to provision a FL setup with such _external_ silos.**


## Nomenclature
- `FL Admin`: this is the person who is responsible for provisioning the FL setup and ensuring the security. He or she works for Contoso, the company who wants to do train a model using FL.
- `Silo Admin`: this is the person who owns the compute and the data of a given silo. He or she works for Fabrikam, a company that has agreed to share their data with Contoso for FL purposes. 
- `Subscription`: in all that follows, unless specified otherwise, when we talk about an _Azure subscription_ we mean the subscription where the orchestrator will be deployed. This subscription belongs to Contoso.
- For common FL terms such as `silo` or `orchestrator`, please refer to the [glossary](./glossary.md).


## Prerequisites
- Some Kubernetes (k8s) clusters (at least one) with version <= 1.24.6, either on-premises, or in Azure (in a different tenant from that of the orchestrator). The cluster should have a minimum of 4 vCPU cores and 8-GB memory
  - For creating a k8s cluster on-premises one can use [Kind](https://kind.sigs.k8s.io/), for instance.
  - For creating a k8s cluster in Azure (in a different tenant otherwise we'd be dealing with _internal_ silos), one can use [Azure Kubernetes Service (AKS)](https://portal.azure.com/#create/microsoft.aks).
- FL Admin needs to have Owner role in the subscription, or at least in a resource group. This is because some Role Assignments will have to be created.
- Silo Admin must be given (temporary) Contributor role to the subscription, or at least the [Azure Arc Onboarding](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#kubernetes-cluster---azure-arc-onboarding) built-in role. This is because some steps will require access to both the orchestrator subscription, and to the k8s cluster. It assumed that _the FL Admin shouldn't have direct access to the k8s cluster_.


## Procedure

### A. Create the orchestrator workspace
**FL Admin** creates the orchestrator workspace with no silos (see the value of the `siloRegions` parameter below). FL Admin needs Owner permissions in `<workspace-resource-group>`, since Role Assignments will be created.
```bash 
az deployment group create --template-file ./mlops/bicep/open_sandbox_setup.bicep --resource-group <workspace-resource-group> --parameters demoBaseName="fldemo" siloRegions=[]
```
The orchestrator workspace will be named `aml-fldemo` (the value of the `demoBaseName` parameter, appended to "`aml-`").
### B. Connect the existing cluster to Azure Arc
Detailed instructions for this phase, including steps for verification or for slightly different use cases can be found [there](https://learn.microsoft.com/en-us/azure/azure-arc/kubernetes/quickstart-connect-cluster?tabs=azure-cli). Here is a summary, which should be all you need.

0. In addition to the prerequisites listed [above](#prerequisites), make sure you have the following:
    - an identity that can be used to log in to the Azure CLI and connect the k8s cluster to Azure Arc;
    - [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) with version >= 2.24.0;
    - the **connectedk8s** Azure CLI extension with version >= 1.2.0;
      - install _via_ `az extension add --name connectedk8s`
    - a [kubeconfig file](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) and context pointing to the k8s cluster;
    - [Helm 3](https://helm.sh/docs/intro/install) with version < 3.7.0;
1. Register providers for Azure Arc-enabled Kubernetes. **This step should be performed by the FL Admin.** It only needs to be performed once, not for every silo.
    - Enter the following commands.
      ```bash
      az provider register --namespace Microsoft.Kubernetes
      az provider register --namespace Microsoft.KubernetesConfiguration
      az provider register --namespace Microsoft.ExtendedLocation
      ```
    - Monitor the registration process. Registration may take up to 10 minutes.
      ```bash
      az provider show -n Microsoft.Kubernetes -o table
      az provider show -n Microsoft.KubernetesConfiguration -o table
      z provider show -n Microsoft.ExtendedLocation -o table
      ```
    - Once registered, you should see the `RegistrationState` state for these namespaces change to `Registered`.
2. Create a `<connected-cluster-resource-group>` resource group for the connected clusters. **This step should be performed by the FL Admin.** Several connected clusters pointing to different k8s clusters can be added to this group - no need to create a separate group for each silo. The location of this group <connected-cluster-resource-group-location> is not critical, but should preferably be the same as that of the orchestrator workspace.
    - Enter the following command.
      ```bash
      az group create --name <connected-cluster-resource-group> --location <connected-cluster-resource-group-location>
      ```
3. Connect an existing k8s cluster to Azure Arc by creating an Azure Arc-enabled Kubernetes resource named `<Azure-Arc-enabled-k8s-resource-name>`. **This step should be performed by the Silo Admin, since it requires access to the k8s cluster (via the kube config file).** 
    - Enter the following command.
      ```bash
      az connectedk8s connect --name <Azure-Arc-enabled-k8s-resource-name> --resource-group <connected-cluster-resource-group>
      ```
    - If the default kube config and context do not point to the k8s cluster, then the `--kube-config` and `--kube-context` parameters can be used to specify the correct values.


### C. Deploy the Azure ML extension on the k8s cluster
Detailed instructions for this phase, including current limitations, steps for verification or for slightly different use cases can be found [there](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-kubernetes-extension?tabs=deploy-extension-with-cli)). Here is a summary, which should be all you need.

**This step should be run by the FL Admin.** The Azure CLI extension `k8s-extension` with version >= 1.2.3 is required. It can be installed _via_ `az extension add --name k8s-extension`. 

To deploy the Azure ML extension on the k8s cluster, enter the following command.
```bash
az k8s-extension create --name <extension-name> --extension-type Microsoft.AzureML.Kubernetes --config enableTraining=True --cluster-type connectedClusters --cluster-name <Azure-Arc-enabled-k8s-resource-name> --resource-group <connected-cluster-resource-group> --scope cluster
```

The deployment can be verified by the following.
```bash
az k8s-extension show --name <extension-name> --cluster-type connectedClusters --cluster-name <Azure-Arc-enabled-k8s-resource-name> --resource-group <connected-cluster-resource-group>
```

In the response, look for `"name"` and `"provisioningState": "Succeeded"`. Note that it might show `"provisioningState": "Pending"` for the first few minutes.


### D. Attach the k8s cluster to the orchestrator workspace
Detailed instructions for this phase, including steps for verification or for slightly different use cases can be found [there](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-attach-kubernetes-to-workspace?tabs=cli)). Here is a summary, which should be all you need.

**This step should be run by the FL Admin.** The `ml` Azure CLI extension (_a.k.a._ Azure ML CLI v2) will be required. It can be installed _via_ `az extension add --name ml`. See [over there](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public) for more details on installation.

To attach the k8s cluster to the orchestrator workspace, enter the following command:
```bash
az ml compute attach --resource-group <workspace-resource-group> --workspace-name <workspace-name> --type Kubernetes --name <azureml-compute-name> --resource-id "/subscriptions/<subscription-id>/resourceGroups/<connected-cluster-resource-group>/providers/Microsoft.Kubernetes/connectedClusters/<Azure-Arc-enabled-k8s-resource-name> " --identity-type SystemAssigned --no-wait
```
where:
- `<workspace-resource-group>` is the resource group of the orchestrator workspace that you used in [step A](#a-create-the-orchestrator-workspace);
- `<workspace-name>` is the name of the orchestrator workspace that was created during [step A](#a-create-the-orchestrator-workspace);
  - if you kept all defaults, it should be `aml-fldemo`;
- `<azureml-compute-name>` is the name that you _choose_ for the silo compute in the orchestrator workspace (arbitrary);
- `<subscription-id>` is the Id of the orchestrator subscription;
- `<connected-cluster-resource-group>` and `<Azure-Arc-enabled-k8s-resource-name> ` have been defined in the [previous step](#c-deploy-the-azure-ml-extension-on-the-k8s-cluster).


### E. Run a test job
Run a test job. Provide shorter, targeted instructions.


### F. Secure the silo
Secure the silo by creating the storage accounts and UAI's. Need to modify the silo bicep script to accommodate already-existing computes.


### G. Run another test job
Run a test job. Provide shorter, targeted instructions.


## Add more silos 
Well, that's easy - just rinse and repeat.


## TODO's
- Clarify what needs to be done by the FL Admin vs by the Silo Admin.
  - This ties in with understanding which commands rely on the kube config file. 
  - It seems like some actions require both subscription access, and k8s access: those should be taken by the Silo Admin.
- Provide docker files meeting all the prerequisites already?
- Include some schematic at the beginning, showing all the resources that are created, mappings to these instructions, and the FL/Silo Admin personas?
- Take care of inputs. For both _internal_ and _external_ silos. We will probably need a separate "Connect the sensitive silo data" document for that.
  - **Pay attention to the case of inputs, and how they are treated slightly differently between on-prem and in-Azure...**