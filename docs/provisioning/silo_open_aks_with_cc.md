# Create an open sandbox silo using AKS and confidential compute

:warning: This should be used for **development purpose only**. The following tutorial is intended as a proof of concept only.

## Prerequisites

To run these deployment options, you first need:

- an existing Azure ML workspace (see [cookbook](README.md#create-an-azure-ml-workspace))
- an existing orchestrator (see [tutorial](orchestrator_open.md))
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- Optional: [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

## Create a compute and storage pair for the silo

> Note: this deployment can take up to 15-20 minutes.
> Note: please do not forget create InstanceType that is necessary to be able to fully utilize provisioned cluster, the tutorial is at the end of this document

### Option 1 : one click deployment

1. Click on [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fopen_aks_with_confcomp_storage_pair.json)

2. Adjust parameters, in particular:

    - Region: this will be set by Azure to the region of your resource group.
    - Machine Learning Name: need to match the name of the AzureML workspace in the resource group.
    - Machine Learning Region: the region in which the AzureML workspace was deployed (default: same as resource group).
    - Pair Region: the region where the compute and storage will be deployed (default: same as resource group).
    - Pair Base Name: a unique name for the **silo**, example `silo1-westus`. This will be used to create all other resources (storage name, compute name, etc.).

### Option 2 : deployment using az cli

In the resource group of your AzureML workspace, use the following command with parameters corresponding to your setup:

```bash
az deployment group create --template-file ./mlops/bicep/modules/fl_pairs/open_aks_with_confcomp_storage_pair.bicep --resource-group <resource group name> --parameters pairBaseName="silo1-westus" pairRegion="westus" machineLearningName="aml-fldemo" machineLearningRegion="eastus"
```

### Option 3 : run each step manually

The option 1 wraps up multiple provisioning steps from multiple sections of the Azure documentation. If you want to reproduce this manually, here's the steps you can use:

- [Quickstart: Deploy an AKS cluster with confidential computing Intel SGX agent nodes by using the Azure CLI](https://learn.microsoft.com/en-us/azure/confidential-computing/confidential-enclave-nodes-aks-get-started),
- [How to deploy Kubernetes extension](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-kubernetes-extension?tabs=deploy-extension-with-cli),
- [How to attach Kubernetes to Workspace](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-attach-kubernetes-to-workspace?tabs=cli).

## Set permissions for the silo's compute to R/W from/to the orchestrator

1. Navigate the Azure portal to find your resource group.

2. Look for a resource of type **Managed Identity** in the region of the silo named like `uai-<pairBaseName>`. It should have been created by the instructions above.

3. Open this identity and click on **Azure role assignments**. You should see the list of assignments for this identity.

    It should contain 3 roles towards the storage account of the silo itself:
    - **Storage Blob Data Contributor**
    - **Reader and Data Access**
    - **Storage Account Key Operator Service Role**

4. Click on **Add role assignment** and add each of these same role towards the storage account of your orchestrator.

## Create InstanceType

InstanceType sets restrictions for each job running on the AKS cluster. You can create multiple InstanceType(s) for different type of jobs. For example, job for pre-processing data is usually less demanding than a training job and thus the InstanceType can provide process with less resources. You can find example InstanceType definition in [`mlops/k8_templates/instance-type.yaml`](../../mlops/k8s_templates/instance-type.yaml). To create InstanceType follow these steps:

> Note: Make sure you have `kubectl` tool installed: <https://kubernetes.io/docs/tasks/tools/>

1. Update `mlops/k8_templates/instance-type.yaml` file to reflect minimum and limit resources for the job you intend to deploy (for simplicity you can just set the limit to resources provided by provisioned node in the AKS cluster)
2. Update `name` property under `metadata` section in the `mlops/k8_templates/instance-type.yaml` file. Please remember this name as you will need it later on.
3. Run `az login`
4. Run `az account set --subscription <your-subscription-id>`
5. Run `az aks get-credentials --resource-group <rg-name> --name <aks-name>`
6. Navigate to `mlops/k8s_templates` folder and run: `kubectl apply -f instance-type.yaml`
7. Add `instance_type` property to your pipeline config for the AKS silo and set value to the name set in the step 2
