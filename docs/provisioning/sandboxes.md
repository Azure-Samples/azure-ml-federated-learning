# Azure ML Federated Learning Sandboxes

This page describes the different sandboxes that you can **fully provision and use out-of-the-box** with our samples. Each sandbox has distinct properties depending on what you'd like to test.

- [Open Sandbox (eyes-on)](#open-sandbox-eyes-on)
- [Open workspace, Azure ML compute (cpu+gpu), eyes-off storage](#open-workspace-azure-ml-compute-cpugpu-eyes-off-storage)
- [Open workspace, Azure Kubernetes Service (with confidential compute), eyes-off storage](#open-workspace-azure-kubernetes-service-with-confidential-compute-eyes-off-storage)


## Open Sandbox (eyes-on)

Deploy a completely open sandbox to allow you to try things out in an eyes-on environment. This setup is intended **only for demo** purposes. The data is still accessible by the users of your subscription when opening the storage accounts, and data exfiltration is possible.

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fopen_sandbox_setup.json)

### Relevant Documentation

To manually reproduce this full provisioning, see relevant documentation:
- [Create workspace resources you need to get started with Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources)
- [Manage user-assigned managed identities](https://learn.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/how-manage-user-assigned-managed-identities)

## Open workspace, Azure ML compute (cpu+gpu), eyes-off storage

Deploy a sandbox where the silos storages are kept eyes-off by a private service endpoint, accessible only by the computes through a vnet.

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fvnet_publicip_sandbox_setup.json)

This sandbox is typical of a cross-geo federated learning scenario. Each silo is provisioned with a single-tenant, but in different regions. Each silo has a distinct virtual network enabling private communication between the silo compute and the silo storage.

### :exclamation: Important parameters

| Parameter | Description | Values |
| --- | --- | --- |
| **compute2** | Provision a second compute within each silo. | `true` or `false` |
| **compute2SKU** | SKU of the second compute to provision. For a hybrid pipeline CPU+GPU, set `compute1SKU` and `compute2SKU` accordingly. | ex: `STANDARD_NC6` |
| **applyVNetPeering** | Peer the silo networks to the orchestrator network to allow for live private communication between jobs (required for Vertical FL). | `true` or `false` |
| **siloRegions** | List of regions used for the silos. All our samples work with 3 regions. | ex: `["australiaeast", "eastus", "westeurope"]` |

### Architecture

![](../pics/sandbox_aml_vnet.png)

### Relevant Documentation

To manually reproduce this full provisioning, see relevant documentation:
- [Secure an Azure Machine Learning workspace with virtual networks](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-workspace-vnet)
- [Compute instance/cluster with public IP](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-training-vnet?tabs=cli%2Crequired#compute-instancecluster-with-public-ip)
- create a new [vnet and subnet](https://learn.microsoft.com/en-us/azure/virtual-network/virtual-networks-overview), with a [network security group](https://learn.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview)
- create a new [managed identity](https://learn.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/overview) (User Assigned) to manage permissions of the compute
- create a new [storage account](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-overview) in a given region, with a [private endpoint](https://learn.microsoft.com/en-us/azure/storage/common/storage-private-endpoints) inside the vnet

## Open workspace, Azure Kubernetes Service (with confidential compute), eyes-off storage

Deploy an eyes-off sandbox where the computes leverage confidential computing to keep your training and processing within an enclave.

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-ml-federated-learning%2Fmain%2Fmlops%2Farm%2Fvnet_publicip_sandbox_aks_confcomp_setup.json)

Note: to take full benefit of the VMs, you will need to finalize the setup of the AKS cluster by [creating an instance type and use it in pipeline configs](./silo_open_aks_with_cc.md#create-instancetype).

### :exclamation: Important parameters

| Parameter | Description | Values |
| --- | --- | --- |
| **computeSKU** | VM to provision in the AKS cluster (default will use [a CVM from dcasv5](https://learn.microsoft.com/en-us/azure/virtual-machines/dcasv5-dcadsv5-series)). You can also use any non-confidential SKU. | ex: `Standard_DC4as_v5` |
| **applyVNetPeering** | Peer the silo networks to the orchestrator network to allow for live private communication between jobs (required for Vertical FL). | `true` or `false` |
| **siloRegions** | List of regions used for the silos. All our samples work with 3 regions. :exclamation: make sure you have quota in those regions for confidential compute in particular. | ex: `["australiaeast", "eastus", "westeurope"]` |

### Architecture

Note: in current sandbox, we're provisioning only in the `eastus` region by default, to allow for capacity and quick deployment.

![](../pics/sandbox_aks_cc_vnet.png)

### Relevant Documentation

To manually reproduce this full provisioning, see relevant documentation:
- [Secure an Azure Machine Learning workspace with virtual networks](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-workspace-vnet)
- [Introduction to Kubernetes compute target in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-attach-kubernetes-anywhere)
- [DCasv5 and DCadsv5-series confidential VMs](https://learn.microsoft.com/en-us/azure/virtual-machines/dcasv5-dcadsv5-series)
- [Quickstart: Deploy an AKS cluster with confidential computing Intel SGX agent nodes by using the Azure CLI](https://learn.microsoft.com/en-us/azure/confidential-computing/confidential-enclave-nodes-aks-get-started)
- [Network concepts for applications in Azure Kubernetes Service (AKS)](https://learn.microsoft.com/en-us/azure/aks/concepts-network)
- [How to deploy Kubernetes extension](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-kubernetes-extension?tabs=deploy-extension-with-cli)
- [How to attach Kubernetes to Workspace](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-attach-kubernetes-to-workspace?tabs=cli)
- [Manage user-assigned managed identities](https://learn.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/how-manage-user-assigned-managed-identities)
