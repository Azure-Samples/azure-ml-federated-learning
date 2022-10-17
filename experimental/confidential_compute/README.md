# Run a sample AzureML job on confidential compute nodes in AKS

:warning: EXPERIMENTAL :warning:

In this tutorial, you will:
* Provision an [Azure Kubernetes Service (AKS) cluster with confidential compute nodes](https://learn.microsoft.com/en-us/azure/confidential-computing/confidential-nodes-aks-overview)
* Attach this AKS to an existing AzureML workspace
* Run a sample job showing on which AKS node pool it runs

Current limitations:
* No storage used in this sample job
* AKS is not secured through vNets, identities, etc

## Prerequisites

To enjoy this quickstart, you will need to:
- have an **existing AzureML workspace** in an active [Azure subscription](https://azure.microsoft.com) that you can use for development purposes,
- have permissions to create resources, set permissions, and create identities in this subscription (or at least in one resource group),
  - Note that to set permissions, you typically need _Owner_ role in the subscription or resource group - _Contributor_ role is not enough. This is key for being able to _secure_ the setup.
- [install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

## Deploy AKS Cluster with confidential compute nodes

> This simplifies existing detailed manual instructions contained in [Quickstart: Deploy an AKS cluster with confidential computing Intel SGX agent nodes by using the Azure CLI](https://learn.microsoft.com/en-us/azure/confidential-computing/confidential-enclave-nodes-aks-get-started).

1. Create a new resource group for the demo resources. Having a new group would make it easier to delete the resources afterwards (deleting this RG will delete all resources within).

    ```bash
    # create a resource group for the resources
    az group create --name <resource group name> --location <region>
    ```

2. Run the bicep deployment script in a resource group you own:

    ```bash
    # deploy the aks cluster in your resource group
    az deployment group create --template-file ./bicep/aks_with_confcomp.bicep --resource-group <resource group name> --parameters clusterName="akswithcceus2" region='eastus2'
    ```

    This script will deploy an AKS cluster in your resource group, with a single node pool `confcompool` with confidential compute nodes.

3. Deploy the AzureML extension in this cluster

    ```bash
    az k8s-extension create --name azmlext --extension-type Microsoft.AzureML.Kubernetes --config enableTraining=True enableInference=False inferenceRouterServiceType=LoadBalancer allowInsecureConnections=True inferenceLoadBalancerHA=False --cluster-type managedClusters --cluster-name akswithcceus2 --scope cluster --resource-group <resource group name>
    ```

    > See detailed instructions at [How to deploy Kubernetes extension](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-kubernetes-extension?tabs=deploy-extension-with-cli).

4. Check the status of the extension

    ```bash
    az k8s-extension show --name azmlext --cluster-type managedClusters --cluster-name akswithcceus2 --resource-group <resource group name>
    ```

## Manually attach to AzureML workspace

> See detailed instructions in [How to attach Kubernetes to Workspace](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-attach-kubernetes-to-workspace?tabs=cli).

1. In your AzureML workspace, go to **Compute** then to the **Attached Compute** panel.

2. Click **+ New** and select **Kubernetes**.

3. Look for the cluster you created in previous step, click on it, in the panel showing up set **Compute Name** to `akswithcceus2`.

## Run sample job

1. In this repository folder `/experimental/confidential_compute`, run the following command to submit a job to the cluster:

    ```bash
    az ml job create --file ./pipelines/test_ip.yaml -w <workspace name> -g <resource group name>
    ```

2. The job prints all environment variables in `user_logs/std_log.txt`, you should see this:

    ```
    ENV: NODE_NAME=aks-confcompool-19739343-vmss000000
    ```
