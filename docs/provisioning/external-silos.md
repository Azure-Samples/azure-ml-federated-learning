# Advanced - Provisioning a setup with _external_ silos

## Contents

Many real-world Federated Learning (FL) applications will rely on silos that are not in the same Azure tenant as the orchestrator. This is the case when the silos are owned by different companies. Furthermore, those silos might not even be in Azure at all - they might be on different cloud platforms, or on-premises.

We refer to those types of silos as _external_ silos. The goal of this document is to **provide guidance on how to provision a FL setup with _external_ silos.**

## Scenario

The _Contoso_ corporation wants to train a model using a FL scheme. The underlying data belong to partner companies, and reside on-premises. One such company is _Fabrikam_.

At Contoso, one person is responsible for provisioning the FL setup and ensuring the security. We'll call that person the **FL Admin**.

At Fabrikam (one of the silos), one person owns the compute and the data. We'll call that person the **Silo Admin**.

> Both **FL Admin** and **Silo Admin** have some prerequisites to meet. The **Prerequisites** section explains what is required of whom. After that, the title of every subsection in the **Procedure** section clearly indicates who of the **FL Admin** or **Silo Admin** should be performing the tasks. **Silo Admin** will only be involved in [step C](#c-silo-admin-connects-the-existing-k8s-cluster-to-azure-arc).

In all that follows, when we talk about an **Azure subscription** we mean the subscription where the Azure ML workspace and the orchestrator will be deployed. This subscription belongs to Contoso.

For common FL terms such as **silo** or **orchestrator**, please refer to the [glossary](../concepts/glossary.md).

## Prerequisites

- **Some Kubernetes (k8s) cluster** (at least one) with version <= 1.24.6, either on-premises, or in Azure (in a different tenant from that of the orchestrator). The cluster should have a minimum of 4 vCPU cores and 8-GB memory.
  - For creating a k8s cluster on-premises one can use [Kind](https://kind.sigs.k8s.io/), for instance.
    - If you want your k8s cluster to have access to _local_ data that reside on the same machine, you can create your cluster following [this tutorial](../tutorials/read-local-data-in-k8s-silo.md).
  - Alternatively, if you are not familiar with Kind/Kubernetes, you can use [AKS Edge Essentials](https://learn.microsoft.com/en-us/azure/aks/hybrid/aks-edge-overview) to create your on-premises k8s cluster. Start by [setting up your machine](https://learn.microsoft.com/en-us/azure/aks/hybrid/aks-edge-howto-setup-machine), then [create a single machine deployment](https://learn.microsoft.com/en-us/azure/aks/hybrid/aks-edge-howto-single-node-deployment). After that, if you need your cluster to have access to local data, you need to [add a local storage binding](https://learn.microsoft.com/en-us/azure/aks/hybrid/aks-edge-howto-use-storage-local-path).
  - For creating a k8s cluster in Azure (in a different tenant otherwise we'd be dealing with _internal_ silos), one can use [Azure Kubernetes Service (AKS)](https://portal.azure.com/#create/microsoft.aks).
- **FL Admin** needs to have **Owner** role in the subscription, or at least in the resource group where the Azure ML workspace will be created. This is because some Role Assignments will have to be created.
- **Silo Admin** must be given (temporary) Contributor role to the subscription, or at least the [Azure Arc Onboarding](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#kubernetes-cluster---azure-arc-onboarding) built-in role. This is because one step will require access to both the orchestrator subscription, and to the k8s cluster. It assumed that _the FL Admin shouldn't have direct access to the k8s cluster_.
- An identity that can be used to log in to the Azure CLI and connect the k8s cluster to Azure Arc (that would be the **Silo Admin**'s identity).
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) with version >= 2.24.0.
- **FL Admin** will need the following Azure CLI extensions.
  - The **ml** Azure CLI extension (_a.k.a._ Azure ML CLI v2).
    - Install _via_ `az extension add --name ml`. See [over there](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public) for more details on installation.
  - The **k8s-extension** Azure CLI extension with version >= 1.2.3.
    - Install _via_ `az extension add --name k8s-extension`.
- **Silo Admin** will need the following:
  - The **connectedk8s** Azure CLI extension with version >= 1.2.0.
    - Install _via_ `az extension add --name connectedk8s`.
  - A [kubeconfig file](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) and context pointing to the k8s cluster.
  - [Helm 3](https://helm.sh/docs/intro/install) with version < 3.7.0.

## Procedure

### A. **FL Admin** creates the Azure ML workspace and the orchestrator
>
> This is all explained in the first sections of the [cookbook](./README.md) but repeated here for convenience.

Create an open Azure ML workspace named `<workspace-name>`. Owner permissions in `<workspace-resource-group>` are required, since Role Assignments will need to be created later on. (The `<base-name>` value will be used when creating associated resources and can be chosen arbitrarily, but note that it should be unique in the subscription.)

```bash
az deployment group create --template-file ./mlops/bicep/modules/azureml/open_azureml_workspace.bicep --resource-group <workspace-resource-group> --parameters machineLearningName=<workspace-name> baseName=<base-name>
```

After that, create the compute and storage corresponding to the orchestrator (the value of the `pairBaseName` parameter will need to be adjusted if you have already created an orchestrator with this base name in the subscription).

```bash
az deployment group create --template-file ./mlops/bicep/modules/fl_pairs/open_compute_storage_pair.bicep --resource-group <workspace-resource-group> --parameters pairBaseName="orch" machineLearningName=<workspace-name>
```

### B. **FL Admin** prepares for connecting the existing k8s cluster to Azure Arc

> Detailed instructions for this phase, including steps for verification or for slightly different use cases can be found [there](https://learn.microsoft.com/en-us/azure/azure-arc/kubernetes/quickstart-connect-cluster?tabs=azure-cli). Here is a summary, which should be all you need.

1. Register providers for Azure Arc-enabled Kubernetes. It only needs to be performed once, not for every silo.
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
      az provider show -n Microsoft.ExtendedLocation -o table
      ```

    - Once registered, you should see the `RegistrationState` state for these namespaces change to `Registered`.
2. Create a `<connected-cluster-resource-group>` resource group for the connected clusters. Several connected clusters pointing to different k8s clusters can be added to this group - no need to create a separate group for each silo that will be created in the future. The location of this group `<connected-cluster-resource-group-location>` is not critical, but should preferably be the same as that of the orchestrator workspace.
    - Enter the following command.

      ```bash
      az group create --name <connected-cluster-resource-group> --location <connected-cluster-resource-group-location>
      ```

### C. **Silo Admin** connects the existing k8s cluster to Azure Arc

The connection is established by creating an Azure Arc-enabled Kubernetes resource named `<Azure-Arc-enabled-k8s-resource-name>`. This step should be performed by the Silo Admin, since it requires access to the k8s cluster (happening implicitly _via_ the kube config file). It also requires Contributor role (at least) in the resource group `<connected-cluster-resource-group>` created in the previous step.

- Enter the following command.

    ```bash
    az connectedk8s connect --name <Azure-Arc-enabled-k8s-resource-name> --resource-group <connected-cluster-resource-group>
    ```

- If the default kube config and context do not point to the k8s cluster, then the `--kube-config` and `--kube-context` parameters can be used to specify the correct values.

### D. **FL Admin** deploys the Azure ML extension on the k8s cluster _via_ Azure Arc
>
> Detailed instructions for this phase, including current limitations, steps for verification or for slightly different use cases can be found [there](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-kubernetes-extension?tabs=deploy-extension-with-cli). Here is a summary, which should be all you need.

To deploy the Azure ML extension on the k8s cluster, enter the following command.

```bash
az k8s-extension create --name <extension-name> --extension-type Microsoft.AzureML.Kubernetes --config enableTraining=True --cluster-type connectedClusters --cluster-name <Azure-Arc-enabled-k8s-resource-name> --resource-group <connected-cluster-resource-group> --scope cluster
```

where `<extension-name>` can be chosen arbitrarily.

> Note: if you're using an AKS cluster (as opposed to a local k8s cluster), you'll need to change the `--cluster-type` parameter value from `connectedClusters` to `managedClusters`.

The deployment can be verified by the following.

```bash
az k8s-extension show --name <extension-name> --cluster-type connectedClusters --cluster-name <Azure-Arc-enabled-k8s-resource-name> --resource-group <connected-cluster-resource-group>
```

In the response, look for `"name"` and `"provisioningState": "Succeeded"`. Note that this step can take 10-15 minutes and will show `"provisioningState": "Pending"` at first.

### E. **FL Admin** attaches the Arc cluster to the orchestrator workspace
>
> (Detailed instructions for this phase, including steps for verification or for slightly different use cases can be found [there](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-attach-kubernetes-to-workspace?tabs=cli).)

1. Create a user-assigned identity (UAI) that will later be assigned to the Azure ML attached compute:

    ```bash
    az identity create --name uai-<azureml-compute-name> --resource-group <workspace-resource-group>
    ```

2. Attach the Arc cluster to the orchestrator workspace, or in other words _create an Azure ML attached compute pointing to the Arc cluster_:

    ```bash
    az ml compute attach --resource-group <workspace-resource-group> --workspace-name <workspace-name> --type Kubernetes --name <azureml-compute-name> --resource-id "/subscriptions/<subscription-id>/resourceGroups/<connected-cluster-resource-group>/providers/Microsoft.Kubernetes/connectedClusters/<Azure-Arc-enabled-k8s-resource-name>" --identity-type UserAssigned --user-assigned-identities "subscriptions/<subscription-id>/resourceGroups/<workspace-resource-group>/providers/Microsoft.ManagedIdentity/userAssignedIdentities/uai-<azureml-compute-name>" --no-wait    
    ```

    where:
    - `<workspace-resource-group>` is the resource group of the orchestrator workspace that you used in [step A](#a-fl-admin-creates-the-azure-ml-workspace-and-the-orchestrator);
    - `<workspace-name>` is the name of the orchestrator workspace that was created during [step A](#a-fl-admin-creates-the-azure-ml-workspace-and-the-orchestrator);
    - `<azureml-compute-name>` is the name that you _choose_ for the silo compute in the orchestrator workspace (arbitrary);
    - `<subscription-id>` is the Id of the orchestrator subscription;
    - `<connected-cluster-resource-group>` and `<Azure-Arc-enabled-k8s-resource-name>` have been defined in steps [C](#c-silo-admin-connects-the-existing-k8s-cluster-to-azure-arc) and [D](#d-fl-admin-deploys-the-azure-ml-extension-on-the-k8s-cluster-via-azure-arc);
    - `uai-<azureml-compute-name>` should be the name of the user-assigned identity you just created.
3. Create a storage account for this external silo.

    ```bash
    az deployment group create --template-file ./mlops/bicep/modules/storages/new_blob_storage_datastore.bicep --resource-group <workspace-resource-group> --parameters machineLearningName=<workspace-name> storageName=<storage-account-name> storageRegion=<workspace-region> publicNetworkAccess="Enabled" tags={}
    ```

    where:
    - `<storage-account-name>` is the name that you _choose_ for the silo storage account (arbitrary, but will be automatically formatted to remove forbidden characters);
    - `<workspace-region>` is the region of the orchestrator workspace (we recommend creating this storage account in the same region as the orchestrator workspace).
4. Set permissions for the silo's compute to R/W from/to the orchestrator and silo storage accounts.

    4.1. Navigate the Azure portal to find your workspace resource group.

    4.2. Look for a resource of type **Managed Identity** named like `uai-<azureml-compute-name>`. It should have been created by the instructions above.

    4.3. Open this identity and click on **Azure role assignments**.

    4.4. Click on **Add role assignment** and add the 3 roles below towards the silo storage account, which should be named `<storage-account-name>` (or something slightly different if you used any forbidden characters; in any case, you should be able to easily locate it from the Azure portal).
      - **Storage Blob Data Contributor**
      - **Reader and Data Access**
      - **Storage Account Key Operator Service Role**

    4.5. Repeat the same steps for the storage account of your orchestrator (this storage account should be named `storch` if you kept the default value for the `pairBaseName` parameter suggested in [step A](#a-fl-admin-creates-the-azure-ml-workspace-and-the-orchestrator), otherwise it will be the value you chose for `pairBaseName`, appended to `st`).

### F. **FL Admin** runs a test job

To validate everything is wired properly we are going to run a degenerate (using only one silo) HELLOWORLD-type FL job.

First, open the [config.yaml](../../examples/pipelines/fl_cross_silo_literal/config.yaml) file located in the `examples/pipelines/fl_cross_silo_literal` directory, and do the following.

- In the `aml` section, adjust the values of `subscription_id`, `resource_group_name`, and `workspace_name` to the proper values corresponding to your workspace.
- In the `orchestrator` section, adjust the values of `compute` and `datastore` to those corresponding to the orchestrator you created in [step A](#a-fl-admin-creates-the-azure-ml-workspace-and-the-orchestrator).
  - The `compute` value should be `cpu-cluster-orch` if you kept the default value for the `pairBaseName` , otherwise it will be the value you chose for `pairBaseName`, appended to `cpu-cluster-`.
  - The `datastore` value should be `datastore_orch` if you kept the default value for the `pairBaseName` , otherwise it will be the value you chose for `pairBaseName`, appended to `datastore_` (if you had '`-`' characters in `pairBaseName`, they will be replaced by '`_`').
- In the `silo` section:
  - Just keep one silo, by deleting or commenting out all entries but one (3 entries to start with, each with a `compute`, `datastore`, `training_data`, and `testing_data` parameter).
  - Adjust the remaining values of `compute` and `datastore` to those corresponding to the silo you created in [step E](#e-fl-admin-attaches-the-arc-cluster-to-the-orchestrator-workspace). The value for `compute` will be `<azureml-compute-name>`, and the value for `datastore` will be `datastore_<storage-account-name>` (with '`-`' characters replaced by '`_`' if applicable).

Submit the job by running the following command.

```bash
python ./examples/pipelines/fl_cross_silo_literal/submit.py --example HELLOWORLD
```

> Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.
