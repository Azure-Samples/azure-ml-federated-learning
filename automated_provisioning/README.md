# Resource Provisioning for Contoso

## Contents

This document proposes a procedure to provision the resources required for Contoso to be running Federated Learning experiments. (Contoso is just a fictional organization.)

The resources can then be used to explore Federated Learning in Azure ML, for instance by running the [example experiment](../fl_arc_k8s/README.md).

## Requirements

For running _external_ cross-_silo_ Federated Learning experiments, the Contoso team needs the following ingredients:

- an "orchestrator" Azure ML workspace;
- some Kubernetes (K8s) clusters;
- connections between the K8s clusters and the Azure ML workspace;
- computes and datastores to run the DataTransfer steps.

The procedure outlined in this document will provision resources that meet the requirements above.

## Prerequisites

We are providing a [docker file](./Dockerfile) which contains all the prerequisites below. We strongly recommend you use this dockerfile to ensure you have all the required dependencies.

> Reading the rest of this **Prerequisites** section is completely optional, as all these steps have been incorporated in the docker file.

Taken from [here](https://github.com/Azure/AML-Kubernetes#prerequisites) (along with the K8s cluster creation/connection steps).

0. Have PowerShell (or PowerShell Core if not on Windows).
1. Have access to an Azure subscription.
2. Install the [latest release of Helm 3](https://helm.sh/docs/intro/install/) - for Windows, we recommend going the _chocolatey_ route.
3. Meet the pre-requisites listed under the [generic cluster extensions documentation](https://docs.microsoft.com/azure/azure-arc/kubernetes/extensions#prerequisites).
    - Azure CLI version >=2.24.0
    - Azure CLI extension k8s-extension version >=1.0.0.
    - Only focus on the extensions, no need to create the k8s cluster yet.
4. Install and setup the v2 version of the _Azure ML CLI extension_ following [these instructions](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli).
5. Install the [Bicep CLI](https://docs.microsoft.com/en-us/azure/azure-resource-manager/bicep/install) and the [Bicep extension in VS Code](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-bicep).

## Procedure

The procedure is fairly simple (detailed instructions are given in the numbered subsections below, this section is just a preview of what you'll have to do).

0. Use the docker image that contains all the prerequisites.
1. You will first run the `CreateK8sCluster.ps1` script (with the appropriate arguments) and log in when prompted. This will create your k8s cluster, _a,k,a,_ your silo.
2. Then you will run the `ConnectSiloToOrchestrator.ps1` script (with the appropriate arguments) and log in when prompted. This will create an Azure ML workspace (if it doesn't exist already) and connect your silo to it.
3. You will have the option to change the silo compute size by running `ChangeComputeSize.ps1` (with the appropriate arguments).
4. Add more silos by repeating the steps above.
5. Run a simple silo validation job. All resources are provided, you'll just need to point at your silo.
6. Create compute resources in the orchestrator.
7. Create the resources required for Data Transfer steps.
8. Upload some (non-sensitive) data

### 0. Use the docker image that contains all the prerequisites
- Clone the current repository and set `automated_provisioning` as your working directory.
- Build a docker image based on the [docker file](./Dockerfile) by running: 
  ```ps1
  docker build -t fl-rp .
  ```
  ("fl-rp" will be the name of the docker image and stands for FL-Resource Provisioning).
- Run the docker image with the following command:
    ```ps1
    docker run -i fl-rp
    ```
> All the steps below will need to be carried out **from within the docker image**.
### 1. Set up a silo

For starters, you need to create a K8s cluster and the associated resource group if they don't exist already. Then you'll need to connect to this cluster and create the _kube_ config file (which will be used implicitly by the second script to point at this particular cluster). There is a script that does all that for you: `CreateK8sCluster.ps1`. It takes the following input arguments.
- `SubscriptionId`: the Id of the subscription where the silo will be created. 
- `K8sClusterName`: the name of the K8s cluster to be created (default: "cont-k8s-01"). It will live in a resource group named like the cluster, with "-rg" appended.
- `RGLocation`: the location of the K8s cluster and its corresponding resource group (default: "westus2").
- `AgentCount`: the number of agents in the K8s cluster (default: 1 - beware, this should be an _int_, not a _string_).
- `AgentVMSize`: The agent VM SKU (default: "Standard_A2_v2"). Here, we suggest to use a weak VM as the default pool of K8s, since we will introduce more powerful compute instances for silo training. 

The command should look something like the below (with the parameters replaced by your own values of course).

```ps
./ps/CreateK8sCluster.ps1 -SubscriptionId "Your-Silo-SubscriptionId" -K8sClusterName "Name-of-Cluster-to-Create" -RGLocation "Location-of-Cluster-to-Create" -AgentCount Number-of-Agents -AgentVMSize "VM-SKU"
```

### 2. Connect the silo to the orchestrator workspace

To connect a silo to an orchestrator Azure ML workspace, the following needs to happen:
- create the Azure ML workspace if it doesn't already exist;
- connect the K8s to Azure Arc; 
- deploy the Azure ML extension on the Arc cluster;
- attach the Arc cluster to the Azure ML workspace.

Here again, there is a script that does all of that for you: `ConnectSiloToOrchestrator.ps1`. It takes the following input arguments.
- `SubscriptionId_Orchestrator`: the Id of the subscription to which the orchestrator will belong.
- `AMLWorkspaceName`: the name of the orchestrator Azure ML workspace to create, if it doesn't exist already.
- `AMLWorkspaceRGName`: the name of the orchestrator resource group to create, if it doesn't exist already.
- `AMLWorkspaceLocation`: the location of the orchestrator Azure ML workspace (default: "westus2").
- `K8sClusterName`: the name of the K8s cluster to connect to the orchestrator Azure ML workspace (default: "cont-k8s-01"). **Note that this is just used to create the name of the Arc cluster and its resource group. The K8s cluster is referenced implicitly by the kube config file that was created during the previous step.**
- `AMLComputeName`: the name of the Azure ML compute to be created (default: "cont-01-compute" - must be between 2-16 characters and only contain alphanumeric characters or dashes). **This is the compute name you will be using when submitting jobs.**

The command should look something like the below (with the parameters replaced by your own values of course).

```ps
./ps/ConnectSiloToOrchestrator.ps1 -SubscriptionId_Orchestrator "Your-Orchestrator-SubscriptionId" -AMLWorkspaceName "Your-Orchestrator-Workspace-Name" -AMLWorkspaceRGName "Your-Orchestrator-Resource-Group-Name" -AMLWorkspaceLocation "Your-Orchestrator-Location" -K8sClusterName "Name-of-K8s-Cluster-to-Connect" -AMLComputeName "AML-Compute-Name-to-Create"
```

### 3. Change Compute Instance size (optional)

> If you're just exploring Federated Learning on Azure ML (for instance by running the [example experiment](../fl_arc_k8s/README.md)), you probably won't need a lot of compute resources, so you can skip this step.

The default compute instance only provisions a small portion of your K8s cluster (specifically, 1.5G memory and 0.6 cpu). You may need to override this by taking the following steps.
- Setup your local kubectl environment and connect to the AKS.
- Get the node name and apply a label.
- Deploy a new compute instance with your choice of compute power.
**Resource units explanation can be found [here](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu). More information on requests and limits can be found [there](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).**

Here again, there is a script that does all of that for you: `ChangeComputeSize.ps1`. It takes the following input arguments.
- `SubscriptionId`: the Id of the subscription to which the K8s cluster belongs.
- `K8sClusterName`: the name of the K8s cluster whose size we want to change (default: "cont-k8s-01"). 

The command should look something like the below (with the parameters replaced by your own values of course).

```ps
./ps/ChangeComputeSize.ps1 -SubscriptionId "Your-Silo-SubscriptionId" -K8sClusterName "Name-of-K8s-Cluster-to-Resize" 
```

To verify, go to the orchestrator AML workspace, and find the attached cluster by clicking "Compute" &mapsto; "Attached computers" and search your cluster with the `AMLComputeName` you used in the previous step.

### 4. Add more silos
Just repeat the steps above for every silo you want to create. For running the [example experiment](../fl_arc_k8s/README.md), **you will need to create 3 silos**.

> You need to create a cluster, then connect it. If you first create several clusters, then try to connect them, you will run into issues. This is because the connection script implicitly uses the cluster reference from the first step. 

> If you want to have your K8s cluster and the orchestrator Azure ML workspace in different subscriptions, this is possible. Just use a different subscription in each of the 2 steps, and log in accordingly when prompted to. 



### 5. Run a simple validation job

> This simple validation job currently just tests **one** silo. You will need top run it on every one of them.

To double check that you can actually run Azure ML jobs on the Arc Cluster, we provide all the files required for a sample job, following the example [here](https://github.com/Azure/AML-Kubernetes/blob/master/docs/simple-train-cli.md). First, you'll need to open `./sample_job/job.yml` - this is the file where the job you are going to run is defined. Adjust the compute name (the part after `compute: azureml:`) to the name of your Azure ML compute.

The PowerShell script `RunSampleJob.ps1` will submit the job and upload the classic MNIST train and test data for you, from the files locally available in the repository. It takes the following arguments.
- `SubscriptionId`: the Id of the subscription to which the Azure ML orchestrator workspace belongs.
- `WorkspaceName`: the name of the Azure ML orchestrator workspace.
- `ResourceGroup`: the resource group of the Azure ML orchestrator workspace.

The command should look something like the below (with the parameters replaced by your own values of course).

```ps
./sample_job/RunSampleJob.ps1 -SubscriptionId "Your-Orchestrator-SubscriptionId" -WorkspaceName "Your-Orchestrator-Workspace-Name" -ResourceGroup "Your-Orchestrator-Resource-Group"
```

Note that there is nothing special about this job, except for the fact that it will run on the new Arc-enabled K8s cluster; if you feel more comfortable using one of your own jobs to achieve that, it is perfectly acceptable.

### 6. Create compute resources in the orchestrator
By default, the orchestrator workspace might not come with any CPU compute cluster. We need to create one (for running the aggregation or preprocessing steps). To do so, just run the following command with the appropriate parameter values for your orchestrator workspace name and resource group.

```ps1
az ml compute create --file .\yaml\cpu-cluster.yml --resource-group "Your-Orchestrator-Resource-Group" --workspace-name "Your-Orchestrator-Workspace-Name"
```

You will likely need a GPU cluster too. Just re-run the command above, but this time pointing at the `.\yaml\gpu-cluster.yml` file.

After that, you will have 2 new compute clusters in your orchestrator workspace, named `cpu-cluster` and `gpu-cluster`. (For more details about compute cluster creation, see [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=azure-cli#create).)

### 7. Create resources for data transfer steps

A typical Federated Learning experiment involves transferring some model weights back and forth between the orchestrator and the silos. This is achieved using a particular Azure ML component: the [Data Transfer Component](https://componentsdk.azurewebsites.net/components/data_transfer_component.html). This component requires a Data Factory compute, along with some properly configured datastores and storage accounts.

Attaching a Data Factory compute should be straightforward. Just install the `azureml-core` package _via_ `pip install azureml-core` and run the [./python/attach_data_factory.py](./python/attach_data_factory.py) script _after_ having entered the proper values of the workspace name, resource group, and subscription Id for your setup.

The next step after that is to create some datastores and the underlying storage accounts (these will be used for communicating model weights between the silos and the orchestrator).

First, let's create a storage account for the first silo, using the following command.
```ps1
az storage account create -n "Your-Account-Name-For-This-Silo" -g "Your-Orchestrator-Resource-Group" -l "Your-Silo-Location" --kind StorageV2 --access-tier Hot
```

A few notes on the parameter values you need to provide:
- the name of your storage account (`-n` parameter) has to be globally unique in Azure;
- although not mandatory, we recommend attaching the storage account to the resource group containing the orchestrator Azure ML workspace (`-g` parameter);
- we recommend using the same location as the silo (`-l` parameter).

Now that we have a storage account, we need to create an Azure ML datastore and connect it to the storage account (more information on datastores can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-datastore?tabs=cli-identity-based-access%2Ccli-adls-identity-based-access%2Ccli-azfiles-account-key%2Ccli-adlsgen1-identity-based-access)). To do so, we will first update the [./yaml/datastore.yml](./yaml/datastore.yml) file with the proper values for the datastore name and description, and for the storage account we just created. Then, we will run the following command with the appropriate parameter values for the workspace name and resource group.

```ps1
az ml datastore create --file .\yaml\datastore.yml --resource-group "Your-Orchestrator-Resource-Group" --workspace "Your-Orchestrator-Workspace-Name"
```

Last, we need to update the datastore credentials so it can connect to the storage account. This will be done through the UI, as follows.
- Locate your storage account in the [Azure portal](https://portal.azure.com/). Click on "Access Keys" in the left-hand menu, and copy the value of the access key.
- Locate your datastore in the [Azure ML workspace](https://ml.azure.com). In the top menu, click "Update authentication". Enter your subscription Id and the name of the resource group containing the storage resource (it should be your orchestrator resource group if you followed our recommendations). Finally, for "Authentication type" select "Account Key", paste the key you copied previously, and click "Save".

That's it, you now should have a datastore and a storage account configured properly for exchanging model weights between your first silo and the orchestrator. Repeat the 3 steps above (storage account creation, datastore creation, and credentials update) for the remaining silos.

### 8. Add some data

> This section is still pretty much work in progress. We haven't introduced the parts to control access to the _silo_ data yet. This is enough to get you going with the [example experiment](../fl_arc_k8s/README.md), but please do not use any sensitive data for now!

In general, a Federated Learning experiment needs two kinds of data:
- the _silo_ data (_e.g._ the sensitive data on which to train the local models);
- the _primer_ data (_e.g._ some pre-trained model weights, or some public data on which to train the baseline model in the orchestrator).

We now explain how to create datasets that we will use for running the [example experiment](../fl_arc_k8s/README.md). We are going to use the public [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database), and are going to create several copies of it to mimic the various kinds of data mentioned above. The data are included in the current repository, so all we have to do is upload them, and package them as an Azure ML dataset (more information [here](https://docs.microsoft.com/en-us/azure/machine-learning/v1/how-to-create-register-datasets) if needed).

This can be done by running the following command, which will create an Azure ML dataset named `mnist` (see the value of the `--name` parameter). We will use this dataset as _primer_ data.

```ps1
az ml dataset create --name mnist --description "MNIST dataset" --local-path ./sample_job/src/mnist-data --resource-group "Your-Orchestrator-Resource-Group" --workspace-name "Your-Orchestrator-Workspace-Name"
```

For the time being, we create the _silo_ data in the exact same fashion. Just re-run the above command once per silo, giving a unique `--name` each time. (We recommend naming the datasets something like `mnist_01`, `mnist_02`, etc...)


## Further reading
- The part about creating/connecting K8s clusters is based on [these instructions](https://github.com/Azure/AML-Kubernetes). A summary can also be found in [this deck](https://microsoft.sharepoint.com/:p:/t/AMLDataScience/EQSxAxYrjX1BiOh3s23GpJUB81sgQfNQJFTWCRR0T8pODg?e=6hcvRL).
