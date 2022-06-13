# Simple example of using shrike Federated Learning API + Arc + Kubernetes + AML to submit a Federated Learning experiment

## Scope
:warning:
Note that the goal of this experiment is to demonstrate the feasibility of using the [shrike Federated Learning API](https://shrike-docs.com/pipeline/federated-learning-doc/) to run a Federated Learning experiment where the silos are Kubernetes clusters connected to the Azure ML orchestrator workspace _via_ Azure Arc. 

The goal is NOT to train a _real_ model on _real_ data, it is just to demonstrate how the various components interact with each other, and to show what the Federated Learning experience on Azure ML might feel like. No _real_ training is happening, and although we suggest one particular dataset, any dataset can be used.

## Setup

- Clone the current repository and set `fl_arc_k8s` as your working directory.
- Set up and activate a new conda environment with python version 3.8 and shrike dependencies (the environment is defined in `environment.yml` and `requirements.txt`):

  `conda env create --file environment.yml`

## How to run the example

Make sure you are in the `fl_arc_k8s` directory, and the conda environment 'fl-demo-env-py38' described in the Setup step above is activated. If you have access to the `aml1p-ml-wus2` Azure ML workspace, you can simply run the following command to submit the experiment:

```
python pipelines/experiments/demo_federated_learning_k8s.py --config-dir pipelines/config --config-name experiments/demo_federated_learning_k8s +run.submit=True
```

Here is an [example successful experiment](https://ml.azure.com/runs/7d2e979a-0785-4ff0-a5d0-b1d43f6c8467?wsid=/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourcegroups/aml1p-rg/workspaces/aml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#).

Note that the example here is using the AIMS team's `aml1p-ml-wus2` workspace. If you don't have access to this workspace and are a Microsoft employee, you can get access by joining the "aims-contrib" security group in idweb.

If you are not a Microsoft employee, then you will not be able to get access and you will have to modify the corresponding config files under directory `fl_arc_k8s/pipelines/config/` to point to your own AML workspace. You will also have to provision some computes and some input datasets before you can submit the experiment. All of this is explained in the 3 sections below.

### Computes, workspace, and data
For provisioning the orchestrator workspace and the silos computes, we recommend using the scripts in the `automated_provisioning` folder. See the [README](../automated_provisioning/README.md) for the details. For running the current example experiment, _you will need 3 silos_.

By default, the orchestrator workspace will not come with any compute cluster. We need to create one for  running the aggregation step. To do so, just run the following command with the appropriate parameter values for your orchestrator workspace name and resource group.

```ps1
az ml compute create --file .\utils\template\compute_cluster.yml --resource-group <Your-Orchestrator-Resource-Group> --workspace-name <Your-Orchestrator-Workspace-Name>
```

It will create a basic compute cluster in your orchestrator workspace and name it `cpu-cluster`. For more details about the ARM template used to create the cluster, see [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=azure-cli#create).

Now that we have a compute resource, we only need one last thing before we can run a FL job: some data! As discussed above this is not a _real_ job, so _any_ data can be used. If you want to use some of your custom data, you can create a dataset following the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/v1/how-to-create-register-datasets).

If you don't have any data, you can use the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database), for instance. Just run the following command with the appropriate parameter values for your orchestrator workspace name and resource group.

```
az ml dataset create --name mnist --description "MNIST dataset" --local-path ../automated_provisioning/sample_job/src/mnist-data --resource-group <Your-Orchestrator-Resource-Group> --workspace-name <Your-Orchestrator-Workspace-Name>
```

This will create a dataset with name `mnist` in your workspace. That's the data we will use in this example.

### Configuration files
Since you won't be using the default `aml1p-ml-wus2` workspace, you need to update the [aml/public_workspace.yaml](./pipelines/config/aml/public_workspace.yaml) config file in the `fl_arc_k8s/pipelines/config/` directory to point to your workspace of choice.

Once that is done, you need to update the [experiments/demo_federated_learning_k8s.yaml](./pipelines/config/experiments/demo_federated_learning_k8s.yaml) config file to point to your dataset of choice, and to your silos. For the silos, you will need to make sure that each silo's `compute` value matches the name of the compute attached to the orchestrator workspace. The silos name in the config file is arbitrary.

Datastores?