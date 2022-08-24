# Simple example of using shrike Federated Learning API + Arc + Kubernetes + AML to submit a Federated Learning experiment

## Scope
:warning:
Note that the goal of this experiment is to demonstrate the feasibility of using the [shrike Federated Learning API](https://shrike-docs.com/pipeline/federated-learning-doc/) to run a Federated Learning experiment where the silos are Kubernetes clusters connected to the Azure ML orchestrator workspace _via_ Azure Arc. 

The goal is NOT to train a _real_ model on _real_ data, it is just to demonstrate how the various components interact with each other, and to show what the Federated Learning experience on Azure ML might feel like. No _real_ training is happening, and any dataset can be used.

## Setup

- Clone the current repository and set `fl_arc_k8s` as your working directory.
- Set up a new conda environment with python version 3.8 and shrike dependencies (the environment is defined in `environment.yml` and `requirements.txt`) by running the following command.

  `conda env create --file environment.yml`

- Activate the newly created environment.

## How to run the example

Make sure you are in the `fl_arc_k8s` directory, and the conda environment 'fl-demo-env-py38' described in the Setup step above is activated. If you have access to the `aml1p-ml-wus2` Azure ML workspace, you can simply run the following command to submit the experiment:

```ps1
python pipelines/experiments/demo_federated_learning_k8s.py --config-dir pipelines/config --config-name experiments/demo_federated_learning_k8s +run.submit=True
```

Here is an [example successful experiment](https://ml.azure.com/runs/e5fe7afa-0884-492a-a36e-2f242946d066?wsid=/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourcegroups/aml1p-rg/workspaces/aml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#). It does 3 iterations of (mock) local training and model aggregation, using 3 different silos. Each silo uses a different (dummy) dataset based on the [MNIST public data](https://en.wikipedia.org/wiki/MNIST_database): mnist_01, mnist_02, and mnist_03. Due to lack of capacity in the underlying Azure subscription we have not created silos in different regions, but this is indeed possible (if you have access to the "ML Lifecycle PM" Azure subscription, you can find such an example [over there](https://ml.azure.com/experiments/id/97f21904-e7c7-404a-835a-1b21ee026d67/runs/e4b26654-4ba9-4153-8270-e3a1cd8a0c33?wsid=/subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourcegroups/thopo-orchestrator-rg/workspaces/thopo-orchestrator&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#)).

Note that the example here is using the AIMS team's `aml1p-ml-wus2` workspace. If you don't have access to this workspace and are a Microsoft employee, you can get access by joining the "aims-contrib" security group in idweb.

If you are not a Microsoft employee, then you will not be able to get access. You will first need to create all the resources yourself; we recommend using the scripts in the  `mlops` folder, as explained in the associated [README](../mlops/README.md). Then you will have to modify the configuration files under directory `fl_arc_k8s/pipelines/config/` to point to your own AML workspace, silos, etc... This last part is explained in the section below.

### Modify the configuration files
Since you won't be using the default `aml1p-ml-wus2` workspace, you need to update the [aml/public_workspace.yaml](./pipelines/config/aml/public_workspace.yaml) config file in the `fl_arc_k8s/pipelines/config/` directory to point to your workspace of choice.

Once that is done, you need to update the [experiments/demo_federated_learning_k8s.yaml](./pipelines/config/experiments/demo_federated_learning_k8s.yaml) config file to point to your dataset of choice, and to your silos. For the silos, you will need to make sure that each silo's `compute` value in the config file matches the name of the compute attached to the orchestrator workspace. The silo's name in the config file is arbitrary.
