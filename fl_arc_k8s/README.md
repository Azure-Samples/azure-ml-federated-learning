# Simple example of using shrike Federated Learning API + Arc + Kubernetes + AML to submit a Federated Learning pipeline experiment

## Setup

- Clone the current repository and set `fl_arc_k8s` as your working directory.
- Set up and activate a new Conda environment with python version 3.8 and shrike dependencies (environment is defined in `environment.yml` and `requirements.txt`):

  `conda env create --file environment.yml`

## How to run the example

Assume you're already in the directory `fl_arc_k8s` and the conda environment 'fl-shrike-examples-env-py38' in the Setup step above is activated. If you have access to  you could simply run below command within the Anaconda Powershell prompt window to submit the experiment:

```
python pipelines/experiments/demo_federated_learning_k8s.py --config-dir pipelines/config --config-name experiments/demo_federated_learning_k8s +run.submit=True
```

Here is an [example successful experiment](https://ml.azure.com/runs/7d2e979a-0785-4ff0-a5d0-b1d43f6c8467?wsid=/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourcegroups/aml1p-rg/workspaces/aml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#) submitted to AIMS team's `aml1p-ml-wus2` workspace using this command.

Note that, the example here is using AIMS team's `aml1p-ml-wus2` workspace for demonstration purpose. If you don't have access to AIMS team's `aml1p-ml-wus2` workspace, then you would have to modify/update the corresponding config files under directory `fl_arc_k8s\pipelines\config\` to use your own AML workspace and provision corresponding computes and input datasets before you could run above command to successfully submit the experiment.