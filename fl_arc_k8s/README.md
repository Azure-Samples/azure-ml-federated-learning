# Simple example of using shrike Federated Learning API + Arc + Kubernetes + AML to submit a Federated Learning experiment

## Setup

- Clone the current repository and set `fl_arc_k8s` as your working directory.
- Set up and activate a new conda environment with python version 3.8 and shrike dependencies (the environment is defined in `environment.yml` and `requirements.txt`):

  `conda env create --file environment.yml`

## How to run the example

Make sure you are in the `fl_arc_k8s` directory, and the conda environment 'fl-shrike-examples-env-py38' described in the Setup step above is activated. If you have access to the `aml1p-ml-wus2` Azure ML workspace, you can simply run the following command to submit the experiment:

```
python pipelines/experiments/demo_federated_learning_k8s.py --config-dir pipelines/config --config-name experiments/demo_federated_learning_k8s +run.submit=True
```

Here is an [example successful experiment](https://ml.azure.com/runs/7d2e979a-0785-4ff0-a5d0-b1d43f6c8467?wsid=/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourcegroups/aml1p-rg/workspaces/aml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#).

Note that the example here is using the AIMS team's `aml1p-ml-wus2` workspace. If you don't have access to this workspace and are a Microsoft employee, you can get access by joining the "aims-contrib" security group in idweb. If you are not a Microsoft employee, then you will not be able to get access and you will have to modify the corresponding config files under directory `fl_arc_k8s\pipelines\config\` to point to your own AML workspace. You will also have to provision the corresponding computes (for instance using the scripts in the [automated_provisioning](../automated_provisioning/README.md) folder) and input datasets before you can submit the experiment.