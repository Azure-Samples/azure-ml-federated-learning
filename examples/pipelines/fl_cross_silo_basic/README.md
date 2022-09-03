# How to submit a Federated Learning experiment

## Scope

The goal of the experiment is to showcase what a federated learning experiment looks like on the Azure Machine Learning platform. A user can either consume pre-existing examples or can have a custom setup following the similar structure defined below.

## Setup

- Clone the current repository and set `examples/pipelines/fl_cross_silo_basic` as your working directory.
- Set up a new conda environment (defined in the `environment.yml` and `requirements.txt`) by running the following command.

  `conda env create --file environment.yml`

- Activate the newly created environment.

## How to run the example

Make sure you are in the `fl_cross_silo_basic` directory, and the conda environment `fl_experiment_conda_env` described in the Setup step above is activated. Follow the below steps:

1. Provide the AML subscription name, workspace name and resource group name in the `config.yaml` file.
2. Update the orchestrator as well as silos datastore and compute names. (Note: This step assumes that you followed the instructions given in the `mlops/README.md` file.)
3. Execute the following command by providing a example name you want to play around with. (Note: Current existing examples: [`"MNIST"`])
   
    ```ps1
    python submit.py --example <name> --submit
    ```


By default, it does 2 rounds of model aggregation along with 3 iterations of local training at each round.

Note: If you provisioned your setup using the resources in the `mlops/internal_silos` directory, you won't need to change the names of the computes and of the data assets.