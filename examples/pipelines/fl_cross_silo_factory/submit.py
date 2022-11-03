# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Federated Learning Cross-Silo pipeline built by a factory class.

This scripts wraps up the FL pipeline orchestration code in a "factory" class.
This class is a draft API to build a pipeline based on simple steps
(preprocessing, training, aggregation). The factory could be extended
to cover more advanced scenarios.

In sequence, this script will:
A) reads a config file in yaml specifying the number of silos and their parameters,
B) reads the components from a given folder,
C) allow developers to write FL pipeline steps as pythonic functions,
D) call the factory class to build the full FL pipeline based on custom user code.


To adapt this script to your scenario, you can:
- modify the config file to change the number of silos
  and their parameters (see section A and config.yaml file),
- modify the components directly in the components folder (see section B),
- modify the silo_preprocessing()m silo_training() and orchestrator_aggregation() functions
  to change the steps behaviors (see section C and D.2 D.3),
- modify the affinity map according to a custom permission model (see section D.4).
"""
import os
import argparse
import random
import string
import datetime
import webbrowser
import time
import json
import sys
import subprocess

# Azure ML sdk v2 imports
import azure
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.entities._job.pipeline._io import NodeOutput

# to handle yaml config easily
from omegaconf import OmegaConf

from typing import List, Optional, Union
from typing import Callable, Dict
from dataclasses import dataclass
import itertools

# local imports
from fl_factory import FederatedLearningPipelineFactory


# Note: This code is using subgraphs (a.k.a. pipeline component) which is currently a PrivatePreview feature subject to change.
# For an FL experience relying only on GA features, please refer to the literal version of the code.
os.environ["AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED"] = "true"

###############################
### A. CONFIGURE THE SCRIPT ###
###############################

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--config",
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    help="path to a config yaml file",
)
parser.add_argument(
    "--submit",
    default=False,
    action="store_true",
    help="actually submits the experiment to AzureML",
)
parser.add_argument(
    "--ignore_validation",
    default=False,
    action="store_true",
    help="bypass soft validation warnings and submit the experiment",
)

parser.add_argument(
    "--example",
    required=False,
    choices=["MNIST", "HELLOWORLD"],
    default="MNIST",
    help="dataset name",
)

parser.add_argument(
    "--subscription_id",
    type=str,
    required=False,
    help="Subscription ID",
)
parser.add_argument(
    "--resource_group",
    type=str,
    required=False,
    help="Resource group name",
)

parser.add_argument(
    "--workspace_name",
    type=str,
    required=False,
    help="Workspace name",
)

parser.add_argument(
    "--wait",
    default=False,
    action="store_true",
    help="Wait for the pipeline to complete",
)

args = parser.parse_args()

# load the config from a local yaml file
YAML_CONFIG = OmegaConf.load(args.config)

# dict of training parameters
training_kwargs = {
    "lr": YAML_CONFIG.training_parameters.lr,
    "batch_size": YAML_CONFIG.training_parameters.batch_size,
    "epochs": YAML_CONFIG.training_parameters.epochs,
}

###########################
### CONNECT TO AZURE ML ###
###########################


def connect_to_aml():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()

    # Get a handle to workspace
    try:
        # tries to connect using local config.json
        ML_CLIENT = MLClient.from_config(credential=credential)

    except Exception as ex:
        print(
            "Could not find config.json, using config.yaml refs to Azure ML workspace instead."
        )

        # tries to connect using cli args if provided else using config.yaml
        ML_CLIENT = MLClient(
            subscription_id=args.subscription_id or YAML_CONFIG.aml.subscription_id,
            resource_group_name=args.resource_group
            or YAML_CONFIG.aml.resource_group_name,
            workspace_name=args.workspace_name or YAML_CONFIG.aml.workspace_name,
            credential=credential,
        )
    return ML_CLIENT


ML_CLIENT = connect_to_aml()


#######################################
### B. LOAD THE PIPELINE COMPONENTS ###
#######################################

# path to the components
COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", args.example
)

# Loading the component from their yaml specifications
preprocessing_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "preprocessing", "preprocessing.yaml")
)

training_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "traininsilo", "traininsilo.yaml")
)

aggregate_component = load_component(
    source=os.path.join(
        COMPONENTS_FOLDER, "aggregatemodelweights", "aggregatemodelweights.yaml"
    )
)


#########################
### C. CUSTOM DS CODE ###
#########################

# This is your section, please modify anything here following the guidelines
# in the docstrings.

# The idea is that each of the following method is a "contract" you own
# and specify. The factory will read this contract and assemble the
# pieces of your pipeline based on input and output keys.

@pipeline(
    name="Silo Federated Learning Subgraph",
    description="It includes preprocessing and training components",
)
def silo_scatter_subgraph(
    # user defined inputs
    raw_train_data: Input,
    raw_test_data: Input,

    # user defined accumulator
    aggregated_checkpoint: Input(optional=True),

    # factory inputs (contract)
    scatter_compute: str,
    scatter_datastore: str,
    gather_datastore: str,
    iteration_num: int,

    # user defined training arguments
    lr: float = 0.01,
    epochs: int = 3,
    batch_size: int = 64,
):
    """Create silo/training subgraph.

    Args:
        raw_train_data (Input): raw train data
        raw_test_data (Input): raw test data
        aggregated_checkpoint (Input): if not None, the checkpoint obtained from previous iteration (see orchestrator_aggregation())
        iteration_num (int): Iteration number
        compute (str): Silo compute name
        datastore (str): Silo datastore name
        model_datastore (str): Model datastore name

    Returns:
        Dict[str, Outputs]: a map of the outputs
    """
    # we're using preprocessing component directly
    silo_pre_processing_step = preprocessing_component(
        raw_training_data=raw_train_data,
        raw_testing_data=raw_test_data,
        metrics_prefix=scatter_compute,
    )

    # we're using training component directly
    silo_training_step = training_component(
        # with the train_data from the pre_processing step
        train_data=silo_pre_processing_step.outputs.processed_train_data,
        # with the test_data from the pre_processing step
        test_data=silo_pre_processing_step.outputs.processed_test_data,
        # and the checkpoint from previous iteration (or None if iteration == 1)
        checkpoint=aggregated_checkpoint,
        # Learning rate for local training
        lr=lr,
        # Number of epochs
        epochs=epochs,
        # Dataloader batch size
        batch_size=batch_size,
        # Silo name/identifier
        metrics_prefix=scatter_compute,
        # Iteration number
        iteration_num=iteration_num,
    )

    # IMPORTANT: use outputs only for data that can be exfiltrated into the orchestrator
    return {
        # IMPORTANT: key needs to be consistent with expected numbered inputs of gather component/pipeline
        "input_silo" : silo_training_step.outputs.model
    }


#######################
### D. FACTORY CODE ###
#######################

# In this section, we're using a wrapper to build the full FL pipeline
# based on the custom methods you implemented in section C.

# 1. create an instance of the factory
builder = FederatedLearningPipelineFactory()

# 2. feed it with FL parameters

builder.set_orchestrator(
    # provide settings for orchestrator
    YAML_CONFIG.federated_learning.orchestrator.compute,
    YAML_CONFIG.federated_learning.orchestrator.datastore,
)

for silo_config in YAML_CONFIG.federated_learning.silos:
    builder.add_silo(
        # provide settings for this silo
        silo_config.compute,
        silo_config.datastore,
        # any additional custom kwarg will be sent to silo_preprocessing() as is
        raw_train_data=Input(
            type=silo_config.training_data.type,
            mode=silo_config.training_data.mode,
            path=silo_config.training_data.path,
        ),
        raw_test_data=Input(
            type=silo_config.testing_data.type,
            mode=silo_config.testing_data.mode,
            path=silo_config.testing_data.path,
        ),
    )

# 3. use a pipeline factory method
# 3.1 build flexible fl pipeline

pipeline_job = builder.build_flexible_fl_pipeline(
    # building requires all 2 functions provided as argument below
    scatter=silo_scatter_subgraph,
    gather=aggregate_component,

    accumulator={
        # this key needs to be the name of the output of gather component/pipeline
        # AND be an acceptable input key for scatter component/pipeline
        "aggregated_checkpoint" : Input(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=Input(
                type="uri_folder",
                mode="mount",
                path="azureml://{YAML_CONFIG.federated_learning.orchestrator.datastore}/foo"
            ),
        )
    },
    iterations=YAML_CONFIG.training_parameters.num_of_iterations,

    # any additional kwarg is considered constant and given to scatter as is
    lr=YAML_CONFIG.training_parameters.lr,
    batch_size=YAML_CONFIG.training_parameters.batch_size,
    epochs=YAML_CONFIG.training_parameters.epochs,
)

# 4. Validate the pipeline using soft rules

print(pipeline_job)  # print yaml for visual debugging

# use a default set of rules
builder.set_default_affinity_map()

# run affinity map validation
builder.soft_validate(
    pipeline_job,
    raise_exception=not (
        args.ignore_validation
    ),  # set to False if you know what you're doing
)

# 5. Submit to Azure ML

if args.submit:
    print("Submitting the pipeline job to your AzureML workspace...")

    pipeline_job = ML_CLIENT.jobs.create_or_update(
        pipeline_job, experiment_name="fl_dev"
    )

    print("The url to see your live job running is returned by the sdk:")
    print(pipeline_job.services["Studio"].endpoint)

    webbrowser.open(pipeline_job.services["Studio"].endpoint)

    if args.wait:
        job_name = pipeline_job.name
        status = pipeline_job.status

        while status not in ["Failed", "Completed", "Canceled"]:
            print(f"Job current status is {status}")

            # check status after every 100 sec.
            time.sleep(100)
            pipeline_job = ML_CLIENT.jobs.get(name=job_name)
            status = pipeline_job.status

        print(f"Job finished with status {status}")
        if status in ["Failed", "Canceled"]:
            sys.exit(1)

else:
    print("The pipeline was NOT submitted, use --submit to send it to AzureML.")
