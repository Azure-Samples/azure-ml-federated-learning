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

# Azure ML sdk v2 imports
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# to handle yaml config easily
from omegaconf import OmegaConf

from typing import List, Optional, Union
from typing import Callable, Dict
from dataclasses import dataclass
import itertools

# local imports
from fl_factory import FederatedLearningPipelineFactory

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

args = parser.parse_args()

# load the config from a local yaml file
YAML_CONFIG = OmegaConf.load(args.config)

# CONNECT TO AZURE ML

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

    # tries to connect using provided references in config.yaml
    ML_CLIENT = MLClient(
        subscription_id=YAML_CONFIG.aml.subscription_id,
        resource_group_name=YAML_CONFIG.aml.resource_group_name,
        workspace_name=YAML_CONFIG.aml.workspace_name,
        credential=credential,
    )


#######################################
### B. LOAD THE PIPELINE COMPONENTS ###
#######################################

# path to the components
COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", args.example
)

# Loading the component from their yaml specifications
preprocessing_component = load_component(
    path=os.path.join(COMPONENTS_FOLDER, "preprocessing", "preprocessing.yaml")
)

training_component = load_component(
    path=os.path.join(COMPONENTS_FOLDER, "traininsilo", "traininsilo.yaml")
)

aggregate_component = load_component(
    path=os.path.join(
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


def silo_preprocessing(raw_train_data: Input, raw_test_data: Input) -> Dict[str, Input]:
    """Create steps for running FL preprocessing in the silo.

    Args:
        raw_train_data (Input): preprocessed data (see FederatedLearningPipelineFactory.add_silo())
        raw_test_data (Input): preprocessed data (see FederatedLearningPipelineFactory.add_silo())

    Returns:
        PipelineStep: the preprocessing step of the FL pipeline
        Dict[str, Input]: a map of the inputs expected as kwargs by silo_training()
    """
    # run the pre-processing component once
    silo_pre_processing_step = preprocessing_component(
        raw_training_data=raw_train_data,
        raw_testing_data=raw_test_data,
    )

    return silo_pre_processing_step, {
        # IMPORTANT: use a key that is consistent with kwargs of silo_training()
        "train_data": silo_pre_processing_step.outputs.processed_train_data,
        "test_data": silo_pre_processing_step.outputs.processed_test_data,
    }


def silo_training(
    train_data: Input = None,  # output from silo_preprocessing()
    test_data: Input = None,  # output from silo_preprocessing()
    running_checkpoint: Input = None,  # output from orchestrator_aggregation()
    lr: int = 0.01,  # custom param given to factory build_basic_fl_pipeline()
    batch_size: int = 64,  # custom param given to factory build_basic_fl_pipeline()
    epochs: int = 1,  # custom param given to factory build_basic_fl_pipeline()
):
    """Create steps for running FL training in the silo.

    Args:
        train_data (Input): preprocessed data (see outputs of silo_preprocessing())
        test_data (Input): preprocessed data (see outputs of silo_preprocessing())
        running_checkpoint (Input): if not None, the checkpoint obtained from previous iteration (see orchestrator_aggregation())
        lr (int): learning rate for training component
        batch_size (int): batch size for training component
        epochs (int): epochs for training component

    Returns:
        PipelineStep: the training step of the FL pipeline
        Dict[str, Input]: a map of the inputs expected as kwargs by orchestrator_aggregation()
    """
    # we're using training component here
    silo_training_step = training_component(
        # with the train_data from the pre_processing step
        train_data=train_data,
        # with the test_data from the pre_processing step
        test_data=test_data,
        # and the checkpoint from previous iteration (or None if iteration == 1)
        checkpoint=running_checkpoint,
        # Learning rate for local training
        lr=lr,
        # Number of epochs
        epochs=epochs,
        # Dataloader batch size
        batch_size=batch_size,
    )

    return silo_training_step, {
        # IMPORTANT: use a key that is consistent with kwargs of orchestrator_aggregation()
        "weights": silo_training_step.outputs.model
    }


def orchestrator_aggregation(weights=[]):
    """Create steps for running FL training in the silo.

    Args:
        train_data (Input): preprocessed data (see silo_training())
        test_data (Input): preprocessed data (see silo_training())
        running_checkpoint (Input): if not None, the checkpoint obtained from previous iteration (see orchestrator_aggregation())
        lr (int): learning rate for training component
        batch_size (int): batch size for training component
        epochs (int): epochs for training component

    Returns:
        PipelineStep: the training step of the FL pipeline
        Dict[str, Input]: a map of the inputs expected as kwargs by orchestrator_aggregation()
    """
    # create some custom map from silo_outputs_map to expected inputs
    aggregation_inputs = dict(
        [
            (f"input_silo_{silo_index+1}", silo_output)
            for silo_index, silo_output in enumerate(weights)
        ]
    )

    # aggregate all silo models into one
    aggregate_weights_step = aggregate_component(**aggregation_inputs)

    return aggregate_weights_step, {
        # IMPORTANT: use a key that is consistent with kwargs of silo_training()
        "running_checkpoint": aggregate_weights_step.outputs.aggregated_output
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

pipeline_job = builder.build_basic_fl_pipeline(
    # building requires all 3 functions provided as argument below
    silo_preprocessing,
    silo_training,
    orchestrator_aggregation,
    # RESERVED: this kwarg is for building iterations
    iterations=YAML_CONFIG.training_parameters.num_of_iterations,
    # any additional custom kwarg will be sent to silo_training() as is
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
else:
    print("The pipeline was NOT submitted, use --submit to send it to AzureML.")
