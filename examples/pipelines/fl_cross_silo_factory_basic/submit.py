"""Federated Learning Cross-Silo basic pipeline.

This script:
1) reads a config file in yaml specifying the number of silos and their parameters,
2) reads the components from a given folder,
3) builds a flexible pipeline depending on the config,
4) configures each step of this pipeline to read/write from the right silo.
"""
import os
import argparse
import random
import string
import datetime

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

############################
### CONFIGURE THE SCRIPT ###
############################

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
    "--example", required=False, choices=["MNIST"], default="MNIST", help="dataset name"
)

args = parser.parse_args()

# load the config from a local yaml file
YAML_CONFIG = OmegaConf.load(args.config)

# path to the components
COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", args.example
)

###########################
### CONNECT TO AZURE ML ###
###########################

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


####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

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


######################
### CUSTOM DS CODE ###
######################


def silo_inputs(silo_config: dataclass) -> Dict[str, Input]:
    """Create AzureML SDK v2 objects for the inputs required by silo_preprocessing().

    Args:
        silo_config (namespace): content of config.yaml file silos[i] section

    Returns:
        Dict[Input]: a map of the inputs expected as kwargs by silo_preprocessing()
    """
    return {
        # IMPORTANT: use a key that is consistent with kwargs of silo_preprocessing()
        "raw_train_data": Input(
            type=AssetTypes.URI_FILE,
            mode="mount",
            path=silo_config.training_data_path,
        ),
        "raw_test_data": Input(
            type=AssetTypes.URI_FILE,
            mode="mount",
            path=silo_config.training_data_path,
        ),
    }


def silo_preprocessing(raw_train_data: Input, raw_test_data: Input) -> Dict[str, Input]:
    """Create steps for running FL preprocessing in the silo.

    Args:
        raw_train_data (Input): preprocessed data (see silo_inputs())
        raw_test_data (Input): preprocessed data (see silo_inputs())

    Returns:
        Dict[Input]: a map of the inputs expected as kwargs by silo_training()
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
    training_parameters: dataclass,
    train_data: Input = None,
    test_data: Input = None,
    running_checkpoint: Input = None,
):
    """Create steps for running FL training in the silo.

    Args:
        training_parameters (dataclass): the section of config.yaml
        train_data (Input): preprocessed data (see silo_inputs())
        test_data (Input): preprocessed data (see silo_inputs())
        running_checkpoint (Input): if not None, the checkpoint obtained from previous iteration (see orchestrator_aggregation())

    Returns:
        Dict[Input]: a map of the inputs expected as kwargs by orchestrator_aggregation()
    """
    # we're using training component here
    silo_training_step = training_component(
        # with the train_data from the pre_processing step
        train_data=train_data,
        # with the test_data from the pre_processing step
        test_data=test_data,
        # and the checkpoint from previous round (or None if round == 1)
        checkpoint=running_checkpoint,
        # Learning rate for local training
        lr=training_parameters.lr,
        # Number of epochs
        epochs=training_parameters.epochs,
        # Dataloader batch size
        batch_size=training_parameters.batch_size,
    )

    return silo_training_step, {
        # IMPORTANT: use a key that is consistent with kwargs of orchestrator_aggregation()
        "weights": silo_training_step.outputs.model
    }


def orchestrator_aggregation(weights=[]):
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


####################
### FACTORY CODE ###
####################

# DO NOT MODIFY :)


def custom_fl_data_output(
    datastore_name, output_name, unique_id="${{name}}", round=None
):
    """Produces a path to store the data during FL training.

    Args:
        datastore_name (str): name of the Azure ML datastore
        output_name (str): a name unique to this output
        unique_id (str): a unique id for the run (default: inject run id with ${{name}})
        round (str): an round id if relevant

    Returns:
        data_path (str): direct url to the data path to store the data
    """
    data_path = f"azureml://datastores/{datastore_name}/paths/federated_learning/{output_name}/{unique_id}/"
    if round:
        data_path += f"round_{round}/"

    return Output(type=AssetTypes.URI_FOLDER, mode="mount", path=data_path)


def getUniqueIdentifier(length=8):
    """Generates a random string and concatenates it with today's date

    Args:
        length (int): length of the random string (default: 8)

    """
    str = string.ascii_lowercase
    date = datetime.date.today().strftime("%Y_%m_%d_")
    return date + "".join(random.choice(str) for i in range(length))


def anchor_step_in_silo(pipeline_step, silo_config, tags={}, description=None):
    """Takes a step and enforces the right compute/datastore config"""
    # make sure the compute corresponds to the silo
    pipeline_step.compute = silo_config.compute

    # make sure every output data is written in the right datastore
    for key in pipeline_step.outputs:
        setattr(
            pipeline_step.outputs,
            key,
            custom_fl_data_output(silo_config.datastore, key),
        )

    return pipeline_step


pipeline_identifier = getUniqueIdentifier()

from azure.ai.ml.entities._job.pipeline._io import PipelineOutputBase

@pipeline(
    description=f'FL cross-silo basic pipeline and the unique identifier is "{pipeline_identifier}" that can help you to track files in the storage account.',
)
def _fl_cross_silo_factory_pipeline():
    ######################
    ### PRE-PROCESSING ###
    ######################

    # once per silo, we're running a pre-processing step

    # map of preprocessed outputs
    silo_preprocessed_outputs = {}

    for silo_index, silo_config in enumerate(YAML_CONFIG.federated_learning.silos):
        # building the inputs as specified by developer
        preprocessing_inputs = silo_inputs(silo_config)

        # verify the outputs from the developer code
        assert isinstance(
            preprocessing_inputs, dict
        ), f"your silo_inputs() function should return a dictionary (currently returns a {type(preprocessing_inputs)}"
        for key in preprocessing_inputs.keys():
            assert isinstance(
                preprocessing_inputs[key], Input
            ), f"silo_inputs() returned dict contains a key {key} that should map to an Input class from Azure ML SDK v2 (current type is {type(preprocessing_inputs[key])})."

        # building the preprocessing as specified by developer
        preprocessing_step, preprocessing_outputs = silo_preprocessing(
            **preprocessing_inputs  # feed kwargs as produced by silo_inputs()
        )

        # TODO: verify _step is an actual step

        # verify the outputs from the developer code
        assert isinstance(
            preprocessing_outputs, dict
        ), f"your silo_preprocessing() function should return a step,outputs tuple with outputs a dictionary (currently a {type(preprocessing_outputs)})"
        for key in preprocessing_outputs.keys():
            assert isinstance(
                preprocessing_outputs[key], PipelineOutputBase
            ), f"silo_preprocessing() returned outputs has a key '{key}' not mapping to an PipelineOutputBase class from Azure ML SDK v2 (current type is {type(preprocessing_outputs[key])})."

        # make sure the compute corresponds to the silo
        # make sure the data is written in the right datastore
        anchor_step_in_silo(preprocessing_step, silo_config)

        # each output is indexed to be fed into training_component as a distinct input
        silo_preprocessed_outputs[silo_index] = preprocessing_outputs

    ################
    ### TRAINING ###
    ################

    running_outputs = {}  # for round 1, we have no pre-existing checkpoint

    # now for each round, run training
    for round in range(1, YAML_CONFIG.training_parameters.num_rounds + 1):
        # collect all outputs in a dict to be used for aggregation
        silo_training_outputs = []

        # for each silo, run a distinct training with its own inputs and outputs
        for silo_index, silo_config in enumerate(YAML_CONFIG.federated_learning.silos):
            # building the training steps as specified by developer
            training_step, training_outputs = silo_training(
                YAML_CONFIG.training_parameters,  # providing training params
                **silo_preprocessed_outputs[
                    silo_index
                ],  # feed kwargs as produced by silo_preprocessing()
                **running_outputs,  # feed optional running kwargs as produced by aggregate_component()
            )

            # TODO: verify _step is an actual step

            # verify the outputs from the developer code
            assert isinstance(
                training_outputs, dict
            ), f"your silo_training() function should return a step,outputs tuple with outputs a dictionary (currently a {type(training_outputs)})"
            for key in training_outputs.keys():
                assert isinstance(
                    training_outputs[key], PipelineOutputBase
                ), f"silo_training() returned outputs has a key '{key}' not mapping to an PipelineOutputBase class from Azure ML SDK v2 (current type is {type(training_outputs[key])})."

            # make sure the compute corresponds to the silo
            # make sure the data is written in the right datastore
            anchor_step_in_silo(training_step, silo_config)

            # each output is indexed to be fed into aggregate_component as a distinct input
            silo_training_outputs.append(training_outputs)

        # a couple of basic tests before aggregating
        # do we even have outputs?
        assert (
            len(silo_training_outputs) > 0
        ), "The list of silo outputs is empty, did you define a list of silos in config.yaml:federated_learning.silos section?"
        # do we have enough?
        assert len(silo_training_outputs) == len(
            YAML_CONFIG.federated_learning.silos
        ), "The list of silo outputs has length that doesn't match length of config.yaml:federated_learning.silos section."

        # does every output have the same keys?
        reference_keys = set(silo_training_outputs[0].keys())
        for _index, _output in enumerate(silo_training_outputs):
            if not (reference_keys == set(_output.keys())):
                raise Exception(
                    f"""The output returned by silo at index {_index} has keys {set(_output.keys())} that differ from keys of silo 0 ({reference_keys}).
                    Please make sure your silo_training() function returns a consistent set of keys for every silo."""
                )

        # pivot the outputs from index->key to key->index
        aggregation_kwargs_inputs = dict(
            [
                (
                    key,  # for every key in the outputs
                    [
                        # create a list of all silos outputs
                        _outputs[key]
                        for _outputs in silo_training_outputs
                    ],
                )
                for key in reference_keys
            ]
        )

        # aggregate all silo models into one
        aggregation_step, aggregation_outputs = orchestrator_aggregation(
            **aggregation_kwargs_inputs
        )

        # TODO: verify _step is an actual step

        # verify the outputs from the developer code
        assert isinstance(
            aggregation_outputs, dict
        ), f"your orchestrator_aggregation() function should return a (step,outputs) tuple with outputs a dictionary (current type a {type(aggregation_outputs)})"
        for key in aggregation_outputs.keys():
            assert isinstance(
                aggregation_outputs[key], PipelineOutputBase
            ), f"orchestrator_aggregation() returned outputs has a key '{key}' not mapping to an PipelineOutputBase class from Azure ML SDK v2 (current type is {type(aggregation_outputs[key])})."

        # this is done in the orchestrator compute/datastore
        anchor_step_in_silo(
            aggregation_step, YAML_CONFIG.federated_learning.orchestrator
        )

        # let's keep track of the running outputs (dict) to be used as input for next round
        running_outputs = aggregation_outputs

    return running_outputs


pipeline_job = _fl_cross_silo_factory_pipeline()

# Inspect built pipeline
print(pipeline_job)

if args.submit:
    print("Submitting the pipeline job to your AzureML workspace...")

    pipeline_job = ML_CLIENT.jobs.create_or_update(
        pipeline_job, experiment_name="fl_dev"
    )

    print("The url to see your live job running is returned by the sdk:")
    print(pipeline_job.services["Studio"].endpoint)
else:
    print("The pipeline was NOT submitted, use --submit to send it to AzureML.")
