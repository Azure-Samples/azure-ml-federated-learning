"""Federated Learning Cross-Silo basic pipeline.

This script:
1) reads a config file in yaml specifying the number of silos and their parameters,
2) reads the components from a given folder,
3) builds a flexible pipeline depending on the config,
4) configures each step of this pipeline to read/write from the right silo.
"""
import os
import sys
import uuid
import argparse

# Azure ML sdk v2 imports
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# to handle yaml config easily
from omegaconf import OmegaConf


############################
### CONFIGURE THE SCRIPT ###
############################

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--config", type=str, required=False, default=os.path.join(os.path.dirname(__file__), "config.yaml"), help="path to a config yaml file")
parser.add_argument("--submit", default=False, action='store_true', help="actually submits the experiment to AzureML")

args = parser.parse_args()

# load the config from a local yaml file
YAML_CONFIG = OmegaConf.load(args.config)

# create a unique id for a folder on our datastore
# TODO: one major issue here is that we'll never use caching
UNIQUE_FOLDER_ID = str(uuid.uuid4())

# path to the components
COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components"
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
    logging.info(
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
training_component = load_component(
    path=os.path.join(COMPONENTS_FOLDER, "traininsilo", "traininsilo.yaml")
)
preprocessing_component = load_component(
    path=os.path.join(COMPONENTS_FOLDER, "preprocessing", "preprocessing.yaml")
)
aggregate_component = load_component(
    path=os.path.join(
        COMPONENTS_FOLDER, "aggregatemodelweights", "aggregatemodelweights.yaml"
    )
)


########################
### BUILD A PIPELINE ###
########################


def custom_fl_data_path(datastore_name, unique_id, data_name, epoch=None):
    """This method produces a path to store the data during FL training"""
    base_path = f"azureml://datastores/{datastore_name}/paths/federated_learning/{unique_id}/{data_name}/"
    if epoch:
        base_path += f"epoch_{epoch}/"

    return base_path


@pipeline(
    description="FL cross-silo basic pipeline",
)
def fl_cross_silo_internal_basic():
    ######################
    ### PRE-PROCESSING ###
    ######################

    # once per silo, we're running a pre-processing step

    silo_preprocessed_train_data = (
        []
    )  # list of preprocessed train datasets for each silo
    silo_preprocessed_test_data = []  # list of preprocessed test datasets for each silo

    for silo_index, silo_config in enumerate(YAML_CONFIG.federated_learning.silos):
        # run the pre-processing component once
        silo_pre_processing_step = preprocessing_component(
            raw_data=Input(
                type=AssetTypes.URI_FOLDER,
                mode="mount",
                path=silo_config.training_data_path,
            )
        )
        # make sure the compute corresponds to the silo
        silo_pre_processing_step.compute = silo_config.compute

        # make sure the data is written in the right datastore
        silo_pre_processing_step.outputs.processed_train_data = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                silo_config.datastore, UNIQUE_FOLDER_ID, "train_data"
            ),
        )
        silo_pre_processing_step.outputs.processed_test_data = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                silo_config.datastore, UNIQUE_FOLDER_ID, "test_data"
            ),
        )

        # store a handle to the train data for this silo
        silo_preprocessed_train_data.append(
            silo_pre_processing_step.outputs.processed_train_data
        )
        # store a handle to the test data for this silo
        silo_preprocessed_test_data.append(
            silo_pre_processing_step.outputs.processed_test_data
        )

    ################
    ### TRAINING ###
    ################

    running_checkpoint = None  # for epoch 0, we have no pre-existing checkpoint

    # now for each epoch, run training
    for epoch in range(YAML_CONFIG.training_parameters.epochs):
        # collect all outputs in a dict to be used for aggregation
        silo_weights_outputs = {}

        # for each silo, run a distinct training with its own inputs and outputs
        for silo_index, silo_config in enumerate(YAML_CONFIG.federated_learning.silos):
            # we're using training component here
            silo_training_step = training_component(
                # with the train_data from the pre_processing step
                train_data=silo_preprocessed_train_data[silo_index],
                # with the test_data from the pre_processing step
                test_data=silo_preprocessed_test_data[silo_index],
                # and the checkpoint from previous epoch (or None if epoch == 0)
                checkpoint=running_checkpoint,
            )

            # make sure the compute corresponds to the silo
            silo_training_step.compute = silo_config.compute

            # make sure the data is written in the right datastore
            silo_training_step.outputs.model = Output(
                type=AssetTypes.URI_FOLDER,
                mode="mount",
                path=custom_fl_data_path(
                    silo_config.datastore, UNIQUE_FOLDER_ID, "silo_model", epoch=epoch
                ),
            )

            # each output is indexed to be fed into aggregate_component as a distinct input
            silo_weights_outputs[
                f"model_silo_{silo_index+1}"
            ] = silo_training_step.outputs.model

        # aggregate all silo models into one
        aggregate_weights_step = aggregate_component(**silo_weights_outputs)
        # this is done in the orchestrator compute
        aggregate_weights_step.compute = (
            YAML_CONFIG.federated_learning.orchestrator.compute
        )

        # make sure the data is written in the right datastore
        aggregate_weights_step.outputs.aggregated_model = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                YAML_CONFIG.federated_learning.orchestrator.datastore,
                UNIQUE_FOLDER_ID,
                "aggregated_model",
                epoch=epoch,
            ),
        )

        # let's keep track of the checkpoint to be used as input for next epoch
        running_checkpoint = aggregate_weights_step.outputs.aggregated_model

    # NOTE: a pipeline returns a dictionary of outputs
    # keys will code for the pipeline output identifier
    return {}


pipeline_job = fl_cross_silo_internal_basic()

# Inspect built pipeline
print(pipeline_job)

if args.submit:
    print("Submitting the pipeline job to your AzureML workspace...")

    pipeline_job = ML_CLIENT.jobs.create_or_update(pipeline_job, experiment_name="fl_dev")

    print("The url to see your live job running is returned by the sdk:")
    print(pipeline_job.services["Studio"].endpoint)
else:
    print("The pipeline was NOT submitted, use --submit to send it to AzureML.")
