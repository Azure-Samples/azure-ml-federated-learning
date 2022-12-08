"""Federated Learning Cross-Silo basic pipeline for pneumonia detection on chest xrays.

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
import webbrowser
import time
import sys

# Azure ML sdk v2 imports
import azure
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# to handle yaml config easily
from omegaconf import OmegaConf


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

# path to the components
COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", "PNEUMONIA"
)

# path to the shared components
SHARED_COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", "utils"
)

###########################
### CONNECT TO AZURE ML ###
###########################


def connect_to_aml():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential does not work
        credential = InteractiveBrowserCredential()

    # Get a handle to workspace
    try:
        # tries to connect using cli args if provided else using config.yaml
        ML_CLIENT = MLClient(
            subscription_id=args.subscription_id or YAML_CONFIG.aml.subscription_id,
            resource_group_name=args.resource_group
            or YAML_CONFIG.aml.resource_group_name,
            workspace_name=args.workspace_name or YAML_CONFIG.aml.workspace_name,
            credential=credential,
        )

    except Exception as ex:
        print("Could not find either cli args or config.yaml.")
        # tries to connect using local config.json
        ML_CLIENT = MLClient.from_config(credential=credential)

    return ML_CLIENT


####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
training_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "traininsilo", "spec.yaml")
)

aggregate_component = load_component(
    source=os.path.join(SHARED_COMPONENTS_FOLDER, "aggregatemodelweights", "spec.yaml")
)


########################
### BUILD A PIPELINE ###
########################


def custom_fl_data_path(
    datastore_name, output_name, unique_id="${{name}}", iteration_num=None
):
    """Produces a path to store the data during FL training.

    Args:
        datastore_name (str): name of the Azure ML datastore
        output_name (str): a name unique to this output
        unique_id (str): a unique id for the run (default: inject run id with ${{name}})
        iteration_num (str): an iteration number if relevant

    Returns:
        data_path (str): direct url to the data path to store the data
    """
    data_path = f"azureml://datastores/{datastore_name}/paths/federated_learning/{output_name}/{unique_id}/"
    if iteration_num:
        data_path += f"iteration_{iteration_num}/"

    return data_path


def getUniqueIdentifier(length=8):
    """Generates a random string and concatenates it with today's date

    Args:
        length (int): length of the random string (default: 8)

    """
    str = string.ascii_lowercase
    date = datetime.date.today().strftime("%Y_%m_%d_")
    return date + "".join(random.choice(str) for i in range(length))


pipeline_identifier = getUniqueIdentifier()


@pipeline(
    description=f'FL cross-silo basic pipeline for pneumonia detection. The unique identifier is "{pipeline_identifier}" that can help you to track files in the storage account.',
)
def fl_pneumonia_basic():
    ################
    ### TRAINING ###
    ################

    running_checkpoint = None  # for iteration 1, we have no pre-existing checkpoint

    # now for each iteration, run training
    for iteration in range(1, YAML_CONFIG.training_parameters.num_of_iterations + 1):
        # collect all outputs in a dict to be used for aggregation
        silo_weights_outputs = {}

        # for each silo, run a distinct training with its own inputs and outputs
        for silo_index, silo_config in enumerate(YAML_CONFIG.federated_learning.silos):
            # we're using training component here
            silo_training_step = training_component(
                # with the train_data from the pre_processing step
                dataset_name=Input(
                    type=silo_config.silo_data.type,
                    mode=silo_config.silo_data.mode,
                    path=silo_config.silo_data.path,
                ),
                # and the checkpoint from previous iteration (or None if iteration == 1)
                checkpoint=running_checkpoint,
                # Learning rate for local training
                lr=YAML_CONFIG.training_parameters.lr,
                # Number of epochs
                epochs=YAML_CONFIG.training_parameters.epochs,
                # Silo name/identifier
                metrics_prefix=silo_config.compute,
                # Iteration number
                iteration_num=iteration,
            )
            # add a readable name to the step
            silo_training_step.name = f"silo_{silo_index}_training"

            # make sure the compute corresponds to the silo
            silo_training_step.compute = silo_config.compute

            # make sure the data is written in the right datastore
            silo_training_step.outputs.model = Output(
                type=AssetTypes.URI_FOLDER,
                mode="mount",
                path=custom_fl_data_path(
                    # IMPORTANT: writing the output of training into the orchestrator datastore
                    YAML_CONFIG.federated_learning.orchestrator.datastore,
                    f"model/silo{silo_index}",
                    iteration_num=iteration,
                ),
            )

            # each output is indexed to be fed into aggregate_component as a distinct input
            silo_weights_outputs[
                f"input_silo_{silo_index+1}"
            ] = silo_training_step.outputs.model

        # aggregate all silo models into one
        aggregate_weights_step = aggregate_component(**silo_weights_outputs)
        # this is done in the orchestrator compute
        aggregate_weights_step.compute = (
            YAML_CONFIG.federated_learning.orchestrator.compute
        )
        # add a readable name to the step
        aggregate_weights_step.name = f"iteration_{iteration}_aggregation"

        # make sure the data is written in the right datastore
        aggregate_weights_step.outputs.aggregated_output = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                YAML_CONFIG.federated_learning.orchestrator.datastore,
                "aggregated_output",
                unique_id=pipeline_identifier,
                iteration_num=iteration,
            ),
        )

        # let's keep track of the checkpoint to be used as input for next iteration
        running_checkpoint = aggregate_weights_step.outputs.aggregated_output

    return {"final_aggregated_model": running_checkpoint}


pipeline_job = fl_pneumonia_basic()

# Inspect built pipeline
print(pipeline_job)

if args.submit:
    print("Submitting the pipeline job to your AzureML workspace...")
    ML_CLIENT = connect_to_aml()
    pipeline_job = ML_CLIENT.jobs.create_or_update(
        pipeline_job, experiment_name="fl_demo_pneumonia"
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
            try:
                pipeline_job = ML_CLIENT.jobs.get(name=job_name)
            except azure.identity._exceptions.CredentialUnavailableError as e:
                print(f"Token expired or Credentials unavailable: {e}")
                sys.exit(5)
            status = pipeline_job.status

        print(f"Job finished with status {status}")
        if status in ["Failed", "Canceled"]:
            sys.exit(1)
else:
    print("The pipeline was NOT submitted, use --submit to send it to AzureML.")
