"""Federated Learning Cross-Silo Vertical basic pipeline for MNIST.

This script:
1) reads a config file in yaml specifying the number of silos and their parameters,
2) reads the components from a given folder,
3) builds a flexible pipeline depending on the config,
4) configures each step of this pipeline to read/write from the right silo.
"""
import os
import json
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
    os.path.dirname(__file__), "..", "..", "components", "MNIST_VERTICAL"
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
training_contributor_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "traininsilo", "contributor_spec.yaml")
)

training_host_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "traininsilo", "host_spec.yaml")
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
    description=f'FL cross-silo basic pipeline and the unique identifier is "{pipeline_identifier}" that can help you to track files in the storage account.',
)
def fl_mnist_vertical_basic():
    ################
    ### TRAINING ###
    ################

    outputs = {}
    # for each silo, run a distinct training with its own inputs and outputs
    for silo_index, silo_config in enumerate(
        [YAML_CONFIG.federated_learning.host] + YAML_CONFIG.federated_learning.silos
    ):
        if silo_index == 0:
            # we're using training component here
            silo_training_step = training_host_component(
                train_data=Input(
                    type=silo_config.training_data.type,
                    mode=silo_config.training_data.mode,
                    path=silo_config.training_data.path,
                ),
                test_data=Input(
                    type=silo_config.testing_data.type,
                    mode=silo_config.testing_data.mode,
                    path=silo_config.testing_data.path,
                ),
                # Learning rate for local training
                lr=YAML_CONFIG.training_parameters.lr,
                # Number of epochs
                epochs=YAML_CONFIG.training_parameters.epochs,
                # Dataloader batch size
                batch_size=YAML_CONFIG.training_parameters.batch_size,
                # Silo name/identifier
                metrics_prefix=silo_config.compute,
                global_size=len(YAML_CONFIG.federated_learning.silos) + 1,
                global_rank=silo_index,
                local_size=1,
                local_rank=0,
            )
            # add a readable name to the step
            silo_training_step.name = f"host_training"
            outputs[f"host_output"] = silo_training_step.outputs.model
        else:
            # we're using training component here
            silo_training_step = training_contributor_component(
                train_data=Input(
                    type=silo_config.training_data.type,
                    mode=silo_config.training_data.mode,
                    path=silo_config.training_data.path,
                ),
                test_data=Input(
                    type=silo_config.testing_data.type,
                    mode=silo_config.testing_data.mode,
                    path=silo_config.testing_data.path,
                ),
                # Learning rate for local training
                lr=YAML_CONFIG.training_parameters.lr,
                # Number of epochs
                epochs=YAML_CONFIG.training_parameters.epochs,
                # Dataloader batch size
                batch_size=YAML_CONFIG.training_parameters.batch_size,
                # Silo name/identifier
                metrics_prefix=silo_config.compute,
                global_size=len(YAML_CONFIG.federated_learning.silos) + 1,
                global_rank=silo_index,
                local_size=1,
                local_rank=0,
            )
            # add a readable name to the step
            silo_training_step.name = f"contributor_{silo_index}_training"
            outputs[
                f"contributor_{silo_index}_output"
            ] = silo_training_step.outputs.model

        # make sure the compute corresponds to the silo
        silo_training_step.compute = silo_config.compute

        # assign instance type for AKS, if available
        if hasattr(silo_config, "instance_type"):
            if silo_training_step.resources is None:
                silo_training_step.resources = {}
            silo_training_step.resources["instance_type"] = silo_config.instance_type

        # make sure the data is written in the right datastore
        model_file_name = "host" if silo_index == 0 else f"contributor_{silo_index}"
        silo_training_step.outputs.model = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                # IMPORTANT: writing the output of training into the host datastore
                YAML_CONFIG.federated_learning.host.datastore,
                f"model/{model_file_name}",
            ),
        )

    return outputs


pipeline_job = fl_mnist_vertical_basic()

# Inspect built pipeline
print(pipeline_job)

if args.submit:
    print("Submitting the pipeline job to your AzureML workspace...")

    ML_CLIENT = connect_to_aml()
    pipeline_job = ML_CLIENT.jobs.create_or_update(
        pipeline_job, experiment_name="fl_demo_mnist_vertical"
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
