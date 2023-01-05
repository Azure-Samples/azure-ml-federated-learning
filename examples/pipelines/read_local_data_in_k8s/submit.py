"""Basic pipeline demonstrating how to read local data from a k8s cluster. 

This script:
1) reads a config file in yaml specifying the silo and parameters,
2) loads the 'readlocaldata' component from a given folder,
3) builds a pipeline depending on the config,
4) configures the pipeline to read/write from the right silo.
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

####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
readlocaldata_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "readlocaldata", "spec.yaml")
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
def read_local_data_basic():
    ######################
    ### PRE-PROCESSING ###
    ######################

    # we'll run just one component in one silo
    silo_config = YAML_CONFIG.federated_learning.silos[0]
    read_local_data_step = readlocaldata_component(
        local_data_path=silo_config.local_data_path
    )

    # add a readable name to the step
    read_local_data_step.name = "silo_read_local_data_and_preprocess"

    # make sure the compute corresponds to the silo
    read_local_data_step.compute = silo_config.compute

    # make sure the data are written to the right datastore
    read_local_data_step.outputs.preprocessed_local_data = Output(
        type=AssetTypes.URI_FOLDER,
        mode="mount",
        path=custom_fl_data_path(
            silo_config.datastore, "preprocessed_data", unique_id=pipeline_identifier
        ),
    )

    # let's keep track of the checkpoint to be used as input for next iteration
    preprocessed_output = read_local_data_step.outputs.preprocessed_local_data

    return {"preprocessed_output": preprocessed_output}


pipeline_job = read_local_data_basic()

# Inspect built pipeline
print(pipeline_job)

if args.submit:
    print("Submitting the pipeline job to your AzureML workspace...")

    pipeline_job = ML_CLIENT.jobs.create_or_update(
        pipeline_job, experiment_name="test_onprem_input"
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
