"""FLWR"""
import os
import argparse
import random
import string
import datetime
import webbrowser
import time
import sys
import uuid

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
    "--offline",
    default=False,
    action="store_true",
    help="Sets flag to not submit the experiment to AzureML",
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
    os.path.dirname(__file__), "..", "..", "components", "FLWR"
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


#############################################
### GET ML_CLIENT AND COMPUTE INFORMATION ###
#############################################

if not args.offline:
    ML_CLIENT = connect_to_aml()

####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
server_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "server", "spec.yaml")
)
client_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "client", "spec.yaml")
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


@pipeline(
    description=f"Flower experimental FL pipeline using AzureML",
)
def fl_pipeline_flwr_basic():
    ## SCATTER ##

    fed_identifier = str(uuid.uuid4())

    # for each silo, run a distinct client with its own inputs and outputs
    for silo_index, silo_config in enumerate(YAML_CONFIG.federated_learning.silos):
        # create a client component for the silo
        silo_client_step = client_component(
            federation_identifier=fed_identifier,
            client_data=Input(
                type=silo_config.silo_data.type,
                mode=silo_config.silo_data.mode,
                path=silo_config.silo_data.path,
            ),
            # Learning rate for local training
            lr=YAML_CONFIG.training_parameters.lr,
            # Number of epochs
            epochs=YAML_CONFIG.training_parameters.epochs,
            # Silo name/identifier
            metrics_prefix=silo_config.name,
        )
        # add a readable name to the step
        silo_client_step.name = f"silo_{silo_index}_client"

        # make sure it runs in the silo itself
        silo_client_step.compute = silo_config.computes[0]

    ## GATHER ##

    # create a server component as a "gather" step
    server_step = server_component(
        federation_identifier=fed_identifier,
        # and a number of clients to wait before job submission
        expected_clients=len(YAML_CONFIG.federated_learning.silos),
    )
    # this server job will run in the orchestrator compute
    server_step.compute = YAML_CONFIG.federated_learning.orchestrator.compute

    # make sure the data is written in the right datastore
    server_step.outputs.job_artefacts = Output(
        type=AssetTypes.URI_FOLDER,
        mode="mount",
        path=custom_fl_data_path(
            YAML_CONFIG.federated_learning.orchestrator.datastore, "job_artefacts"
        ),
    )

    # no return value yet
    return {}


# build the pipeline
pipeline_job = fl_pipeline_flwr_basic()

# Inspect built pipeline
print(pipeline_job)

if not args.offline:
    print("Submitting the pipeline job to your AzureML workspace...")
    pipeline_job = ML_CLIENT.jobs.create_or_update(
        pipeline_job, experiment_name="fl_pipeline_flwr_basic"
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
    print("The pipeline was NOT submitted, omit --offline to send it to AzureML.")
