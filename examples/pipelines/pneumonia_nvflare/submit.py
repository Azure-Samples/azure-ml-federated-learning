"""NVFlare FL Cross-Silo pipeline for pneumonia detection on chest xrays.

This script:
1) reads an NVFlare project config file in yaml specifying the number of silos and their parameters,
2) loads some NVFlare provision/server/client components from a local folder,
3) builds an AzureML pipeline depending on the config,
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
import hashlib

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
    "--project_config",
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "project.yaml"),
    help="path to an NVFlare project.yaml file",
)
parser.add_argument(
    "--nvflare_app",
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "pneumonia_federated"),
    help="path to an NVFlare application folder",
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
PROJECT_CONFIG = OmegaConf.load(args.project_config)

# use a hash to check if there are changes in config (or else, reuse provisioning)
with open(args.project_config, "r", encoding="utf-8") as f:
    PROJECT_CONFIG_HASH = hashlib.md5(f.read().encode("utf-8")).hexdigest()

# path to the components
COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", "NVFLARE"
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
            subscription_id=args.subscription_id or PROJECT_CONFIG.aml.subscription_id,
            resource_group_name=args.resource_group
            or PROJECT_CONFIG.aml.resource_group_name,
            workspace_name=args.workspace_name or PROJECT_CONFIG.aml.workspace_name,
            credential=credential,
        )

    except Exception as ex:
        print("Could not find either cli args or config.yaml.")
        # tries to connect using local config.json
        ML_CLIENT = MLClient.from_config(credential=credential)

    return ML_CLIENT


if not args.offline:
    ML_CLIENT = connect_to_aml()

####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
provision_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "provision", "spec.yaml")
)
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
    description=f"NVFlare experimental FL pipeline using AzureML",
)
def fl_pneumonia_nvflare():
    # get the server config from project yaml
    server_config = [
        participant
        for participant in PROJECT_CONFIG.participants
        if participant.type == "server"
    ][0]
    # get all client configs from project yaml
    client_configs = [
        participant
        for participant in PROJECT_CONFIG.participants
        if participant.type == "client"
    ]
    # get the admin config from project yaml
    admin_config = [
        participant
        for participant in PROJECT_CONFIG.participants
        if participant.type == "admin"
    ][0]

    # run a provisioning component
    provision_step = provision_component(
        # with the config file
        project_config=Input(type=AssetTypes.URI_FILE, path=args.project_config)
    )
    # run it in the orchestrator
    provision_step.compute = server_config.azureml.compute

    # set a specific path to produce the NVFlare workspace config folder
    nvflare_workspace_datapath = custom_fl_data_path(
        server_config.azureml.datastore,  # store it on the orchestrator
        "nvflare_workspace",
        # below will ensure we reuse previous provision job run if config is unchanged
        unique_id=PROJECT_CONFIG_HASH,
    )
    provision_step.outputs.workspace = Output(
        type=AssetTypes.URI_FOLDER, mode="mount", path=nvflare_workspace_datapath
    )
    # use a dummy "start" output to synchronize start of server/client
    provision_step.outputs.start = Output(
        type=AssetTypes.URI_FOLDER,
        mode="mount",
        path=nvflare_workspace_datapath + "/start.txt",
    )

    ## SCATTER ##

    # for each silo, run a distinct client with its own inputs and outputs
    for client_index, client_config in enumerate(client_configs):
        # some preprocessing CAN happen here :)
        # in this demo we're just using a previously created dataset
        client_preprocessed_data = Input(
            type=AssetTypes.URI_FOLDER, mode="mount", path=client_config.azureml.data
        )

        # create a client component for the silo
        silo_client_step = client_component(
            # it will be given this client's workspace config folder
            client_config=Input(
                type=AssetTypes.URI_FOLDER,
                path=nvflare_workspace_datapath
                + f"{PROJECT_CONFIG.name}/prod_00/{client_config.name}/",
            ),
            # some input data from blob
            client_data=client_preprocessed_data,
            # passed as an env variable for NVFlare training code to consume
            client_data_env_var="CLIENT_DATA_PATH",
            # use the start signal
            start=provision_step.outputs.start,
        )
        # add a readable name to the step
        silo_client_step.name = f"silo_client_{client_index}"

        # make sure it runs in the silo itself
        silo_client_step.compute = client_config.azureml.compute

    ## GATHER ##

    # create a server component as a "gather" step
    server_step = server_component(
        # it will be given the server workspace config folder
        server_config=Input(
            path=nvflare_workspace_datapath
            + f"{PROJECT_CONFIG.name}/prod_00/{server_config.name}/",
            type=AssetTypes.URI_FOLDER,
        ),
        # ... and the application folder (training code)
        app_dir=Input(type=AssetTypes.URI_FOLDER, path=args.nvflare_app),
        # ... and the admin folder (so that it can submit the job to itself)
        admin_config=Input(
            path=nvflare_workspace_datapath
            + f"{PROJECT_CONFIG.name}/prod_00/{admin_config.name}/",
            type=AssetTypes.URI_FOLDER,
        ),
        # will start after provisioning is complete
        start=provision_step.outputs.start,
        # server is given a hostname for /etc/hosts
        server_name=server_config.name,
        # and a number of clients to wait before job submission
        expected_clients=len(client_configs),
    )
    # this server job will run in the orchestrator compute
    server_step.compute = server_config.azureml.compute
    # and its outputs are stored in the orchestrator datastore
    server_step.outputs.job_artefacts = Output(
        type=AssetTypes.URI_FOLDER,
        path=nvflare_workspace_datapath + "runs/${{name}}/job_artefacts/",
    )

    # no return value yet
    return {}


# build the pipeline
pipeline_job = fl_pneumonia_nvflare()

# Inspect built pipeline
print(pipeline_job)

if not args.offline:
    print("Submitting the pipeline job to your AzureML workspace...")
    pipeline_job = ML_CLIENT.jobs.create_or_update(
        pipeline_job, experiment_name="fl_demo_pneumonia_nvflare"
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
