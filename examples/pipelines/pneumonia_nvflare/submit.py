"""NVFLARE experimental"""
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
    "--project_config",
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "project.yaml"),
    help="path to an NVFlare project.yaml file",
)
parser.add_argument(
    "--nvflare_app",
    type=str,
    required=True,
    help="path to an NVFlare application folder",
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
PROJECT_CONFIG = OmegaConf.load(args.project_config)

# use a hash to check if there are changes in config (or else, reuse provisioning)
with open(args.project_config, "r") as f:
    PROJECT_CONFIG_HASH = hash(f.read())

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
    description=f"EXPERIMENTAL",
)
def fl_pneumonia_nvflare():
    server_config = [
        participant
        for participant in PROJECT_CONFIG.participants
        if participant.type == "server"
    ][0]
    admin_config = [
        participant
        for participant in PROJECT_CONFIG.participants
        if participant.type == "admin"
    ][0]
    client_configs = [
        participant
        for participant in PROJECT_CONFIG.participants
        if participant.type == "client"
    ]

    # run a provisioning component
    provision_step = provision_component(
        # with the config file
        project_config=Input(type=AssetTypes.URI_FILE, path=args.project_config)
    )
    provision_step.compute = server_config.azureml.compute
    nvflare_workspace_datapath = custom_fl_data_path(
        server_config.azureml.datastore,
        "nvflare_workspace",
        unique_id=PROJECT_CONFIG_HASH, # reuse previous provision run if config is unchanged
    )
    provision_step.outputs.workspace = Output(
        type=AssetTypes.URI_FOLDER, mode="mount", path=nvflare_workspace_datapath
    )
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

        silo_client_step = client_component(
            client_config=Input(
                type=AssetTypes.URI_FOLDER,
                path=nvflare_workspace_datapath
                + f"{PROJECT_CONFIG.name}/prod_00/{client_config.name}/",
            ),
            client_data=client_preprocessed_data,
            client_data_env_var="CLIENT_DATA_PATH",
            start=provision_step.outputs.start,
        )
        # add a readable name to the step
        silo_client_step.name = f"silo_client_{client_index}"

        # make sure the compute corresponds to the silo
        silo_client_step.compute = client_config.azureml.compute

    ## GATHER ##

    server_step = server_component(
        server_config=Input(
            path=nvflare_workspace_datapath
            + f"{PROJECT_CONFIG.name}/prod_00/{server_config.name}/",
            type=AssetTypes.URI_FOLDER,
        ),
        admin_config=Input(
            path=nvflare_workspace_datapath
            + f"{PROJECT_CONFIG.name}/prod_00/{admin_config.name}/",
            type=AssetTypes.URI_FOLDER,
        ),
        app_dir=Input(type=AssetTypes.URI_FOLDER, path=args.nvflare_app),
        start=provision_step.outputs.start,
        server_name=server_config.name,
        expected_clients=len(client_configs),
    )
    # this is done in the orchestrator compute
    server_step.compute = server_config.azureml.compute
    server_step.outputs.job_artefacts = Output(
        type=AssetTypes.URI_FOLDER, path=nvflare_workspace_datapath + "job_artefacts/"
    )

    # no return value yet
    return {}


pipeline_job = fl_pneumonia_nvflare()

# Inspect built pipeline
print(pipeline_job)

if args.submit:
    print("Submitting the pipeline job to your AzureML workspace...")
    ML_CLIENT = connect_to_aml()
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
    print("The pipeline was NOT submitted, use --submit to send it to AzureML.")
