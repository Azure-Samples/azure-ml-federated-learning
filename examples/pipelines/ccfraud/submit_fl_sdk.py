"""Federated Learning Cross-Silo basic pipeline for Credit Card Fraud example using PrPr FL SDK.

This script:
1) reads a config file in yaml specifying the number of silos and their parameters,
2) reads the components from a given folder,
3) builds a flexible pipeline depending on the config,
4) configures each step of this pipeline to read/write from the right silo.
"""
import os
import argparse
import webbrowser
import time
import sys

# IMPORTANT: Set environment variable to enable private features before importing azure-ml
os.environ["AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED"] = "True"

# Azure ML sdk v2 imports
import azure
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# to handle yaml config easily
from omegaconf import OmegaConf

# Azure ML PrPr FL SDK imports
from azure.ai.ml.entities._assets.federated_learning_silo import FederatedLearningSilo
import azure.ai.ml.dsl._fl_scatter_gather_node as fl

############################
### CONFIGURE THE SCRIPT ###
############################

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--config",
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "config_fl_sdk.yaml"),
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
    os.path.dirname(__file__), "..", "..", "components", "CCFRAUD"
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


#############################################
### GET ML_CLIENT AND COMPUTE INFORMATION ###
#############################################

if not args.offline:
    ML_CLIENT = connect_to_aml()
    COMPUTE_SIZES = ML_CLIENT.compute.list_sizes()


def get_gpus_count(compute_name):
    if not args.offline:
        ws_compute = ML_CLIENT.compute.get(compute_name)
        if hasattr(ws_compute, "size"):
            silo_compute_size_name = ws_compute.size
            silo_compute_info = next(
                (
                    x
                    for x in COMPUTE_SIZES
                    if x.name.lower() == silo_compute_size_name.lower()
                ),
                None,
            )
            if silo_compute_info is not None and silo_compute_info.gpus >= 1:
                return silo_compute_info.gpus
    return 1


####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
preprocessing_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "preprocessing", "spec.yaml")
)

training_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "traininsilo", "spec.yaml")
)

aggregate_component = load_component(
    source=os.path.join(SHARED_COMPONENTS_FOLDER, "aggregatemodelweights_mltable", "spec.yaml")
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

# Prepare the list of silos
silo_list = [
    FederatedLearningSilo(
        compute=silo_config["computes"][0],
        datastore=silo_config["datastore"],
        inputs= {
            "silo_name": silo_config["name"],
            "raw_train_data": Input(**silo_config["training_data"]),
            "raw_test_data": Input(**silo_config["training_data"]),
            
        },
    )
    for silo_config in YAML_CONFIG.federated_learning.silos
]

# Prepare the args for the pipeline
silo_to_aggregation_argument_map = {"model" : "from_silo_input"}
aggregation_to_silo_argument_map = {"aggregated_output" : "checkpoint"}

# Prepare kwargs for the pipeline
silo_kwargs = dict(YAML_CONFIG.training_parameters)
agg_kwargs = {}

@pipeline(
    name="CCFRAUD Silo FL Subgraph",
    description="It includes all steps that needs to be executing in silo",
)
def silo_scatter_subgraph(
    # user defined inputs
    raw_train_data: Input,
    raw_test_data: Input,
    checkpoint: Input(optional=True),
    silo_name: str,
    # user defined training arguments
    model_name: str = 'SimpleLinear',
    lr: float = 0.01,
    epochs: int = 3,
    batch_size: int = 64,
    dp: bool = False,
    dp_target_epsilon: float = 50.0,
    dp_target_delta: float = 1e-5,
    dp_max_grad_norm: float = 1.0,
) -> dict:
    """Create scatter/silo subgraph.

    Args:
        raw_train_data (Input): raw train data
        raw_test_data (Input): raw test data
        checkpoint (Input): if not None, the checkpoint obtained from previous iteration
        scatter_compute1 (str): Silo compute1 name
        scatter_compute2 (str): Silo compute2 name
        iteration_num (int): Iteration number
        lr (float, optional): Learning rate. Defaults to 0.01.
        epochs (int, optional): Number of epochs. Defaults to 3.
        batch_size (int, optional): Batch size. Defaults to 64.
        dp (bool, optional): Differential Privacy
        dp_target_epsilon (float, optional): DP target epsilon
        dp_target_delta (float, optional): DP target delta
        dp_max_grad_norm (float, optional): DP max gradient norm
        num_of_iterations (int, optional): Total number of iterations

    Returns:
        Dict[str, Outputs]: a map of the outputs
    """
    
    ######################
    ### PRE-PROCESSING ###
    ######################

    silo_pre_processing_step = preprocessing_component(
        raw_training_data=raw_train_data,
        raw_testing_data=raw_test_data,
        metrics_prefix=silo_name,
    )
    # if confidentiality is enabled, add the keyvault and key name as environment variables
    if hasattr(YAML_CONFIG, "confidentiality"):
        silo_pre_processing_step.environment_variables = {
            "CONFIDENTIALITY_DISABLE": str(not YAML_CONFIG.confidentiality.enable),
            "CONFIDENTIALITY_KEYVAULT": YAML_CONFIG.confidentiality.keyvault,
            "CONFIDENTIALITY_KEY_NAME": YAML_CONFIG.confidentiality.key_name,
        }
    else:
        silo_pre_processing_step.environment_variables = {
            "CONFIDENTIALITY_DISABLE": "True",
        }

    # we're using our own training component
    silo_training_step = training_component(
        # with the train_data from the pre_processing step
        train_data=silo_pre_processing_step.outputs.processed_train_data,
        # with the test_data from the pre_processing step
        test_data=silo_pre_processing_step.outputs.processed_test_data,
        # and the checkpoint from previous iteration (or None if iteration == 1)
        checkpoint=checkpoint,
        # Learning rate for local training
        lr=lr,
        # Number of epochs
        epochs=epochs,
        # Dataloader batch size
        batch_size=batch_size,
        # Differential Privacy
        dp=dp,
        model_name=model_name,
        # DP target epsilon
        dp_target_epsilon=dp_target_epsilon,
        # DP target delta
        dp_target_delta=dp_target_delta,
        # DP max gradient norm
        dp_max_grad_norm=dp_max_grad_norm,
        # Silo name/identifier
        metrics_prefix=silo_name,
    )
    # if confidentiality is enabled, add the keyvault and key name as environment variables
    if hasattr(YAML_CONFIG, "confidentiality"):
        silo_training_step.environment_variables = {
            "CONFIDENTIALITY_DISABLE": str(
                not YAML_CONFIG.confidentiality.enable
            ),
            "CONFIDENTIALITY_KEYVAULT": YAML_CONFIG.confidentiality.keyvault,
            "CONFIDENTIALITY_KEY_NAME": YAML_CONFIG.confidentiality.key_name,
        }
    else:
        silo_training_step.environment_variables = {
            "CONFIDENTIALITY_DISABLE": "True",
        }

    # IMPORTANT: we will assume that any output provided here can be exfiltrated into the orchestrator/gather
    return {
        # NOTE: the key you use is custom
        # a map function scatter_to_gather_map needs to be provided
        # to map the name here to the expected input from gather
        "model": silo_training_step.outputs.model
    }


fl_node = fl.fl_scatter_gather(
    silo_configs=silo_list,
    silo_component=silo_scatter_subgraph,
    aggregation_component=aggregate_component,
    aggregation_compute=YAML_CONFIG.federated_learning.orchestrator.compute,
    aggregation_datastore=YAML_CONFIG.federated_learning.orchestrator.datastore,
    shared_silo_kwargs=silo_kwargs,
    aggregation_kwargs=agg_kwargs,
    silo_to_aggregation_argument_map=silo_to_aggregation_argument_map,
    aggregation_to_silo_argument_map=aggregation_to_silo_argument_map,
    max_iterations=YAML_CONFIG.federated_learning.num_of_iterations, 
)

# Inspect built pipeline
# print(fl_node.scatter_gather_graph)

if not args.offline:
    print("Submitting the pipeline job to your AzureML workspace...")
    pipeline_job = ML_CLIENT.jobs.create_or_update(
        fl_node.scatter_gather_graph, experiment_name="fl_sdk_ccfraud"
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
