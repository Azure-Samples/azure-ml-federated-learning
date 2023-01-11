"""Initializes and launches an NVFlare job in AzureML."""
import logging
import time
import os
import sys
import argparse
import uuid
import shutil
import tempfile
import subprocess
import socket

# to handle yaml config easily
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from azure.ai.ml import command
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ManagedIdentityCredential,
)
from azure.ai.ml import MLClient


def get_arg_parser(parser=None):
    """Parse the command line arguments for merge using argparse.
    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser):
        an argument parser instance
    Returns:
        ArgumentParser: the argument parser instance
    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the component
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    group = parser.add_argument_group("MCD Launcher Inputs")
    group.add_argument("--project_config", type=str, required=True, help="")
    group.add_argument("--app_folder", type=str, required=True, help="")
    group.add_argument("--overseer", type=str, required=False, default=None)

    group = parser.add_argument_group("Azure ML References (for local launch)")
    group.add_argument(
        "--subscription_id",
        type=str,
        required=False,
        help="Subscription ID",
    )
    group.add_argument(
        "--resource_group",
        type=str,
        required=False,
        help="Resource group name",
    )

    group.add_argument(
        "--workspace_name",
        type=str,
        required=False,
        help="Workspace name",
    )

    return parser


def connect_to_aml(
    subscription_id: str = None, resource_group: str = None, workspace_name: str = None
) -> MLClient:
    """Get MLClient using the adapted auth / environment.

    Args:
        subscription_id (str, optional): Azure subscription ID. Defaults to None.
        resource_group (str, optional): Azure resource group name. Defaults to None.
        workspace_name (str, optional): AzureML workspace name. Defaults to None.

    Returns:
        MLClient: MLClient instance
    """
    if "DEFAULT_IDENTITY_CLIENT_ID" in os.environ:
        # if we're running this within an AzureML job (as intended)
        credential = ManagedIdentityCredential(
            client_id=os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
        )
    else:
        # if not, use the default
        credential = DefaultAzureCredential()

    if subscription_id:
        # if we provide the references manually (local use)
        logging.info("Workspace references provided as arguments")
        ml_client = MLClient(
            credential, subscription_id, resource_group, workspace_name
        )
    elif "AZUREML_ARM_WORKSPACE_NAME" in os.environ:
        # if not, let's try to get them from inside the AzureML job using env vars
        logging.info("Workspace references found in os.environ")
        ml_client = MLClient(
            credential,
            os.environ.get("AZUREML_ARM_SUBSCRIPTION", None),
            os.environ.get("AZUREML_ARM_RESOURCEGROUP", None),
            os.environ.get("AZUREML_ARM_WORKSPACE_NAME", None),
        )
    else:
        # else, try local config
        logging.info("No workspace references provided, using local config")
        ml_client = MLClient.from_config(credential)

    return ml_client


class NVFlareLauncher:
    """Launcher for NVFlare jobs in AzureML."""
    RUNTIME_LOGS_PATH = "outputs/"
    def __init__(self, project_config, app_folder, ml_client, overseer=None):
        """Initialize the launcher.

        Args:
            project_config (str): Path to the project config file
            app_folder (str): Path to the app folder
            ml_client (MLClient): MLClient instance
        """
        self.logger = logging.getLogger(__name__)
        self.ml_client = ml_client

        # run id is either the MLFLOW run id or a random string (for local use)
        self.run_id = os.environ.get("MLFLOW_RUN_ID") or str(uuid.uuid4())[:8]
        self.logger.info(f"MCD root run ID: {self.run_id}")

        # saving all jobs created
        self.jobs = []

        # let's save the path to the config file (project.yml)
        self.project_config_path = project_config

        # will contain the config into a DictConfig object
        self.project_config = None

        # storing some important paths
        self.app_folder = app_folder  # os.path.join(self.app_folder_tmp.name, "src")

        # we'll create a workspace folder for provisioning
        self.workspace_folder = tempfile.TemporaryDirectory().name
        os.makedirs(self.workspace_folder, exist_ok=True)
        self.admin_name = "admin@azure.ml"

        # we'll create a folder to host the components we generate
        self.component_folder = tempfile.TemporaryDirectory().name
        os.makedirs(self.component_folder, exist_ok=True)

        if overseer is None:
            self.overseer = None
        elif overseer == "self":
            _local_hostname = socket.gethostname()
            self.overseer = "self"
            self.overseer_ip = socket.gethostbyname(_local_hostname)
            self.logger.warning(f"Overseer will be local ip: {self.overseer}")
        else:
            self.overseer = overseer
            self.overseer_ip = overseer

    def run_cli_command(self, cli_command: list, timeout: int = None) -> int:
        """Runs subprocess for a cli command"""
        self.logger.debug(f"Launching cli with command: {cli_command}")
        cli_command_call = subprocess.run(
            cli_command,
            # stdout=PIPE,
            # stderr=PIPE,
            universal_newlines=True,
            check=False,  # will not raise an exception if subprocess fails
            timeout=timeout,  # TODO: more than a minute would be weird?
            # env=custom_env
        )
        self.logger.info(f"Command returned code: {cli_command_call.returncode}")

        if cli_command_call.returncode != 0:
            raise RuntimeError("Cli command returned code != 0")

        return cli_command_call.returncode

    def load_and_validate_config(self, config_path):
        """Load and validates the config file."""
        self.logger.info(
            "Loading and validating config file from {}".format(config_path)
        )
        _config = OmegaConf.load(config_path)
        self.logger.debug(f"Loaded config: {_config}")

        errors = []

        if "azureml" not in _config or _config.azureml is None:
            errors.append("please add a section 'azureml' in your project config to specify which environment to use")

        if "participants" not in _config or _config.participants is None:
            errors.append("please add a section 'participants' in your project config to specify which participants to use")
        if "builders" not in _config or _config.builders is None:
            errors.append("please add a section 'builders' in your project config to specify which builders to use")

        for participant in _config.participants:
            if "name" not in participant or participant.name is None:
                errors.append(f"participant {participant} has no name")
            if "type" not in participant or participant.type is None:
                errors.append(f"participant {participant} has no type")
            if "type" in participant and participant.type == "admin":
                errors.append("when using this launching component, please do NOT include an admin participant in your project config.")
            if "azureml_compute" not in participant or participant.azureml_compute is None:
                errors.append(f"each participant needs an azureml_compute config, could not find it in participant={participant}")
    
        if errors:
            raise ValueError("Validating the NVFlare provisioning config file {} has led to critical errors:\n-- {}".format(
                config_path, "\n-- ".join(errors)
            ))
        else:
            return _config

    def launch(self):
        """Takes care of everything"""
        # load the project config
        self.project_config = self.load_and_validate_config(self.project_config_path)

        # add an admin participant
        self.project_config.participants.append(
            {
                "name": self.admin_name,
                "type": "admin",
                "org": "azureml",
                "role": "project_admin",
            }
        )
        # add overseer if requested
        if self.overseer:
            self.project_config.participants.append(
                {"name": "overseer", "type": "overseer", "org": "azureml"}
            )
            raise NotImplementedError("overseer support is not implemented yet")

        # save as a new config
        new_config_file = os.path.join(self.workspace_folder, "project.yml")
        OmegaConf.save(config=self.project_config, f=new_config_file)
        self.project_config_path = new_config_file

        # use project config to provision the workspace in the workspace folder
        try:
            self.logger.info("Running NVFlare provisioning...")
            self.run_cli_command(
                cli_command=[
                    "nvflare",
                    "provision",
                    "-p",
                    self.project_config_path,
                    "-w",
                    self.workspace_folder,
                ],
                timeout=60,  # this is supposed to take less than a minute
            )
        except RuntimeError as e:
            err_msg = "Running the nvflare provision command failed, see user_logs/std_log.txt logs to debug."
            raise RuntimeError(err_msg)

        @dsl.pipeline(
            description="NVFlare/AzureML orchestration pipeline",
        )
        def nvflare_pipeline():
            # get list of server/client participants
            participants = [
                p
                for p in self.project_config.participants
                if p.type in ["server", "client"]
            ]

            # launch a job for each of them in their respective computes
            for index, participant in enumerate(participants):
                self.logger.info("Launching participant {}".format(participant))
                component = self.build_participant_component(
                    participant=participant, rank=index, size=len(participants)
                )

                step = component()
                step.compute = participant.azureml_compute
                step.name = "participant_{}_{}".format(participant.type, index)
    
        pipeline_job = nvflare_pipeline()
        pipeline_job.display_name = f"NVFlare pipeline ({self.project_config.name})"
        # submit the command
        self.logger.info("Submitting NVFlare graph... {}".format(pipeline_job))
        returned_job = self.ml_client.jobs.create_or_update(
            pipeline_job,
            experiment_name=os.environ.get("AZUREML_ARM_PROJECT_NAME", "mcd_dev"),
        )
        # get a URL for the status of the job
        self.logger.info("Access job at : {}".format(returned_job.studio_url))
        self.jobs.append(returned_job)


    def build_participant_component(self, participant, rank, size):
        """Launch an AzureML job for a given participant.

        Args:
            participant (dict): participant config
            rank (int): rank of the participant
            size (int): total number of participants
        
        Returns:
            None
        """
        # locate the "prod" folder in the workspace folder
        prod_folder = os.path.join(
            self.workspace_folder, self.project_config.name, "prod_00"
        )
        # locate the participant folder
        participant_folder = os.path.join(prod_folder, participant.name)

        # create a new component folder for this participant
        component_folder = os.path.join(self.component_folder, participant.name)

        # copy whatever is in the workspace folder in the component of the participant
        shutil.copytree(participant_folder, component_folder)

        # also copy the necessary python files (runtime)
        shutil.copy(
            os.path.join(
                os.path.dirname(__file__), "nvflare_host_runtime", "nvflare_host.py"
            ),
            component_folder,
        )
        shutil.copy(
            os.path.join(
                os.path.dirname(__file__),
                "nvflare_host_runtime",
                "service_bus_driver.py",
            ),
            component_folder,
        )

        if participant.type == "server":
            # copy the admin folder in the server job snapshot
            shutil.copytree(
                os.path.join(prod_folder, self.admin_name),
                os.path.join(component_folder, self.admin_name),
            )
            # copy the app folder in the server job snapshot
            shutil.copytree(self.app_folder, os.path.join(component_folder, self.admin_name, "app"))

        # create a command to launch the participant
        command_line = [
            "python",
            "nvflare_host.py",
            "--run_id", self.run_id,
            "--rank", str(rank),
            "--size", str(size),
            "--type", participant.type,
            "--name", participant.name,
        ]
        if self.overseer:
            command.append("--overseer")
            command.append(self.overseer_ip)

        # create the AzureML job for it
        participant_job = command(
            # compute=participant.azureml_compute,
            code=component_folder,
            command=" ".join(command_line),
            environment=self.project_config.azureml.environment.lstrip("azureml:"),
            # name="nvflare_{}_{}_{}".format(self.project_config.name, participant.name.replace, participant.type),
            display_name=f"NVFlare project={self.project_config.name} name={participant.name} type={participant.type} r={rank}/s={size}"
        )

        return participant_job.component


def main(cli_args=None):
    """Component main function.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)
    print(f"Running script with arguments: {args}")

    # Set logging to sys.out
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    # verify the project config file exists
    if not os.path.exists(args.project_config):
        raise Exception(f"Project file {args.project_config} does not exist")
    if not os.path.isfile(args.project_config):
        raise Exception(f"Project file {args.project_config} is not a file")

    # verify the app directory exists
    if not os.path.exists(args.app_folder):
        raise Exception(f"App folder {args.app_folder} does not exist")
    if not os.path.isdir(args.app_folder):
        raise Exception(f"App folder {args.app_folder} is not a directory")

    ml_client = connect_to_aml(
        args.subscription_id, args.resource_group, args.workspace_name
    )

    # Parse the config file and launch sequence
    launcher = NVFlareLauncher(
        project_config=args.project_config,
        app_folder=args.app_folder,
        ml_client=ml_client,
        overseer=args.overseer,
    )

    # Launch the process
    launcher.launch()


if __name__ == "__main__":
    main()
