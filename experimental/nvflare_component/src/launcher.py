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

# to handle yaml config easily
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, ManagedIdentityCredential
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
):
    if "DEFAULT_IDENTITY_CLIENT_ID" in os.environ:
        credential = ManagedIdentityCredential(
            client_id=os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
        )
    else:
        credential = DefaultAzureCredential()
    if subscription_id:
        logging.info("Workspace references provided as arguments")
        ml_client = MLClient(
            credential, subscription_id, resource_group, workspace_name
        )
    elif "AZUREML_ARM_WORKSPACE_NAME" in os.environ:
        logging.info("Workspace references found in os.environ")
        ml_client = MLClient(
            credential,
            os.environ.get("AZUREML_ARM_SUBSCRIPTION", None),
            os.environ.get("AZUREML_ARM_RESOURCEGROUP", None),
            os.environ.get("AZUREML_ARM_WORKSPACE_NAME", None),
        )
    else:
        logging.info("No workspace references provided, using local config")
        ml_client = MLClient.from_config(credential)

    return ml_client


class NVFlareLauncher:
    def __init__(self, project_config, app_folder, ml_client):
        self.logger = logging.getLogger(__name__)
        self.head = None
        self.workers = []

        self.run_id = os.environ.get("MLFLOW_RUN_ID") or str(uuid.uuid4())[:8]
        self.logger.info(f"MCD root run ID: {self.run_id}")

        self.project_config_path = project_config
        self.project_config = self._load_and_validate_config(project_config)
        self.app_folder_tmp = tempfile.TemporaryDirectory()
        self.app_folder = os.path.join(self.app_folder_tmp.name, "src")
        self.workspace_folder = tempfile.TemporaryDirectory().name
        self.prod_folder = os.path.join(self.workspace_folder, self.project_config.name, "prod_00")
        self.component_folder = tempfile.TemporaryDirectory().name
    
        shutil.copytree(app_folder, self.app_folder)

        self.shared_inputs = {}
        self.shared_outputs = {}
        self.jobs = []

        self.ml_client = ml_client

    def run_cli_command(self, cli_command: list, timeout: int = 60):
        """Runs subprocess for a cli setup command"""
        self.logger.info(f"Launching cli with command: {cli_command}")
        cli_command_call = subprocess.run(
            cli_command,
            # stdout=PIPE,
            # stderr=PIPE,
            universal_newlines=True,
            check=False,  # will not raise an exception if subprocess fails
            timeout=timeout,  # TODO: more than a minute would be weird?
            # env=custom_env
        )
        self.logger.info(f"return code: {cli_command_call.returncode}")

        if cli_command_call.returncode != 0:
            raise RuntimeError("Cli command returned code != 0")

        return cli_command_call.returncode

    def _provision(self):
        self.run_cli_command(
            cli_command=["nvflare", "provision", "-p", self.project_config_path, "-w", self.workspace_folder]
        )

    def _load_and_validate_config(self, config_path):
        """Load and validates the config file."""
        self.logger.info(
            "Loading and validating config file from {}".format(config_path)
        )
        _config = OmegaConf.load(config_path)
        self.logger.debug(f"Loaded config: {_config}")

        assert (
            "participants" in _config and _config.participants is not None
        ), "participants config not found"
        assert (
            "builders" in _config and _config.builders is not None
        ), "builders config not found"

        
        return _config

    def _build_io(self, io_config, io_class):
        """Build inputs and outputs from the config file."""
        if isinstance(io_config, DictConfig):
            # data
            io = io_class(**io_config)
        elif isinstance(io_config, str) and io_config.startswith("${{inputs."):
            io = self.shared_inputs[io_config[10:-2]]
        elif isinstance(io_config, str) and io_config.startswith("${{outputs."):
            io = self.shared_outputs[io_config[11:-2]]
        else:
            # float, str, bool, etc.
            io = io_config
        return io

    def _build_shared_io(self):
        """Build shared inputs and outputs from the config file."""
        if "inputs" in self.config and self.config.inputs is not None:
            self.logger.info(
                "Building shared inputs from config {}".format(self.config.inputs)
            )
            for input_key in self.config.inputs:
                self.shared_inputs[input_key] = self._build_io(
                    self.config.inputs[input_key], Input
                )
                self.logger.debug(
                    f"Shared input {input_key}={self.shared_inputs[input_key]}"
                )
        else:
            self.logger.info("No shared inputs found in config")

        if "outputs" in self.config and self.config.outputs is not None:
            self.logger.info(
                "Building shared outputs from config {}".format(self.config.outputs)
            )
            for output_key in self.config.outputs:
                self.shared_outputs[output_key] = self._build_io(
                    self.config.outputs[output_key], Output
                )
                self.logger.debug(
                    f"Shared output {output_key} = {self.shared_outputs[output_key]}"
                )
        else:
            self.logger.info("No shared outputs found in config")

    def _build_command(self, command_config, run_id, rank, size):
        """Build the command to run from the config file."""
        inputs = {}
        if "inputs" in command_config and command_config.inputs is not None:
            for k in command_config.inputs:
                inputs[k] = self._build_io(command_config.inputs[k], Input)
        inputs.update(self.shared_inputs)

        outputs = {}
        if "outputs" in command_config and command_config.outputs is not None:
            for k in command_config.outputs:
                outputs[k] = self._build_io(command_config.outputs[k], Output)
        outputs.update(self.shared_outputs)

        job_config = {}
        for k in command_config:
            if k not in ["inputs", "outputs"]:
                job_config[k] = command_config[k]
        job_config["inputs"] = inputs
        job_config["outputs"] = outputs

        job_config["compute"] = job_config["compute"].lstrip("azureml:/")

        if "code" in job_config:
            job_config["code"] = os.path.join(self.source_dir, job_config["code"])
            shutil.copy(
                os.path.join(os.path.dirname(__file__), "mcd_runtime", "launch.py"),
                job_config["code"],
            )
            shutil.copy(
                os.path.join(
                    os.path.dirname(__file__), "mcd_runtime", "service_bus_driver.py"
                ),
                job_config["code"],
            )

        if "command" in job_config:
            job_config["command"] = (
                f"python launch.py {run_id} {rank} {size} " + job_config["command"]
            )

        job = command(**job_config)

        return job

    def launch(self):
        self._provision()
        # self._build_shared_io()

        participants = [ p for p in self.project_config.participants if p.type in ["server", "client"]]

        for index, participant in enumerate(participants):
            self._launch_participant(participant, index, len(participants))

    def _launch_participant(self, participant, rank, size):
        participant_folder = os.path.join(self.prod_folder, participant.name)
        component_folder = os.path.join(self.component_folder, participant.name)
        shutil.copytree(participant_folder, component_folder)
        shutil.copy(os.path.join(os.path.dirname(__file__), "nvflare_host_runtime", "nvflare_host.py"), component_folder)
        shutil.copy(os.path.join(os.path.dirname(__file__), "nvflare_host_runtime", "service_bus_driver.py"), component_folder)

        if participant.type == "server":
            shutil.copytree(self.prod_folder+"/admin@nvidia.com", component_folder+"/admin@nvidia.com")
            shutil.copytree(self.app_folder, component_folder+"/app")

        participant_job = command(
            compute=participant.name,
            code=component_folder,
            command=f"python nvflare_host.py --run_id {self.run_id} --rank {rank} --size {size} --type {participant.type} --name {participant.name}",
            environment=f"nvflare-pt@latest"
        )

        participant_job.display_name = self.run_id + "_" + participant.type
        # submit the command
        self.logger.debug("Submitting job {}".format(participant_job))
        returned_job = self.ml_client.jobs.create_or_update(
            participant_job,
            experiment_name=os.environ.get("AZUREML_ARM_PROJECT_NAME", "mcd_dev"),
        )
        # get a URL for the status of the job
        print(returned_job.studio_url)
        self.jobs.append(returned_job)


    def _launch_head(self):
        self.logger.info("Launching head from config {}".format(self.config.head))
        job = self._build_command(
            self.config.head,
            run_id=self.run_id,
            rank=0,
            size=len(self.config.workers) + 1,
        )
        job.display_name = self.run_id + "_head"
        # submit the command
        self.logger.debug("Submitting job {}".format(job))
        returned_job = self.ml_client.jobs.create_or_update(
            job,
            experiment_name=os.environ.get("AZUREML_ARM_PROJECT_NAME", "mcd_dev"),
        )
        # get a URL for the status of the job
        print(returned_job.studio_url)
        self.jobs.append(returned_job)

    def _launch_workers(self):
        for worker_rank, (worker_key, worker_conf) in enumerate(
            self.config.workers.items()
        ):
            self.logger.info(f"Launching worker {worker_key} from config {worker_conf}")
            job = self._build_command(
                worker_conf,
                run_id=self.run_id,
                rank=(worker_rank + 1),
                size=len(self.config.workers) + 1,
            )
            job.display_name = self.run_id + f"_worker_{worker_key}"
            # submit the command
            self.logger.debug(f"Submitting job f{worker_key}={job}")
            returned_job = self.ml_client.jobs.create_or_update(
                job,
                experiment_name=os.environ.get("AZUREML_ARM_PROJECT_NAME", "mcd_dev"),
            )
            # get a URL for the status of the job
            print(returned_job.studio_url)
            self.jobs.append(returned_job)


def run(args: argparse.Namespace):
    """Run the component."""
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
        project_config=args.project_config, app_folder=args.app_folder, ml_client=ml_client
    )
    launcher.launch()


def main(cli_args=None):
    """Component main function.

    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)

    print(f"Running script with arguments: {args}")
    run(args)


if __name__ == "__main__":
    # Set logging to sys.out
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    main()
