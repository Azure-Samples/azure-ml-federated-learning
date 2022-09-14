"""Factory to build a basic FL pipeline."""
import os
import argparse
import random
import string
import datetime
import logging

# Azure ML sdk v2 imports
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

from typing import List, Optional, Union
from typing import Callable, Dict
from dataclasses import dataclass
import itertools
from azure.ai.ml.entities._job.pipeline._io import PipelineOutputBase
from azure.ai.ml._ml_exceptions import ValidationException


class FederatedLearningPipelineFactory:
    def __init__(self):
        self.silos = []
        self.orchestrator = {}
        self.unique_identifier = self.getUniqueIdentifier()

        # see soft_validate()
        self.affinity_map = {}

    def set_orchestrator(self, compute: str, datastore: str):
        self.orchestrator = {"compute": compute, "datastore": datastore}

    def add_silo(self, compute: str, datastore: str, **custom_input_args):
        self.silos.append(
            {
                "compute": compute,
                "datastore": datastore,
                "custom_input_args": custom_input_args or {},
            }
        )

    def custom_fl_data_output(
        self, datastore_name, output_name, unique_id="${{name}}", round=None
    ):
        """Produces a path to store the data during FL training.

        Args:
            datastore_name (str): name of the Azure ML datastore
            output_name (str): a name unique to this output
            unique_id (str): a unique id for the run (default: inject run id with ${{name}})
            round (str): an round id if relevant

        Returns:
            data_path (str): direct url to the data path to store the data
        """
        data_path = f"azureml://datastores/{datastore_name}/paths/federated_learning/{output_name}/{unique_id}/"
        if round:
            data_path += f"round_{round}/"

        return Output(type=AssetTypes.URI_FOLDER, mode="mount", path=data_path)

    def getUniqueIdentifier(self, length=8):
        """Generates a random string and concatenates it with today's date

        Args:
            length (int): length of the random string (default: 8)

        """
        str = string.ascii_lowercase
        date = datetime.date.today().strftime("%Y_%m_%d_")
        return date + "".join(random.choice(str) for i in range(length))

    def anchor_step_in_silo(
        self, pipeline_step, silo_config, tags={}, description=None
    ):
        """Takes a step and enforces the right compute/datastore config"""
        # make sure the compute corresponds to the silo
        pipeline_step.compute = silo_config["compute"]

        # make sure every output data is written in the right datastore
        for key in pipeline_step.outputs:
            setattr(
                pipeline_step.outputs,
                key,
                self.custom_fl_data_output(silo_config["datastore"], key),
            )

        return pipeline_step

    def build_basic_fl_pipeline(
        self,
        silo_inputs,
        silo_preprocessing,
        silo_training,
        orchestrator_aggregation,
        iterations=1,
        **training_kwargs,
    ):
        @pipeline(
            description=f'FL cross-silo basic pipeline and the unique identifier is "{self.unique_identifier}" that can help you to track files in the storage account.',
        )
        def _fl_cross_silo_factory_pipeline():
            ######################
            ### PRE-PROCESSING ###
            ######################

            # once per silo, we're running a pre-processing step

            # map of preprocessed outputs
            silo_preprocessed_outputs = {}

            for silo_index, silo_config in enumerate(self.silos):
                # building the inputs as specified by developer
                preprocessing_inputs = silo_inputs(**silo_config["custom_input_args"])

                # verify the outputs from the developer code
                assert isinstance(
                    preprocessing_inputs, dict
                ), f"your silo_inputs() function should return a dictionary (currently returns a {type(preprocessing_inputs)}"
                for key in preprocessing_inputs.keys():
                    assert isinstance(
                        preprocessing_inputs[key], Input
                    ), f"silo_inputs() returned dict contains a key {key} that should map to an Input class from Azure ML SDK v2 (current type is {type(preprocessing_inputs[key])})."

                # building the preprocessing as specified by developer
                preprocessing_step, preprocessing_outputs = silo_preprocessing(
                    **preprocessing_inputs  # feed kwargs as produced by silo_inputs()
                )

                # TODO: verify _step is an actual step

                # verify the outputs from the developer code
                assert isinstance(
                    preprocessing_outputs, dict
                ), f"your silo_preprocessing() function should return a step,outputs tuple with outputs a dictionary (currently a {type(preprocessing_outputs)})"
                for key in preprocessing_outputs.keys():
                    assert isinstance(
                        preprocessing_outputs[key], PipelineOutputBase
                    ), f"silo_preprocessing() returned outputs has a key '{key}' not mapping to an PipelineOutputBase class from Azure ML SDK v2 (current type is {type(preprocessing_outputs[key])})."

                # make sure the compute corresponds to the silo
                # make sure the data is written in the right datastore
                self.anchor_step_in_silo(preprocessing_step, silo_config)

                # each output is indexed to be fed into training_component as a distinct input
                silo_preprocessed_outputs[silo_index] = preprocessing_outputs

            ################
            ### TRAINING ###
            ################

            running_outputs = {}  # for round 1, we have no pre-existing checkpoint

            # now for each round, run training
            for round in range(1, iterations + 1):
                # collect all outputs in a dict to be used for aggregation
                silo_training_outputs = []

                # for each silo, run a distinct training with its own inputs and outputs
                for silo_index, silo_config in enumerate(self.silos):
                    # building the training steps as specified by developer
                    training_step, training_outputs = silo_training(
                        **silo_preprocessed_outputs[
                            silo_index
                        ],  # feed kwargs as produced by silo_preprocessing()
                        **running_outputs,  # feed optional running kwargs as produced by aggregate_component()
                        **training_kwargs,  # # providing training params
                    )

                    # TODO: verify _step is an actual step

                    # verify the outputs from the developer code
                    assert isinstance(
                        training_outputs, dict
                    ), f"your silo_training() function should return a step,outputs tuple with outputs a dictionary (currently a {type(training_outputs)})"
                    for key in training_outputs.keys():
                        assert isinstance(
                            training_outputs[key], PipelineOutputBase
                        ), f"silo_training() returned outputs has a key '{key}' not mapping to an PipelineOutputBase class from Azure ML SDK v2 (current type is {type(training_outputs[key])})."

                    # make sure the compute corresponds to the silo
                    # make sure the data is written in the right datastore
                    self.anchor_step_in_silo(training_step, silo_config)

                    # each output is indexed to be fed into aggregate_component as a distinct input
                    silo_training_outputs.append(training_outputs)

                # a couple of basic tests before aggregating
                # do we even have outputs?
                assert (
                    len(silo_training_outputs) > 0
                ), "The list of silo outputs is empty, did you define a list of silos in config.yaml:federated_learning.silos section?"
                # do we have enough?
                assert len(silo_training_outputs) == len(
                    self.silos
                ), "The list of silo outputs has length that doesn't match length of config.yaml:federated_learning.silos section."

                # does every output have the same keys?
                reference_keys = set(silo_training_outputs[0].keys())
                for _index, _output in enumerate(silo_training_outputs):
                    if not (reference_keys == set(_output.keys())):
                        raise Exception(
                            f"""The output returned by silo at index {_index} has keys {set(_output.keys())} that differ from keys of silo 0 ({reference_keys}).
                            Please make sure your silo_training() function returns a consistent set of keys for every silo."""
                        )

                # pivot the outputs from index->key to key->index
                aggregation_kwargs_inputs = dict(
                    [
                        (
                            key,  # for every key in the outputs
                            [
                                # create a list of all silos outputs
                                _outputs[key]
                                for _outputs in silo_training_outputs
                            ],
                        )
                        for key in reference_keys
                    ]
                )

                # aggregate all silo models into one
                aggregation_step, aggregation_outputs = orchestrator_aggregation(
                    **aggregation_kwargs_inputs
                )

                # TODO: verify _step is an actual step

                # verify the outputs from the developer code
                assert isinstance(
                    aggregation_outputs, dict
                ), f"your orchestrator_aggregation() function should return a (step,outputs) tuple with outputs a dictionary (current type a {type(aggregation_outputs)})"
                for key in aggregation_outputs.keys():
                    assert isinstance(
                        aggregation_outputs[key], PipelineOutputBase
                    ), f"orchestrator_aggregation() returned outputs has a key '{key}' not mapping to an PipelineOutputBase class from Azure ML SDK v2 (current type is {type(aggregation_outputs[key])})."

                # this is done in the orchestrator compute/datastore
                self.anchor_step_in_silo(aggregation_step, self.orchestrator)

                # let's keep track of the running outputs (dict) to be used as input for next round
                running_outputs = aggregation_outputs

            return running_outputs

        return _fl_cross_silo_factory_pipeline()

    ###########################
    ### AFFINITY VALIDATION ###
    ###########################

    OPERATION_READ = "READ"
    OPERATION_WRITE = "WRITE"
    DATASTORE_UNKNOWN = "UNKNOWN"

    def set_default_affinity_map(self):
        """Build a map of affinities between computes and datastores for soft validation."""
        self.affinity_map = {}

        # orchestrator permissions
        self.set_affinity(
            self.orchestrator["compute"], self.orchestrator["datastore"], self.OPERATION_READ, True
        )
        self.set_affinity(
            self.orchestrator["compute"], self.orchestrator["datastore"], self.OPERATION_WRITE, True
        )

        # silo permissions
        for silo in self.silos:
            self.set_affinity(silo["compute"], silo["datastore"], self.OPERATION_READ, True)
            self.set_affinity(silo["compute"], silo["datastore"], self.OPERATION_WRITE, True)

            # it's actually ok to read from anywhere?
            self.set_affinity(silo["compute"], self.DATASTORE_UNKNOWN, self.OPERATION_READ, True)

            self.set_affinity(
                silo["compute"], self.orchestrator["datastore"], self.OPERATION_READ, True
            )  # OK to get data in
            self.set_affinity(
                silo["compute"], self.orchestrator["datastore"], self.OPERATION_WRITE, False
            )  # NOT OK to exfiltrate from the silo

            self.set_affinity(
                self.orchestrator["compute"], silo["datastore"], self.OPERATION_READ, False
            )  # NOT OK to read from silo
            self.set_affinity(
                self.orchestrator["compute"], silo["datastore"], self.OPERATION_WRITE, False
            )  # NOT OK to write in silo?

        return self.affinity_map

    def set_affinity(
        self, compute: str, datastore: str, operation: str, affinity: bool
    ) -> None:
        """Set the affinity of a given compute and datastore for this operation."""
        if operation not in [self.OPERATION_READ, self.OPERATION_WRITE]:
            raise ValueError(f"set_affinity() for operation {affinity} is not allowed, only READ and WRITE.")

        affinity_key = (compute, datastore, operation)
        self.affinity_map[affinity_key] = affinity

    def check_affinity(self, compute: str, datastore: str, operation: str) -> bool:
        """Verify the affinity of a given compute and datastore for this operation."""
        affinity_key = (compute, datastore, operation)

        if affinity_key not in self.affinity_map:
            return False
        else:
            return self.affinity_map[affinity_key]

    def soft_validate(self, pipeline_job, raise_exception=True):
        """Runs a soft validation to verify computes and datastores have affinity.

        Args:
            pipeline_job (Pipeline): returned by factory methods.
            raise_exception (bool): fail hard if we do not validate.

        Returns:
            bool: result of validation
        """
        # build an affinity of compute-datastore for READ/WRITE
        if len(self.affinity_map) == 0:
            raise Exception("Affinity map hasn't been built, use set_affinity() or default_affinity_map() to set.")

        # accumulate errors found in a list for debugging
        soft_validation_report = []

        # loop on all the jobs
        for job_key in pipeline_job.jobs:
            job = pipeline_job.jobs[job_key]
            compute = job.compute

            # loop on all the inputs
            for input_key in job.inputs:
                try:
                    # get the path of this input
                    input_path = job.inputs[input_key].path
                except ValidationException:
                    continue

                # extract the datastore
                if input_path.startswith("azureml://datastores/"):
                    datastore = input_path[21:].split("/")[0]
                else:
                    # if using a registered dataset, let's consider datastore UNKNOWN
                    datastore = self.DATASTORE_UNKNOWN

                # verify affinity and log errors
                if not self.check_affinity(compute, datastore, self.OPERATION_READ):
                    soft_validation_report.append(
                        f"In job {job_key}, input={input_key} is located on datastore={datastore} which should not have READ access by compute={compute}"
                    )

            # loop on all the outputs
            for output_key in job.outputs:
                try:
                    # get the path of this output
                    output_path = job.outputs[output_key].path
                except ValidationException:
                    continue

                # extract the datastore
                if output_path.startswith("azureml://datastores/"):
                    datastore = output_path[21:].split("/")[0]
                else:
                    soft_validation_report.append(
                        f"In job {job_key}, output={output_key} does not start with azureml://datastores/"
                    )
                    continue

                # verify affinity and log errors
                if not self.check_affinity(compute, datastore, self.OPERATION_WRITE):
                    soft_validation_report.append(
                        f"In job {job_key}, output={output_key} is located on datastore={datastore} which is should have WRITE access by compute={compute}"
                    )

        # when looping through all jobs is done
        if soft_validation_report:
            # if any error is found
            soft_validation_report.insert(
                0,
                "Soft validation could not validate pipeline job due to the following issues:\n",
            )

            soft_validation_report.append("\nAccording to the affinity_map:")
            for key in self.affinity_map:
                soft_validation_report.append(f" -- {key}=>{self.affinity_map[key]}")

            if raise_exception:
                raise Exception("\n".join(soft_validation_report))
            else:
                logging.critical("\n".join(soft_validation_report))
                return False

        return True
