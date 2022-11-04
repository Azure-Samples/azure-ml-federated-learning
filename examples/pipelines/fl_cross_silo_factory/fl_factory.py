# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script provides a class to help building Federated Learning pipelines in AzureML.

We invite you to NOT MODIFY THIS SCRIPT unless you know what you are doing, and you
are trying to achieve a particular edge case scenario.
"""
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
from azure.ai.ml.entities._job.pipeline._io import NodeOutput
from azure.ai.ml.exceptions import ValidationException


class FederatedLearningPipelineFactory:
    def __init__(self):
        """Constructor"""
        self.silos = []
        self.orchestrator = {}
        self.unique_identifier = self.getUniqueIdentifier()

        # see soft_validate()
        self.affinity_map = {}

        self.logger = logging.getLogger(__name__)

    def set_orchestrator(self, compute: str, datastore: str):
        """Set the internal configuration of the orchestrator.

        Args:
            compute (str): name of the compute target
            datastore (str): name of the datastore
        """
        self.orchestrator = {"compute": compute, "datastore": datastore}

    def add_silo(self, compute: str, datastore: str, **custom_input_args):
        """Add a silo to the internal configuration of the builder.

        Args:
            compute (str): name of the compute target
            datastore (str): name of the datastore
            **custom_input_args: any of those will be passed to the preprocessing step as-is
        """
        self.silos.append(
            {
                "compute": compute,
                "datastore": datastore,
                "custom_input_args": custom_input_args or {},
            }
        )

    def custom_fl_data_output(
        self, datastore_name, output_name, unique_id="${{name}}", iteration_num=None
    ):
        """Returns an Output pointing to a path to store the data during FL training.

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

        return Output(type=AssetTypes.URI_FOLDER, mode="mount", path=data_path)

    def getUniqueIdentifier(self, length=8):
        """Generate a random string and concatenates it with today's date

        Args:
            length (int): length of the random string (default: 8)
        """
        str = string.ascii_lowercase
        date = datetime.date.today().strftime("%Y_%m_%d_")
        return date + "".join(random.choice(str) for i in range(length))

    def anchor_step_in_silo(
        self,
        pipeline_step,
        compute,
        output_datastore,
        model_output_datastore=None,
        tags={},
        description=None,
        _path="root",
    ):
        """Take a step and enforces the right compute/datastore config.

        Args:
            pipeline_step (PipelineStep): a step to anchor
            compute (str): name of the compute target
            output_datastore (str): name of the datastore for the outputs of this step
            model_output_datastore (str): name of the datastore for the model/weights outputs of this step
            tags (dict): tags to add to the step in AzureML UI
            description (str): description of the step in AzureML UI

        Returns:
            pipeline_step (PipelineStep): the anchored step
        """
        self.logger.debug(f"{_path}: anchoring type={pipeline_step.type}")
        if pipeline_step.type == "pipeline":
            if hasattr(pipeline_step, "component"):
                # pipeline component!
                self.logger.debug(f"{_path} --  pipeline component detected")
                self.anchor_step_in_silo(
                    pipeline_step.component,
                    compute,
                    output_datastore,
                    model_output_datastore=model_output_datastore,
                    tags=tags,
                    description=description,
                    _path=f"{_path}.component",
                )

                # make sure every output data is written in the right datastore
                for key in pipeline_step.outputs:
                    self.logger.debug(f"{_path}.outputs.{key} -- type={pipeline_step.outputs[key].type} class={type(pipeline_step.outputs[key])}")
                    setattr(
                        pipeline_step.outputs,
                        key,
                        self.custom_fl_data_output(output_datastore, key),
                    )

            else:
                self.logger.debug(f"{_path}: pipeline (non-component) detected")
                for key in pipeline_step.outputs:
                    self.logger.debug(f"{_path}.outputs.{key} -- type={pipeline_step.outputs[key].type} class={type(pipeline_step.outputs[key])}")
                    pipeline_step.outputs[key] = self.custom_fl_data_output(self.orchestrator["datastore"], key, unique_id="pipelineoutput")

                for job_key in pipeline_step.jobs:
                    job = pipeline_step.jobs[job_key]
                    self.anchor_step_in_silo(
                        job,
                        compute,
                        output_datastore,
                        model_output_datastore=model_output_datastore,
                        tags=tags,
                        description=description,
                        _path=f"{_path}.jobs.{job_key}",
                    )

            return pipeline_step

        elif pipeline_step.type == "command":
            self.logger.debug(f"{_path}: command detected")
            # make sure the compute corresponds to the silo
            if pipeline_step.compute is None:
                self.logger.debug(f"{_path}: compute is None, using {compute} instead")
                pipeline_step.compute = compute

            for key in pipeline_step.outputs:
                self.logger.debug(f"{_path}.outputs.{key} -- type={pipeline_step.outputs[key].type} class={type(pipeline_step.outputs[key])}")

                if pipeline_step.outputs[key]._data is None:
                    # means intermediary output
                    self.logger.debug(f"{_path}.outputs.{key}: direct ouptut detected, forcing datastore {output_datastore}")
                    setattr(
                        pipeline_step.outputs,
                        key,
                        self.custom_fl_data_output(output_datastore, key, unique_id="commanddirectoutput"),
                    )
                else:
                    # means internal reference to parent
                    self.logger.debug(f"{_path}.outputs.{key}: reference ouptut detected, leaving as is")
                    pass

            return pipeline_step

        else:
            raise NotImplementedError(f"under path={_path}: step type={pipeline_step.type} is not supported")


    def build_flexible_fl_pipeline(
        self,
        scatter,
        gather,
        accumulator: dict,
        iterations=1,
        **constant_scatter_args
    ):
        """Build a typical FL pipeline based on the provided steps.

        Args:
            scatter (Pipeline): Silo/Training subgraph step contains components such as pre-processing, training, etc
            gather (func): aggregation step to run in the orchestrator
            accumulator (Dict[Input]): a dictionary of inputs to pass to the gather step
            iterations (int): number of iterations to run (default: 1)
        """
        # type checking
        assert isinstance(accumulator, dict), "accumulator must be an dict"
        assert "name" in accumulator, "accumulator must have a key 'name'"

        # assert isinstance(scatter, function), f"scatter must be a {scatter.__class__.__name__}"
        # assert isinstance(gather, PipelineStep), "gather must be a PipelineStep"

        # prepare keys for building
        accumulator_key = accumulator["name"]
        scatter_outputs_keys = None

        @pipeline(
            name="FL Scatter-Gather Iteration",
            description="Pipeline includes preprocessing, training and aggregation components",
        )
        def fl_scatter_gather_iteration(
            running_accumulator: Input(optional=True),
            iteration_num: int
        ):
            # collect all outputs in a list to be used for aggregation
            silo_subgraphs_outputs = []

            # for each silo, run a distinct training with its own inputs and outputs
            for silo_config in self.silos:

                scatter_arguments = {}
                # custom data inputs
                scatter_arguments.update(silo_config["custom_input_args"])

                # custom accumulator input
                scatter_arguments[accumulator_key] = running_accumulator

                # custom training args
                scatter_arguments.update(constant_scatter_args)

                # reserved scatter inputs
                scatter_arguments["iteration_num"] = iteration_num
                scatter_arguments["scatter_compute"] = silo_config["compute"]
                scatter_arguments["scatter_datastore"] = silo_config["datastore"]
                scatter_arguments["gather_datastore"] = self.orchestrator["datastore"]

                silo_subgraph_step = scatter(**scatter_arguments)

                # every step within the silo_subgraph_step needs to be
                # anchored in the silo
                self.anchor_step_in_silo(
                    silo_subgraph_step,
                    compute=silo_config["compute"],
                    output_datastore=silo_config["datastore"],
                    _path="silo_subgraph_step"
                )

                # except the outputs of the subgraph itself
                for key in silo_subgraph_step.outputs:
                    setattr(
                        silo_subgraph_step.outputs,
                        key,
                        self.custom_fl_data_output(self.orchestrator["datastore"], key),
                    )

                # each output is indexed to be fed into aggregate_component as a distinct input
                silo_subgraphs_outputs.append(silo_subgraph_step.outputs)
                scatter_outputs_keys = list(silo_subgraph_step.outputs.keys())

            # a couple of basic tests before aggregating
            # do we even have outputs?
            assert (
                len(silo_subgraphs_outputs) > 0
            ), "The list of silo outputs is empty, did you define a list of silos in config.yaml:federated_learning.silos section?"
            # do we have enough?
            assert len(silo_subgraphs_outputs) == len(
                self.silos
            ), "The list of silo outputs has length that doesn't match length of config.yaml:federated_learning.silos section."

            # prepare inputs for the gather step
            gather_inputs_list = []
            for key in scatter_outputs_keys:
                for i, silo_output in enumerate(silo_subgraphs_outputs):
                    gather_inputs_list.append(
                        ( f"{key}_{i+1}", silo_output[key] )
                    )
            gather_inputs_dict = dict(gather_inputs_list)

            # aggregate all silo models into one
            aggregation_step = gather(**gather_inputs_dict)

            # this is done in the orchestrator compute/datastore
            self.anchor_step_in_silo(
                aggregation_step,
                compute=self.orchestrator["compute"],
                output_datastore=self.orchestrator["datastore"],
                model_output_datastore=self.orchestrator["datastore"],
                _path="aggregation_step"
            )

            return {
                accumulator_key: aggregation_step.outputs[accumulator_key]
            }


        @pipeline(
            description=f'FL cross-silo factory pipeline and the unique identifier is "{self.unique_identifier}" that can help you to track files in the storage account.',
        )
        def _fl_cross_silo_factory_pipeline():

            running_accumulator = None

            # now for each iteration, run training
            # Note: The Preprocessing will be done once. For 'n-1' iterations, the cached states/outputs will be used.
            for iteration_num in range(1, iterations + 1):

                # call pipeline for each iteration
                output_iteration = fl_scatter_gather_iteration(running_accumulator, iteration_num)
                output_iteration.name = f"scatter_gather_iteration_{iteration_num}"

                # the running accumulator is on the orchestrator side
                for key in output_iteration.outputs:
                    setattr(
                        output_iteration.outputs,
                        key,
                        self.custom_fl_data_output(self.orchestrator["datastore"], key),
                    )

                # let's keep track of the checkpoint to be used as input for next iteration
                running_accumulator = output_iteration.outputs[accumulator_key]

            return {
                # the only output is the accumulator
                accumulator_key : running_accumulator
            }

        pipeline_job = _fl_cross_silo_factory_pipeline()

        # for some reason, we also need that here, making sure
        for key in pipeline_job.outputs:
            setattr(
                pipeline_job.outputs,
                key,
                self.custom_fl_data_output(self.orchestrator["datastore"], key),
            )

        return pipeline_job

    ###########################
    ### AFFINITY VALIDATION ###
    ###########################

    OPERATION_READ = "READ"
    OPERATION_WRITE = "WRITE"
    DATASTORE_UNKNOWN = "UNKNOWN_DATASTORE"
    DATATYPE_UNKNOWN = "UNKNOWN_DATATYPE"

    def set_default_affinity_map(self):
        """Build a map of affinities between computes and datastores for soft validation."""
        self.affinity_map = {}

        # orchestrator permissions
        self.set_affinity(
            self.orchestrator["compute"],
            self.orchestrator["datastore"],
            self.OPERATION_READ,
            True,
        )
        self.set_affinity(
            self.orchestrator["compute"],
            self.orchestrator["datastore"],
            self.OPERATION_WRITE,
            True,
        )

        # silo permissions
        for silo in self.silos:
            self.set_affinity(
                silo["compute"], silo["datastore"], self.OPERATION_READ, True
            )
            self.set_affinity(
                silo["compute"], silo["datastore"], self.OPERATION_WRITE, True
            )

            # it's actually ok to read from anywhere?
            self.set_affinity(
                silo["compute"], self.DATASTORE_UNKNOWN, self.OPERATION_READ, True
            )

            self.set_affinity(
                silo["compute"],
                self.orchestrator["datastore"],
                self.OPERATION_READ,
                True,
                data_type=AssetTypes.CUSTOM_MODEL,
            )  # OK to get a model our of the orchestrator
            self.set_affinity(
                silo["compute"],
                self.orchestrator["datastore"],
                self.OPERATION_WRITE,
                True,
                data_type=AssetTypes.CUSTOM_MODEL,
            )  # OK to write a model into the orchestrator

            self.set_affinity(
                self.orchestrator["compute"],
                silo["datastore"],
                self.OPERATION_READ,
                False,
            )  # NOT OK to read from silo
            self.set_affinity(
                self.orchestrator["compute"],
                silo["datastore"],
                self.OPERATION_WRITE,
                False,
            )  # NOT OK to write in silo?

        return self.affinity_map

    def set_affinity(
        self,
        compute: str,
        datastore: str,
        operation: str,
        affinity: bool,
        data_type=None,
    ) -> None:
        """Set the affinity of a given compute and datastore for this operation."""
        if operation not in [self.OPERATION_READ, self.OPERATION_WRITE]:
            raise ValueError(
                f"set_affinity() for operation {affinity} is not allowed, only READ and WRITE."
            )

        affinity_key = (
            compute,
            datastore,
            operation,
            data_type or self.DATATYPE_UNKNOWN,
        )
        self.affinity_map[affinity_key] = affinity

    def check_affinity(
        self, compute: str, datastore: str, operation: str, data_type=None
    ) -> bool:
        """Verify the affinity of a given compute and datastore for this operation."""
        # check the specific affinity as provided
        affinity_key = (
            compute,
            datastore,
            operation,
            data_type or self.DATATYPE_UNKNOWN,
        )

        if affinity_key in self.affinity_map:
            return self.affinity_map[affinity_key]

        # if we don't have a specific affinity, let's check if we have a generic one
        affinity_key = (compute, datastore, operation, self.DATATYPE_UNKNOWN)

        if affinity_key in self.affinity_map:
            return self.affinity_map[affinity_key]

        return False


    #######################
    ## WORK IN PROGRESS ###
    #######################

    def _resolve_pipeline_data(self, data_key, data_def, inputs_map={}, outputs_map={}, _path="ROOT"):
        self.logger.debug(f"{_path}: resolving data_key={data_key} type={data_def.type} class={data_def.__class__.__name__} with inputs_map={list(inputs_map.keys())} outputs_map={list(outputs_map.keys())}")
        if data_def.type in ['string', 'boolean', 'integer', 'number']:
            self.logger.debug(f"{_path}: job i/o key={data_key} is not data")
            return data_def.type, None

        if isinstance(data_def._data, Input):
            self.logger.debug(f"{_path}: job i/o key={data_key} is input")
            return data_def._data.type, data_def._data.path
        if isinstance(data_def._data, Output):
            self.logger.debug(f"{_path}: job i/o key={data_key} is output")
            return data_def._data.type, data_def._data.path


        if data_def._data is not None:
            # internal reference inside the graph
            self.logger.debug(f"{_path}: job i/o key={data_key} is an internal reference to pipeline level data name={data_def._data._name}")

            if data_def._data._data is not None:
                # to avoid infinite recursion
                return data_def._data._data.type, data_def._data._data.path

            ref_key = data_def._data._name

            if ref_key in inputs_map:
                _data_def = inputs_map[ref_key]
            elif ref_key in outputs_map:
                _data_def = outputs_map[ref_key]
            else:
                raise ValueError(f"{_path}: internal reference {ref_key} not found in inputs_map (keys={list(inputs_map.keys())}) or outputs_map (keys={list(outputs_map.keys())})")

            self.logger.debug(f"{_path}: job i/o key={data_key} resolved to data definition name={data_def._data._name}")
            return self._resolve_pipeline_data(
                data_key=data_def._data._name,
                data_def=_data_def,
                inputs_map=inputs_map,
                outputs_map=outputs_map,
                _path=f"{_path}->{data_key}",
            )
        else:
            self.logger.debug(f"{_path}: job i/o key={data_key} is pointing to None (optional input)")
            return None, None


    def _recursive_validate(self, job, _path="ROOT", inputs_map={}, outputs_map={}):
        soft_validation_report = []
        self.logger.debug(f"{_path}: recursive validation of job name={job.name} with inputs_map={list(inputs_map.keys())} outputs_map={list(outputs_map.keys())}")

        if job.type == "pipeline":
            compute = job.compute

            for key in job.inputs:
                # validate inputs somehow
                self.logger.debug(f"{_path}: job input={key} <> {compute}")
                inputs_map[key] = job.inputs[key]

            for key in job.outputs:
                # validate outputs somehow
                self.logger.debug(f"{_path}: job output={key} <> {compute}")
                outputs_map[key] = job.outputs[key]

            if hasattr(job, "component"):
                # pipeline component
                self.logger.debug(f"{_path}: pipeline component detected for job name={job.name}")

                job_level_inputs_map = inputs_map.copy()
                job_level_outputs_map = outputs_map.copy()
                for job_key in job.component.jobs:
                    for key in job.component.jobs[job_key].inputs:
                        job_level_inputs_map[key] = job.component.jobs[job_key].inputs[key]
                    for key in job.component.jobs[job_key].outputs:
                        job_level_outputs_map[key] = job.component.jobs[job_key].outputs[key]

                    soft_validation_report.extend(
                        self._recursive_validate(
                            job.component.jobs[job_key],
                            _path=f"{_path}.component.jobs.{job_key}",
                            inputs_map=job_level_inputs_map,
                            outputs_map=job_level_outputs_map,
                        ),
                    )

                return soft_validation_report
            else:
                # regular pipeline
                self.logger.debug(f"{_path}: regular pipeline detected for job name={job.name}")
                for job_key in job.jobs:
                    soft_validation_report.extend(
                        self._recursive_validate(
                            job.component.jobs[job_key],
                            _path=f"{_path}.jobs.{job_key}",
                            inputs_map=inputs_map,
                            outputs_map=outputs_map,
                        ),
                    )

                return soft_validation_report

        elif job.type == "command":
            # make sure the compute corresponds to the silo
            job_key = job.name

            if job.compute is None:
                soft_validation_report.append(f"{_path}: job name={job.name} has no compute")

            # loop on all the inputs
            for input_key in job.inputs:
                input_type, input_path = self._resolve_pipeline_data(
                    data_key=input_key,
                    data_def=job.inputs[input_key],
                    _path=f"{_path}.inputs.{input_key}",
                    inputs_map=inputs_map,
                    outputs_map=outputs_map,
                )

                if input_path is None:
                    # optional input
                    continue

                self.logger.debug(f"{_path}: job input={input_key} is pointing to to path={input_path}")

                # extract the datastore
                if input_path and input_path.startswith("azureml://datastores/"):
                    datastore = input_path[21:].split("/")[0]
                else:
                    # if using a registered dataset, let's consider datastore UNKNOWN
                    datastore = self.DATASTORE_UNKNOWN

                self.logger.debug(f"{_path}: validating job input={input_key} on datastore={datastore} against compute={job.compute}")

                # verify affinity and log errors
                if not self.check_affinity(
                    job.compute, datastore, self.OPERATION_READ, job.inputs[input_key].type
                ):
                    soft_validation_report.append(
                        f"In job {_path}, input={input_key} of type={job.inputs[input_key].type} is located on datastore={datastore} which should not have READ access by compute={job.compute}"
                    )

            # loop on all the outputs
            for output_key in job.outputs:
                output_type, output_path = self._resolve_pipeline_data(
                    data_key=output_key,
                    data_def=job.outputs[output_key],
                    _path=f"{_path}.outputs.{output_key}",
                    inputs_map=inputs_map,
                    outputs_map=outputs_map,
                )

                self.logger.debug(f"{_path}: job output={output_key} is pointing to to path={output_path}")

                # extract the datastore
                if output_path and output_path.startswith("azureml://datastores/"):
                    datastore = output_path[21:].split("/")[0]
                else:
                    soft_validation_report.append(
                        f"In job {_path}, output={output_key} does not start with azureml://datastores/ but is {output_path}"
                    )
                    continue

                # verify affinity and log errors
                if not self.check_affinity(
                    job.compute,
                    datastore,
                    self.OPERATION_WRITE,
                    job.outputs[output_key].type,
                ):
                    soft_validation_report.append(
                        f"In job {_path}, output={output_key} of type={job.outputs[output_key].type} will be saved on datastore={datastore} which should not have WRITE access by compute={job.compute}"
                    )

            return soft_validation_report

        else:
            raise NotImplementedError(f"{_path}: job name={job.name} has type={job.type} that is not supported")

     
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
            raise Exception(
                "Affinity map hasn't been built, use set_affinity() or default_affinity_map() to set."
            )

        # accumulate errors found in a list for debugging
        soft_validation_report = []

        # check if there's overlap in the configured datastores
        silo_datastore_names = set([_silo["datastore"] for _silo in self.silos])
        if len(silo_datastore_names) == 0:
            soft_validation_report.append("No silo datastores have been configured.")
        elif len(silo_datastore_names) < len(self.silos):
            soft_validation_report.append(
                "You have multiple silos using the same datastore, please fix your config."
            )
        if self.orchestrator["datastore"] in silo_datastore_names:
            soft_validation_report.append(
                "You have the orchestrator and silos using the same datastore, please fix your config."
            )

        # check if there's overlap in the configured computes
        silo_computes_names = set([_silo["compute"] for _silo in self.silos])
        if len(silo_computes_names) == 0:
            soft_validation_report.append("No silo computes have been configured.")
        elif len(silo_computes_names) < len(self.silos):
            soft_validation_report.append(
                "You have multiple silos using the same compute, please fix your config."
            )
        if self.orchestrator["compute"] in silo_computes_names:
            soft_validation_report.append(
                "You have the orchestrator and silos using the same compute, please fix your config."
            )

        # recurse through all the jobs
        soft_validation_report.extend(self._recursive_validate(pipeline_job))

        # when looping through all jobs is done
        if soft_validation_report:
            # if any error is found
            soft_validation_report.insert(
                0,
                "Soft validation could not validate pipeline job due to the following issues:\n",
            )

            soft_validation_report.append(
                "\nAccording to the affinity_map rules below:\n"
            )
            soft_validation_report.append(
                "CLUSTER\tDATASTORE\tOPERATION\tDATATYPE\tAFFINITY"
            )
            for key in self.affinity_map:
                soft_validation_report.append(
                    "{c}\t{d}\t{t}\t{o}\t{a}".format(
                        c=key[0],
                        d=key[1],
                        t=key[3],
                        o=key[2],
                        a="ALLOW" if self.affinity_map[key] else "DENY",
                    )
                )

            if raise_exception:
                raise Exception("\n".join(soft_validation_report))
            else:
                logging.critical("\n".join(soft_validation_report))
                return False

        return True
