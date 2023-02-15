# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script provides an EXPERIMENTAL class to help building Federated Learning pipelines in AzureML.

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
from azure.ai.ml.entities._job.pipeline._io import NodeOutput, PipelineInput
from azure.ai.ml.exceptions import ValidationException

logger = logging.getLogger(__name__)


class FLValidationEngine:
    def __init__(self, scatter_config, gather_config):
        """Constructor"""
        self.silos = []
        self.orchestrator = {}

        self.set_orchestrator(gather_config)
        self.set_silos(scatter_config)

        # see soft_validate()
        self.affinity_map = {}
        self.set_default_affinity_map()

    def set_orchestrator(self, gather_config):
        """Set the internal configuration of the orchestrator.

        Args:
            gather_config (dict): Contains config such as compute, datastore, etc.
        """
        self.orchestrator = gather_config

    def set_silos(self, scatter_config):
        """Set the internal configuration of the silos.

        Args:
            scatter_config (dict): Contains the config for each silo.
        """
        for silo_config in scatter_config:
            self.add_silo(silo_config)

    def add_silo(self, silo_config):
        """Add a silo to the internal configuration.

        Args:
            silo_config (dict): dict of silo's config that includes compute, datastore, inputs, etc.
        """
        self.silos.append(silo_config)

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
            for silo_compute in silo["computes"]:
                self.set_affinity(
                    silo_compute, silo["datastore"], self.OPERATION_READ, True
                )
                self.set_affinity(
                    silo_compute, silo["datastore"], self.OPERATION_WRITE, True
                )

                # it's actually ok to read from anywhere?
                self.set_affinity(
                    silo_compute, self.DATASTORE_UNKNOWN, self.OPERATION_READ, True
                )

                self.set_affinity(
                    silo_compute,
                    self.orchestrator["datastore"],
                    self.OPERATION_READ,
                    True,
                    data_type=AssetTypes.URI_FOLDER,
                )  # OK to get a model our of the orchestrator
                self.set_affinity(
                    silo_compute,
                    self.orchestrator["datastore"],
                    self.OPERATION_WRITE,
                    True,
                    data_type=AssetTypes.URI_FOLDER,
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

    def _resolve_pipeline_data_path(
        self, data_key, data_def, inputs_map={}, outputs_map={}, _path="ROOT"
    ):
        """Recursively resolves a given data input/output definition within the pipeline graph.

        Args:
            data_key (str): the key of the data input/output
            data_def (Any): the definition of the data input/output (ex: Input)
            inputs_map (dict): the map of inputs in the context of the current job
            outputs_map (dict): the map of outputs in the context of the current job
            _path (str): the path of the current job in the graph

        Returns:
            str: the type of the data
            str: the resolved data input/output path or value
        """
        logger.debug(
            f"{_path}: resolving data_key={data_key} type={data_def.type} class={data_def.__class__.__name__} with inputs_map={list(inputs_map.keys())} outputs_map={list(outputs_map.keys())}"
        )

        # if data is not data at all, just return it
        if data_def.type in ["string", "boolean", "integer", "number"]:
            logger.debug(f"{_path}: job i/o key={data_key} is not data")
            return data_def.type, None

        # if data is an actual input/output, just return the path directly
        if isinstance(data_def._data, Input):
            logger.debug(f"{_path}: job i/o key={data_key} is input")
            return data_def._data.type, data_def._data.path
        if isinstance(data_def._data, Output):
            logger.debug(f"{_path}: job i/o key={data_key} is output")
            return data_def._data.type, data_def._data.path

        if data_def._data is not None:
            # if data is an internal reference inside the graph
            logger.debug(
                f"{_path}: job i/o key={data_key} is an internal reference to parent level data name={data_def._data._name}"
            )

            if data_def._data._data is not None:
                # if that reference is a direct link with a path, return it
                if "path" in data_def._data._data.__dict__:
                    logger.debug(
                        f"{_path}: job i/o key={data_key} is a direct link to path={data_def._data._data.path}"
                    )
                    return data_def._data._data.type, data_def._data._data.path

                # if not, it is a reference to a parent level data that we'll look up in the context
                ref_key = data_def._data._data._name
            else:
                # if data is None, we need to look it up in the context of the job

                ref_key = data_def._data._name

            # we try as an input or output first
            if ref_key in inputs_map:
                _data_def = inputs_map[ref_key]
            elif ref_key in outputs_map:
                _data_def = outputs_map[ref_key]
            else:
                # if not found, we need to except
                raise ValueError(
                    f"{_path}: internal reference {ref_key} not found in inputs_map (keys={list(inputs_map.keys())}) or outputs_map (keys={list(outputs_map.keys())})"
                )

            # when we find the actual reference
            logger.debug(
                f"{_path}: job i/o key={data_key} resolved to data definition name={data_def._data._name}"
            )

            if ref_key == data_key and id(_data_def) == id(data_def):
                return None, None
            # we recursively resolve it (in case there's multiple levels of references)
            return self._resolve_pipeline_data_path(
                data_key=data_def._data._name,
                data_def=_data_def,
                inputs_map=inputs_map,
                outputs_map=outputs_map,
                _path=f"{_path}->{data_key}",
            )
        else:
            # if data ref is None, it's likely an optional input
            logger.debug(
                f"{_path}: job i/o key={data_key} is pointing to None (optional input)"
            )
            return None, None

    def _recursive_validate(self, job, inputs_map={}, outputs_map={}, _path="ROOT"):
        """Recursively soft-validate a job against the affinity map.

        Args:
            job (Job): the job to validate
            inputs_map (dict): the map of inputs in the context of the current job
            outputs_map (dict): the map of outputs in the context of the current job
            _path (str): the path of the current job in the graph

        Returns:
            report (List[str]): the list of errors found
        """
        # let's create a report (hopefully empty by the end of the process)
        soft_validation_report = []

        # debug logs
        logger.debug(
            f"{_path}: recursive validation of job name={job.name} with inputs_map={list(inputs_map.keys())} outputs_map={list(outputs_map.keys())}"
        )

        if job.type == "pipeline":
            # if the job is a pipeline, we need to validate the pipeline itself

            # let's first record the inputs into the context of the job
            for key in job.inputs:
                # TODO: validate inputs somehow?
                logger.debug(f"{_path}: job input={key}")
                inputs_map[key] = job.inputs[key]

            for key in job.outputs:
                # TODO: validate outputs somehow?
                logger.debug(f"{_path}: job output={key}")
                outputs_map[key] = job.outputs[key]

            if hasattr(job, "component"):
                # if the job is a pipeline component
                logger.debug(
                    f"{_path}: pipeline component detected for job name={job.name}"
                )

                # let's prepare to recurse into the component
                # create a context for inputs/outputs
                job_level_inputs_map = inputs_map.copy()
                job_level_outputs_map = outputs_map.copy()

                # loop over the component jobs
                for job_key in job.component.jobs:
                    # context is cumulative to allow for a job N+1 to reference outputs of job N
                    for key in job.component.jobs[job_key].inputs:
                        job_level_inputs_map[key] = job.component.jobs[job_key].inputs[
                            key
                        ]
                    for key in job.component.jobs[job_key].outputs:
                        job_level_outputs_map[key] = job.component.jobs[
                            job_key
                        ].outputs[key]

                    # recurse into the job
                    soft_validation_report.extend(
                        self._recursive_validate(
                            job.component.jobs[job_key],
                            inputs_map=job_level_inputs_map,
                            outputs_map=job_level_outputs_map,
                            _path=f"{_path}.component.jobs.{job_key}",  # to help with debug logging
                        ),
                    )

                # return the report
                return soft_validation_report
            else:
                # the job is a regular pipeline (root?)
                logger.debug(
                    f"{_path}: regular pipeline detected for job name={job.name}"
                )
                # let's recurse into the pipeline jobs
                for job_key in job.jobs:
                    soft_validation_report.extend(
                        self._recursive_validate(
                            job.component.jobs[job_key],
                            inputs_map=inputs_map,
                            outputs_map=outputs_map,
                            _path=f"{_path}.jobs.{job_key}",  # to help with debug logging
                        ),
                    )

                # return the report
                return soft_validation_report

        elif job.type == "command":
            # if the job is an actual command, we need to validate the command itself

            # validate that the compute is anchored (unspecified compute is not accepted)
            job_compute = (
                job.compute._data
                if isinstance(job.compute, PipelineInput)
                else job.compute
            )
            if job.compute is None:
                soft_validation_report.append(
                    f"{_path}: job name={job.name} has no compute"
                )

            # loop on all the inputs
            for input_key in job.inputs:
                # resolve the input path by recursing through the references
                input_type, input_path = self._resolve_pipeline_data_path(
                    data_key=input_key,
                    data_def=job.inputs[input_key],
                    inputs_map=inputs_map,
                    outputs_map=outputs_map,
                    _path=f"{_path}.inputs.{input_key}",  # to help with debug logging
                )

                if input_path is None:
                    # assuming this is an optional input here
                    logger.debug("{_path}: optional input detected, passing")
                    continue

                logger.debug(
                    f"{_path}: job input={input_key} is pointing to to path={input_path}"
                )

                # extract the datastore from the input path
                if input_path and input_path.startswith("azureml://datastores/"):
                    datastore = input_path[21:].split("/")[0]
                else:
                    # if using a registered dataset, let's consider datastore UNKNOWN
                    datastore = self.DATASTORE_UNKNOWN

                logger.debug(
                    f"{_path}: validating job input={input_key} on datastore={datastore} against compute={job_compute}"
                )

                # verify affinity and log errors if they occur
                if not self.check_affinity(
                    job_compute,
                    datastore,
                    self.OPERATION_READ,
                    job.inputs[input_key].type,
                ):
                    soft_validation_report.append(
                        f"In job {_path}, input={input_key} of type={job.inputs[input_key].type} is located on datastore={datastore} which should not have READ access by compute={job_compute}"
                    )

            # loop on all the outputs
            for output_key in job.outputs:
                job_compute = (
                    job.compute._data
                    if isinstance(job.compute, PipelineInput)
                    else job.compute
                )
                # resolve the output path by recursing through the references
                output_type, output_path = self._resolve_pipeline_data_path(
                    data_key=output_key,
                    data_def=job.outputs[output_key],
                    inputs_map=inputs_map,
                    outputs_map=outputs_map,
                    _path=f"{_path}.outputs.{output_key}",  # to help with debug logging
                )

                logger.debug(
                    f"{_path}: job output={output_key} is pointing to to path={output_path}"
                )

                # extract the datastore
                if output_path and output_path.startswith("azureml://datastores/"):
                    datastore = output_path[21:].split("/")[0]
                else:
                    continue

                # verify affinity and log errors if they occur
                if not self.check_affinity(
                    job_compute,
                    datastore,
                    self.OPERATION_WRITE,
                    job.outputs[output_key].type,
                ):
                    soft_validation_report.append(
                        f"In job {_path}, output={output_key} of type={job.outputs[output_key].type} will be saved on datastore={datastore} which should not have WRITE access by compute={job_compute}"
                    )

            # return the report
            return soft_validation_report

        else:
            raise NotImplementedError(
                f"{_path}: job name={job.name} has type={job.type} that is not supported"
            )

    def soft_validate(self, pipeline_job, raise_exception=True):
        """Runs a soft validation to verify computes and datastores have affinity.

        Args:
            pipeline_job (Pipeline): returned by the scatter_gather function.
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
        silo_computes_names = set(
            [silo_compute for _silo in self.silos for silo_compute in _silo["computes"]]
        )
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

        # recurse through all the jobs to identify errors
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


def custom_fl_data_output(
    datastore_name, output_name, unique_id="${{name}}", iteration_num=None
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


def getUniqueIdentifier(length=8):
    """Generate a random string and concatenates it with today's date

    Args:
        length (int): length of the random string (default: 8)
    """
    str = string.ascii_lowercase
    date = datetime.date.today().strftime("%Y_%m_%d_")
    return date + "".join(random.choice(str) for i in range(length))


def anchor_step_in_silo(
    pipeline_step,
    compute,
    output_datastore,
    tags={},
    description=None,
    orchestrator_datastore=None,
    _path="root",
):
    """Take a step and recursively enforces the right compute/datastore config.

    Args:
        pipeline_step (PipelineStep): a step to anchor
        compute (str): name of the compute target
        output_datastore (str): name of the datastore for the outputs of this step
        tags (dict): tags to add to the step in AzureML UI
        description (str): description of the step in AzureML UI
        orchestrator_datastore (str): name of the orchestrator datastore
        _path (str): for recursive anchoring, codes the "path" inside the pipeline

    Returns:
        pipeline_step (PipelineStep): the anchored step
    """
    logger.debug(f"{_path}: anchoring node of type={pipeline_step.type}")

    if pipeline_step.type == "pipeline":  # if the current step is a pipeline
        if hasattr(pipeline_step, "component"):
            # current step is a pipeline component
            logger.debug(f"{_path} --  pipeline component detected")

            # then anchor the component inside the current step
            anchor_step_in_silo(
                pipeline_step.component,
                compute,
                output_datastore,
                tags=tags,
                description=description,
                orchestrator_datastore=orchestrator_datastore,
                _path=f"{_path}.component",  # pass the path for the debug logs
            )

            # and make sure every output data is anchored to the right datastore
            for key in pipeline_step.outputs:
                logger.debug(
                    f"{_path}.outputs.{key}: has type={pipeline_step.outputs[key].type} class={type(pipeline_step.outputs[key])}, anchoring to datastore={output_datastore}"
                )
                setattr(
                    pipeline_step.outputs,
                    key,
                    custom_fl_data_output(output_datastore, key),
                )

        else:
            # current step is a (regular) pipeline (likely the root of the graph)
            logger.debug(f"{_path}: pipeline (regular) detected")

            # let's anchor each outputs of the pipeline to the right datastore
            for key in pipeline_step.outputs:
                logger.debug(
                    f"{_path}.outputs.{key}: has type={pipeline_step.outputs[key].type} class={type(pipeline_step.outputs[key])}, anchoring to datastore={output_datastore}"
                )
                pipeline_step.outputs[key] = custom_fl_data_output(
                    orchestrator_datastore, key
                )

            # then recursively anchor each job inside the pipeline
            for job_key in pipeline_step.jobs:
                job = pipeline_step.jobs[job_key]
                anchor_step_in_silo(
                    job,
                    compute,
                    output_datastore,
                    tags=tags,
                    description=description,
                    orchestrator_datastore=orchestrator_datastore,
                    _path=f"{_path}.jobs.{job_key}",  # pass the path for the debug logs
                )

        # return the anchored pipeline
        return pipeline_step

    elif pipeline_step.type == "command":
        # if the current step is a command
        logger.debug(f"{_path}: command detected")

        # make sure the compute corresponds to the silo
        if pipeline_step.compute is None:
            logger.debug(f"{_path}: compute is None, forcing compute={compute} instead")
            pipeline_step.compute = compute

        # then anchor each of the job's outputs to the right datastore
        for key in pipeline_step.outputs:
            logger.debug(
                f"{_path}.outputs.{key}: has type={pipeline_step.outputs[key].type} class={type(pipeline_step.outputs[key])}, anchoring to datastore={output_datastore}"
            )

            if pipeline_step.outputs[key]._data is None:
                # if the output is an intermediary output
                logger.debug(
                    f"{_path}.outputs.{key}: intermediary output detected, forcing datastore {output_datastore}"
                )
                setattr(
                    pipeline_step.outputs,
                    key,
                    custom_fl_data_output(output_datastore, key),
                )
            else:
                # if the output is an internal reference to a parent output
                # let's trust that the parent has been anchored properly
                logger.debug(
                    f"{_path}.outputs.{key}: reference ouptut detected, leaving as is"
                )

        # return the anchored pipeline
        return pipeline_step

    else:
        raise NotImplementedError(
            f"under path={_path}: step type={pipeline_step.type} is not supported"
        )


def scatter_gather(
    scatter,
    gather,
    scatter_strategy,
    gather_strategy,
    scatter_to_gather_map: Callable,
    gather_to_scatter_map: Callable,
    iterations=1,
    scatter_constant_inputs=None,
):
    """Build a typical FL pipeline based on the provided steps.

    Args:
        scatter (Pipeline): Silo/Training subgraph step contains components such as pre-processing, training, etc
        gather (Pipeline): aggregation step to run in the orchestrator
        scatter_strategy(Dict): Scatter/Silo config details including computes, datastore, etc
        gather_strategy(Dict): Gather/orchestrator config details including compute, datastore, etc
        scatter_to_gather_map (Callable): function to map the outputs of the scatter step to the inputs of the gather step
        gather_to_scatter_map (Callable): function to map the outputs of the gather step to the scatter input
        iterations (int): number of iterations to run (default: 1)
        scatter_constant_inputs (Dict[Any]): any constant custom arguments passed to every scatter step (ex: training params)
    """

    # type checking
    assert callable(scatter_to_gather_map), "scatter_to_gather_map must be a function"
    assert callable(gather_to_scatter_map), "gather_to_scatter_map must be a function"
    assert isinstance(iterations, int), "iterations must be an int"
    assert iterations > 0, "iterations must be > 0"

    @pipeline(
        name="FL Scatter-Gather Iteration",
        description="Pipeline includes preprocessing, training and aggregation components",
    )
    def scatter_gather_iteration(iteration_input: Input(optional=True), **kwargs):
        """Pipeline for a single iteration of the scatter-gather graph.

        Args:
            iteration_input (Input): accumulator input to the iteration (ex: checkpoint)
        """
        # collect all outputs in a list to be used for aggregation
        scatter_subgraphs_outputs = []

        # for each silo, run a distinct training with its own inputs and outputs
        for silo_index, silo_config in enumerate(scatter_strategy):
            scatter_arguments = {}
            # custom scatter data inputs
            scatter_arguments.update(silo_config["inputs"])

            # initial checkpoint
            scatter_arguments["checkpoint"] = iteration_input

            # custom training args
            scatter_arguments.update(**scatter_constant_inputs)

            # reserved scatter inputs
            scatter_arguments["iteration_num"] = kwargs.get("iteration_num", 1)
            scatter_arguments["silo_compute1"] = silo_config["computes"][0]
            scatter_arguments["silo_compute2"] = (
                silo_config["computes"][1]
                if len(silo_config["computes"]) > 1
                else silo_config["computes"][0]
            )

            silo_subgraph_step = scatter(**scatter_arguments)
            silo_subgraph_step.name = f"silo_subgraph_{silo_index}"

            for silo_compute in silo_config["computes"]:
                # every step within the silo_subgraph_step needs to be anchored in the SILO
                anchor_step_in_silo(
                    silo_subgraph_step,
                    compute=silo_compute,
                    output_datastore=silo_config["datastore"],
                    orchestrator_datastore=gather_strategy["datastore"],
                    _path="silo_subgraph_step",  # to help with debug logging
                )

            # BUT the outputs of the scatter() subgraph/component
            # are exfiltrated to the orchestrator instead

            for key in silo_subgraph_step.outputs:
                setattr(
                    silo_subgraph_step.outputs,
                    key,
                    custom_fl_data_output(gather_strategy["datastore"], key),
                )

            # each output is added to a list to be fed into gather() as a distinct inputs
            scatter_subgraphs_outputs.append(silo_subgraph_step.outputs)

        # a couple of basic tests before gathering
        # do we even have outputs?
        assert (
            len(scatter_subgraphs_outputs) > 0
        ), "The list of scattered outputs is empty."
        # do we have enough?
        assert len(scatter_subgraphs_outputs) == len(
            scatter_strategy
        ), "The list of scattered outputs has length that doesn't match the number of silos."

        # prepare inputs with their respective keys for the gather() step
        gather_inputs_list = []
        for i, scatter_output in enumerate(scatter_subgraphs_outputs, 1):
            # map keys of outputs of scatter subgraph into inputs of gather
            gather_inputs_list.append(
                (scatter_to_gather_map(key, i), scatter_output[key])
            )
        # create a dict to feed to gather() using **
        gather_inputs_dict = dict(gather_inputs_list)

        # call gather() to aggregate all scattered outputs into one
        aggregation_step = gather(**gather_inputs_dict)

        # and let's anchor the output of gather() in the ORCHESTRATOR
        anchor_step_in_silo(
            aggregation_step,
            compute=gather_strategy["compute"],
            output_datastore=gather_strategy["datastore"],
            orchestrator_datastore=gather_strategy["datastore"],
            _path="aggregation_step",  # to help with debug logging
        )
        # now let's map the output of gather() to the accumulator for next iteration
        iteration_outputs = {}
        for key in aggregation_step.outputs:
            iteration_outputs[gather_to_scatter_map(key)] = aggregation_step.outputs[
                key
            ]
        # and return that as the output of the iteration pipeline
        return iteration_outputs

    @pipeline(
        description=f'FL cross-silo scatter-gather pipeline and the unique identifier is "{getUniqueIdentifier()}" that can help you to track files in the storage account.',
    )
    def fl_pipeline():
        """The entire scatter-gather pipeline."""
        # initialize the checkpoint
        checkpoint = None

        # now for each iteration, run scatter-gather subgraph
        for iteration_num in range(1, iterations + 1):
            # call scatter-gather() for each iteration
            iteration_step = scatter_gather_iteration(
                iteration_input=checkpoint, iteration_num=iteration_num
            )

            # set a readable name for the iteration step
            iteration_step.name = f"scatter_gather_iteration_{iteration_num}"

            # the running checkpoint needs to be anchored to the orchestrator
            for key in iteration_step.outputs:
                setattr(
                    iteration_step.outputs,
                    key,
                    custom_fl_data_output(gather_strategy["datastore"], key),
                )

            # let's keep track of the checkpoint to be used as input for next iteration
            checkpoint = iteration_step.outputs["checkpoint"]

        return {
            # the only output is the checkpoint with a custom key
            "checkpoint": checkpoint
        }

    # finally create an instance of the job
    pipeline_job = fl_pipeline()

    # NOTE: for some reason, we need to anchor the output of that job as well
    # this is anchored to the orchestrator
    for key in pipeline_job.outputs:
        setattr(
            pipeline_job.outputs,
            key,
            custom_fl_data_output(gather_strategy["datastore"], key),
        )

    return pipeline_job
