from azureml.core import Workspace, Experiment, Run, ScriptRunConfig, Environment, Dataset
import time


class TesApi:
    def __init__(self, workspace: Workspace):
        self.ws = workspace

    def list_tasks(self, view="mini", task_group="test_TES_API", include_children=False):
        """Returns the list of "tasks" (a.k.a. "runs" in Azure ML jargon) in a given "task group" (a.k.a. "experiment" in Azure3 ML jargon).
        Inputs:
          - view: "mini" to only get Task Id and Status (default), "full" to get all the details
          - task_group: the name of the Azure ML experiment whose runs we want to grab ("test_TES_API" by default)
          - include_children: whether to include the children of the runs in the list (False by default)
        """
        tic = time.perf_counter()
        runs = []
        # the experiment we're restricting ourselves to:
        exps = Experiment.list(self.ws, experiment_name=task_group)
        # iterating over all exps so the code will still run even if we don't filter down to a single experiment
        for exp in exps:
            exp_runs = exp.get_runs(include_children=include_children)
            for exp_run in exp_runs:
                runs.append(exp_run)
        runs_descriptors = []
        for run in runs:
            runs_descriptors.append(TesApi.get_descriptors(run, view))
        toc = time.perf_counter()
        print(f"Fetched the list of tasks in {toc - tic:0.4f} seconds")
        return runs_descriptors

    def get_task(self, task_id, view="mini"):
        """Returns the details of a given "task" (a.k.a. "run" in Azure ML jargon).
        Inputs:
          - task_id: the id of the task of which we want to get the details
          - view: "mini" to only get Task Id and Status (default), "full" to get all the details
        """
        tic = time.perf_counter()
        run = Run.get(self.ws, task_id)
        run_descriptors = TesApi.get_descriptors(run, view)
        toc = time.perf_counter()
        print(f"Fetched the task in {toc - tic:0.4f} seconds")
        return run_descriptors

    def cancel_task(self, task_id):
        tic = time.perf_counter()
        run = Run.get(self.ws, task_id)
        run.cancel()
        toc = time.perf_counter()
        print(f"Marked the task as canceled in {toc - tic:0.4f} seconds")

    def create_task(
        self,
        name: str,
        description: str,
        inputs=[],
        outputs=[],
        compute_target="cpu-cluster",
        environment="AzureML-minimal-ubuntu18.04-py37-cpu-inference",
        executors={
            'source_directory': './tests/hello_world',
            'script': 'hello.py',
            'command': [],
            'arguments': [],
        },
        volumes=None,
        tags={'test_tag_1': 'test_value_1', 'test_tag_2': 'test_value_2'},
        task_group="test_TES_API",
    ):
        """Creates a new "task" (a.k.a. "run" in Azure ML jargon) in a given "task group" (a.k.a. "experiment" in Azure3 ML jargon). Currently, only supports tasks with a single component.
        Inputs:
          - name: the display name of the task
          - description: the description of the task
          - inputs: a list of existing datasets, to be treated as inputs to the task (to-do: allow non-existing datasets?)
          #- outputs: a dictionary of outputs to the task
          - compute_target: the name of the existing compute target (to-do: allow compute target creation if it doesn't exist already)
          - environment: the name of the existing environment (to-do: allow environment creation if it doesn't exist already)
          - executors: a dictionary describing the task to run (currently only simple tasks with one step are supported). It must have the following keys:
            - source_directory: the path to the directory containing the source code to run
            - script: the name of the script to run (to-do: introduce support for arguments)
            - arguments: a list of arguments to pass to the script
            - command: the command to run the script with (Note: only command or script needs to be specified)
          #- volumes:
          - tags: a dictionary of tags to add to the task
          - task_group: the name of the Azure ML experiment in which to create the task ("test_TES_API" by default)
        """
        tic = time.perf_counter()
        experiment = Experiment(self.ws, task_group)
        command, arguments = executors["command"], executors["arguments"]
        datasets_inputs = [Dataset.get_by_name(self.ws, input).as_mount() for input in inputs]
        command, arguments = TesApi.update_command_and_arguments(
            command, arguments, datasets_inputs
        )
        src = ScriptRunConfig(
            source_directory=executors["source_directory"],
            script=executors["script"],
            command=command,
            compute_target=compute_target,
            arguments=arguments,
            environment=Environment.get(workspace=self.ws, name=environment),
        )
        run = experiment.submit(config=src, tags=tags)
        run.display_name = name
        run.description = description
        toc = time.perf_counter()
        run_id = TesApi.get_descriptors(run)["runId"]
        print(f"Submitted the task with id '{run_id}' in {toc - tic:0.4f} seconds")
        return run_id

    @staticmethod
    def update_command_and_arguments(command, arguments, inputs):
        """Updates the command and arguments to be run with the inputs.
        Inputs:
        - command: the command provided by the user (a list).
        - arguments: the arguments provided by the user (a list).
        - inputs: the inputs provided by the user (a list).
        """
        if len(inputs) == 0:
            return command, arguments
        # if no command given, just update the list of arguments
        if len(command) == 0:
            new_arguments = TesApi.update_list(arguments, inputs)
            return command, new_arguments
        # otherwise update the command
        else:
            new_command = TesApi.update_list(command, inputs)
            return new_command, arguments

    @staticmethod
    def update_list(list, inputs):
        """Appends the following to list: ['--input_data_1', inputs[0], '--input_data_2', inputs[1], ...]"""
        N_inputs = len(inputs)
        for i in range(N_inputs):
            list.append(f"--input_data_{i+1}")
            list.append(inputs[i])
        return list

    @staticmethod
    def get_descriptors(run: Run, view="mini"):
        """Returns the details of a given "task" (a.k.a. "run" in Azure ML jargon).
        Inputs:
          - run: the run object we want to get the details of
          - view: "mini" to only get Task Id and Status (default), "full" to get all the details
        """
        if view == "mini":
            return {
                "runId": run.__getattribute__("_run_id"),
                "status": run.__getattribute__("status"),
            }
        elif view == "full":
            return run.get_details()
        else:
            raise ValueError("'view' parameter must be either 'mini' or 'full'")
