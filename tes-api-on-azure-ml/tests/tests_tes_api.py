from azureml.core import Workspace, Run, Experiment
from api import TesApi
import pytest
from azureml.exceptions import ServiceException
import datetime


def test_init():
    """Tests if init function works"""
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    tes_api = TesApi(workspace)
    assert tes_api.ws._subscription_id == "48bbc269-ce89-4f6f-9a12-c6f91fcb772d"
    assert tes_api.ws._resource_group == "aml1p-rg"
    assert tes_api.ws._workspace_name == "aml1p-ml-wus2"


@pytest.mark.parametrize(
    "test_run_id,view",
    [
        ("dcc407c8-d905-4b54-9441-9e8772abdf8a", "mini"),
        ("dcc407c8-d905-4b54-9441-9e8772abdf8a", "full"),
        ("dcc407c8-d905-4b54-9441-9e8772abdf8a", "wrong_view"),
    ],
)
def test_get_descriptors(test_run_id, view):
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    test_run = Run.get(workspace, test_run_id)
    if view not in ("mini", "full"):
        with pytest.raises(ValueError):
            descriptors = TesApi.get_descriptors(test_run, view)
        return
    else:
        descriptors = TesApi.get_descriptors(test_run, view)
        assert descriptors["runId"] == "dcc407c8-d905-4b54-9441-9e8772abdf8a"
        assert descriptors["status"] == "Failed"
        if view == "full":
            assert descriptors["startTimeUtc"] == "2022-05-11T01:53:20.168964Z"
            assert descriptors["endTimeUtc"] == "2022-05-11T02:04:51.030079Z"
            assert descriptors["services"] == {}
            assert descriptors["properties"] == {
                "azureml.runsource": "azureml.PipelineRun",
                "runSource": "Designer",
                "runType": "HTTP",
                "azureml.parameters": "{}",
                "azureml.continue_on_step_failure": "False",
                "azureml.pipelineComponent": "pipelinerun",
            }
            assert descriptors["submittedBy"] == "Fuhui Fang"
            assert descriptors["inputDatasets"].__len__() == 0
            assert descriptors["outputDatasets"].__len__() == 0
            assert descriptors["logFiles"]["logs/azureml/executionlogs.txt"].startswith(
                "https://aml1pmlwus27954171068.blob.core.windows.net/azureml/ExperimentRun/dcid.dcc407c8-d905-4b54-9441-9e8772abdf8a/logs/azureml/executionlogs.txt?sv=2019-07-07"
            )
            assert descriptors["logFiles"]["logs/azureml/stderrlogs.txt"].startswith(
                "https://aml1pmlwus27954171068.blob.core.windows.net/azureml/ExperimentRun/dcid.dcc407c8-d905-4b54-9441-9e8772abdf8a/logs/azureml/stderrlogs.txt?sv=2019-07-07"
            )
            assert descriptors["logFiles"]["logs/azureml/stdoutlogs.txt"].startswith(
                "https://aml1pmlwus27954171068.blob.core.windows.net/azureml/ExperimentRun/dcid.dcc407c8-d905-4b54-9441-9e8772abdf8a/logs/azureml/stdoutlogs.txt?sv=2019-07-07"
            )


@pytest.mark.parametrize("view", ["mini", "full", "wrong_view_value"])
def test_list_tasks(view):
    """Tests if list_tasks returns a non-empty list of tasks, with non-empty values for the main keys"""
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    tes_api = TesApi(workspace)
    if (view == "mini") or (view == "full"):
        tasks = tes_api.list_tasks(view=view)
    else:
        with pytest.raises(ValueError):
            tasks = tes_api.list_tasks(view=view)
        return
    assert len(tasks) > 0
    assert tasks[0]["runId"]
    assert tasks[0]["status"]
    if view == "full":
        assert tasks[0]["submittedBy"]


@pytest.mark.parametrize(
    "task_id,view",
    [
        ("dcc407c8-d905-4b54-9441-9e8772abdf8a", "mini"),
        ("dcc407c8-d905-4b54-9441-9e8772abdf8a", "full"),
        ("wrong_task_id", "mini"),
        ("wrong_task_id", "full"),
    ],
)
def test_get_task(task_id, view):
    """Tests if get_task returns a non-empty dictionary of task details"""
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    tes_api = TesApi(workspace)
    if task_id == "wrong_task_id":
        with pytest.raises(ServiceException):
            task = tes_api.get_task(task_id, view)
        return
    else:
        task = tes_api.get_task(task_id, view)
    assert len(task) > 0
    assert task["runId"]
    assert task["status"]
    if view == "full":
        assert task["submittedBy"]


def test_cancel_existing_task():
    """Tests if we can cancel an existing task."""
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    # first submit a run and grab its id
    experiment = Experiment(workspace, "test_TES_API")
    run = experiment.start_logging(outputs=None, snapshot_directory=".")
    task_id = TesApi.get_descriptors(run, "mini")["runId"]
    # cancel it
    tes_api = TesApi(workspace)
    tes_api.cancel_task(task_id)
    # check it has been canceled
    task = tes_api.get_task(task_id, "mini")
    assert task["status"] == "Canceled"


def test_cancel_nonexisting_task():
    """Tests that trying to cancel a non-existing task throws an exception, as expected"""
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    tes_api = TesApi(workspace)
    with pytest.raises(ServiceException):
        tes_api.cancel_task("wrong_task_id")


def test_create_task_script():
    """Tests that we can create a basic task (run a script)."""
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    tes_api = TesApi(workspace)
    # first we submit the task
    run_id = tes_api.create_task(
        name="mock_task" + "_" + str(datetime.datetime.now()),
        description="A mock task created during unit tests (script with no arguments).",
        inputs=[],
        outputs=[],
        compute_target="cpu-cluster",
        environment="AzureML-minimal-ubuntu18.04-py37-cpu-inference",
        executors={
            "source_directory": "./tests/hello_world",
            "script": "hello.py",
            "command": [],
            "arguments": [],
        },
        volumes=None,
        tags={"test_type": "script"},
        task_group="test_TES_API",
    )
    # then we test that we can grab it, and finally we delete it
    descriptors = tes_api.get_task(run_id, "full")
    assert descriptors["runId"]
    assert descriptors["status"]
    tes_api.cancel_task(run_id)


def test_create_task_command_with_args():
    """Tests that we can create a basic task (run a command using some arguments)."""
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    tes_api = TesApi(workspace)
    # first we submit the task
    run_id = tes_api.create_task(
        name="mock_task_command_" + str(datetime.datetime.now()),
        description="A mock task created during unit tests (command with arguments).",
        inputs=[],
        outputs=[],
        compute_target="cpu-cluster",
        environment="AzureML-minimal-ubuntu18.04-py37-cpu-inference",
        executors={
            "source_directory": "./tests/add_and_print",
            "script": "",
            "command": [
                "python",
                "add_and_print.py",
                "--operand_1",
                "2",
                "--operand_2",
                "3",
            ],
            "arguments": [],
        },
        volumes=None,
        tags={"test_type": "command_with_arguments"},
        task_group="test_TES_API",
    )
    # then we test that we can grab it, and finally we delete it
    descriptors = tes_api.get_task(run_id, "full")
    assert descriptors["runId"]
    assert descriptors["status"]
    tes_api.cancel_task(run_id)


def test_create_task_command_with_inputs():
    """Tests that we can create a basic task (run a command using some inputs)."""
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    tes_api = TesApi(workspace)
    # first we submit the task
    run_id = tes_api.create_task(
        name="mock_task_command_inputs_" + str(datetime.datetime.now()),
        description="A mock task created during unit tests (command with inputs).",
        inputs=["mnist", "irisdata"],
        outputs=[],
        compute_target="cpu-cluster",
        environment="AzureML-minimal-ubuntu18.04-py37-cpu-inference",
        executors={
            "source_directory": "./tests/count_files",
            "script": "",
            "command": ["python", "count_files.py", "--argument_1", "foo"],
            "arguments": [],
        },
        volumes=None,
        tags={"test_type": "command_with_inputs"},
        task_group="test_TES_API",
    )
    # then we test that we can grab it, and finally we delete it
    descriptors = tes_api.get_task(run_id, "full")
    assert descriptors["runId"]
    assert descriptors["status"]
    tes_api.cancel_task(run_id)


def test_create_task_script_with_inputs():
    """Tests that we can create a basic task (run a script using some inputs)."""
    workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
    tes_api = TesApi(workspace)
    # first we submit the task
    run_id = tes_api.create_task(
        name="mock_task_script_inputs_" + str(datetime.datetime.now()),
        description="A mock task created during unit tests (script with inputs).",
        inputs=["mnist", "irisdata"],
        outputs=[],
        compute_target="cpu-cluster",
        environment="AzureML-minimal-ubuntu18.04-py37-cpu-inference",
        executors={
            "source_directory": "./tests/count_files",
            "script": "count_files.py",
            "command": [],
            "arguments": ["--argument_1", "foo"],
        },
        volumes=None,
        tags={"test_type": "script_with_inputs"},
        task_group="test_TES_API",
    )
    # then we test that we can grab it, and finally we delete it
    descriptors = tes_api.get_task(run_id, "full")
    assert descriptors["runId"]
    assert descriptors["status"]
    tes_api.cancel_task(run_id)


@pytest.mark.parametrize(
    "list,inputs,expected_result",
    [
        ([], [], []),
        ([], ["dataset_1"], ["--input_data_1", "dataset_1"]),
        (["foo", "bar"], ["dataset_1"], ["foo", "bar", "--input_data_1", "dataset_1"]),
    ],
)
def test_update_list(list, inputs, expected_result):
    result = TesApi.update_list(list, inputs)
    assert result == expected_result
