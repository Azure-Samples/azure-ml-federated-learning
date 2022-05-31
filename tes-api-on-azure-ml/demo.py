from api.tes_api import TesApi
from azureml.core import Workspace
import datetime

# First we point at our workspace
# (adjust '_file_name' to the name of your config file)
workspace = Workspace.from_config(path="./config", _file_name="workspace.json")
# and we define the "experiment name", a.k.a. the name of the container in which the tasks will be grouped.
task_group_name = "demo_TES_API"

# Then, we create an instance of the TES API.
tes_api = TesApi(workspace)

# Let's see what kind of tasks are/were running (will return an empty list on the very first run)
tasks = tes_api.list_tasks(view="mini", task_group=task_group_name)
print(tasks)
print("")

# If some tasks were found, let's show the details of the first one
if len(tasks) > 0:
    task_details = tes_api.get_task(tasks[0]["runId"], view="full")
    print(task_details)
    print("")

# Let's create a task! See the create_task() docstring for more info.
created_task_id = tes_api.create_task(
    name="demo_task_command_inputs_" + str(datetime.datetime.now()),
    description="A mock task created during unit tests (command with inputs).",
    inputs=["mnist", "irisdata"],  # the name of the datasets we'll be using
    outputs=[],
    compute_target="cpu-cluster",  # the name of the (existing) compute target we'll be using
    environment="AzureML-minimal-ubuntu18.04-py37-cpu-inference",  # the name of the (existing) environment we'll be using
    executors={  # the definition of the executor - provide either script or command, not both
        "source_directory": "./tests/count_files",
        "script": "",
        "command": ["python", "count_files.py", "--argument_1", "foo"],
        "arguments": [],
    },
    volumes=None,
    tags={"test_type": "command_with_inputs"},
    task_group=task_group_name,
)
print("")

# Let's grab the details of the newly created task
created_task_details = tes_api.get_task(created_task_id, view="full")
print(created_task_details)
print("")

# And finally let's cancel it
tes_api.cancel_task(created_task_id)
