from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, DataFactoryCompute
from azureml.exceptions import ComputeTargetException


def get_or_create_data_factory(workspace, factory_name):
    try:
        return DataFactoryCompute(workspace, factory_name)
    except ComputeTargetException as e:
        if "ComputeTargetNotFound" in e.message:
            print("Data factory not found, creating...")
            provisioning_config = DataFactoryCompute.provisioning_configuration()
            data_factory = ComputeTarget.create(
                workspace, factory_name, provisioning_config
            )
            data_factory.wait_for_completion()
            return data_factory
        else:
            raise e


# Enter your workspace name, resource group, and subscription Id here
ws = Workspace.get(
    name="Your-Workspace-Name",
    subscription_id="Your-Workspace-Subscription-Id",
    resource_group="Your-Workspace-Resource-Group",
)
data_factory_name = "data-factory"
data_factory_compute = get_or_create_data_factory(ws, data_factory_name)
print("Setup Azure Data Factory account complete")
