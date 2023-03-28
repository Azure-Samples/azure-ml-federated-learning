# Troubleshooting guide

## Table of Contents

- [Deployment failures](#deployment-failures)
  - [Issue: The storage account named staml... is already taken](#issue-the-storage-account-named-staml-is-already-taken)
  - [Issue: A vault with the same name already exists in deleted state](#issue-a-vault-with-the-same-name-already-exists-in-deleted-state)
- [Experiment failures](#experiment-failures)
  - [Issue: Dataset initialization failed DataAccessError(PermissionDenied)](#issue-dataset-initialization-failed-dataaccesserrorpermissiondenied)
  - [Issue: DataAccessError in an isolated environment](#issue-dataaccesserror-in-an-isolated-environment)

## Deployment failures

### Issue: The client does not have permission to perform action 'Microsoft.Authorization/roleAssignments/write'

During deployment, you may encounter the following exception:

```json
{
    "code": "InvalidTemplateDeployment",
    "message": "Deployment failed with multiple errors: 'Authorization failed for template resource '<UUID>' of type 'Microsoft.Authorization/roleAssignments'. The client '<USERNAME>' with object id '<UUID>' does not have permission to perform action 'Microsoft.Authorization/roleAssignments/write' at scope '/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Storage/storageAccounts/<STORAGENAME>/providers/Microsoft.Authorization/roleAssignments/<UUID>'"
}
```

**Root cause**: To provision a sandbox workspace from our tutorials, you need to have permissions to set role assignments in a given resource group. Creating your own resource group is not enough, you need to be the _Owner_ of this Azure resource group.

**Fix**: Ask your subscription admin to set you _Owner_ of the resource group you are deploying to.

### Issue: The storage account named staml... is already taken

Referenced in [issue 130](https://github.com/Azure-Samples/azure-ml-federated-learning/issues/130).

During deployment, you may encounter the following exception:

```json
{
    "code": "InvalidTemplateDeployment",
    "message": "The template deployment 'sandbox_minimal' is not valid according to the validation procedure. The tracking id is 'f6e64397-6b33-4990-a7b3-48b4ba92c4c8'. See inner errors for details."
}
"Inner Errors": {
    "code": "PreflightValidationCheckFailed",
    "message": "Preflight validation failed. Please refer to the details for the specific errors."
}
"Inner Errors": {
    "code": "StorageAccountAlreadyTaken",
    "target": "stamlfldemofghjk4",
    "message": "The storage account named stamlfldemofghjk4 is already taken."
}
```

**Root cause**: The name for the storage account (either for the workspace, for the orchestrator or the silo) conflicts with an existing name. Storage account names are **global**, so even if you create a storage account in a different subscription or resource group, it will still conflict.

**Fix**: pick a different name for the storage account or delete the existing one.

### Issue: A vault with the same name already exists in deleted state

Referenced in [issue 130](https://github.com/Azure-Samples/azure-ml-federated-learning/issues/130).

During deployment, you may encounter the following exception, even if you cannot find an existing keyvault with the name you specified:

```json
{
    "status":"Failed",
    "error": {
        "code":"DeploymentFailed",
        "message":"At least one resource deployment operation failed. Please list deployment operations for details. Please see https://aka.ms/DeployOperations for usage details.",
        "details":[
            {
                "code":"Conflict",
                "message":"{\r\n \"status\": \"Failed\",\r\n \"error\": {\r\n \"code\": \"ResourceDeploymentFailure\",\r\n \"message\": \"The resource operation completed with terminal provisioning state 'Failed'.\",\r\n \"details\": [\r\n {\r\n \"code\": \"DeploymentFailed\",\r\n \"message\": \"At least one resource deployment operation failed. Please list deployment operations for details. Please see https://aka.ms/DeployOperations for usage details.\",\r\n \"details\": [\r\n {\r\n \"code\": \"Conflict\",\r\n \"message\": \"{\\r\\n \\\"error\\\": {\\r\\n \\\"code\\\": \\\"ConflictError\\\",\\r\\n \\\"message\\\": \\\"A vault with the same name already exists in deleted state. You need to either recover or purge existing key vault. Follow this link https://go.microsoft.com/fwlink/?linkid=2149745 for more information on soft delete.\\\"\\r\\n }\\r\\n}\"\r\n }\r\n ]\r\n }\r\n ]\r\n }\r\n}"
            }
        ]
    }
}
```

**Root cause**: The reason is likely because you created a keyvault before with the same name, and soft-deleted it. [As documented here, soft-deletion](https://learn.microsoft.com/en-us/azure/key-vault/general/soft-delete-overview) allows you to recover a deleted keyvault. The name will be reserved for 90 days by default.

**Fix**: either [hard delete the keyvault](https://learn.microsoft.com/en-us/azure/key-vault/general/soft-delete-overview) or pick a different name.

## Experiment failures

### Issue: Dataset initialization failed DataAccessError(PermissionDenied)

During an experiment, you may encounter the following exception:

```log
Dataset initialization failed: AzureMLException:
 Message: DataAccessError(PermissionDenied)
 InnerException None
 ErrorResponse
{
    "error": {
        "message": "DataAccessError(PermissionDenied)"
    }
}
```

**Root cause**: This occurs usually in the training phase of our tutorials when the silo compute tries to mount the orchestrator storage, but doesn't have the permissions to do so. In some of our tutorials, it happens routinely if, after provisioning a silo or orchestrator, you skipped the section of the tutorial to set the RBAC R/W permissions (example [for open silo](./provisioning/silo_open.md#set-permissions-for-the-silos-compute-to-rw-fromto-the-orchestrator)). This can also happen for other reasons, but what this exception indicates is that the compute's identity doesn't have the required permissions to access/mount the data on the given datastore.

**Fix**: Follow the instructions to set the permissions right for this compute to access the orchestrator storage. In particular, set the RBAC roles for the user assigned identity towards the storage account (example [for open silo](./provisioning/silo_open.md#set-permissions-for-the-silos-compute-to-rw-fromto-the-orchestrator)).

### Issue: DataAccessError in an isolated environment

Referenced in [issue 195](https://github.com/Azure-Samples/azure-ml-federated-learning/issues/195).

During an experiment, if you encounter the following exception:

```json
{
    "NonCompliant":"DataAccessError(ConnectionFailure { source: Some(Custom { kind: TimedOut, error: 'Request timeout' }) })"
}
{
    "code": "data-capability.UriMountSession.PyFuseError",
    "target": "",
    "category": "UserError",
    "error_details": [
        {
            "key": "NonCompliantReason",
            "value": "DataAccessError(ConnectionFailure { source: Some(Custom { kind: TimedOut, error: 'Request timeout' }) })"
        },
        {
            "key": "StackTrace",
            "value": " File '/opt/miniconda/envs/data-capability/lib/python3.7/site-packages/data_capability/capability_session.py', line 70, in start\n (data_path, sub_data_path) = session.start()\n\n File '/opt/miniconda/envs/data-capability/lib/python3.7/site-packages/data_capability/data_sessions.py', line 386, in start\n options=mnt_options\n\n File '/opt/miniconda/envs/data-capability/lib/python3.7/site-packages/azureml/dataprep/fuse/dprepfuse.py', line 696, in rslex_uri_volume_mount\n raise e\n\n File '/opt/miniconda/envs/data-capability/lib/python3.7/site-packages/azureml/dataprep/fuse/dprepfuse.py, line 690, in rslex_uri_volume_mount\n mount_context = RslexDirectURIMountContext(mount_point, uri, options)\n"
        }
    ]
}
```

**Root cause**: You are running this experiment using a compute behind a vnet, with a storage accessed through a private endpoint. The compute is not able to connect to the storage because it cannot find its private IP address in the vnet. The root cause is likely because the IP address in the private DNS zone is wrong, or the endpoint private IP address does not match what is in the DNS records.

**Fix**: If the private DNS zone records are wrong, you should be able to modify them to reflect the real IP address of the private endpoint. If the problem comes from the private endpoint NIC and IP address, you might need to re-create the endpoint to set it correctly. Also, see the reference [issue 195](https://github.com/Azure-Samples/azure-ml-federated-learning/issues/195) for details.
