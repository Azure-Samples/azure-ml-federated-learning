
# 0. Extract setup information, load dependencies, set subscription
# 0.a Read the YAML file
[string[]]$fileContent = Get-Content "./YAML/setup_info.yml"
$content = ''
foreach ($line in $fileContent) { $content = $content + "`n" + $line }
$yaml = ConvertFrom-YAML $content
# 0.b Collect the values from the YAML file
$SubscriptionId = $yaml['subscription_id']
$WorkspaceName = $yaml['workspace']['name']
$WorkspaceResourceGroup = $yaml['workspace']['resource_group']
$Silos = $yaml['silos']
$NumSilos = $Silos.Count
# 0.c Display summary of what will be created
# 0.c.ii Per-silo summary
# 0.d Load useful functions
. "$PSScriptRoot/../../external_silos/ps/AzureUtilities.ps1"
# 0.e Log in and make sure we're in the right subscription
az login
az account set --subscription $SubscriptionId


# SECURE TRAINING INPUTS
$SiloIndex = 1
foreach ($Silo in $Silos)
{
    $SiloName = $Silo['name']
    $SiloRegion = $Silo['region']
    $SiloInputStorageAccount = $Silo['storage_account']
    $SiloContainer = $Silo['container']
    $SiloDatastoreName = $SiloName.replace('-', '') + "ds"
    Write-Output "Securing silo '$SiloName'..."
    # 1. create silo input datastore
    Write-Output "Creating input datastore '$SiloDatastoreName'"
    # 1.1 build the definition in YAML 
    $DatastoreYAML="`$schema: https://azuremlschemas.azureedge.net/latest/azureBlob.schema.json
name: $SiloDatastoreName
type: azure_blob
description: Credential-less datastore pointing to a blob container.
account_name: $SiloInputStorageAccount
container_name: $SiloContainer"
    # 1.2 write the YAML
    $DatastoreYAML | Out-File -FilePath ./datastore.yml
    # 1.3 use the CLI to create the datastore
    az ml datastore create --file datastore.yml --resource-group $WorkspaceResourceGroup --workspace-name $WorkspaceName
    # 1.4 delete the temporary YAML file
    Remove-Item -Path ./datastore.yml
    
    # 2. create data assets
    Write-Output "Creating data assets"
    # 2.1 build the definitions in YAML
    $TrainingDataYAML="`$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: mnist-train-$SiloRegion
description: Data asset created from file in cloud.
type: uri_file
path: azureml://datastores/$SiloDatastoreName/paths/train.csv" # adjust path and type accordingly
    $TestDataYAML="`$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: mnist-test-$SiloRegion
description: Data asset created from file in cloud.
type: uri_file
path: azureml://datastores/$SiloDatastoreName/paths/t10k.csv" # adjust path and type accordingly
    # 2.2 write the YAML's into 2 files
    $TrainingDataYAML | Out-File -FilePath ./mnist_train.yml
    $TestDataYAML | Out-File -FilePath ./mnist_test.yml
    # 2.3 use the CLI to create the data assets
    az ml data create --file ./mnist_train.yml --resource-group $WorkspaceResourceGroup --workspace-name $WorkspaceName --subscription $SubscriptionId
    az ml data create --file ./mnist_test.yml --resource-group $WorkspaceResourceGroup --workspace-name $WorkspaceName --subscription $SubscriptionId
    # 2.4 and finally we delete the temporary files
    Remove-Item -Path ./mnist_train.yml
    Remove-Item -Path ./mnist_test.yml
    
    # 3. create a user-assigned managed identity
    $SiloIdentityName = $SiloName + "-identity"
    Write-Output "Creating user-assigned identity '$SiloIdentityName'"
    az identity create --name $SiloIdentityName --resource-group $WorkspaceResourceGroup --location $SiloRegion --subscription $SubscriptionId

    # 4. give managed identity to the silo compute
    $SiloComputeName = "cpu-" + $SiloName
    Write-Output "Giving the user-assigned identity '$SiloIdentityName' to silo compute '$SiloComputeName'"
    $IdentityResourceId = "/subscriptions/$SubscriptionId/resourcegroups/$WorkspaceResourceGroup/providers/Microsoft.ManagedIdentity/userAssignedIdentities/$SiloIdentityName"
    az ml compute update --name $SiloComputeName --resource-group $WorkspaceResourceGroup --workspace-name $WorkspaceName --identity-type UserAssigned --user-assigned-identities $IdentityResourceId --subscription $SubscriptionId

    # Increment the silo index to keep it accurate
    Write-Output "Done securing silo '$SiloName'."
    $SiloIndex=$SiloIndex+1
}
