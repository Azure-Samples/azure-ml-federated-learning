# 0. Extract setup information, load dependencies, set subscription
# 0.a Read the YAML file
[string[]]$fileContent = Get-Content "./YAML/setup_info.yml"
$content = ''
foreach ($line in $fileContent) { $content = $content + "`n" + $line }
$yaml = ConvertFrom-YAML $content
# 0.b Collect the values from the YAML file
$SubscriptionId = $yaml['subscription_id']
$WorkspaceName = $yaml['workspace']['name']
$WorkspaceRegion = $yaml['workspace']['region']
$WorkspaceResourceGroup = $yaml['workspace']['resource_group']
$OrchestratorComputeName = $yaml['workspace']['orchestrator_compute_name']
$Silos = $yaml['silos']
$NumSilos = $Silos.Count
# 0.c Display summary of what will be created
# 0.c.i Overall summary
$Summary = "We will create a new workspace named '$WorkspaceName' in the '$WorkspaceRegion' region. The workspace will be created in the '$WorkspaceResourceGroup' resource group of the subscription '$SubscriptionId'. It will have $NumSilos silos."
Write-Output $Summary
# 0.c.ii Per-silo summary
$SiloIndex = 1
foreach ($Silo in $Silos)
{
    $SiloName = $Silo['name']
    $SiloRegion = $Silo['region']
    $SiloSummary = "Silo number $SiloIndex will be named '$SiloName' and will be created in the '$SiloRegion' region."
    Write-Output $SiloSummary
    # Increment the silo index to keep it accurate
    $SiloIndex=$SiloIndex+1
}
# 0.d Load useful functions
. "$PSScriptRoot/../../external_silos/ps/AzureUtilities.ps1"
# 0.e Log in and make sure we're in the right subscription
az login
az account set --subscription $SubscriptionId

# 1. Create workspace            
# 1.a Validate the required name of the Azure ML workspace
Confirm-Name $WorkspaceName "AMLWorkspace"
# 1.b Create the orchestrator workspace if it does not exist already
$Workspaces =  az ml workspace list --resource-group $WorkspaceResourceGroup --query "[?name=='$WorkspaceName']" | ConvertFrom-Json
if ($Workspaces.Length -eq 0){
    Deploy-RGIfInexistent $WorkspaceResourceGroup $WorkspaceRegion "AML workspace"
    Write-Output "Creating the workspace '$WorkspaceName'..."
    az deployment group create --resource-group $WorkspaceResourceGroup --template-file ../external_silos/bicep/AMLWorkspace.bicep --parameters workspacename=$WorkspaceName  location=$WorkspaceRegion
} else {
    Write-Output "The AML workspace $WorkspaceName already exists."
}
# 1.c Create compute cluster for the orchestrator
$OrchestratorComputes = az ml compute list -g $WorkspaceResourceGroup -w $WorkspaceName --query "[?name=='$OrchestratorComputeName']" | ConvertFrom-Json
if ($OrchestratorComputes.Length -eq 0){
    Write-Output "Creating the '$OrchestratorComputeName' compute in the orchestrator workspace."
    az ml compute create --name $OrchestratorComputeName --type AmlCompute --size STANDARD_DS3_v2 --min-instances 0 --max-instances 4 --idle-time-before-scale-down 120 --resource-group $WorkspaceResourceGroup --workspace-name $WorkspaceName
} else {
    Write-Output "The orchestrator compute $OrchestratorComputeName already exists."
}
Write-Output "Done with workspace creation"

# 2. Create silos (in the vanilla case, silos are just computes)
$SiloIndex = 1
foreach ($Silo in $Silos)
{
    # Get silo properties
    $SiloName = $Silo['name']
    $SiloRegion = $Silo['region'] # unused for now due to bug https://dev.azure.com/msdata/Vienna/_workitems/edit/1953178
    # Derive the compute name
    $SiloComputeName = "cpu-" + $SiloName
    # Validate the compute name
    Confirm-Name $SiloComputeName "Compute"
    # The kind of GPU we want
    $ComputeSKU = "STANDARD_DS3_v2"
    # Create it if it does not exist already
    $SiloComputes = az ml compute list -g $WorkspaceResourceGroup -w $WorkspaceName --query "[?name=='$SiloComputeName']" | ConvertFrom-Json
    if ($SiloComputes.Length -eq 0){
        Write-Output "Creating the '$SiloComputeName' silo compute in the orchestrator workspace."
        az ml compute create --name $SiloComputeName --type AmlCompute --size $ComputeSKU --min-instances 0 --max-instances 2 --idle-time-before-scale-down 120 --resource-group $WorkspaceResourceGroup --workspace-name $WorkspaceName
    } else {
        Write-Output "The silo compute $SiloComputeName already exists."
    }
    # Increment the silo index to keep it accurate
    $SiloIndex=$SiloIndex+1
}
Write-Output "Done with silos creation."

# 3. Upload some MNIST data to each silo (one training set and one test set). At this point, the data in all 3 silos is identical, but we'll pre-process the data so each silo only considers a different subset.
$SiloIndex = 1
foreach ($Silo in $Silos)
{
    Write-Output "Uploading training and test data to silo $SiloIndex..."
    # Pad the silo number with a zero (for nicer display name)
    $FormattedSiloIndex = $SiloIndex.ToString("D2")
    # We prepare the data assets definition in YAML.
    # First for the training set...
    $TrainingDataYAML="`$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: mnist-train-$FormattedSiloIndex
description: Dataset created from folder in cloud using https URL.
type: uri_file
path: https://azureopendatastorage.blob.core.windows.net/mnist/processed/train.csv"
    # Then for the test set.
    $TestDataYAML="`$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: mnist-test-$FormattedSiloIndex
description: Dataset created from folder in cloud using https URL.
type: uri_file
path: https://azureopendatastorage.blob.core.windows.net/mnist/processed/t10k.csv"
    # We write the YAML's into 2 files
    $TrainingDataYAML | Out-File -FilePath ./mnist_train.yml
    $TestDataYAML | Out-File -FilePath ./mnist_test.yml
    # We use the CLI to create the data assets
    az ml data create --file ./mnist_train.yml --resource-group $WorkspaceResourceGroup --workspace-name $WorkspaceName --subscription $SubscriptionId
    az ml data create --file ./mnist_test.yml --resource-group $WorkspaceResourceGroup --workspace-name $WorkspaceName --subscription $SubscriptionId
    # And finally we delete the temporary files
    Remove-Item -Path ./mnist_train.yml
    Remove-Item -Path ./mnist_test.yml
    # Increment the silo index to keep it accurate
    $SiloIndex=$SiloIndex+1
}
Write-Output "Done with data assets creation."