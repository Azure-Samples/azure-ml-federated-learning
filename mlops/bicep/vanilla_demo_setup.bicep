// This BICEP script will fully provision a functional federated learning demo
// based on simple internal silos secured with UAI.

// For a given set of regions, it will provision:
// - an AzureML workspace and compute cluster for orchestration
// - per region, a silo (1 storage with 1 dedicated containers, 1 compute, 1 UAI)

// Usage (sh):
// > az login
// > az account set --name <subscription name>
// > az deployment sub create --template-file .\mlops\bicep\vanilla_demo_setup.bicep \
//                            --location eastus \
//                            --parameters demoBaseName="fldemo1"

targetScope = 'subscription'

// please specify the base name for all resources
@description('Base name of the demo, used for creating all resources as prefix')
param demoBaseName string

// below parameters are optionals and have default values
@description('Name of the workspace to create.')
param workspaceName string = '${demoBaseName}-aml'

@description('Name of the resource group to create.')
param resourceGroupName string = '${demoBaseName}-rg'

@description('Location of the orchestrator (workspace, central storage and compute).')
param location string = 'eastus'

@description('List of each region in which to create an internal silo.')
param siloRegions array = ['eastus', 'westus', 'westus2']

@description('Tags to curate the resources in Azure.')
param tags object = {
  Owner: 'AzureML Samples'
  Project: 'azure-ml-federated-learning'
  Environment: 'dev'
  Toolkit: 'bicep'
  Docs: 'https://github.com/Azure-Samples/azure-ml-federated-learning'
}

var siloCount = length(siloRegions)

// Create resource group for the demo (cold start)
resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

// Create Azure Machine Learning workspace for orchestration
// with an orchestration compute
module orchestratorDeployment './modules/orchestrators/orchestrator_with_uai.bicep' = {
  name: '${demoBaseName}deployazuremlworkspace${location}'
  params: {
    workspaceName: workspaceName
    location: location
    orchestratorComputeName: 'cpu-cluster-orchestrator'
  }
  scope: resourceGroup
}

// Create all vanilla silos using a provided bicep module
module siloDeployments './modules/silos/internal_blob_with_uai.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}deploysilo${siloRegions[i]}'
  params: {
    workspaceName: workspaceName
    region: siloRegions[i]
  }
  scope: resourceGroup
  dependsOn: [
    orchestratorDeployment
  ]
}]

// Set permissions model
module permissionModelDeployment './modules/permissions/model_uai_single_direction_write.bicep' = {
  name: '${demoBaseName}permissionmodeldeployment'
  params: {
    orchestratorConfig: orchestratorDeployment.outputs.orchestratorConfig
    siloConfigArray: [ for i in range(0, siloCount): siloDeployments[i].outputs.siloConfig ]
  }
  scope: resourceGroup
  dependsOn: [
    orchestratorDeployment
    siloDeployments
  ]
}


// TODO: output the config for local submit???
output subscription_id string = subscription().subscriptionId
output resource_group string = resourceGroupName
output workspace_name string = workspaceName
