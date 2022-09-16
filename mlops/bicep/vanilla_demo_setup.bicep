// This BICEP script will fully provision a functional federated learning demo
// based on simple internal silos secured with UAI.

targetScope = 'subscription'

// please specify the base name for all resources
@description('Base name of the demo, used for creating all resources as prefix')
param demoBaseName string

// below parameters are optionals and have default values
@description('Name of the workspace to create.')
param workspaceName string = '${demoBaseName}-aml'

@description('Name of the resource group to create.')
param resourceGroupName string = '${demoBaseName}-deleteme20220916rc2-rg'

@description('Location of the orchestrator (workspace, central storage and compute).')
param orchestratorLocation string = 'eastus'

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

// Create resource group for the demo (cold start)
resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resourceGroupName
  location: orchestratorLocation
  tags: tags
}

// Create Azure Machine Learning workspace for orchestrator
module workspaceDeployment './modules/azureml_workspace.bicep' = {
  name: '${demoBaseName}deployazuremlworkspace${orchestratorLocation}'
  params: {
    workspaceName: workspaceName
    location: orchestratorLocation
  }
  scope: resourceGroup
}

// Create vanilla orchestrator "silo" with its own storage and compute
// TODO

// Create all vanilla silos using a provided bicep module
module siloDeployments './modules/silos/internal_blob_uai.bicep' = [for region in siloRegions: {
  name: '${demoBaseName}deploysilo${region}'
  params: {
    workspaceName: workspaceName
    region: region
    // all other parameters left default
  }
  scope: resourceGroup
  dependsOn: [
    workspaceDeployment
  ]
}]
