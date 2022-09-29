// This BICEP script will fully provision a functional federated learning sandbox
// based on simple internal silos secured with only Managed Identities.

// IMPORTANT: This setup is intended only for demo purpose. The data is still accessible
// by the users when opening the storage accounts, and data exfiltration is easy.

// For a given set of regions, it will provision:
// - an AzureML workspace and compute cluster for orchestration
// - per region, a silo (1 storage with 1 dedicated containers, 1 compute, 1 Managed Identity)

// The demo permission model is represented by the following matrix:
// |               | orch.compute | siloA.compute | siloB.compute |
// |---------------|--------------|---------------|---------------|
// | orch.storage  |     R/W      |      R/W      |      R/W      |
// | siloA.storage |      -       |      R/W      |       -       |
// | siloB.storage |      -       |       -       |      R/W      |

// Usage (sh):
// > az login
// > az account set --name <subscription name>
// > az group create --name <resource group name> --location <region>
// > az deployment group create --template-file .\mlops\bicep\open_sandbox_setup.bicep \
//                              --resource-group <resource group name>
//                              --parameters demoBaseName="fldemo"

targetScope = 'resourceGroup'

// please specify the base name for all resources
@description('Base name of the demo, used for creating all resources as prefix')
param demoBaseName string = 'fldemo'

// below parameters are optionals and have default values
@description('Location of the orchestrator (workspace, central storage and compute).')
param orchestratorRegion string = resourceGroup().location

@description('List of each region in which to create an internal silo.')
param siloRegions array = [
  'westus'
  'francecentral'
  'brazilsouth'
]

@description('The VM used for creating compute clusters in orchestrator and silos.')
param computeSKU string = 'Standard_DS13_v2'

@description('The number of nodes used for creating compute clusters in orchestrator.')
param computeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
@description('The type of identity to use for the compute clusters.')
param identityType string = 'UserAssigned'

@description('Tags to curate the resources in Azure.')
param tags object = {
  Owner: 'AzureML Samples'
  Project: 'azure-ml-federated-learning'
  Environment: 'dev'
  Toolkit: 'bicep'
  Docs: 'https://github.com/Azure-Samples/azure-ml-federated-learning'
}

// Create Azure Machine Learning workspace for orchestration
module workspace './modules/resources/open_azureml_workspace.bicep' = {
  name: '${demoBaseName}-deploy-aml-${orchestratorRegion}'
  scope: resourceGroup()
  params: {
    machineLearningName: 'aml-${demoBaseName}'
    location: orchestratorRegion
    tags: tags
  }
}

// Create an orchestrator compute+storage pair and attach to workspace
module orchestrator './modules/orchestrators/open_orchestrator_blob.bicep' = {
  name: '${demoBaseName}-deploy-orchestrator-${orchestratorRegion}'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspace
    region: orchestratorRegion
    tags: tags

    computeName: 'cpu-cluster-orchestrator'
    computeSKU: computeSKU
    computeNodes: computeNodes

    identityType: identityType
  }
  dependsOn: [
    workspace
  ]
}

var siloCount = length(siloRegions)

// Create all vanilla silos using a provided bicep module
module silos './modules/silos/open_internal_blob.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-deploy-silo-${i}-${siloRegions[i]}'
  scope: resourceGroup()
  params: {
    siloName: '${demoBaseName}-silo${i}-${siloRegions[i]}'
    machineLearningName: 'aml-${demoBaseName}'
    region: siloRegions[i]
    tags: tags

    computeName: 'cpu-silo${i}-${siloRegions[i]}'
    computeSKU: computeSKU
    datastoreName: 'datastore_silo${i}_${siloRegions[i]}'

    identityType: identityType

    // reference of the orchestrator to set permissions
    orchestratorStorageAccountName: orchestrator.outputs.storage
  }
  dependsOn: [
    orchestrator
    workspace
  ]
}]
