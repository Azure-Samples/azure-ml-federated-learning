// This BICEP script will fully provision a functional federated learning sandbox
// based on simple internal silos secured with only Managed Identities.

targetScope = 'resourceGroup'

// please specify the base name for all resources
@description('Base name of the demo, used for creating all resources as prefix')
param demoBaseName string = 'fldemo'

// below parameters are optionals and have default values
@allowed(['UserAssigned','SystemAssigned'])
@description('Type of identity to use for permissions model')
param identityType string = 'UserAssigned'

@description('Region of the orchestrator (workspace, central storage and compute).')
param orchestratorRegion string = resourceGroup().location

@description('List of each region in which to create an internal silo.')
param siloRegions array = [
  'australiaeast'
  'eastus'
  'westeurope'
]

@description('The VM used for creating compute clusters in orchestrator and silos.')
param compute1SKU string = 'Standard_DS4_v2'

@description('Flag whether to create a second compute or not')
param compute2 bool = false

@description('The VM used for creating a second compute cluster in orchestrator and silos.')
param compute2SKU string = 'Standard_NC6'

@description('Name of the keyvault to use for storing actual secrets (ex: encryption at rest).')
param confidentialityKeyVaultName string = 'kv-${demoBaseName}'

@description('Tags to curate the resources in Azure.')
param tags object = {
  Owner: 'AzureML Samples'
  Project: 'azure-ml-federated-learning'
  Environment: 'dev'
  Toolkit: 'bicep'
  Docs: 'https://github.com/Azure-Samples/azure-ml-federated-learning'
}

// Create Azure Machine Learning workspace
module workspace './modules/azureml/open_azureml_workspace.bicep' = {
  name: '${demoBaseName}-aml-${orchestratorRegion}'
  scope: resourceGroup()
  params: {
    baseName: demoBaseName
    machineLearningName: 'aml-${demoBaseName}'
    machineLearningDescription: 'Azure ML demo workspace for federated learning (use for dev purpose only)'
    location: orchestratorRegion
    tags: tags
  }
}

// Create an orchestrator compute+storage pair and attach to workspace
module orchestrator './modules/fl_pairs/open_compute_storage_pair.bicep' = {
  name: '${demoBaseName}-openpair-orchestrator'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion

    pairRegion: orchestratorRegion
    tags: tags

    pairBaseName: '${demoBaseName}-orch'

    compute1Name: 'orchestrator-01' // let's not use demo base name in cluster name
    compute1SKU: compute1SKU
    computeNodes: 2
    compute2: false
    compute2SKU: compute2SKU
    compute2Name: 'orchestrator-02'

    datastoreName: 'datastore_orchestrator' // let's not use demo base name

    // identity for permissions model
    identityType: identityType

    // set R/W permissions for orchestrator UAI towards orchestrator storage
    applyDefaultPermissions: true
  }
  dependsOn: [
    workspace
  ]
}

var siloCount = length(siloRegions)

// Create all silos using a provided bicep module
module silos './modules/fl_pairs/open_compute_storage_pair.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-openpair-silo-${i}'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion
    pairRegion: siloRegions[i]
    tags: tags

    pairBaseName: '${demoBaseName}-silo${i}'

    compute1Name: 'silo${i}-01' // let's not use demo base name
    compute1SKU: compute1SKU
    computeNodes: 2
    compute2: compute2
    compute2SKU: compute2SKU
    compute2Name: 'silo${i}-02'

    datastoreName: 'datastore_silo${i}' // let's not use demo base name

    // identity for permissions model
    identityType: identityType

    // set R/W permissions for orchestrator UAI towards orchestrator storage
    applyDefaultPermissions: true
  }
  dependsOn: [
    workspace
  ]
}]

// set R/W permissions for silo identity towards orchestrator storage
module siloToOrchPermissions './modules/permissions/msi_storage_rw.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-rw-perms-silo${i}-to-orch'
  scope: resourceGroup()
  params: {
    storageAccountName: orchestrator.outputs.storageName
    identityPrincipalId: silos[i].outputs.identityPrincipalId
  }
  dependsOn: [
    silos
  ]
}]


// Create a "confidentiality" keyvault external to the workspace
// This keyvault will be used to store actual secrets (ex: encryption at rest)
var siloIdentities = [ for i in range(0, siloCount) : '${silos[i].outputs.identityPrincipalId}' ]

module confidentialityKeyVault './modules/resources/confidentiality_keyvault.bicep' = {
  name: '${demoBaseName}-kv-confidentiality'
  params: {
    keyVaultName: confidentialityKeyVaultName
    tags: tags
    region: orchestratorRegion
    identitiesEnabledCryptoOperations: siloIdentities
    // for some reason, concat doesn't work here, using secondary list
    secondaryIdentitiesEnabledCryptoOperations: [ '${orchestrator.outputs.identityPrincipalId}' ]
  }
  dependsOn: [
    silos
    orchestrator
  ]
}
