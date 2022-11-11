targetScope = 'resourceGroup'

// please specify the base name for all resources
@description('Base name of the demo, used for creating all resources as prefix')
param demoBaseName string = 'fldemo'

@description('Region of the orchestrator (workspace, central storage and compute).')
param orchestratorRegion string = resourceGroup().location

@description('List of each region in which to create an internal silo (Note: DCsv3/DCdsv5 not available in all regions).')
param siloRegions array = [
  'southeastasia' // use dcv3 series
  'australiaeast' // use dcv3 series
  // 'canadacentral' // not supported by sub
  // 'northeurope' // unknown
  // 'westeurope' // unknown
  // 'japaneast' // not supported by sub
  // 'switzerlandnorth' // not supported by sub
  // 'centralus' // not supported by sub
  //'eastus'
  'eastus2' // use dcv3 series or DC2as_v5
  'southcentralus' // use dcv3 series
  // 'westus' // not supported by sub
  'westus2' // use dcv3 series
]

@description('The VM used for creating compute clusters in orchestrator and silos.')
@allowed([
  // see https://learn.microsoft.com/en-us/azure/virtual-machines/dcv2-series
  'Standard_DC1s_v2'
  'Standard_DC2s_v2'
  'Standard_DC4s_v2'
  'Standard_DC8_v2'
  // see https://learn.microsoft.com/en-us/azure/virtual-machines/dcv3-series
  // DCdsv3-series
  'Standard_DC1ds_v3'
  'Standard_DC2ds_v3'
  'Standard_DC4ds_v3'
  'Standard_DC8ds_v3'
  'Standard_DC16ds_v3'
  'Standard_DC24ds_v3'
  'Standard_DC32ds_v3'
  'Standard_DC48ds_v3'
  // see https://learn.microsoft.com/en-us/azure/virtual-machines/dcasv5-dcadsv5-series
  'Standard_DC2as_v5'
  'Standard_DC4as_v5'
  'Standard_DC8as_v5'
  'Standard_DC16as_v5'
  'Standard_DC32as_v5'
  'Standard_DC48as_v5'
  'Standard_DC64as_v5'
  'Standard_DC96as_v5'
  'Standard_DC2ads_v5'
  'Standard_DC4ads_v5'
  'Standard_DC8ads_v5'
  'Standard_DC16ads_v5'
  'Standard_DC32ads_v5'
  'Standard_DC48ads_v5'
  'Standard_DC64ads_v5'
  'Standard_DC96ads_v5'
])
param computeSKU string = 'Standard_DC4ads_v5'

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
    machineLearningName: 'aml-${demoBaseName}'
    machineLearningDescription: 'Azure ML demo workspace for federated learning using confidential compute in AKS'
    baseName: demoBaseName
    location: orchestratorRegion
    tags: tags
  }
}

// Create an orchestrator compute+storage pair and attach to workspace
module orchestrator './modules/fl_pairs/open_aks_with_confcomp_storage_pair.bicep' = {
  name: '${demoBaseName}-openccpair-orchestrator'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion

    pairRegion: orchestratorRegion
    tags: tags

    pairBaseName: '${demoBaseName}-orch'

    aksClusterName: 'akscc-orch' // let's not use demo base name in cluster name
    computeSKU: computeSKU
    computeNodes: 4
    datastoreName: 'datastore_orchestrator' // let's not use demo base name

    // set R/W permissions for orchestrator UAI towards orchestrator storage
    applyDefaultPermissions: true
  }
  dependsOn: [
    workspace
  ]
}

var siloCount = length(siloRegions)

// Create all silos using a provided bicep module
module silos './modules/fl_pairs/open_aks_with_confcomp_storage_pair.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-openccpair-silo-${i}'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion
    pairRegion: siloRegions[i]
    tags: tags

    pairBaseName: '${demoBaseName}-silo${i}-${siloRegions[i]}'

    aksClusterName: 'akscc-silo${i}-${siloRegions[i]}' // let's not use demo base name
    computeSKU: computeSKU
    computeNodes: 4
    datastoreName: 'datastore_silo${i}_${siloRegions[i]}' // let's not use demo base name

    // set R/W permissions for orchestrator UAI towards orchestrator storage
    applyDefaultPermissions: true
  }
  dependsOn: [
    workspace
  ]
}]

// Set R/W permissions for silo identity towards (eyes-on) orchestrator storage
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
