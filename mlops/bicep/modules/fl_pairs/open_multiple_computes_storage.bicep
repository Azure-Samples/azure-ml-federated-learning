// This BICEP script will provision a compute+storage pair
// in a given AzureML workspace, without any specific security settings.

// IMPORTANT: This setup is intended only for demo purpose. The data is still accessible
// by the users when opening the storage accounts, and data exfiltration is easy.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

@description('Specifies the location of the pair resources.')
param pairRegion string = resourceGroup().location

@description('Base name used for creating all pair resources.')
param pairBaseName string

@description('Name of the storage account resource to create for the pair')
param storageAccountName string = replace('st${pairBaseName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = replace('datastore_${pairBaseName}','-','_')

@description('Name of the default cpu compute cluster for the pair')
param cpuComputeName string = 'cpu-cluster-${pairBaseName}'

@description('VM size for the cpu compute cluster')
param cpuComputeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the cpu compute cluster')
param cpuComputeNodes int = 4

@description('Name of the default gpu compute cluster for the pair')
param gpuComputeName string = 'gpu-cluster-${pairBaseName}'

@description('VM size for the gpu compute cluster')
param gpuComputeSKU string = 'Standard_NC6'

@description('VM nodes for the gpu compute cluster')
param gpuComputeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param identityType string = 'UserAssigned'

@description('Name of the UAI for the pair compute cluster (if identityType==UserAssigned)')
param cpuUaiName string = 'uai-${pairBaseName}-cpu'

@description('Name of the UAI for the pair compute cluster (if identityType==UserAssigned)')
param gpuUaiName string = 'uai-${pairBaseName}-gpu'

@description('Allow compute cluster to access storage account with R/W permissions (using UAI)')
param applyDefaultPermissions bool = true

@description('Tags to curate the resources in Azure.')
param tags object = {}

// create new blob storage and datastore
module storageDeployment '../storages/new_blob_storage_datastore.bicep' = {
  name: '${pairBaseName}-open-storage'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion
    storageName: storageAccountName
    storageRegion: pairRegion
    datastoreName: datastoreName
    publicNetworkAccess: 'Enabled'
    tags: tags
  }
}

// create new Azure ML compute
module cpuComputeDeployment '../computes/open_new_aml_compute.bicep' = {
  name: '${pairBaseName}-cpu-open-aml-compute'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion
    computeName: cpuComputeName
    computeRegion: pairRegion
    computeSKU: cpuComputeSKU
    computeNodes: cpuComputeNodes
    computeIdentityType: identityType
    computeUaiName: cpuUaiName
    tags: tags
  }
}

// create new Azure ML compute
module gpuComputeDeployment '../computes/open_new_aml_compute.bicep' = {
  name: '${pairBaseName}-gpu-open-aml-compute'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion
    computeName: gpuComputeName
    computeRegion: pairRegion
    computeSKU: gpuComputeSKU
    computeNodes: gpuComputeNodes
    computeIdentityType: identityType
    computeUaiName: gpuUaiName
    tags: tags
  }
}

// Set R/W permissions for orchestrator UAI towards orchestrator storage
module pairInternalPermissionsCpu '../permissions/msi_storage_rw.bicep' = if(applyDefaultPermissions) {
  name: '${pairBaseName}-cpu-internal-rw-perms'
  scope: resourceGroup()
  params: {
    storageAccountName: storageDeployment.outputs.storageName
    identityPrincipalId: cpuComputeDeployment.outputs.identityPrincipalId
  }
  dependsOn: [
    storageDeployment
    cpuComputeDeployment
  ]
}

// Set R/W permissions for orchestrator UAI towards orchestrator storage
module pairInternalPermissionsGpu '../permissions/msi_storage_rw.bicep' = if(applyDefaultPermissions) {
  name: '${pairBaseName}-gpu-internal-rw-perms'
  scope: resourceGroup()
  params: {
    storageAccountName: storageDeployment.outputs.storageName
    identityPrincipalId: gpuComputeDeployment.outputs.identityPrincipalId
  }
  dependsOn: [
    storageDeployment
    gpuComputeDeployment
  ]
}

// output the pair config for next actions (permission model)
output identityPrincipalId string = cpuComputeDeployment.outputs.identityPrincipalId
output storageName string = storageDeployment.outputs.storageName
output storageServiceId string = storageDeployment.outputs.storageId
output cpuComputeName string = cpuComputeDeployment.outputs.compute
output gpuComputeName string = gpuComputeDeployment.outputs.compute
output region string = pairRegion
