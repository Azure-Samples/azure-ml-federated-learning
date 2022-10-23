// This BICEP script will provision an AKS cluster with confidential computes
// a new storage account, attached to a given AzureML workspace, without any specific security settings.

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

@description('The name of the Managed Cluster resource.')
param aksClusterName string = 'aks-${pairBaseName}'

// see https://learn.microsoft.com/en-us/azure/virtual-machines/dcv3-series
@description('VM size for the compute cluster')
param computeSKU string = 'Standard_DC4ds_v3'

@description('VM nodes for the compute cluster')
@minValue(1)
@maxValue(50)
param computeNodes int = 4

@description('Name of the UAI for the pair compute cluster')
param uaiName string = 'uai-${aksClusterName}'

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

module computeDeployment '../computes/open_new_aks_with_confcomp.bicep' = {
  name: '${pairBaseName}-open-aks-confcomp'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion
    aksClusterName: aksClusterName
    amlComputeName: aksClusterName
    computeRegion: pairRegion
    agentVMSize: computeSKU
    agentCount: computeNodes
    computeUaiName: uaiName
    tags: tags
  }
}

// Set R/W permissions for orchestrator UAI towards orchestrator storage
module pairInternalPermissions '../permissions/msi_storage_rw.bicep' = if(applyDefaultPermissions) {
  name: '${pairBaseName}-internal-rw-perms'
  scope: resourceGroup()
  params: {
    storageAccountName: storageDeployment.outputs.storageName
    identityPrincipalId: computeDeployment.outputs.identityPrincipalId
  }
  dependsOn: [
    storageDeployment
    computeDeployment
  ]
}

// output the pair config for next actions (permission model)
output identityPrincipalId string = computeDeployment.outputs.identityPrincipalId
output storageName string = storageDeployment.outputs.storageName
output storageServiceId string = storageDeployment.outputs.storageId
output computeName string = computeDeployment.outputs.compute
output region string = pairRegion
