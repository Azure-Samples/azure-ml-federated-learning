
// This BICEP script will provision a compute having access to given storage accounts.

// IMPORTANT: This setup is intended only for demo purpose.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

@description('Name of the storage accounts that are to be linked to the compute')
param storageAccountNames array 

@description('Name of the compute cluster')
param computeName string

@description('Specifies the location of the compute.')
param computeRegion string = resourceGroup().location

@description('VM size for the compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the compute cluster')
param computeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param identityType string = 'UserAssigned'

@description('Name of the UAI for the pair compute cluster (if identityType==UserAssigned)')
param uaiName string = 'uai-${computeName}'

@description('Tags to curate the resources in Azure.')
param tags object = {}

// create new Azure ML compute
module computeDeployment '../../../../mlops/bicep/modules/computes/open_new_aml_compute.bicep' = {
  name: '${computeName}-aml-compute'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion
    computeName: computeName
    computeRegion: computeRegion
    computeSKU: computeSKU
    computeNodes: computeNodes
    computeIdentityType: identityType
    computeUaiName: uaiName
    tags: tags
  }
}

// Set R/W permissions
module pairInternalPermissions '../../../../mlops/bicep/modules/permissions/msi_storage_rw.bicep' = [ for storageAccountName in storageAccountNames: {
  name: '${computeName}-${storageAccountName}-rw-perms'
  scope: resourceGroup()
  params: {
    storageAccountName: storageAccountName
    identityPrincipalId: computeDeployment.outputs.identityPrincipalId
  }
  dependsOn: [
    computeDeployment
  ]
}]
