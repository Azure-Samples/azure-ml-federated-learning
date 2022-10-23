// This BICEP script will provision a compute+storage pair
// in a given AzureML workspace, using a vnet and subnet to secure
// the communication between compute and storage, plus managed identity
// for permissions management.

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

@description('Name of the default compute cluster for the pair')
param computeName string = 'compute-${pairBaseName}'

@description('VM size for the default compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param identityType string = 'UserAssigned'

@description('Name of the UAI for the pair compute cluster (if identityType==UserAssigned)')
param uaiName string = 'uai-${pairBaseName}'

@description('Name of the existing private DNS zone for storage in this resource group')
param privateStorageDnsZoneName string = 'privatelink.blob.${environment().suffixes.storage}'

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${pairBaseName}'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${pairBaseName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Subnet address prefix')
param subnetPrefix string = '10.0.0.0/24'

@description('Subnet name')
param subnetName string = 'snet-training'

@description('Allow other subnets into the storage (need to be in the same region)')
param allowedSubnetIds array = []

@description('Enable compute node public IP')
param enableNodePublicIp bool = true

@allowed(['Enabled','vNetOnly','Disabled'])
@description('Allow or disallow public network access to Storage Account.')
param storagePublicNetworkAccess string = 'Disabled'

@description('Allow compute cluster to access storage account with R/W permissions (using UAI)')
param applyDefaultPermissions bool = true

@description('Tags to curate the resources in Azure.')
param tags object = {}

// create new Azure ML compute
module computeDeployment '../computes/vnet_new_aml_compute.bicep' = {
  name: '${pairBaseName}-vnet-aml-compute'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion

    // compute
    computeName: computeName
    computeRegion: pairRegion
    computeSKU: computeSKU
    computeNodes: computeNodes

    // identity
    computeIdentityType: identityType
    computeUaiName: uaiName

    // networking
    nsgResourceName: nsgResourceName
    vnetResourceName: vnetResourceName
    vnetAddressPrefix: vnetAddressPrefix
    subnetPrefix: subnetPrefix
    subnetName: subnetName
    enableNodePublicIp: enableNodePublicIp

    tags: tags
  }
}

// create new blob storage and datastore
module storageDeployment '../storages/new_blob_storage_datastore.bicep' = {
  name: '${pairBaseName}-vnet-storage'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion
    storageName: storageAccountName
    storageRegion: pairRegion
    datastoreName: datastoreName
    publicNetworkAccess: storagePublicNetworkAccess
    subnetIds: concat(
      ['${computeDeployment.outputs.vnetId}/subnets/${computeDeployment.outputs.vnetName}'],
      allowedSubnetIds
    )
    tags: tags
  }
}

// Look for existing private DNS zone for all our private endpoints
resource privateStorageDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: privateStorageDnsZoneName
}

// Create a private service endpoints internal to each pair for their respective storages
module pairStoragePrivateEndpoint '../networking/private_endpoint.bicep' = if (storagePublicNetworkAccess == 'Disabled') {
  name: '${pairBaseName}-endpoint-to-instorage'
  scope: resourceGroup()
  params: {
    location: pairRegion
    tags: tags
    privateLinkServiceId: storageDeployment.outputs.storageId
    storagePleRootName: 'ple-${storageDeployment.outputs.storageName}-to-${pairBaseName}-st-blob'
    subnetId: '${computeDeployment.outputs.vnetId}/subnets/${computeDeployment.outputs.subnetName}'
    virtualNetworkId: computeDeployment.outputs.vnetId
    privateDNSZoneName: privateStorageDnsZone.name
    privateDNSZoneId: privateStorageDnsZone.id
    groupIds: [
      'blob'
      //'file'
    ]
  }
  dependsOn: [
    storageDeployment
  ]
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
output vnetName string = computeDeployment.outputs.vnetName
output vnetId string = computeDeployment.outputs.vnetId
output subnetId string = computeDeployment.outputs.subnetId
