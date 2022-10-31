// This BICEP script will provision a compute to connect
// to an existing storage located within the same tenant
// attach to given AzureML workspace, using a vnet and subnet to secure
// the communication between compute and storage, plus managed identity
// for permissions management.

// NOTE: setting up R/W permissions with a storage in another sub is
// not possible without tenant-level deployment
// please see tutorial to achieve manually instead.

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

@description('Existing storage account name to attach to the pair.')
param existingStorageAccountName string

@description('Resource group of the existing storage account to attach to the pair.')
param existingStorageAccountResourceGroup string

@description('SubscriptionId of the existing storage account to attach to the pair.')
param existingStorageAccountSubscriptionId string

@description('Name of the storage container resource to create for the pair')
param existingStorageContainerName string = 'private'

@description('Name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = replace('datastore_${pairBaseName}','-','_')

@description('Name of the default compute cluster for the pair')
param computeName string = 'cpu-cluster-${pairBaseName}'

@description('VM size for the default compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param identityType string = 'UserAssigned'

@description('Name of the UAI for the pair compute cluster (if identityType==UserAssigned)')
param uaiName string = 'uai-${pairBaseName}'

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

@description('Enable compute node public IP')
param enableNodePublicIp bool = true

@description('Name of the private DNS zone for blob')
param blobPrivateDNSZoneName string = 'privatelink.blob.${environment().suffixes.storage}'

@description('Location of the private DNS zone for blob')
param blobPrivateDNSZoneLocation string = 'global'

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

// attach existing blob storage and create datastore
module storageDeployment '../storages/existing_blob_storage_datastore.bicep' = {
  name: '${pairBaseName}-attach-existing-storage'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    storageAccountName: existingStorageAccountName
    storageAccountResourceGroup: existingStorageAccountResourceGroup
    storageAccountSubscriptionId: existingStorageAccountSubscriptionId
    storageRegion: pairRegion
    datastoreName: datastoreName
    tags: tags
  }
}

// Create a private service endpoints internal to each pair for their respective storages
module privateEndpoint '../networking/private_endpoint.bicep' = {
  name: '${pairBaseName}-endpoint-to-existing-storage'
  scope: resourceGroup()
  params: {
    location: pairRegion
    tags: tags
    resourceServiceId: storageDeployment.outputs.storageId
    resourceName: storageDeployment.outputs.storageName
    pleRootName: 'ple-remoteaccount-${existingStorageAccountName}-to-${pairBaseName}-st-blob'
    virtualNetworkId: computeDeployment.outputs.vnetId
    subnetId: '${computeDeployment.outputs.vnetId}/subnets/${computeDeployment.outputs.subnetName}'
    privateDNSZoneName: blobPrivateDNSZoneName
    privateDNSZoneLocation: blobPrivateDNSZoneLocation
    groupId: 'blob'
  }
  dependsOn: [
    storageDeployment
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
