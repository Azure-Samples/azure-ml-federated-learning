// Connects the main dependent resources of a given AzureML workspace
// to a given virtual network and subnet using Private Endpoints
// NOTE: private DNS Zones need to exist

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Machine learning workspace name')
param machineLearningName string

// optional parameters
@description('Specifies the location of the Azure Machine Learning workspace and dependent resources.')
param pleRegion string

@description('Name of virtual network to add the PLEs')
param virtualNetworkName string

@description('Subnet name in which to add the PLEs')
param subnetName string

@description('Virtual network ID for the workspace resources PLEs')
param virtualNetworkId string = '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.Network/virtualNetworks/${virtualNetworkName}'

@description('Whether to link the private DNS zones to the virtual network')
param linkKeyvaultDnsToVirtualNetwork bool = true

param createKeyVaultPLE bool = true

@description('Static IP address for the KeyVault PLE')
param keyVaultPLEStaticIP string = ''

@description('Whether to link the private DNS zones to the virtual network')
param linkAcrDnsToVirtualNetwork bool = true

param createAcrPLE bool = true

@description('Static IP address for the ACR PLE (needs 1 for registy, one for data)')
param acrPLEStaticIP string = ''

@description('Whether to link the private DNS zones to the virtual network')
param linkBlobDnsToVirtualNetwork bool = true

param createBlobPLE bool = true

@description('Static IP address for the workspace blob PLE')
param blobPLEStaticIP string = ''

@description('Tags to curate the resources in Azure.')
param tags object = {}

resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2022-12-01-preview' existing = {
  name: machineLearningName
}
var subnetId = '${virtualNetworkId}/subnets/${subnetName}'


var keyVaultPrivateDnsZoneName = 'privatelink${environment().suffixes.keyvaultDns}'
resource keyVaultPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: keyVaultPrivateDnsZoneName
}
resource keyVaultPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = if (linkKeyvaultDnsToVirtualNetwork) {
  name: uniqueString(virtualNetworkId, keyVaultPrivateDnsZoneName, pleRegion)
  parent: keyVaultPrivateDnsZone
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetworkId
    }
  }
}
module keyVaultPrivateEndpoint '../networking/private_endpoint.bicep' = if (createKeyVaultPLE) {
  name: 'ple-${machineLearningName}-kv-in-vnet-${virtualNetworkName}-deploy'
  scope: resourceGroup()
  params: {
    tags: tags
    location: pleRegion
    resourceServiceId: machineLearning.properties.keyVault
    pleRootName: 'ple-${machineLearningName}-kv-in-vnet-${virtualNetworkName}'
    subnetId: subnetId
    privateDNSZoneName: keyVaultPrivateDnsZone.name
    groupId: 'vault'
    memberNames: [ 'default' ]
    useStaticIPAddress: !empty(keyVaultPLEStaticIP)
    privateIPAddress: keyVaultPLEStaticIP
  }
  dependsOn:[
    keyVaultPrivateDnsZoneVnetLink
  ]
}


var acrPrivateDnsZoneName = 'privatelink${environment().suffixes.acrLoginServer}'
resource acrPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: acrPrivateDnsZoneName
}
resource acrPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = if (linkAcrDnsToVirtualNetwork) {
  name: uniqueString(virtualNetworkId, acrPrivateDnsZoneName, pleRegion)
  parent: acrPrivateDnsZone
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetworkId
    }
  }
}
module acrPrivateEndpoint '../networking/private_endpoint.bicep' = if (createAcrPLE) {
  name: 'ple-${machineLearningName}-cr-in-vnet-${virtualNetworkName}-deploy'
  scope: resourceGroup()
  params: {
    tags: tags
    location: pleRegion
    resourceServiceId: machineLearning.properties.containerRegistry
    pleRootName: 'ple-${machineLearningName}-cr-in-vnet-${virtualNetworkName}'
    subnetId: subnetId
    privateDNSZoneName: acrPrivateDnsZoneName
    groupId: 'registry'
    memberNames: [ 'registry', 'registry_data_${machineLearning.location}']
    useStaticIPAddress: !empty(acrPLEStaticIP)
    privateIPAddress: acrPLEStaticIP
  }
  dependsOn:[
    acrPrivateDnsZoneVnetLink
  ]
}


var blobPrivateDnsZoneName = 'privatelink.blob.${environment().suffixes.storage}'
resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: blobPrivateDnsZoneName
}
resource blobPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = if (linkBlobDnsToVirtualNetwork) {
  name: uniqueString(virtualNetworkId, blobPrivateDnsZoneName, pleRegion)
  parent: blobPrivateDnsZone
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetworkId
    }
  }
}
module blobPrivateEndpoint '../networking/private_endpoint.bicep' = if (createBlobPLE) {
  name: 'ple-${machineLearningName}-blob-in-vnet-${virtualNetworkName}-ple'
  scope: resourceGroup()
  params: {
    tags: tags
    location: pleRegion
    resourceServiceId: machineLearning.properties.storageAccount
    pleRootName: 'ple-${machineLearningName}-blob-in-vnet-${virtualNetworkName}-ple'
    subnetId: subnetId
    privateDNSZoneName: blobPrivateDnsZoneName
    groupId: 'blob'
    useStaticIPAddress: !empty(blobPLEStaticIP)
    privateIPAddress: blobPLEStaticIP
  }
  dependsOn: [
    blobPrivateDnsZoneVnetLink
  ]
}
