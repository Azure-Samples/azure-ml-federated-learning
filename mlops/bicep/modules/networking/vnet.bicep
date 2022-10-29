// Creates a virtual network

targetScope = 'resourceGroup'

@description('Azure region of the deployment')
param location string = resourceGroup().location

@description('Name of the virtual network resource')
param virtualNetworkName string

@description('Group ID of the network security group')
param networkSecurityGroupId string

@description('Virtual network address prefix')
param vnetAddressPrefix string = '192.168.0.0/16'

@description('Training subnets names and address prefix')
param subnets array = [
  {
    name: 'snet-training'
    addressPrefix: '192.168.0.0/24'
  }
]

@description('List of service endpoints expected on this vnet')
param serviceEndpoints array = [
  'Microsoft.KeyVault'
  'Microsoft.ContainerRegistry'
  'Microsoft.Storage'
]

@description('Tags to add to the resources')
param tags object = {}

var serviceEndpointsDefinition = [for service in serviceEndpoints: { service: service }]
var subnetsDefinition = [for subnet in subnets: {
  name: subnet.name
  properties: {
    addressPrefix: subnet.addressPrefix
    privateEndpointNetworkPolicies: 'Disabled'
    privateLinkServiceNetworkPolicies: 'Disabled'
    serviceEndpoints: serviceEndpointsDefinition
    networkSecurityGroup: {
      id: networkSecurityGroupId
    }
  }
}]

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2022-01-01' = {
  name: virtualNetworkName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        vnetAddressPrefix
      ]
    }
    subnets: subnetsDefinition
  }
}

output id string = virtualNetwork.id
output name string = virtualNetwork.name
