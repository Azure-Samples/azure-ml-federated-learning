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

@description('Training subnet address prefix')
param trainingSubnetPrefix string = '192.168.0.0/24'

@description('Training subnet name')
param trainingSubnetName string = 'snet-training'

@description('Scoring subnet address prefix')
param scoringSubnetPrefix string = '192.168.1.0/24'

@description('Scoring subnet name')
param scoringSubnetName string = 'snet-scoring'

@description('Tags to add to the resources')
param tags object = {}


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
    subnets: [
      { 
        name: trainingSubnetName
        properties: {
          addressPrefix: trainingSubnetPrefix
          privateEndpointNetworkPolicies: 'Disabled'
          privateLinkServiceNetworkPolicies: 'Disabled'
          networkSecurityGroup: {
            id: networkSecurityGroupId
          }
        }
      }
      { 
        name: scoringSubnetName
        properties: {
          addressPrefix: scoringSubnetPrefix
          privateEndpointNetworkPolicies: 'Disabled'
          privateLinkServiceNetworkPolicies: 'Disabled'
          serviceEndpoints: [
            {
              service: 'Microsoft.KeyVault'
            }
            {
              service: 'Microsoft.ContainerRegistry'
            }
            {
              service: 'Microsoft.Storage'
            }
          ]
          networkSecurityGroup: {
            id: networkSecurityGroupId
          }
        }
      }
    ]
  }
}

output id string = virtualNetwork.id
output name string = virtualNetwork.name
