// Creates a virtual network with service endpoint for Storage

targetScope = 'resourceGroup'

@description('Azure region of the deployment')
param location string = resourceGroup().location

@description('Tags to add to the resources')
param tags object = {}

@description('Name of the virtual network resource')
param virtualNetworkName string

@description('Group ID of the network security group')
param networkSecurityGroupId string

@description('Virtual network address prefix')
param vnetAddressPrefix string = '192.168.0.0/16'

@description('Training subnet address prefix')
param subnetPrefix string = '192.168.0.0/24'

@description('Subnet name')
param subnetName string = 'snet-training'


// @description('Scoring subnet address prefix')
// param scoringSubnetPrefix string = '192.168.1.0/24'

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
        name: subnetName
        properties: {
          addressPrefix: subnetPrefix
          privateEndpointNetworkPolicies: 'Disabled'
          privateLinkServiceNetworkPolicies: 'Disabled'
          networkSecurityGroup: {
            id: networkSecurityGroupId
          }
          serviceEndpoints: [
            {
              service: 'Microsoft.Storage'
            }
          ]
        }
      }
      // { 
      //   name: 'snet-scoring'
      //   properties: {
      //     addressPrefix: scoringSubnetPrefix
      //     privateEndpointNetworkPolicies: 'Disabled'
      //     privateLinkServiceNetworkPolicies: 'Disabled'
      //     serviceEndpoints: [
      //       {
      //         service: 'Microsoft.KeyVault'
      //       }
      //       {
      //         service: 'Microsoft.ContainerRegistry'
      //       }
      //       {
      //         service: 'Microsoft.Storage'
      //       }
      //     ]
      //     networkSecurityGroup: {
      //       id: networkSecurityGroupId
      //     }
      //   }
      // }
    ]
  }
}

output id string = virtualNetwork.id
output name string = virtualNetwork.name
