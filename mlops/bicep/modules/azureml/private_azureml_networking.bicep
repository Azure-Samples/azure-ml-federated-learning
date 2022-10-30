// Creates private endpoints and DNS zones for the azure machine learning workspace
@description('Azure region of the deployment')
param location string

@description('Machine learning workspace private link endpoint name')
param machineLearningPleName string

@description('Resource ID of the virtual network resource')
param virtualNetworkId string

@description('Resource ID of the subnet resource')
param subnetId string

@description('Resource ID of the machine learning workspace')
param workspaceArmId string

@description('Tags to add to the resources')
param tags object

param amlPrivateDnsZoneName string
param amlPrivateDnsZoneLocation string = 'global'
param aznbPrivateDnsZoneName string
param aznbPrivateDnsZoneLocation string = 'global'

resource machineLearningPrivateEndpoint 'Microsoft.Network/privateEndpoints@2022-01-01' = {
  name: machineLearningPleName
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [
      {
        name: machineLearningPleName
        properties: {
          groupIds: [
            'amlworkspace'
          ]
          privateLinkServiceId: workspaceArmId
        }
      }
    ]
    subnet: {
      id: subnetId
    }
  }
}

resource amlPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: amlPrivateDnsZoneName
}

resource amlPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  name: uniqueString(workspaceArmId)
  parent: amlPrivateDnsZone
  location: amlPrivateDnsZoneLocation
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetworkId
    }
  }
}

// Notebook
resource notebookPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: aznbPrivateDnsZoneName
}

resource notebookPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  name: uniqueString(workspaceArmId)
  parent: notebookPrivateDnsZone
  location: aznbPrivateDnsZoneLocation
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetworkId
    }
  }
}

resource privateEndpointDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2022-01-01' = {
  name: '${machineLearningPrivateEndpoint.name}/amlworkspace-PrivateDnsZoneGroup'
  properties:{
    privateDnsZoneConfigs: [
      {
        name: amlPrivateDnsZone.name
        properties:{
          privateDnsZoneId: amlPrivateDnsZone.id
        }
      }
      {
        name: notebookPrivateDnsZone.name
        properties:{
          privateDnsZoneId: notebookPrivateDnsZone.id
        }
      }
    ]
  }
}
