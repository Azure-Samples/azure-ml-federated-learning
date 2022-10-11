// Creates a storage account, private endpoints and DNS zones
@description('Azure region of the deployment')
param location string

@description('Tags to add to the resources')
param tags object = {}

@description('Service ID')
param privateLinkServiceId string

@description('Name of the storage blob private link endpoint')
param storagePleRootName string

@description('Resource ID of the vnet')
param virtualNetworkId string

@description('Resource ID of the subnet')
param subnetId string

@description('Name of the DNS zone')
param privateDNSZoneName string

@description('Resource ID of the DNS zone group')
param privateDNSZoneId string

@description('Name of the DNS zone groups to add to the private endpoint')
param groupIds array


resource servicePrivateEndpoint 'Microsoft.Network/privateEndpoints@2022-01-01' = {
  name: storagePleRootName
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [ for groupId in groupIds: {
      name: storagePleRootName
      properties: {
        groupIds: [ groupId ]
        privateLinkServiceId: privateLinkServiceId
        privateLinkServiceConnectionState: {
          status: 'Approved'
          description: 'Auto-Approved'
          actionsRequired: 'None'
        }
      }
  }]
    subnet: {
      id: subnetId
    }
  }
}

resource privateEndpointDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2020-06-01' = [for groupId in groupIds: {
  name: '${servicePrivateEndpoint.name}/${groupId}-PrivateDnsZoneGroup'
  //name: '${storagePrivateEndpointBlob.name}/blob-${uniqueString(storage.id)}-PrivateDnsZoneGroup'
  properties:{
    privateDnsZoneConfigs: [
      {
        name: privateDNSZoneName
        properties:{
          privateDnsZoneId: privateDNSZoneId
        }
      }
    ]
  }
}]

resource privateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = [for groupId in groupIds: {
  name: '${privateDNSZoneName}/${uniqueString(subnetId, privateLinkServiceId, groupId)}'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetworkId
    }
  }
}]

output endpoint string = servicePrivateEndpoint.id
