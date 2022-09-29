// Creates a storage account, private endpoints and DNS zones
@description('Azure region of the deployment')
param location string

@description('Tags to add to the resources')
param tags object = {}

@description('Service ID')
param privateLinkServiceId string

@description('Name of the storage blob private link endpoint')
param storagePleRootName string

@description('Resource ID of the subnet')
param subnetId string

@description('Resource ID of the virtual network')
// param virtualNetworkId string

param groupIds array

// var blobPrivateDnsZoneName = 'privatelink.blob.${storageName}.${environment().suffixes.storage}'

resource servicePrivateEndpoint 'Microsoft.Network/privateEndpoints@2020-06-01' = {
  name: storagePleRootName
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [
      {
        name: storagePleRootName
        properties: {
          groupIds: groupIds
          privateLinkServiceId: privateLinkServiceId
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'Auto-Approved'
            actionsRequired: 'None'
          }
        }
      }
    ]
    subnet: {
      id: subnetId
    }
  }
}

// resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
//   name: blobPrivateDnsZoneName
// }

// resource privateEndpointDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2020-06-01' = {
//   //name: '${storagePrivateEndpointBlob.name}/blob-PrivateDnsZoneGroup'
//   name: '${storagePrivateEndpointBlob.name}/blob-${uniqueString(storage.id)}-PrivateDnsZoneGroup'
//   properties:{
//     privateDnsZoneConfigs: [
//       {
//         name: blobPrivateDnsZoneName
//         properties:{
//           privateDnsZoneId: blobPrivateDnsZone.id
//         }
//       }
//     ]
//   }
// }

// resource blobPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
//   name: '${blobPrivateDnsZone.name}/${uniqueString(storage.id)}'
//   location: 'global'
//   properties: {
//     registrationEnabled: false
//     virtualNetwork: {
//       id: virtualNetworkId
//     }
//   }
// }


output endpoint string = servicePrivateEndpoint.id
