// Creates a storage account, private endpoints and DNS zones
@description('Azure region of the deployment')
param location string

@description('Tags to add to the resources')
param tags object

@description('Name of the existing storage account')
param storageName string

@description('Name of the storage blob private link endpoint')
param storagePleBlobName string = 'ple-${storageName}-st-blob'

@description('Name of the storage file private link endpoint')
param storagePleFileName string = 'ple-${storageName}-st-file'

@description('Resource ID of the subnet')
param subnetId string

@description('Resource ID of the virtual network')
param virtualNetworkId string

var blobPrivateDnsZoneName = 'privatelink.blob.${storageName}.${environment().suffixes.storage}'

resource storage 'Microsoft.Storage/storageAccounts@2021-04-01' existing = {
  name: storageName
  scope: resourceGroup()
}

resource storagePrivateEndpointBlob 'Microsoft.Network/privateEndpoints@2020-06-01' = {
  name: storagePleBlobName
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [
      { 
        name: storagePleBlobName
        properties: {
          groupIds: [
            'blob'
          ]
          privateLinkServiceId: storage.id
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

resource storagePrivateEndpointFile 'Microsoft.Network/privateEndpoints@2020-06-01' = {
  name: storagePleFileName
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [
      {
        name: storagePleFileName
        properties: {
          groupIds: [
            'file'
          ]
          privateLinkServiceId: storage.id
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

resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: blobPrivateDnsZoneName
}

resource privateEndpointDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2020-06-01' = {
  name: '${storagePrivateEndpointBlob.name}/blob-PrivateDnsZoneGroup'
  //name: '${storagePrivateEndpointBlob.name}/blob-${uniqueString(storage.id)}-PrivateDnsZoneGroup'
  properties:{
    privateDnsZoneConfigs: [
      {
        name: blobPrivateDnsZoneName
        properties:{
          privateDnsZoneId: blobPrivateDnsZone.id
        }
      }
    ]
  }
}

resource blobPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  name: '${blobPrivateDnsZone.name}/${uniqueString(storage.id)}'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetworkId
    }
  }
}


output storageId string = storage.id
