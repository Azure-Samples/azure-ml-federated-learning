// Creates a storage account, private endpoints and DNS zones
@description('Name of the storage account')
param storageName string

@description('Azure region of the storage to create')
param storageRegion string

@allowed([
  'Standard_LRS'
  'Standard_ZRS'
  'Standard_GRS'
  'Standard_GZRS'
  'Standard_RAGRS'
  'Standard_RAGZRS'
  'Premium_LRS'
  'Premium_ZRS'
])
@description('Storage SKU')
param storageSKU string = 'Standard_LRS'

@description('Resource ID of the virtual network')
param virtualNetworkId string

@description('Resource ID of the subnets allowed into this storage')
param subnetId string

@description('Name of the storage blob private link endpoint')
param storagePleBlobName string

@description('Name of the storage file private link endpoint')
param storagePleFileName string

@description('Tags to add to the resources')
param tags object = {}

var storageNameCleaned = replace(storageName, '-', '')
var storageAccountCleanName = substring(storageNameCleaned, 0, min(length(storageNameCleaned),24))


resource storage 'Microsoft.Storage/storageAccounts@2022-05-01' = {
  name: storageAccountCleanName
  location: storageRegion
  tags: tags
  sku: {
    name: storageSKU
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'

    // Allow or disallow public access to all blobs or containers in the storage account.
    allowBlobPublicAccess: false

    // Allow or disallow cross AAD tenant object replication.
    allowCrossTenantReplication: false

    // Restrict copy to and from Storage Accounts within an AAD tenant or with Private Links to the same VNet.
    allowedCopyScope: 'PrivateLink'

    // Indicates whether the storage account permits requests to be authorized with the account access key via Shared Key.
    allowSharedKeyAccess: true

    // Allow or disallow public network access to Storage Account.
    publicNetworkAccess: 'Disabled'

    // Network rule set
    networkAcls: {
      // Specifies whether traffic is bypassed for Logging/Metrics/AzureServices.
      // Possible values are any combination of Logging,Metrics,AzureServices
      // (For example, "Logging, Metrics"), or None to bypass none of those traffics.
      bypass: 'AzureServices'

      // Specifies the default action of allow or deny when no other rules match.
      defaultAction: 'Deny'
    }

    // Encryption settings to be used for server-side encryption for the storage account.
    encryption: {
      keySource: 'Microsoft.Storage'
      requireInfrastructureEncryption: false
      services: {
        blob: {
          enabled: true
          keyType: 'Account'
        }
        file: {
          enabled: true
          keyType: 'Account'
        }
        queue: {
          enabled: true
          keyType: 'Service'
        }
        table: {
          enabled: true
          keyType: 'Service'
        }
      }
    }
    
    // Account HierarchicalNamespace enabled if sets to true.
    isHnsEnabled: false

    // NFS 3.0 protocol support enabled if set to true.
    isNfsV3Enabled: false

    // Enables local users feature, if set to true
    isLocalUserEnabled: false

    // Enables Secure File Transfer Protocol, if set to true
    isSftpEnabled: false

    // KeyPolicy assigned to the storage account.
      keyPolicy: {
      keyExpirationPeriodInDays: 7
    }

    // Allow large file shares if sets to Enabled. It cannot be disabled once it is enabled.
    largeFileSharesState: 'Disabled'

    // Set the minimum TLS version to be permitted on requests to storage.
    minimumTlsVersion: 'TLS1_2'

    // Allows https traffic only to storage service if sets to true. 
    supportsHttpsTrafficOnly: true
  }
}

var blobPrivateDnsZoneName = 'privatelink.blob.${environment().suffixes.storage}'

var filePrivateDnsZoneName = 'privatelink.file.${environment().suffixes.storage}'

resource storagePrivateEndpointBlob 'Microsoft.Network/privateEndpoints@2022-01-01' = {
  name: storagePleBlobName
  location: storageRegion
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

resource storagePrivateEndpointFile 'Microsoft.Network/privateEndpoints@2022-01-01' = {
  name: storagePleFileName
  location: storageRegion
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

resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: blobPrivateDnsZoneName
  location: 'global'
}

resource privateEndpointDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2022-01-01' = {
  name: '${storagePrivateEndpointBlob.name}/blob-PrivateDnsZoneGroup'
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

resource filePrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: filePrivateDnsZoneName
  location: 'global'
}

resource filePrivateEndpointDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2022-01-01' = {
  name: '${storagePrivateEndpointFile.name}/flie-PrivateDnsZoneGroup'
  properties:{
    privateDnsZoneConfigs: [
      {
        name: filePrivateDnsZoneName
        properties:{
          privateDnsZoneId: filePrivateDnsZone.id
        }
      }
    ]
  }
}

resource filePrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  name: '${filePrivateDnsZone.name}/${uniqueString(storage.id)}'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetworkId
    }
  }
}

output storageId string = storage.id
