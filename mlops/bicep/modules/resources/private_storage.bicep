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

@description('Name of the private DNS zone for blob')
param blobPrivateDNSZoneName string = 'privatelink.blob.${environment().suffixes.storage}'

@description('Location of the private DNS zone for blob')
param blobPrivateDNSZoneLocation string = 'global'

@description('Name of the private DNS zone for file')
param filePrivateDNSZoneName string = 'privatelink.file.${environment().suffixes.storage}'

@description('Location of the private DNS zone for file')
param filePrivateDNSZoneLocation string = 'global'

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

module blobPrivateEndpoint '../networking/private_endpoint.bicep' = {
  name: '${storage.name}-endpoint-to-vnet'
  scope: resourceGroup()
  params: {
    tags: tags
    location: storage.location
    resourceServiceId: storage.id
    resourceName: storage.name
    virtualNetworkId: virtualNetworkId
    subnetId: subnetId
    privateDNSZoneName: blobPrivateDNSZoneName
    privateDNSZoneLocation: blobPrivateDNSZoneLocation
    groupId: 'blob'
  }
}

module filePrivateEndpoint '../networking/private_endpoint.bicep' = {
  name: '${storage.name}-endpoint-to-vnet'
  scope: resourceGroup()
  params: {
    tags: tags
    location: storage.location
    resourceServiceId: storage.id
    resourceName: storage.name
    virtualNetworkId: virtualNetworkId
    subnetId: subnetId
    privateDNSZoneName: filePrivateDNSZoneName
    privateDNSZoneLocation: filePrivateDNSZoneLocation
    groupId: 'file'
  }
}


output storageId string = storage.id
