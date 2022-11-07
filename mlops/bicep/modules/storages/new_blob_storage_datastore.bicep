// Creates a storage account, private endpoints and DNS zones
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

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

@description('Name of the storage container resource to create for the pair')
param containerName string = 'private'

@description('Name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = replace('datastore_${storageName}','-','_')

@description('Resource ID of the subnets allowed into this storage')
param subnetIds array = []

@allowed(['Enabled','vNetOnly','Disabled'])
@description('Allow or disallow public network access to Storage Account.')
param publicNetworkAccess string = 'Disabled' // for Disabled, you'd need to create private endpoints

@description('Tags to add to the resources')
param tags object = {}

var storageNameCleaned = replace(storageName, '-', '')
var storageAccountCleanName = substring(storageNameCleaned, 0, min(length(storageNameCleaned),24))

// settings depending on publicNetworkAccess
// no need for subnetIds if publicNetworkAccess is Enabled
var storageAllowedSubnetIds = publicNetworkAccess == 'Enabled' ? [] : subnetIds
// if publicNetworkAccess is Enabled, then default action is Allow
var storagedefaultAction = publicNetworkAccess == 'Enabled' ? 'Allow' : 'Deny'
// vNetOnly is just a specific case of Enabled
var storagepublicNetworkAccess = publicNetworkAccess == 'Disabled' ? 'Disabled' : 'Enabled'

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


    // Network rule set
    networkAcls: {
      // Specifies whether traffic is bypassed for Logging/Metrics/AzureServices.
      // Possible values are any combination of Logging,Metrics,AzureServices
      // (For example, "Logging, Metrics"), or None to bypass none of those traffics.
      bypass: 'AzureServices'

      // Specifies the default action of allow or deny when no other rules match.
      defaultAction: storagedefaultAction

      // Sets the IP ACL rules
      // ipRules

      // Sets the resource access rules
      resourceAccessRules: [
        // NOTE: keeping this here for now, to use as reference until we figure out the appropriate settings.
        // {
        //   resourceId: resourceId('Microsoft.Storage/storageAccounts/blobServices', storageAccountCleanName, 'default')
        //   tenantId: tenant().tenantId
        // }
        // {
        //   resourceId: resourceId('Microsoft.Storage/storageAccounts/fileServices', storageAccountCleanName, 'default')
        //   tenantId: tenant().tenantId
        // }
      ]

      // Sets the virtual network rules
      virtualNetworkRules : [ for subnet in storageAllowedSubnetIds: {
          id: subnet
          action: 'Allow'
      }]
    }

    // Allow or disallow public network access to Storage Account.
    publicNetworkAccess: storagepublicNetworkAccess

    // 	Maintains information about the network routing choice opted by the user for data transfer
    routingPreference: {
      // Routing Choice defines the kind of network routing opted by the user.
      routingChoice: 'MicrosoftRouting'

      // A boolean flag which indicates whether internet routing storage endpoints are to be published
      // publishInternetEndpoints: false

      // A boolean flag which indicates whether microsoft routing storage endpoints are to be published
      publishMicrosoftEndpoints: true
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

    // ???
    // customDomain    
  }
}

// create a "private" container in the storage account
// this one will be readable only by silo compute
resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  name: '${storage.name}/default/${containerName}'
  properties: {
    metadata: {}
    publicAccess: 'None'
  }
  dependsOn: [
    storage
  ]
}

// attach as a datastore in AzureML
resource datastore 'Microsoft.MachineLearningServices/workspaces/datastores@2022-06-01-preview' = {
  name: '${machineLearningName}/${datastoreName}'
  properties: {
    tags: tags
    credentials: {
      credentialsType: 'None'
      // For remaining properties, see DatastoreCredentials objects
    }
    description: 'Private storage in region ${storageRegion}'
    properties: {}
    datastoreType: 'AzureBlob'
    // For remaining properties, see DatastoreProperties objects
    accountName: storage.name
    containerName: containerName
    // endpoint: 'string'
    // protocol: 'string'
    resourceGroup: resourceGroup().name
    // serviceDataAccessAuthIdentity: 'string'
    subscriptionId: subscription().subscriptionId
  }
  dependsOn: [
    container
  ]
}

// output storage references
output storageId string = storage.id
output storageName string = storage.name
output containerName string = container.name
output datastoreName string = datastore.name
