// This BICEP script will provision an AKS cluster with confidential computes
// in a given AzureML workspace, using a vnet and subnet to secure
// the communication between compute and storage, plus managed identity
// for permissions management.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

@description('Specifies the location of the pair resources.')
param pairRegion string = resourceGroup().location

@description('Base name used for creating all pair resources.')
param pairBaseName string

@description('Name of the storage account resource to create for the pair')
param storageAccountName string = replace('st${pairBaseName}','-','') // replace because only alphanumeric characters are supported

param storageContainerName string = 'private'

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

@description('The name of the Managed Cluster resource.')
param computeName string = 'aks-${pairBaseName}'

// see https://learn.microsoft.com/en-us/azure/virtual-machines/dcasv5-dcadsv5-series
@description('VM size for the compute cluster')
param computeSKU string = 'Standard_DC2as_v5'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@description('Name of the UAI for the pair compute cluster')
param uaiName string = 'uai-${pairBaseName}'

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${pairBaseName}'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${pairBaseName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '172.19.0.0/16'

@description('Subnet address prefix')
param computeSubnetPrefix string = '172.19.0.0/24'

@description('Subnet address prefix')
param endpointsSubnetPrefix string = '172.19.1.0/24'

@description('Which static IP to use for storage PLE (if useStorageStaticIP is true)')
param storagePLEStaticIP string = ''

@description('Enable compute node public IP')
param enableNodePublicIp bool = true

@allowed(['Enabled','vNetOnly','Disabled'])
@description('Allow or disallow public network access to Storage Account.')
param storagePublicNetworkAccess string = 'Disabled'

@description('Allow compute cluster to access storage account with R/W permissions (using UAI)')
param applyDefaultPermissions bool = true

@description('Name of the private DNS zone for blob')
param blobPrivateDNSZoneName string = 'privatelink.blob.${environment().suffixes.storage}'

@description('Tags to curate the resources in Azure.')
param tags object = {
  Owner: 'AzureML Samples'
  Project: 'azure-ml-federated-learning'
  Environment: 'dev'
  Toolkit: 'bicep'
  Docs: 'https://github.com/Azure-Samples/azure-ml-federated-learning'
}


// Virtual network and network security group
module nsg '../networking/azureml_compute_nsg.bicep' = {
  name: '${nsgResourceName}-deployment'
  params: {
    location: pairRegion
    nsgName: nsgResourceName
    tags: tags
    workspaceRegion: pairRegion
    enableNodePublicIp: enableNodePublicIp
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: '${vnetResourceName}-deployment'
  params: {
    location: pairRegion
    virtualNetworkName: vnetResourceName
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: vnetAddressPrefix
    subnets: [
      {
        name: 'compute'
        addressPrefix: computeSubnetPrefix
      }
      {
        name: 'endpoints'
        addressPrefix: endpointsSubnetPrefix
      }
    ]
    tags: tags
  }
}

var blobStoragePrivateDnsZoneName = 'privatelink.blob.${environment().suffixes.storage}'

module blobStoragePrivateDnsZone '../networking/private_dns_zone.bicep' = {
  name: '${pairBaseName}-blob-storage-private-dns-zone'
  scope: resourceGroup()
  params: {
    name: blobStoragePrivateDnsZoneName
    location: 'global'
    linkToVirtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

// provision a user assigned identify for this compute
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: uaiName
  location: pairRegion
  tags: tags
}


var userAssignedIdentities = {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uaiName}': {}}

resource aks 'Microsoft.ContainerService/managedClusters@2022-05-02-preview' = {
  name: computeName
  location: pairRegion
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: userAssignedIdentities
  }
  properties: {
    dnsPrefix: replace('${computeName}', '-', '')
    //fqdnSubdomain: 'foo'
    addonProfiles: {
      // enable the provisioning of confidential computes nodes
      ACCSGXDevicePlugin: {
        enabled: true
        config: {
          ACCSGXQuoteHelperEnabled: 'false'
        }
      }
    }
    agentPoolProfiles: [
      {
        name: 'confcompool'
        count: computeNodes
        // enableAutoScaling: true
        // maxCount: 5
        // minCount: 2        

        vmSize: computeSKU
        osType: 'Linux'
        mode: 'System'
        osDiskSizeGB: 0
        vnetSubnetID: '${vnet.outputs.id}/subnets/compute'
      }
    ]
    apiServerAccessProfile: {
      // IMPORTANT: use this for demo only, it is not a private AKS cluster
      authorizedIPRanges: []
      enablePrivateCluster: false
      enablePrivateClusterPublicFQDN: false
      enableVnetIntegration: false
    }
    networkProfile:{
      networkPlugin: 'azure'
    }
  }
}


resource storage 'Microsoft.Storage/storageAccounts@2022-09-01' = {
  name: storageAccountName
  location: pairRegion
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
    allowSharedKeyAccess: false // we're using UAI anyway, no need for shared key access


    // Network rule set
    networkAcls: {
      // Specifies whether traffic is bypassed for Logging/Metrics/AzureServices.
      // Possible values are any combination of Logging,Metrics,AzureServices
      // (For example, "Logging, Metrics"), or None to bypass none of those traffics.
      bypass: 'AzureServices'

      // Specifies the default action of allow or deny when no other rules match.
      // if publicNetworkAccess is Enabled, then default action is Allow
      defaultAction: storagePublicNetworkAccess == 'Enabled' ? 'Allow' : 'Deny'

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
      // virtualNetworkRules : [
      //   {
      //     id: '/subscriptions/${subscription().id}/resourceGroups/${resourceGroup().name}/providers/Microsoft.Network/virtualNetworks/${vnet.name}/subnets/endpoints'
      //     action: 'Allow'
      //   }
      // ]
    }

    // Allow or disallow public network access to Storage Account.
    publicNetworkAccess: storagePublicNetworkAccess == 'Disabled' ? 'Disabled' : 'Enabled'

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
  dependsOn: [
    vnet
  ]
}

// create a "private" container in the storage account
// this one will be readable only by silo compute
resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  name: '${storage.name}/default/${storageContainerName}'
  properties: {
    metadata: {}
    publicAccess: 'None'
  }
  dependsOn: [
    storage
  ]
}

// Create a private service endpoints internal to each pair for their respective storages
module pairStoragePrivateEndpoint '../networking/private_endpoint.bicep' = if (storagePublicNetworkAccess == 'Disabled') {
  name: '${pairBaseName}-endpoint-to-pair-storage'
  scope: resourceGroup()
  params: {
    location: pairRegion
    tags: tags
    resourceServiceId: storage.id
    resourceName: storage.name
    pleRootName: 'ple-${storageAccountName}-to-${pairBaseName}-st-blob'
    virtualNetworkId: vnet.outputs.id
    subnetId: '${vnet.outputs.id}/subnets/endpoints'
    useStaticIPAddress: !empty(storagePLEStaticIP)
    privateIPAddress: storagePLEStaticIP
    privateDNSZoneName: blobPrivateDNSZoneName
    privateDNSZoneLocation: 'global'
    groupId: 'blob'
    linkVirtualNetwork: false // already done during blobStoragePrivateDnsZone creation
  }
  dependsOn: [
    storage
    blobStoragePrivateDnsZone
  ]
}

// Set R/W permissions for orchestrator UAI towards orchestrator storage
module pairInternalPermissions '../permissions/msi_storage_rw.bicep' = if(applyDefaultPermissions) {
  name: '${pairBaseName}-internal-rw-perms'
  scope: resourceGroup()
  params: {
    storageAccountName: storageAccountName
    identityPrincipalId: uai.properties.principalId
  }
  dependsOn: [
    storage
  ]
}
