// This BICEP script will provision an AzureML workspace
// with private endpoint to dependent resources
// To access and operate this workspace, you will need a VM inside this vnet.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Base name to create all the resources')
@minLength(2)
@maxLength(20)
param baseName string

@description('Machine learning workspace name')
param machineLearningName string = 'aml-${baseName}'

// optional parameters
@description('Machine learning workspace display name')
param machineLearningFriendlyName string = '${baseName} (private)'

@description('Machine learning workspace description')
param machineLearningDescription string = 'An AzureML workspace behind a private link endpoint.'

@description('Specifies the location of the Azure Machine Learning workspace and dependent resources.')
param location string = resourceGroup().location

@description('Virtual network ID for the workspace')
param virtualNetworkId string

@description('Subnet name for the workspace')
param subnetName string

@description('Name of the container registry resource')
param containerRegistryName string = replace('cr-${baseName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the workspace secrets store (shared keyvault) resource')
param keyVaultName string = 'ws-shkv-${baseName}'

@description('Name of the workspace storage account resource')
param storageAccountName string = replace('st-${baseName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the cluster used to build images for custom environments (required for private workspaces)')
param imageBuildComputeName string = 'image-build-compute'

@description('VM size for cluster used to build images')
param imageBuildComputeSKU string = 'Standard_DS3_v2'

@description('VM nodes for cluster used to build images')
param imageBuildComputeNodes int = 2

@description('Provision new private DNS zones for the workspace dependencies')
param createPrivateDNSZones bool = true

@description('WARNING: Enable access to the AzureML workspace using public azure portal (for debugging)')
@allowed([
  'Enabled'
  'Disabled'
])
param workspacePublicNetworkAccess string = 'Disabled'

@description('Tags to curate the resources in Azure.')
param tags object = {}


// ************************************************************
// Dependent resources for the Azure Machine Learning workspace
// ************************************************************

var blobStoragePrivateDnsZoneName = 'privatelink.blob.${environment().suffixes.storage}'

module blobStoragePrivateDnsZone '../networking/private_dns_zone.bicep' = if (createPrivateDNSZones) {
  name: '${baseName}-blob-storage-private-dns-zone'
  scope: resourceGroup()
  params: {
    name: blobStoragePrivateDnsZoneName
    location: 'global'
    linkToVirtualNetworkId: virtualNetworkId
    tags: tags
  }
}
var fileStoragePrivateDnsZoneName = 'privatelink.file.${environment().suffixes.storage}'
module fileStoragePrivateDnsZone '../networking/private_dns_zone.bicep' = if (createPrivateDNSZones) {
  name: '${baseName}-file-storage-private-dns-zone'
  scope: resourceGroup()
  params: {
    name: fileStoragePrivateDnsZoneName
    location: 'global'
    linkToVirtualNetworkId: virtualNetworkId
    tags: tags
  }
}

module storage '../resources/private_storage.bicep' = {
  name: 'st${baseName}-deployment'
  scope: resourceGroup()
  params: {
    storageRegion: location
    storageName: storageAccountName
    storageSKU: 'Standard_LRS'
    subnetId: '${virtualNetworkId}/subnets/${subnetName}'
    blobPrivateDNSZoneName: blobStoragePrivateDnsZoneName
    filePrivateDNSZoneName: fileStoragePrivateDnsZoneName
    // if workspace is accessible through public API,
    // default storage should also be the same,
    // to allow for component uploads
    publicNetworkAccess: workspacePublicNetworkAccess
    tags: tags
  }
  dependsOn: [
    blobStoragePrivateDnsZone
    fileStoragePrivateDnsZone
  ]
}

var keyVaultPrivateDnsZoneName = 'privatelink${environment().suffixes.keyvaultDns}'
module keyVaultPrivateDnsZone '../networking/private_dns_zone.bicep' = if (createPrivateDNSZones) {
  name: '${baseName}-kv-private-dns-zone'
  scope: resourceGroup()
  params: {
    name: keyVaultPrivateDnsZoneName
    location: 'global'
    linkToVirtualNetworkId: virtualNetworkId
    tags: tags
  }
}
module keyVault '../resources/private_keyvault.bicep' = {
  name: 'kv-${baseName}-deployment'
  scope: resourceGroup()
  params: {
    location: location
    keyvaultName: keyVaultName
    subnetId: '${virtualNetworkId}/subnets/${subnetName}'
    privateDNSZoneName: keyVaultPrivateDnsZoneName
    tags: tags
  }
  dependsOn: [
    keyVaultPrivateDnsZone
  ]
}

var acrPrivateDnsZoneName = 'privatelink${environment().suffixes.acrLoginServer}'
module acrPrivateDnsZone '../networking/private_dns_zone.bicep' = if (createPrivateDNSZones) {
  name: '${baseName}-acr-private-dns-zone'
  scope: resourceGroup()
  params: {
    name: acrPrivateDnsZoneName
    location: 'global'
    linkToVirtualNetworkId: virtualNetworkId
    tags: tags
  }
}
module containerRegistry '../resources/private_acr.bicep' = {
  name: 'cr${baseName}-deployment'
  scope: resourceGroup()
  params: {
    location: location
    containerRegistryName: containerRegistryName
    subnetId: '${virtualNetworkId}/subnets/${subnetName}'
    privateDNSZoneName: acrPrivateDnsZoneName
    tags: tags
  }
}


// ********************************
// Azure Machine Learning workspace
// ********************************

resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2022-12-01-preview' = {
  name: machineLearningName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    // workspace organization
    friendlyName: machineLearningFriendlyName
    description: machineLearningDescription

    // dependent resources
    storageAccount: storage.outputs.storageId
    keyVault: keyVault.outputs.keyvaultId
    containerRegistry: containerRegistry.outputs.containerRegistryId
    hbiWorkspace: true

    // configuration for workspaces with private link endpoint
    publicNetworkAccess: workspacePublicNetworkAccess
    imageBuildCompute: imageBuildComputeName
  }
}

// *******
// Compute
// *******

resource imageBuildCompute 'Microsoft.MachineLearningServices/workspaces/computes@2022-05-01' = {
  name: imageBuildComputeName
  parent: machineLearning
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'AmlCompute'
    computeLocation: location
    description: 'private cluster for building images'
    disableLocalAuth: true

    properties: {
      vmPriority: 'Dedicated'
      vmSize: imageBuildComputeSKU
      osType: 'Linux'

      // how many nodes to provision
      scaleSettings: {
        maxNodeCount: imageBuildComputeNodes
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }
  
      // networking
      subnet: {
        id: '${virtualNetworkId}/subnets/${subnetName}'
      }
      enableNodePublicIp: false
      isolatedNetwork: false
      remoteLoginPortPublicAccess: 'Disabled'
    }
  }
  dependsOn: [
    // to use enableNodePublicIp:Disabled, private endpoint is required
    machineLearningPrivateEndpoint
  ]
}


// *****************************************
// Azure Machine Learning private networking
// *****************************************

var amlPrivateDnsZoneNames =  {
  azureusgovernment: 'privatelink.api.ml.azure.us'
  azurechinacloud: 'privatelink.api.ml.azure.cn'
  azurecloud: 'privatelink.api.azureml.ms'
}
module amlPrivateDnsZone '../networking/private_dns_zone.bicep' = if (createPrivateDNSZones) {
  name: '${baseName}-aml-private-dns-zone'
  scope: resourceGroup()
  params: {
    name: amlPrivateDnsZoneNames[toLower(environment().name)]
    location: 'global'
    linkToVirtualNetworkId: virtualNetworkId
    tags: tags
  }
}

var aznbPrivateAznbDnsZoneName = {
  azureusgovernment: 'privatelink.notebooks.usgovcloudapi.net'
  azurechinacloud: 'privatelink.notebooks.chinacloudapi.cn'
  azurecloud: 'privatelink.notebooks.azure.net'
}
module aznbPrivateDnsZone '../networking/private_dns_zone.bicep' = if (createPrivateDNSZones) {
  name: '${baseName}-aznb-private-dns-zone'
  scope: resourceGroup()
  params: {
    name: aznbPrivateAznbDnsZoneName[toLower(environment().name)]
    location: 'global'
    linkToVirtualNetworkId: virtualNetworkId
    tags: tags
  }
}

resource machineLearningPrivateEndpoint 'Microsoft.Network/privateEndpoints@2022-01-01' = {
  name: 'ple-${baseName}-aml'
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [
      {
        name: 'ple-${baseName}-aml'
        properties: {
          groupIds: [
            'amlworkspace'
          ]
          privateLinkServiceId: machineLearning.id
        }
      }
    ]
    subnet: {
      id: '${virtualNetworkId}/subnets/${subnetName}'
    }
  }
}

resource privateEndpointDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2022-01-01' = {
  name: 'amlworkspace-PrivateDnsZoneGroup'
  parent: machineLearningPrivateEndpoint
  properties:{
    privateDnsZoneConfigs: [
      {
        name: amlPrivateDnsZone.outputs.name
        properties:{
          privateDnsZoneId: amlPrivateDnsZone.outputs.id
        }
      }
      {
        name: aznbPrivateDnsZone.outputs.name
        properties:{
          privateDnsZoneId: aznbPrivateDnsZone.outputs.id
        }
      }
    ]
  }
}


// *******
// Outputs
// *******

output storageName string = storage.name
output workspaceName string = machineLearning.name
output region string = location
output workspaceSecretStoreId string = keyVault.outputs.keyvaultId
output workspaceSecretStoreName string = keyVaultName
