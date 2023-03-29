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

@description('Specifies whether to reduce telemetry collection and enable additional encryption.')
param hbiWorkspace bool = false

@description('Name of the application insights log analytics workspace')
param appInsightLogAnalyticsName string = 'logs-${baseName}'

@description('Name of the application insights resource')
param applicationInsightsName string = 'appi-${baseName}'

@description('Name of the container registry resource')
param containerRegistryName string = replace('cr-${baseName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the workspace secrets store (shared keyvault) resource')
param keyVaultName string = 'ws-shkv-${baseName}'

@description('Name of the storage account resource')
param storageAccountName string = replace('st-${baseName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the cluster used to build images for custom environments (required for private workspaces)')
param imageBuildComputeName string = 'image-build-compute'

@description('VM size for cluster used to build images')
param imageBuildComputeSKU string = 'Standard_DS3_v2'

@description('VM nodes for cluster used to build images')
param imageBuildComputeNodes int = 2

@description('Name of the NSG to create')
param nsgName string = 'nsg-ws-${baseName}'

@description('Name of the virtual network resource')
param vnetName string = 'vnet-ws-${baseName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Workspace resources subnet address prefix')
param wsSubnetPrefix string = '10.0.0.0/24'

@description('Workspace resources subnet name')
param wsSubnetName string = 'snet-ws'

@description('Public network access to the Azure ML workspace itself')
@allowed([
  'Enabled'
  'Disabled'
])
param workspacePublicNetworkAccess string = 'Disabled'

@description('Tags to curate the resources in Azure.')
param tags object = {}


// Virtual network and network security group of the workspace itself
module nsg '../networking/azureml_workspace_nsg.bicep' = { 
  name: 'nsg-${baseName}-aml-compute'
  scope: resourceGroup()
  params: {
    location: location
    nsgName: nsgName
    tags: tags
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: 'vnet-${baseName}-deployment'
  scope: resourceGroup()
  params: {
    location: location
    virtualNetworkName: vnetName
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: vnetAddressPrefix
    subnets: [
      {
        name: wsSubnetName
        addressPrefix: wsSubnetPrefix
      }
    ]
    tags: tags
  }
}


// ************************************************************
// Dependent resources for the Azure Machine Learning workspace
// ************************************************************

resource blobStoragePrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.blob.${environment().suffixes.storage}'
  location: 'global'
  tags: tags 
}
resource fileStoragePrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.file.${environment().suffixes.storage}'
  location: 'global'
  tags: tags 
}

module storage '../resources/private_storage.bicep' = {
  name: 'st${baseName}-deployment'
  scope: resourceGroup()
  params: {
    storageRegion: location
    storageName: storageAccountName
    storageSKU: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/${wsSubnetName}'
    virtualNetworkId: vnet.outputs.id
    blobPrivateDNSZoneName: blobStoragePrivateDnsZone.name
    blobPrivateDNSZoneLocation: blobStoragePrivateDnsZone.location
    filePrivateDNSZoneName: fileStoragePrivateDnsZone.name
    filePrivateDNSZoneLocation: fileStoragePrivateDnsZone.location
    tags: tags
  }
}

resource keyVaultPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink${environment().suffixes.keyvaultDns}'
  location: 'global'
  tags: tags 
}
module keyVault '../resources/private_keyvault.bicep' = {
  name: 'kv-${baseName}-deployment'
  scope: resourceGroup()
  params: {
    location: location
    keyvaultName: keyVaultName
    subnetId: '${vnet.outputs.id}/subnets/${wsSubnetName}'
    virtualNetworkId: vnet.outputs.id
    privateDNSZoneName: keyVaultPrivateDnsZone.name
    privateDNSZoneLocation: keyVaultPrivateDnsZone.location
    tags: tags
  }
}

resource acrPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink${environment().suffixes.acrLoginServer}'
  location: 'global'
  tags: tags 
}
module containerRegistry '../resources/private_acr.bicep' = {
  name: 'cr${baseName}-deployment'
  scope: resourceGroup()
  params: {
    location: location
    containerRegistryName: containerRegistryName
    subnetId: '${vnet.outputs.id}/subnets/${wsSubnetName}'
    virtualNetworkId: vnet.outputs.id
    privateDNSZoneName: acrPrivateDnsZone.name
    privateDNSZoneLocation: acrPrivateDnsZone.location
    tags: tags
  }
}


// ********************************
// Azure Machine Learning workspace
// ********************************

resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2022-10-01' = {
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
    hbiWorkspace: hbiWorkspace

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
        id: '${vnet.outputs.id}/subnets/${wsSubnetName}'
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
      id: '${vnet.outputs.id}/subnets/${wsSubnetName}'
    }
  }
}

var amlPrivateDnsZoneNames =  {
  azureusgovernment: 'privatelink.api.ml.azure.us'
  azurechinacloud: 'privatelink.api.ml.azure.cn'
  azurecloud: 'privatelink.api.azureml.ms'
}
resource amlPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: amlPrivateDnsZoneNames[toLower(environment().name)]
  location: 'global'
  tags: tags 
}
resource amlPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  name: uniqueString(machineLearning.id)
  parent: amlPrivateDnsZone
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.outputs.id
    }
  }
}

var aznbPrivateAznbDnsZoneName = {
  azureusgovernment: 'privatelink.notebooks.usgovcloudapi.net'
  azurechinacloud: 'privatelink.notebooks.chinacloudapi.cn'
  azurecloud: 'privatelink.notebooks.azure.net'
}
resource aznbPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: aznbPrivateAznbDnsZoneName[toLower(environment().name)]
  location: 'global'
  tags: tags 
}
resource notebookPrivateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  name: uniqueString(machineLearning.id)
  parent: aznbPrivateDnsZone
  location: 'global'
  tags: tags 
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.outputs.id
    }
  }
}

resource privateEndpointDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2022-01-01' = {
  name: 'amlworkspace-PrivateDnsZoneGroup'
  parent: machineLearningPrivateEndpoint
  properties:{
    privateDnsZoneConfigs: [
      {
        name: amlPrivateDnsZone.name
        properties:{
          privateDnsZoneId: amlPrivateDnsZone.id
        }
      }
      {
        name: aznbPrivateDnsZone.name
        properties:{
          privateDnsZoneId: aznbPrivateDnsZone.id
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