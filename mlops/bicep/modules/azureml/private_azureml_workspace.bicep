// Execute this main file to configure Azure Machine Learning end-to-end in a moderately secure set up

// Parameters
@minLength(2)
@maxLength(10)
@description('Unique base name for all resource names.')
param baseName string

param machineLearningName string = 'aml-${baseName}'

@description('Azure region used for the deployment of all resources.')
param location string = resourceGroup().location

@description('Set of tags to apply to all resources.')
param tags object = {}

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Training subnet address prefix')
param trainingSubnetPrefix string = '10.0.1.0/24'

@description('Scoring subnet address prefix')
param scoringSubnetPrefix string = '10.0.2.0/24'

@description('Public access to the Azure ML workspace itself')
@allowed([
  'Enabled'
  'Disabled'
])
param workspacePublicNetworkAccess string = 'Disabled'

@description('Enable public IP for Azure Machine Learning compute nodes')
param amlComputePublicIp bool = false

@description('VM size for the default compute cluster')
param amlComputeDefaultVmSize string = 'Standard_DS3_v2'


// Virtual network and network security group
module nsg '../networking/nsg.bicep' = { 
  name: 'nsg-${baseName}-deployment'
  params: {
    location: location
    tags: tags 
    nsgName: 'nsg-${baseName}'
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: 'vnet-${baseName}-deployment'
  params: {
    location: location
    virtualNetworkName: 'vnet-${baseName}'
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: vnetAddressPrefix
    subnets: [
      {
        name: 'snet-training'
        addressPrefix: trainingSubnetPrefix
      }
      {
        name: 'snet-scoring'
        addressPrefix: scoringSubnetPrefix
      }
    ]
    tags: tags
  }
}

// create all DNS zones needed for the next private resources
var amlPrivateDnsZoneNames =  {
  azureusgovernment: 'privatelink.api.ml.azure.us'
  azurechinacloud: 'privatelink.api.ml.azure.cn'
  azurecloud: 'privatelink.api.azureml.ms'
}

var aznbPrivateAznbDnsZoneName = {
    azureusgovernment: 'privatelink.notebooks.usgovcloudapi.net'
    azurechinacloud: 'privatelink.notebooks.chinacloudapi.cn'
    azurecloud: 'privatelink.notebooks.azure.net'
}

var requiredDNSZones = [
  'privatelink${environment().suffixes.acrLoginServer}'
  'privatelink${environment().suffixes.keyvaultDns}'
  'privatelink.blob.${environment().suffixes.storage}'
  'privatelink.file.${environment().suffixes.storage}'
  amlPrivateDnsZoneNames[toLower(environment().name)]
  aznbPrivateAznbDnsZoneName[toLower(environment().name)]
]

resource allDNSZones 'Microsoft.Network/privateDnsZones@2020-06-01' = [for zone in requiredDNSZones: {
  name: zone
  location: 'global'
}]


// Dependent resources for the Azure Machine Learning workspace
module keyvault '../resources/private_keyvault.bicep' = {
  name: 'kv-${baseName}-deployment'
  params: {
    location: location
    keyvaultName: 'kv-${baseName}'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    privateDNSZoneName: 'privatelink${environment().suffixes.keyvaultDns}'
    privateDNSZoneLocation: 'global'
    tags: tags
  }
}

module storage '../resources/private_storage.bicep' = {
  name: 'st${baseName}-deployment'
  params: {
    storageRegion: location
    storageName: 'st${baseName}'
    storageSKU: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    blobPrivateDNSZoneName: 'privatelink.blob.${environment().suffixes.storage}'
    blobPrivateDNSZoneLocation: 'global'
    filePrivateDNSZoneName: 'privatelink.file.${environment().suffixes.storage}'
    filePrivateDNSZoneLocation: 'global'
    tags: tags
  }
}

module containerRegistry '../resources/private_acr.bicep' = {
  name: 'cr${baseName}-deployment'
  params: {
    location: location
    containerRegistryName: 'cr${baseName}'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    privateDNSZoneName: 'privatelink${environment().suffixes.acrLoginServer}'
    privateDNSZoneLocation: 'global'
    tags: tags
  }
}

module applicationInsights '../resources/private_appinsights.bicep' = {
  name: 'appi-${baseName}-deployment'
  params: {
    location: location
    applicationInsightsName: 'appi-${baseName}'
    tags: tags
  }
}

resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2022-05-01' = {
  name: machineLearningName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    // workspace organization
    friendlyName: 'Private link endpoint sample workspace'
    description: 'This is an example workspace having a private link endpoint.'

    // dependent resources
    applicationInsights: applicationInsights.outputs.applicationInsightsId
    containerRegistry: containerRegistry.outputs.containerRegistryId
    keyVault: keyvault.outputs.keyvaultId
    storageAccount: storage.outputs.storageId

    // configuration for workspaces with private link endpoint
    imageBuildCompute: 'default-cluster'
    publicNetworkAccess: workspacePublicNetworkAccess
  }
}

module machineLearningPrivateEndpoint './private_azureml_networking.bicep' = {
  name: 'aml-${baseName}-networking-deployment'
  scope: resourceGroup()
  params: {
    location: location
    tags: tags
    virtualNetworkId: vnet.outputs.id
    workspaceArmId: machineLearning.id
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    machineLearningPleName: 'ple-${baseName}-aml'

    // private DNS zone for Azure Machine Learning workspace
    amlPrivateDnsZoneName: amlPrivateDnsZoneNames[toLower(environment().name)]
    amlPrivateDnsZoneLocation: 'global'
    aznbPrivateDnsZoneName: aznbPrivateAznbDnsZoneName[toLower(environment().name)]
    aznbPrivateDnsZoneLocation: 'global'
  }
}

resource machineLearningDefaultCluster 'Microsoft.MachineLearningServices/workspaces/computes@2022-05-01' = {
  name: '${machineLearning.name}/default-cluster'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'AmlCompute'
    computeLocation: location
    description: 'Machine Learning cluster 001'
    disableLocalAuth: true
    properties: {
      vmPriority: 'Dedicated'
      vmSize: amlComputeDefaultVmSize
      enableNodePublicIp: amlComputePublicIp
      isolatedNetwork: false
      osType: 'Linux'
      remoteLoginPortPublicAccess: 'Disabled'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 5
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      subnet: {
        id: '${vnet.outputs.id}/subnets/snet-training'
      }
    }
  }
  dependsOn: [
    // to use enableNodePublicIp:Disabled, private endpoint is required
    machineLearningPrivateEndpoint
  ]
}

output workspace string = machineLearning.name
