// Execute this main file to configure Azure Machine Learning end-to-end in a moderately secure set up

// Parameters
@minLength(2)
@maxLength(10)
@description('Unique base name for all resource names.')
param baseName string = substring(uniqueString(resourceGroup().id), 0, 4)

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

// Dependent resources for the Azure Machine Learning workspace
module keyvault '../resources/private_keyvault.bicep' = {
  name: 'kv-${baseName}-deployment'
  params: {
    location: location
    keyvaultName: 'kv-${baseName}'
    keyvaultPleName: 'ple-${baseName}-kv'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module storage '../resources/private_storage.bicep' = {
  name: 'st${baseName}-deployment'
  params: {
    storageRegion: location
    storageName: 'st${baseName}'
    storagePleBlobName: 'ple-${baseName}-st-blob'
    storagePleFileName: 'ple-${baseName}-st-file'
    storageSKU: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module containerRegistry '../resources/private_acr.bicep' = {
  name: 'cr${baseName}-deployment'
  params: {
    location: location
    containerRegistryName: 'cr${baseName}'
    containerRegistryPleName: 'ple-${baseName}-cr'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
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
    publicNetworkAccess: 'Disabled'
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
