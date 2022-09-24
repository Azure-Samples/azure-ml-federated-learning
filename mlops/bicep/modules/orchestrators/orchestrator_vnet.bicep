// Creates a machine learning workspace, private endpoints and compute resources
// Compute resources include a GPU cluster, CPU cluster, compute instance and attached private AKS cluster

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Machine learning workspace name')
param machineLearningName string

// optional parameters
@description('Machine learning workspace display name')
param machineLearningFriendlyName string = machineLearningName

@description('Machine learning workspace description')
param machineLearningDescription string = 'Federated Learning Orchestration Workspace'

@description('Specifies the location of the Azure Machine Learning workspace and dependent resources.')
param location string = resourceGroup().location

@description('Specifies whether to reduce telemetry collection and enable additional encryption.')
param hbi_workspace bool = false

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('Name of the application insights resource')
param applicationInsightsName string = 'appi-${machineLearningName}'

@description('Name of the container registry resource')
param containerRegistryName string = replace('cr-${machineLearningName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the key vault resource')
param keyVaultName string = 'kv-${machineLearningName}'

@description('Name of the storage account resource')
param storageAccountName string = replace('st-${machineLearningName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the default compute cluster in orchestrator')
param orchestratorComputeName string = 'cpu-cluster-orchestrator'

@description('VM size for the default compute cluster')
param orchestratorComputeSKU string = 'Standard_DS3_v2'

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${machineLearningName}-orchestrator'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${machineLearningName}-orchestrator'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Training subnet address prefix')
param trainingSubnetPrefix string = '10.0.0.0/24'

@description('Name of the UAI for the default compute cluster')
param orchestratorUAIName string = 'uai-${machineLearningName}-orchestrator'

@description('Role definition ID for the compute towards the internal storage')
param orchToOrchRoleDefinitionId string = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor (read,write,delete)

param allowPublicAccessWhenBehindVnet bool = true
param accessThroughBastion bool = false // TODO


// Virtual network and network security group
module nsg '../networking/nsg.bicep' = { 
  name: '${nsgResourceName}-deployment'
  params: {
    location: location
    tags: tags 
    nsgName: nsgResourceName
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: '${vnetResourceName}-deployment'
  params: {
    location: location
    virtualNetworkName: vnetResourceName
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: vnetAddressPrefix
    trainingSubnetPrefix: trainingSubnetPrefix
    // scoringSubnetPrefix: scoringSubnetPrefix
    tags: tags
  }
}


// Create dependencies within the vnet
module keyvault '../secure_resources/keyvault.bicep'= {
  name: keyVaultName
  params: {
    location: location
    keyvaultName: keyVaultName
    keyvaultPleName: 'ple-${keyVaultName}'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module storage '../secure_resources/storage.bicep' = {
  name: '${storageAccountName}-deployment'
  params: {
    location: location
    storageName: storageAccountName
    storagePleBlobName: 'ple-${machineLearningName}-orchestrator-st-blob'
    storagePleFileName: 'ple-${machineLearningName}-orchestrator-st-file'
    storageSkuName: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module containerRegistry '../secure_resources/containerregistry.bicep' = {
  name: '${containerRegistryName}-deployment'
  params: {
    location: location
    containerRegistryName: containerRegistryName
    containerRegistryPleName: 'ple-${containerRegistryName}-orchestrator-cr'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module applicationInsights '../secure_resources/applicationinsights.bicep' = {
  name: '${applicationInsightsName}-deployment'
  params: {
    location: location
    applicationInsightsName: applicationInsightsName
    tags: tags
  }
}

resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2022-01-01-preview' = {
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
    applicationInsights: applicationInsights.outputs.applicationInsightsId
    containerRegistry: containerRegistry.outputs.containerRegistryId
    keyVault: keyvault.outputs.keyvaultId
    storageAccount: storage.outputs.storageId
    hbiWorkspace: hbi_workspace

    // security end-to-end
    allowPublicAccessWhenBehindVnet: allowPublicAccessWhenBehindVnet

    // configuration for workspaces with private link endpoint TODO
    // imageBuildCompute: 'cluster001'
    // publicNetworkAccess: 'Disabled'
  }
}

// provision a user assigned identify for this silo
resource orchestratorUAI 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: orchestratorUAIName
  location: location
  tags: tags
}

module machineLearningPrivateEndpoint '../networking/aml_ple.bicep' = {
  name: 'aml-ple-${machineLearningName}-deployment'
  scope: resourceGroup()
  params: {
    location: location
    tags: tags
    virtualNetworkId: vnet.outputs.id
    workspaceArmId: machineLearning.id
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    machineLearningPleName: 'ple-${machineLearningName}-orchestrator-mlw'
  }
}

// provision a compute cluster, and assign the user assigned identity to it
resource orchestratorCompute 'Microsoft.MachineLearningServices/workspaces/computes@2022-06-01-preview' = {
  name: '${machineLearningName}/${orchestratorComputeName}'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${orchestratorUAI.name}': {}
    }
  }
  properties: {
    computeType: 'AmlCompute'
    description: 'Orchestration cluster with R/W access to the workspace storage'
    // disableLocalAuth: true
    properties: {
      vmPriority: 'Dedicated'      
      vmSize: orchestratorComputeSKU
      enableNodePublicIp: false
      isolatedNetwork: false
      osType: 'Linux'
      remoteLoginPortPublicAccess: 'Disabled'
      scaleSettings: {
        maxNodeCount: 4
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }
      subnet: {
        id: '${vnet.outputs.id}/subnets/snet-training'
      }
    }
  }
  dependsOn: [
    machineLearning
    machineLearningPrivateEndpoint
  ]
}

// assign the role orch compute should have with orch storage
// resource orchToOrchRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
//   scope: storage.outputs.storageId
//   name: guid(storage.name, orchToOrchRoleDefinitionId, orchestratorUAI.name)
//   properties: {
//     roleDefinitionId: orchToOrchRoleDefinitionId
//     principalId: orchestratorUAI.properties.principalId
//     principalType: 'ServicePrincipal'
//   }
//   dependsOn: [
//     storage
//   ]
// }

output uaiPrincipalId string = orchestratorUAI.properties.principalId
output storage string = storageAccountName
output compute string = orchestratorCompute.name
output workspace string = machineLearning.name
output machineLearningId string = machineLearning.id
output region string = location
output uaiId string = orchestratorUAI.id
output vnetId string = vnet.outputs.id
