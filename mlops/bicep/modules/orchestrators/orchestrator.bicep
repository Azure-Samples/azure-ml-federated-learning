// Creates a machine learning workspace, private endpoints and compute resources
// Compute resources include a GPU cluster, CPU cluster, compute instance and attached private AKS cluster

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

@description('Azure region of the deployment')
param location string = resourceGroup().location

@description('Tags to add to the resources')
param tags object = {}

@description('Machine learning workspace name')
param machineLearningName string

@description('Machine learning workspace display name')
param machineLearningFriendlyName string = machineLearningName

@description('Machine learning workspace description')
param machineLearningDescription string = 'Federated Learning Orchestration Workspace'

@description('Name of the application insights resource')
param applicationInsightsName string = 'appi-${machineLearningName}'

@description('Name of the container registry resource')
param containerRegistryName string = 'cr${machineLearningName}'

@description('Name of the key vault resource')
param keyVaultName string = 'kv-${machineLearningName}'

@description('Name of the storage account resource')
param storageAccountName string = 'st${machineLearningName}'

@description('Name of the orchestrator compute cluster (CPU)')
param orchestratorComputeName string = 'cpu-orchestrator'

@description('VM size for the default compute cluster')
param orchestratorComputeSKU string = 'Standard_DS13_v2'

@description('Name of the vNET resource')
param nsgResourceName string = 'nsg-${machineLearningName}-orchestrator'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${machineLearningName}-orchestrator'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Training subnet address prefix')
param trainingSubnetPrefix string = '10.0.0.0/24'

@description('Virtual network address prefix')
param userAssignedIdentityName string = 'uai-${machineLearningName}-orchestrator'

// @description('Which role the orchestrator compute should have towards its own storage.')
// param orchToOrchRoleDefinitionId string = '' // default Contributor


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

    // IMPORTANT: we still want to allow this
    allowPublicAccessWhenBehindVnet: true

    // configuration for workspaces with private link endpoint
    // imageBuildCompute: 'cluster001'
    // publicNetworkAccess: 'Disabled'
  }
}

// provision a user assigned identify for this silo
resource orchestratorUserAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: userAssignedIdentityName
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
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${orchestratorUserAssignedIdentity.name}': {}
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
//   name: guid(storage.name, orchToOrchRoleDefinitionId, orchestratorUserAssignedIdentity.name)
//   properties: {
//     roleDefinitionId: orchToOrchRoleDefinitionId
//     principalId: orchestratorUserAssignedIdentity.properties.principalId
//     principalType: 'ServicePrincipal'
//   }
//   dependsOn: [
//     storage
//   ]
// }

output machineLearningId string = machineLearning.id
output uaiId string = orchestratorUserAssignedIdentity.id
output uaiPrincipalId string = orchestratorUserAssignedIdentity.properties.principalId
output storageId string = storage.outputs.storageId
