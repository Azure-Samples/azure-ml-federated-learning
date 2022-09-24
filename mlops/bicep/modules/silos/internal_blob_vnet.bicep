// Provision a basic Internal Silo with UAI for permissions management
//
// Given an AzureML workspace, and a specific region, this BICEP script will provision:
// - a new blob storage account in the given region
// - create 1 containers in this storage for private silo data
// - 1 AzureML compute cluster in that same region, attached to the AzureML workspace
// - 2 AzureML datastores for each of the private/shared containers
// - a User Assigned Identity

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Specifies the base name for creating resources.')
param siloBaseName string

@description('Specifies the name of the orchestrator AzureML workspace.')
param machineLearningName string

@description('Specifies the region of the silo (for storage + compute).')
param region string

@description('Specifies the name of the orchestrator storage account.')
param orchestratorStorageAccountName string // needed to set permissions towards orchestrator storage

// optional parameters
@description('Specifies the name of the storage account to provision.')
param storageAccountName string = 'st${replace('${siloBaseName}', '-', '')}'

@description('Specifies the name of the compute cluster to provision.')
param computeClusterName string = 'cpu-cluster-${siloBaseName}'
param siloComputeSKU string = 'Standard_DS3_v2'

@description('Specifies the name of the datastore for attaching the storage to the AzureML workspace.')
param siloDatastoreName string = 'datatore_${replace('${siloBaseName}', '-', '_')}'

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${siloBaseName}'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${siloBaseName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Training subnet address prefix')
param trainingSubnetPrefix string = '10.0.0.0/24'

@description('Specifies the name of the User Assigned Identity to provision.')
param siloUAIName string = 'uai-${siloBaseName}'

@description('Which RBAC role to use for silo compute -> silo storage (default R/W).')
param siloToSiloRoleDefinitionId string = '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor (read,write,delete)

@description('Which RBAC role to use for silo compute -> orchestrator storage (default R/W).')
param siloToOrchRoleDefinitionId string = '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor (read,write,delete)

@description('Tags to curate the resources in Azure.')
param tags object = {}


// Virtual network and network security group
module nsg '../networking/nsg.bicep' = { 
  name: '${nsgResourceName}-deployment'
  params: {
    location: region
    nsgName: nsgResourceName
    tags: tags
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: '${vnetResourceName}-deployment'
  params: {
    location: region
    virtualNetworkName: vnetResourceName
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: vnetAddressPrefix
    trainingSubnetPrefix: trainingSubnetPrefix
    // scoringSubnetPrefix: scoringSubnetPrefix
    tags: tags
  }
}

// deploy a storage account for the silo
module storage '../secure_resources/storage.bicep' = {
  name: '${storageAccountName}-deployment'
  params: {
    location: region
    storageName: storageAccountName
    storagePleBlobName: 'ple-${siloBaseName}-st-blob'
    storagePleFileName: 'ple-${siloBaseName}-st-file'
    storageSkuName: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

// create a "private" container in the storage account
// this one will be readable only by silo compute
resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  name: '${storageAccountName}/default/siloprivate'
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
  name: '${machineLearningName}/${siloDatastoreName}'
  properties: {
    credentials: {
      credentialsType: 'None'
      // For remaining properties, see DatastoreCredentials objects
    }
    description: 'Silo private storage in region ${region}'
    properties: {}
    datastoreType: 'AzureBlob'
    // For remaining properties, see DatastoreProperties objects
    accountName: storageAccountName
    containerName: 'siloprivate'
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


// provision a user assigned identify for this silo
resource siloUserAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: siloUAIName
  location: region
  tags: tags
}

// module machineLearningPrivateEndpoint '../networking/aml_ple.bicep' = {
//   name: 'aml-ple-${siloBaseName}-deployment'
//   scope: resourceGroup()
//   params: {
//     location: region
//     tags: tags
//     virtualNetworkId: vnet.outputs.id
//     workspaceArmId: '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.MachineLearning/${machineLearningName}'
//     subnetId: '${vnet.outputs.id}/subnets/snet-training'
//     machineLearningPleName: 'ple-${machineLearningName}-${siloBaseName}-mlw'
//   }
// }

// provision a compute cluster for the silo and assigned the silo UAI to it
resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2020-09-01-preview' = {
  name: '${machineLearningName}/${computeClusterName}'
  location: region
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${siloUserAssignedIdentity.name}': {}
    }
  }
  properties: {
    computeType: 'AmlCompute'
    description: 'Silo cluster with R/W access to the silo storage'
    properties: {
      vmPriority: 'Dedicated'      
      vmSize: siloComputeSKU
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
  // dependsOn: [
  //   machineLearningPrivateEndpoint
  // ]
}

// role of silo compute -> silo storage
resource siloStorageAccount 'Microsoft.Storage/storageAccounts@2021-06-01' existing = {
  name: storageAccountName
  scope: resourceGroup()
}
resource siloToSiloRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(siloToSiloRoleDefinitionId)) {
  scope: siloStorageAccount
  name: guid(storageAccountName, siloToSiloRoleDefinitionId, siloUserAssignedIdentity.name)
  properties: {
    roleDefinitionId: siloToSiloRoleDefinitionId
    principalId: siloUserAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
  dependsOn: [
    storage
  ]
}

// role of silo compute -> orchestrator storage (for r/w model weights)
resource orchestratorStorageAccount 'Microsoft.Storage/storageAccounts@2021-06-01' existing = {
  name: orchestratorStorageAccountName
  scope: resourceGroup()
}
resource siloToOrchestratorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(siloToOrchRoleDefinitionId)) {
  scope: orchestratorStorageAccount
  name: guid(orchestratorStorageAccount.name, siloToOrchRoleDefinitionId, siloUserAssignedIdentity.name)
  properties: {
    roleDefinitionId: siloToOrchRoleDefinitionId
    principalId: siloUserAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// output the orchestrator config for next actions (permission model)
output uaiPrincipalId string = siloUserAssignedIdentity.properties.principalId
output storage string = storageAccountName
output container string = container.name
output compute string = compute.name
output datastore string = datastore.name
output region string = region
