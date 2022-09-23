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
param workspaceName string

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
param datastoreName string = 'datatore_${replace('${siloBaseName}', '-', '_')}'

@description('Specifies the name of the User Assigned Identity to provision.')
param siloUAIName string = 'uai-${siloBaseName}'

@description('Which RBAC role to use for silo compute -> silo storage (default R/W).')
param siloToSiloRoleDefinitionId string = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor (read,write,delete)

@description('Which RBAC role to use for silo compute -> orchestrator storage (default R/W).')
param siloToOrchRoleDefinitionId string = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor (read,write,delete)

@description('Tags to curate the resources in Azure.')
param tags object = {}


// deploy a storage account for the silo
resource siloStorageAccount 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: storageAccountName
  location: region
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    supportsHttpsTrafficOnly: true
  }
}

// create a "private" container in the storage account
// this one will be readable only by silo compute
resource siloStoragePrivateContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  name: '${siloStorageAccount.name}/default/siloprivate'
  properties: {
    metadata: {}
    publicAccess: 'None'
  }
}

// attach as a datastore in AzureML
resource siloAzureMLPrivateDatastore 'Microsoft.MachineLearningServices/workspaces/datastores@2022-06-01-preview' = {
  name: '${workspaceName}/${datastoreName}_private'
  properties: {
    credentials: {
      credentialsType: 'None'
      // For remaining properties, see DatastoreCredentials objects
    }
    description: 'Silo private storage in region ${region}'
    properties: {}
    datastoreType: 'AzureBlob'
    // For remaining properties, see DatastoreProperties objects
    accountName: siloStorageAccount.name
    containerName: 'siloprivate'
    // endpoint: 'string'
    // protocol: 'string'
    resourceGroup: resourceGroup().name
    // serviceDataAccessAuthIdentity: 'string'
    subscriptionId: subscription().subscriptionId
  }
  dependsOn: [
    siloStoragePrivateContainer
  ]
}


// provision a user assigned identify for this silo
resource siloUserAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: siloUAIName
  location: region
  tags: tags
}

// provision a compute cluster for the silo and assigned the silo UAI to it
resource siloAzureMLCompute 'Microsoft.MachineLearningServices/workspaces/computes@2020-09-01-preview' = {
  name: '${workspaceName}/${computeClusterName}'
  location: region
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${siloUserAssignedIdentity.name}': {}
    }
  }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: siloComputeSKU
      subnet: json('null')
      osType: 'Linux'
      scaleSettings: {
        maxNodeCount: 4
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }
    }
  }
}

// role of silo compute -> silo storage
resource siloToSiloRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(siloToSiloRoleDefinitionId)) {
  scope: siloStoragePrivateContainer
  name: guid(siloStoragePrivateContainer.name, siloToSiloRoleDefinitionId, siloUserAssignedIdentity.name)
  properties: {
    roleDefinitionId: siloToSiloRoleDefinitionId
    principalId: siloUserAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
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
output storage string = siloStorageAccount.name
output container string = siloStoragePrivateContainer.name
output compute string = siloAzureMLCompute.name
output datastore string = siloAzureMLPrivateDatastore.name
output region string = region
