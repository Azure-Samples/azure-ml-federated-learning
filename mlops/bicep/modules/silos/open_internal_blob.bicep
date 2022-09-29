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
@description('Name of AzureML workspace to attach silo to.')
param machineLearningName string

@description('Specifies the region of the silo (for storage + compute).')
param region string

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('Specifies the name of the orchestrator storage account.')
param orchestratorStorageAccountName string // needed to set permissions towards orchestrator storage

// optional parameters
@description('Specifies the base name for creating resources.')
param siloName string = 'silo-${region}'

@description('Specifies the name of the storage account to provision.')
param storageAccountName string = 'st${replace('${siloName}', '-', '')}'

@description('Specifies the name of the compute cluster to provision.')
param computeName string = 'cpu-cluster-${siloName}'

@description('VM size for the default compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@description('Specifies the name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = 'datatore_${replace('${siloName}', '-', '_')}'

@allowed(['UserAssigned','SystemAssigned'])
param identityType string = 'UserAssigned'

@description('Specifies the name of the User Assigned Identity to provision.')
param uaiName string = 'uai-${siloName}'

@description('Which RBAC roles to use for silo compute -> silo storage (default R/W).')
param siloToSiloRoleDefinitionIds array = [
  // see https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles
  // Storage Blob Data Contributor (read,write,delete)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  // Storage Account Key Operator Service Role (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/81a9662b-bebf-436f-a333-f67b29880f12'
  // Reader and Data Access (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/c12c1c16-33a1-487b-954d-41c89c60f349'
]

@description('Which RBAC roles to use for silo compute -> orchestrator storage (default R/W).')
param siloToOrchRoleDefinitionIds array = [
  // see https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles
  // Storage Blob Data Contributor (read,write,delete)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  // Storage Account Key Operator Service Role (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/81a9662b-bebf-436f-a333-f67b29880f12'
  // Reader and Data Access (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/c12c1c16-33a1-487b-954d-41c89c60f349'
]


// deploy a storage account for the silo
resource storage 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: substring(storageAccountName, 0, min(length(storageAccountName),24))
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
resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  name: '${storage.name}/default/siloprivate'
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
    credentials: {
      credentialsType: 'None'
      // For remaining properties, see DatastoreCredentials objects
    }
    description: 'Private storage in region ${region}'
    properties: {}
    datastoreType: 'AzureBlob'
    // For remaining properties, see DatastoreProperties objects
    accountName: storage.name
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
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = if (identityType == 'UserAssigned') {
  name: uaiName
  location: region
  tags: tags
  dependsOn: [
    storage // ensure the storage exists BEFORE we do UAI role assignments
  ]
}

var identityPrincipalId = identityType == 'UserAssigned' ? uai.properties.principalId : compute.identity.principalId
var identityName = identityType == 'UserAssigned' ? uai.name : compute.name
var userAssignedIdentities = identityType == 'SystemAssigned' ? null : {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}

// provision a compute cluster for the silo and assigned the silo UAI to it
resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2020-09-01-preview' = {
  name: '${machineLearningName}/${computeName}'
  location: region
  identity: {
    type: identityType
    userAssignedIdentities: userAssignedIdentities
  }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: computeSKU
      subnet: json('null')
      osType: 'Linux'
      scaleSettings: {
        maxNodeCount: computeNodes
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }
    }
  }
  dependsOn: [
    storage // ensure the storage exists BEFORE we do UAI role assignments
  ]
}

// role of silo compute -> silo storage
resource siloToSiloRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for roleId in siloToSiloRoleDefinitionIds: {
  scope: storage
  name: guid(siloName, region, storage.name, roleId, identityName)
  properties: {
    roleDefinitionId: roleId
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}]

// role of silo compute -> orchestrator storage (for r/w model weights)
resource orchestratorStorageAccount 'Microsoft.Storage/storageAccounts@2021-06-01' existing = {
  name: orchestratorStorageAccountName
  scope: resourceGroup()
}
resource siloToOrchestratorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for roleId in siloToOrchRoleDefinitionIds: {
  scope: orchestratorStorageAccount
  name: guid(siloName, region, orchestratorStorageAccount.name, roleId, identityName)
  properties: {
    roleDefinitionId: roleId
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}]

// output the orchestrator config for next actions (permission model)
output identity string = identityPrincipalId
output storage string = storage.name
output container string = container.name
output compute string = compute.name
output datastore string = datastore.name
output region string = region
