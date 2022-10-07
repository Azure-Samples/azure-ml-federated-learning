// Gets the system identity of an AzureML compute
// and assigns permissions towards a given storage account

@description('Name of AzureML workspace to attach orchestrator to.')
param machineLearningName string

@description('Name of compute cluster attached to the workspace.')
param computeName string

@description('Full path to storage')
param storageAccountId string

@description('Role definition IDs for the compute towards the internal storage')
param computeToStorageRoles array = [
  // see https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles
  // Storage Blob Data Contributor (read,write,delete)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  // Storage Account Key Operator Service Role (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/81a9662b-bebf-436f-a333-f67b29880f12'
  // Reader and Data Access (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/c12c1c16-33a1-487b-954d-41c89c60f349'
]

// role of silo compute -> silo storage
resource storage 'Microsoft.Storage/storageAccounts@2022-05-01' existing = {
  name: storageAccountId
}


resource workspace 'Microsoft.MachineLearningServices/workspaces@2022-05-01' existing = {
  name: machineLearningName
  scope: resourceGroup()
}

resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2022-05-01' existing = {
  name: computeName
  parent: workspace
}

resource roleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for roleId in computeToStorageRoles: {
  scope: storage
  name: guid(resourceGroup().id, storage.id, compute.name, roleId)
  properties: {
    roleDefinitionId: roleId
    principalId: compute.identity.principalId
    principalType: 'ServicePrincipal'
  }
  dependsOn: [
    storage
    compute
  ]
}]
