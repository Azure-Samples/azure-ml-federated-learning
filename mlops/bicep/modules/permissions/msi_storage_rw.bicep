// Assigns roles to a given User Assigned Identity
// towards a given storage account

@description('Full path to storage')
param storageAccountName string

@description('PrincipalId of the managed identity')
param identityPrincipalId string

@description('Role definition IDs for the compute towards the internal storage')
param computeToStorageRoles array = [
  // see https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles
  'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor
  '81a9662b-bebf-436f-a333-f67b29880f12' // Storage Account Key Operator Service Role
  'c12c1c16-33a1-487b-954d-41c89c60f349' // Reader and Data Access
]

resource storage 'Microsoft.Storage/storageAccounts@2022-05-01' existing = {
  name: storageAccountName
}

resource roleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for roleId in computeToStorageRoles: {
  scope: storage
  name: guid(resourceGroup().id, storage.id, identityPrincipalId, roleId)
  properties: {
    roleDefinitionId: '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/${roleId}'
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
  dependsOn: [
    storage
  ]
}]
