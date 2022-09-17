// This BICEP script will set the RBAC for a permission model in our demo.

// This permission model is represented by the following matrix:
// |               | orch.compute | siloA.compute | siloB.compute |
// |---------------|--------------|---------------|---------------|
// | orch.storage  |     R/W      |       W       |       W       |
// | siloA.storage |      W       |      R/W      |       -       |
// | siloB.storage |      W       |       -       |      R/W      |

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

@description('Object config of the orchestrator.')
param orchestratorConfig object

@description('Array containing the config of each silo in an array.')
param siloConfigArray array

var siloCount = length(siloConfigArray)

//  provision a UAI for the orchestrator
resource orchestratorUAI 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: 'orchestrator-uai'
  location: orchestratorConfig.region
}

// provision a distinct UAI for each silo in its region
resource siloUAIArray 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = [for i in range(0, siloCount): {
  name: 'silo-${i}-${siloConfigArray[i].region}-uai'  // add i in name to avoid regional conflict
  location: siloConfigArray[i].region
}]

// create specific roles for those UAIs
module storageWriteOnlyRoleDeployment './role_storage_write_only.bicep' = {
  name: 'fl_demo_write_only_role'
  scope: resourceGroup()
}

module storageReadWriteRoleDeployment './role_storage_read_write.bicep' = {
  name: 'fl_demo_read_write_role'
  scope: resourceGroup()
}

resource orchestratorStorageContainer 'Microsoft.Storage/storageAccounts@2021-06-01' existing = {
  name: orchestratorConfig.storage.id
  scope: resourceGroup()
}

// assign the roles to the orchestrator UAI
resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: orchestratorStorageContainer
  // '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${orchestratorConfig.uai.name}'
  name: guid(orchestratorStorageContainer.id, storageReadWriteRoleDeployment.name, orchestratorConfig.uai.properties.principalId)
  properties: {
    roleDefinitionId: storageReadWriteRoleDeployment.outputs.roleDefinitionId
    principalId: orchestratorConfig.uai.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// resource roleAssignment 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
//   scope: storageAccount
//   name: guid(storageAccount.id, principalId, roleDefinitionResourceId)
//   properties: {
//     roleDefinitionId: roleDefinitionResourceId
//     principalId: principalId
//     principalType: 'ServicePrincipal'
//   }
// }

// var roleAssignmentsToCreate = [{
//   name: guid(siloUserAssignedIdentity.id, resourceGroup().id, 'Storage Blob Data Contributor')
//   roleDefinitionId: 'Storage Blob Data Contributor'
// }]

// // set the UAI role assignment for the silo storage account
// resource roleAssignment 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = [for roleAssignmentToCreate in roleAssignmentsToCreate: {
//   name: roleAssignmentToCreate.name
//   scope: resourceGroup()
//   properties: {
//     description: 'Add Storage Blob Data Contributor role to the silo storage account'
//     principalId: siloUserAssignedIdentity.properties.principalId
//     roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleAssignmentToCreate.roleDefinitionId)
//     principalType: 'ServicePrincipal' // See https://docs.microsoft.com/azure/role-based-access-control/role-assignments-template#new-service-principal to understand why this property is included.
//   }
// }]
