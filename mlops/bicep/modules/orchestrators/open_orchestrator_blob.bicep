// This BICEP script will provision an orchestrator compute+storage pair
// in a given AzureML workspace, without any specific security settings.

// IMPORTANT: This setup is intended only for demo purpose. The data is still accessible
// by the users when opening the storage accounts, and data exfiltration is easy.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach orchestrator to.')
param machineLearningName string

@description('Specifies the location of the orchestrator resources.')
param region string = resourceGroup().location

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('Name of the storage account resource to create for the orchestrator')
param storageAccountName string = replace('st-${machineLearningName}-orch','-','') // replace because only alphanumeric characters are supported

@description('Name of the storage account resource to create for the orchestrator - guaranteed to not have any length issue.')
var truncatedStorageAccountName = substring(storageAccountName, 0, min(length(storageAccountName),24))

@description('Specifies the name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = 'datastore_orchestrator'

@description('Name of the default compute cluster for orchestrator')
param computeName string = 'cpu-cluster-orchestrator'

@description('VM size for the default compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param identityType string = 'UserAssigned'

@description('Name of the Assigned Identity for the default compute cluster')
param orchestratorUAIName string = 'xai-${machineLearningName}-orchestrator'

@description('Role definition IDs for the compute towards the internal storage')
param orchToOrchRoleDefinitionIds array = [
  // see https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles
  // Storage Blob Data Contributor (read,write,delete)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  // Storage Account Key Operator Service Role (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/81a9662b-bebf-436f-a333-f67b29880f12'
  // Reader and Data Access (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/c12c1c16-33a1-487b-954d-41c89c60f349'
]

// deploy a storage account for the orchestrator
resource storage 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: truncatedStorageAccountName
  location: region
  tags: tags
  sku: {
    name: 'Standard_RAGRS'
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
  name: '${storage.name}/default/orchestratorprivate'
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
    containerName: 'orchestratorprivate'
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
  name: orchestratorUAIName
  location: region
  tags: tags
  dependsOn: [
    storage // ensure the storage exists BEFORE we do UAI role assignments
  ]
}

var identityPrincipalId = identityType == 'UserAssigned' ? uai.properties.principalId : compute.identity.principalId
var identityName = identityType == 'UserAssigned' ? uai.name : compute.name
var userAssignedIdentities = identityType == 'SystemAssigned' ? null : {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}

// provision a compute cluster, and assign the user assigned identity to it
resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2022-06-01-preview' = {
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

// assign the role orch compute should have with orch storage
resource orchToOrchRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for roleId in orchToOrchRoleDefinitionIds: {
  scope: storage
  name: guid(machineLearningName, region, storage.id, roleId, identityName)
  properties: {
    roleDefinitionId: roleId
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}]

// output the orchestrator config for next actions (permission model)
output identity string = identityPrincipalId
output storage string = storage.name
output compute string = compute.name
output region string = region
