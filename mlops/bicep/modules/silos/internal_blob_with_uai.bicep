// Provision a basic Internal Silo secured by UAI
//
// Given an AzureML workspace, and a specific region, this BICEP script will provision:
// - a new blob storage account in the given region
// - create 1 containers in this storage for private silo data
// - 1 AzureML compute cluster in that same region, attached to the AzureML workspace
// - 2 AzureML datastores for each of the private/shared containers
// - a User Assigned Identity

// TODO: use UAI to set permissions properly on the containers
// TODO: figure out if we need other params (those that are commented out)

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// these parameters are required
@description('Specifies the name of the workspace.')
param workspaceName string

@description('Specifies the region of the silo (for storage + compute).')
param region string

// below parameters are optionals and have default values
@description('Specifies the name of the storage account to provision.')
param storageAccountName string = '${replace('${workspaceName}', '-', '')}silo${region}'

@description('Specifies the name of the compute cluster to provision.')
param computeClusterName string = 'cpu-cluster-${region}'
param siloComputeSKU string = 'Standard_DS3_v2'

@description('Specifies the name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = 'silo_datatore_${region}'

@description('Specifies the name of the User Assigned Identity to provision.')
param uaiName string = '${replace('${workspaceName}', '-', '')}-uai-${region}'

// permissions model arguments
param orchestratorUAIPrincipalID string
param orchestratorStorageAccountName string
param siloToSiloRoleDefinitionId string = ''
param orchToSiloRoleDefinitionId string = ''
param siloToOrchRoleDefinitionId string = ''


// deploy a storage account for the silo
resource siloStorageAccount 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: storageAccountName
  location: region
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
    // defaultEncryptionScope: 'string'
    // denyEncryptionScopeOverride: bool
    // enableNfsV3AllSquash: bool
    // enableNfsV3RootSquash: bool
    // immutableStorageWithVersioning: {
    //   enabled: bool
    // }
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
  name: uaiName
  location: region
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
        // nodeIdleTimeBeforeScaleDown: '180' // TODO: "The NodeIdleTimeBeforeScaleDown string '180' does not conform to the W3C XML Schema Part for duration."
      }
    }
  }
}

resource siloToSiloRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(siloToSiloRoleDefinitionId)) {
  scope: siloStoragePrivateContainer
  name: guid(siloStoragePrivateContainer.name, siloToSiloRoleDefinitionId, siloUserAssignedIdentity.name)
  properties: {
    roleDefinitionId: siloToSiloRoleDefinitionId
    principalId: siloUserAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// assign the W-only permissions between orchestrator UAI and silo storage containers
resource orchestratorToSiloRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(orchToSiloRoleDefinitionId)) {
  scope: siloStoragePrivateContainer
  name: guid(siloStoragePrivateContainer.name, orchToSiloRoleDefinitionId, orchestratorUAIPrincipalID)
  properties: {
    roleDefinitionId: orchToSiloRoleDefinitionId
    principalId: orchestratorUAIPrincipalID
    principalType: 'ServicePrincipal'
  }
}

// assign the W-only permissions between silo UAI and orchestrator storage containers
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
output siloConfig object = {
  region: region
  storage: siloStorageAccount.name
  container: siloStoragePrivateContainer.name
  compute: siloAzureMLCompute.name
  datastore: siloAzureMLPrivateDatastore.name
  uaiPrincipalId: siloUserAssignedIdentity.properties.principalId
}
