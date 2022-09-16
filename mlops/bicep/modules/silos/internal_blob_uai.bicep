// Provision a basic Internal Silo secured by UAI
//
// Given an AzureML workspace, and a specific region, this BICEP script will provision:
// - a new blob storage account in the given region with a specific container for FL
// - a new AzureML compute cluster in that same region, attached to the AzureML workspace
// - a new AzureML datastore registering the storage account+container in the AzureML workspace
// - a User Assigned Identity

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

@description('Specifies the name of the container for FL demo.')
param storageContainerName string = 'fldemocontainer'

@description('Specifies the name of the compute cluster to provision.')
param computeClusterName string = 'cpu-cluster-${region}'

@description('Specifies the name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = 'silo_datatore_${region}'

@description('Specifies the name of the User Assigned Identity to provision.')
param uaiName string = '${replace('${workspaceName}', '-', '')}-uai-${region}'



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

// create a container in the storage account
resource siloStorageContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  name: '${siloStorageAccount.name}/default/${storageContainerName}'
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

// deploy a user assigned identify for this silo
resource siloUserAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: uaiName
  location: region
}

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

resource siloAzureMLDatastore 'Microsoft.MachineLearningServices/workspaces/datastores@2022-06-01-preview' = {
  name: '${workspaceName}/${datastoreName}'
  properties: {
    credentials: {
      credentialsType: 'None'
      // For remaining properties, see DatastoreCredentials objects
    }
    description: 'Silo storage account in region ${region}'
    properties: {}
    datastoreType: 'AzureBlob'
    // For remaining properties, see DatastoreProperties objects
    accountName: siloStorageAccount.name
    containerName: storageContainerName
    // endpoint: 'string'
    // protocol: 'string'
    resourceGroup: resourceGroup().name
    // serviceDataAccessAuthIdentity: 'string'
    subscriptionId: subscription().subscriptionId
  }
}

resource siloAzureMLCompute 'Microsoft.MachineLearningServices/workspaces/computes@2020-09-01-preview' = {
  name: '${workspaceName}/${computeClusterName}'
  location: region
  // identity: {
  //   type: 'UserAssigned'
  //   userAssignedIdentities: {
  //     '/subscriptions/${subscription()}/resourceGroups/${resourceGroup()}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${siloUserAssignedIdentity.name}': {
  //         // principalId: siloUserAssignedIdentity.properties.principalId
  //         // clientId: siloUserAssignedIdentity.properties.clientId
  //     }
  //   }
  // }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_DS3_v2'
      subnet: json('null')
      osType: 'Linux'
      scaleSettings: {
        maxNodeCount: 4
        minNodeCount: 0
        // nodeIdleTimeBeforeScaleDown: '180' //  "The NodeIdleTimeBeforeScaleDown string '180' does not conform to the W3C XML Schema Part for duration."
      }
    }
  }
}
