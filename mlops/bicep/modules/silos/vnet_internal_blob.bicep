// This BICEP script will provision an internal silo compute+storage pair
// in a given AzureML workspace, using a vnet and subnet to secure
// the communication between compute and storage, plus managed identity
// for permissions management.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach silo to.')
param machineLearningName string

@description('Specifies the location of the silo resources.')
param region string

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('Specifies the name of the silo storage account.')
param orchestratorStorageAccountName string // needed to set permissions towards orchestrator storage

// optional parameters
@description('Specifies the base name for creating resources.')
param siloName string = 'silo-${region}'

@description('Specifies the name of the storage account to provision.')
param storageAccountName string = 'st${replace('${siloName}', '-', '')}'

@description('Specifies the name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = 'datatore_${replace('${siloName}', '-', '_')}'

@description('Specifies the name of the compute cluster to provision.')
param computeName string = 'cpu-cluster-${siloName}'

@description('VM size for the default compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param identityType string = 'UserAssigned'

@description('Specifies the name of the User Assigned Identity to provision.')
param uaiName string = 'uai-${siloName}'

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${siloName}'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${siloName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Training subnet address prefix')
param trainingSubnetPrefix string = '10.0.0.0/24'

@description('Enable compute node public IP')
param enableNodePublicIp bool = true

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

var storageAccountCleanName = substring(storageAccountName, 0, min(length(storageAccountName),24))

// deploy a storage account for the silo
module storageDeployment '../secure_resources/storage.bicep' = {
  name: '${storageAccountCleanName}-deployment'
  params: {
    location: region
    storageName: storageAccountCleanName
    storageSKU: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

// create a "private" container in the storage account
// this one will be readable only by silo compute
resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  name: '${storageAccountCleanName}/default/siloprivate'
  properties: {
    metadata: {}
    publicAccess: 'None'
  }
  dependsOn: [
    storageDeployment
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
    description: 'Silo private storage in region ${region}'
    properties: {}
    datastoreType: 'AzureBlob'
    // For remaining properties, see DatastoreProperties objects
    accountName: storageAccountCleanName
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
    storageDeployment  // ensure the storage exists BEFORE we do UAI role assignments
  ]
}

var identityPrincipalId = identityType == 'UserAssigned' ? uai.properties.principalId : compute.identity.principalId
var identityName = identityType == 'UserAssigned' ? uai.name : compute.name
var userAssignedIdentities = identityType == 'SystemAssigned' ? null : {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}

// provision a compute cluster for the silo and assigned the silo UAI to it
resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2022-06-01-preview' = {
  name: '${machineLearningName}/${computeName}'
  location: region
  identity: {
    type: identityType
    userAssignedIdentities: userAssignedIdentities
  }
  properties: {
    computeType: 'AmlCompute'
    description: 'Silo cluster with R/W access to the silo storage'
    properties: {
      vmPriority: 'Dedicated'      
      vmSize: computeSKU
      enableNodePublicIp: enableNodePublicIp
      isolatedNetwork: false
      osType: 'Linux'
      remoteLoginPortPublicAccess: 'Disabled'
      scaleSettings: {
        maxNodeCount: computeNodes
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }
      subnet: {
        id: '${vnet.outputs.id}/subnets/snet-training'
      }
    }
  }
  dependsOn: [
    storageDeployment  // ensure the storage exists BEFORE we do UAI role assignments
  ]
}

// role of silo compute -> silo storage
resource storage 'Microsoft.Storage/storageAccounts@2021-06-01' existing = {
  name: storageAccountCleanName
  scope: resourceGroup()
}
resource siloToSiloRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for roleId in siloToSiloRoleDefinitionIds: {
  scope: storage
  name: guid(siloName, region, storage.name, roleId, identityName)
  properties: {
    roleDefinitionId: roleId
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
  dependsOn: [
    storageDeployment
    compute
  ]
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
  dependsOn: [
    storageDeployment
    compute
  ]
}]

// output the orchestrator config for next actions (permission model)
output identity string = identityPrincipalId
output storage string = storage.name
output storageServiceId string = storage.id
output container string = container.name
output datastore string = datastore.name
output compute string = compute.name
output region string = region
output vnetId string = vnet.outputs.id
output subnetId string = '${vnet.outputs.id}/subnets/snet-training'
