// Creates a machine learning workspace, private endpoints and compute resources
// Compute resources include a GPU cluster, CPU cluster, compute instance and attached private AKS cluster

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Machine learning workspace name to attach orchestrator compute+storate to')
param machineLearningName string

@description('Specifies the location of the orchestrator.')
param region string = resourceGroup().location

@description('Tags to curate the resources in Azure.')
param tags object = {}

param orchestratorName string = 'orchestrator'

@description('Name of the storage account resource')
param orchestratorStorageAccountName string = replace('st-${machineLearningName}-orch','-','') // replace because only alphanumeric characters are supported

@description('Name of the default compute cluster in orchestrator')
param orchestratorComputeName string = 'cpu-cluster-orchestrator'

@description('VM size for the compute cluster')
param orchestratorComputeSKU string = 'Standard_DS3_v2'

@description('Number of nodes in the compute cluster')
param orchestratorComputeNodes int = 4

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${machineLearningName}-orchestrator'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${machineLearningName}-orchestrator'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Training subnet address prefix')
param trainingSubnetPrefix string = '10.0.0.0/24'

@description('Name of the UAI for the default compute cluster')
param orchestratorUAIName string = 'uai-${machineLearningName}-orchestrator'

@description('Role definition ID for the compute towards the internal storage')
param orchToOrchRoleDefinitionId string = '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor (read,write,delete)


// Virtual network and network security group
module nsg '../networking/nsg.bicep' = { 
  name: '${nsgResourceName}-deployment'
  params: {
    location: region
    tags: tags 
    nsgName: nsgResourceName
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

module orchestratorStorageDeployment '../secure_resources/storage.bicep' = {
  name: '${orchestratorStorageAccountName}-deployment'
  params: {
    location: region
    storageName: orchestratorStorageAccountName
    storagePleBlobName: 'ple-${machineLearningName}-orchestrator-st-blob'
    storagePleFileName: 'ple-${machineLearningName}-orchestrator-st-file'
    storageSkuName: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

// create a "private" container in the storage account
// this one will be readable only by silo compute
resource orchStoragePrivateContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  name: '${orchestratorStorageAccountName}/default/orchprivate'
  properties: {
    metadata: {}
    publicAccess: 'None'
  }
  dependsOn: [
    orchestratorStorageDeployment
  ]
}

// attach as a datastore in AzureML
resource orchAzureMLPrivateDatastore 'Microsoft.MachineLearningServices/workspaces/datastores@2022-06-01-preview' = {
  name: '${machineLearningName}/orchestrator_data'
  properties: {
    credentials: {
      credentialsType: 'None'
      // For remaining properties, see DatastoreCredentials objects
    }
    description: 'Silo private storage in region ${region}'
    properties: {}
    datastoreType: 'AzureBlob'
    // For remaining properties, see DatastoreProperties objects
    accountName: orchestratorStorageAccountName
    containerName: 'orchprivate'
    // endpoint: 'string'
    // protocol: 'string'
    resourceGroup: resourceGroup().name
    // serviceDataAccessAuthIdentity: 'string'
    subscriptionId: subscription().subscriptionId
  }
  dependsOn: [
    orchStoragePrivateContainer
  ]
}


resource workspace 'Microsoft.MachineLearningServices/workspaces@2020-09-01-preview' existing = {
  name: machineLearningName
  scope: resourceGroup()
}
module workspacePrivateEndpointDeployment '../networking/aml_ple.bicep' = {
  name: '${machineLearningName}-ws-orch-ple-deployment'
  params: {
    location: region
    tags: tags
    workspaceArmId: workspace.id
    machineLearningPleName: 'ple-${machineLearningName}-workspace'
    virtualNetworkId: vnet.outputs.id
    subnetId: '${vnet.outputs.id}/subnets/snet-training'
  }
}

// provision a user assigned identify for this silo
resource orchestratorUAI 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: orchestratorUAIName
  location: region
  tags: tags
}

// provision a compute cluster, and assign the user assigned identity to it
resource orchestratorCompute 'Microsoft.MachineLearningServices/workspaces/computes@2022-06-01-preview' = {
  name: '${machineLearningName}/${orchestratorComputeName}'
  location: region
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${orchestratorUAI.name}': {}
    }
  }
  properties: {
    computeType: 'AmlCompute'
    description: 'Orchestration cluster with R/W access to the workspace storage'
    // disableLocalAuth: true
    properties: {
      vmPriority: 'Dedicated'      
      vmSize: orchestratorComputeSKU
      enableNodePublicIp: true
      isolatedNetwork: false
      osType: 'Linux'
      remoteLoginPortPublicAccess: 'Disabled'
      scaleSettings: {
        maxNodeCount: orchestratorComputeNodes
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }
      subnet: {
        id: '${vnet.outputs.id}/subnets/snet-training'
      }
    }
  }
  dependsOn: [
    //workspacePrivateEndpointDeployment
  ]
}

// role of silo compute -> silo storage
resource orchestratorStorage 'Microsoft.Storage/storageAccounts@2021-06-01' existing = {
  name: orchestratorStorageAccountName
  scope: resourceGroup()
}
// assign the role orch compute should have with orch storage
resource orchToOrchRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: orchestratorStorage
  name: guid(orchestratorStorage.name, orchToOrchRoleDefinitionId, orchestratorUAI.name)
  properties: {
    roleDefinitionId: orchToOrchRoleDefinitionId
    principalId: orchestratorUAI.properties.principalId
    principalType: 'ServicePrincipal'
  }
  dependsOn: [
    orchestratorStorageDeployment
  ]
}

output uaiPrincipalId string = orchestratorUAI.properties.principalId
output storage string = orchestratorStorageAccountName
output compute string = orchestratorCompute.name
output region string = region
output uaiId string = orchestratorUAI.id
output vnetId string = vnet.outputs.id
