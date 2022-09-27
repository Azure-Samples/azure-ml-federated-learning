// This BICEP script will provision an orchestrator compute+storage pair
// in a given AzureML workspace, without any specific security settings.

// IMPORTANT: This setup is intended only for demo purpose. The data is still accessible
// by the users when opening the storage accounts, and data exfiltration is easy.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach orchestrator to')
param machineLearningName string

@description('Specifies the location of the orchestrator resources.')
param region string = resourceGroup().location

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('Name of the storage account resource to create for the orchestrator')
param storageAccountName string = replace('st-${machineLearningName}-orch','-','') // replace because only alphanumeric characters are supported

@description('Name of the default compute cluster for orchestrator')
param computeName string = 'cpu-cluster-orchestrator'

@description('VM size for the default compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@description('Name of the UAI for the default compute cluster')
param orchestratorUAIName string = 'uai-${machineLearningName}-orchestrator'

@description('Role definition IDs for the compute towards the internal storage')
param orchToOrchRoleDefinitionIds array = [
  // see https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles
  // Storage Blob Data Contributor (read,write,delete)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  // Storage Account Key Operator Service Role (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/81a9662b-bebf-436f-a333-f67b29880f12'
]


resource storageAccount 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: storageAccountName
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

// provision a user assigned identify for this silo
resource orchestratorUAI 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: orchestratorUAIName
  location: region
  tags: tags
}

// provision a compute cluster, and assign the user assigned identity to it
resource orchestratorCompute 'Microsoft.MachineLearningServices/workspaces/computes@2022-06-01-preview' = {
  name: '${machineLearningName}/${computeName}'
  location: region
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${orchestratorUAI.name}': {}
    }
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
}

// assign the role orch compute should have with orch storage
resource orchToOrchRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for roleId in orchToOrchRoleDefinitionIds: {
  scope: storageAccount
  name: guid(storageAccount.id, roleId, orchestratorUAI.name)
  properties: {
    roleDefinitionId: roleId
    principalId: orchestratorUAI.properties.principalId
    principalType: 'ServicePrincipal'
  }
}]

// output the orchestrator config for next actions (permission model)
output uaiPrincipalId string = orchestratorUAI.properties.principalId
output storage string = storageAccount.name
output compute string = orchestratorCompute.name
output region string = region
