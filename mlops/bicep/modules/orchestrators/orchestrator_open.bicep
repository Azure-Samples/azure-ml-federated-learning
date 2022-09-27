// This BICEP script will provision an AzureML workspace
// without any specific security settings.

// IMPORTANT: This setup is intended only for demo purpose. The data is still accessible
// by the users when opening the storage accounts, and data exfiltration is easy.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Machine learning workspace name')
param machineLearningName string

// optional parameters
@description('Machine learning workspace display name')
param machineLearningFriendlyName string = machineLearningName

@description('Machine learning workspace description')
param machineLearningDescription string = 'Federated Learning open demo orchestrator workspace'

@description('Specifies the location of the Azure Machine Learning workspace and dependent resources.')
param location string = resourceGroup().location

@description('Specifies whether to reduce telemetry collection and enable additional encryption.')
param hbi_workspace bool = false

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('Resource ID of the application insights resource')
param applicationInsightsName string = 'appi-${machineLearningName}'

@description('Resource ID of the container registry resource')
param containerRegistryName string = replace('cr-${machineLearningName}','-','') // replace because only alphanumeric characters are supported

@description('Resource ID of the key vault resource')
param keyVaultName string = 'kv-${machineLearningName}'

@description('Resource ID of the storage account resource')
param storageAccountName string =replace('st-${machineLearningName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the default compute cluster in orchestrator')
param orchestratorComputeName string = 'cpu-cluster-orchestrator'

@description('VM size for the default compute cluster')
param orchestratorComputeSKU string = 'Standard_DS3_v2'

@description('Name of the UAI for the default compute cluster')
param orchestratorUAIName string = '${machineLearningName}-orchestrator-uai'

@description('Role definition ID for the compute towards the internal storage')
param orchToOrchRoleDefinitionIds array = [
  // see https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles
  // Storage Blob Data Contributor (read,write,delete)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  // Storage Account Key Operator Service Role (list keys)
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/81a9662b-bebf-436f-a333-f67b29880f12'
]


resource storageAccount 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: storageAccountName
  location: location
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

resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: []
    enableSoftDelete: true
  }
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: (((location == 'eastus2') || (location == 'westcentralus')) ? 'southcentralus' : location)
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2021-08-01-preview' = {
  sku: {
    name: 'Standard'
  }
  name: containerRegistryName
  location: location
  tags: tags
  properties: {
    adminUserEnabled: true
  }
}

resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2021-07-01' = {
  identity: {
    type: 'SystemAssigned'
  }
  name: machineLearningName
  location: location
  tags: tags
  properties: {
    // workspace organization
    friendlyName: machineLearningFriendlyName
    description: machineLearningDescription

    // dependent resources
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: applicationInsights.id
    containerRegistry: containerRegistry.id
    hbiWorkspace: hbi_workspace
  }
}

// provision a user assigned identify for this silo
resource orchestratorUAI 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: orchestratorUAIName
  location: location
  tags: tags
}

// provision a compute cluster, and assign the user assigned identity to it
resource orchestratorCompute 'Microsoft.MachineLearningServices/workspaces/computes@2022-06-01-preview' = {
  name: '${machineLearningName}/${orchestratorComputeName}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${orchestratorUAI.name}': {}
    }
  }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: orchestratorComputeSKU
      subnet: json('null')
      osType: 'Linux'
      scaleSettings: {
        maxNodeCount: 4
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }
    }
  }
  dependsOn: [
    machineLearning
  ]
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
output workspace string = machineLearning.name
output region string = location
