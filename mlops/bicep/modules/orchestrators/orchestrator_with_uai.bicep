// This BICEP script will provision an AzureML workspace
// copied from initial script in this repo

// TODO: clean up and comment this script

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

@description('Specifies the name of the workspace.')
param workspaceName string
@description('Specifies the location of the Azure Machine Learning workspace and dependent resources.')
param location string = resourceGroup().location
@description('Specifies whether to reduce telemetry collection and enable additional encryption.')
param hbi_workspace bool = false

param orchestratorComputeName string = 'cpu-cluster'

var tenantId = subscription().tenantId
var storageAccountName_var = replace('st-${workspaceName}','-','') // replace because only alphanumeric characters are supported
var keyVaultName_var = 'kv-${workspaceName}'
var applicationInsightsName_var = 'appi-${workspaceName}'
var containerRegistryName_var = replace('cr-${workspaceName}','-','') // replace because only alphanumeric characters are supported
var workspaceName_var = workspaceName
var storageAccount = storageAccountName.id
var keyVault = keyVaultName.id
var applicationInsights = applicationInsightsName.id
var containerRegistry = containerRegistryName.id

resource storageAccountName 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: storageAccountName_var
  location: location
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

resource keyVaultName 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: keyVaultName_var
  location: location
  properties: {
    tenantId: tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: []
    enableSoftDelete: true
  }
}
resource applicationInsightsName 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName_var
  location: (((location == 'eastus2') || (location == 'westcentralus')) ? 'southcentralus' : location)
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}
resource containerRegistryName 'Microsoft.ContainerRegistry/registries@2021-08-01-preview' = {
  sku: {
    name: 'Standard'
  }
  name: containerRegistryName_var
  location: location
  properties: {
    adminUserEnabled: true
  }
}
resource azuremlWorkspace 'Microsoft.MachineLearningServices/workspaces@2021-07-01' = {
  identity: {
    type: 'SystemAssigned'
  }
  name: workspaceName_var
  location: location
  properties: {
    friendlyName: workspaceName_var
    storageAccount: storageAccount
    keyVault: keyVault
    applicationInsights: applicationInsights
    containerRegistry: containerRegistry
    hbiWorkspace: hbi_workspace
  }
}

// provision a user assigned identify for this silo
resource orchestratorUserAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: 'orchestrator-uai'
  location: location
}

// provision a compute cluster, and assign the user assigned identity to it
resource orchestratorCompute 'Microsoft.MachineLearningServices/workspaces/computes@2022-06-01-preview' = {
  name: '${workspaceName}/${orchestratorComputeName}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${orchestratorUserAssignedIdentity.name}': {}
    }
  }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_DS3_v2'
      subnet: json('null')
      osType: 'Linux'
      scaleSettings: {
        maxNodeCount: 4
        minNodeCount: 0
        // nodeIdleTimeBeforeScaleDown: '180' // TODO: "The NodeIdleTimeBeforeScaleDown string '180' does not conform to the W3C XML Schema Part for duration."
      }
    }
  }
  dependsOn: [
    azuremlWorkspace
  ]
}

// output the orchestrator config for next actions (permission model)
output orchestratorConfig object = {
  region: location
  workspace: azuremlWorkspace
  compute: orchestratorCompute.name
  storage: storageAccountName.name
  uaiPrincipalId: orchestratorUserAssignedIdentity.properties.principalId
}
