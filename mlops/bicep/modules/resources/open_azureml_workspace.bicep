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

@description('Name of the application insights resource')
param applicationInsightsName string = 'appi-${machineLearningName}'

@description('Name of the container registry resource')
param containerRegistryName string = replace('cr-${machineLearningName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the key vault resource')
param keyVaultName string = 'kv-${machineLearningName}'

@description('Name of the storage account resource')
param storageAccountName string = replace('st-${machineLearningName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the default compute cluster in orchestrator')
param defaultComputeName string = 'cpu-cluster'

@description('VM size for the default compute cluster')
param defaultComputeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param defaultComputeNodes int = 4


resource storageAccount 'Microsoft.Storage/storageAccounts@2022-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
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

resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' = {
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

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2021-09-01' = {
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

resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2022-05-01' = {
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

    // configuration for workspaces with private link endpoint
    imageBuildCompute: defaultComputeName
    publicNetworkAccess: 'Enabled'    
  }
}

// provision a compute cluster, and assign the user assigned identity to it
resource defaultCompute 'Microsoft.MachineLearningServices/workspaces/computes@2022-05-01' = {
  name: '${machineLearningName}/${defaultComputeName}'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: defaultComputeSKU
      subnet: json('null')
      osType: 'Linux'
      scaleSettings: {
        maxNodeCount: defaultComputeNodes
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }
    }
  }
  dependsOn: [
    machineLearning
  ]
}

// output the orchestrator config for next actions (permission model)
output storage string = storageAccount.name
output compute string = defaultCompute.name
output workspace string = machineLearning.name
output region string = location
