// This BICEP script will provision an AzureML workspace
// without any specific security settings.

// IMPORTANT: This setup is intended only for demo purpose. The data is still accessible
// by the users when opening the storage accounts, and data exfiltration is easy.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Base name to create all the resources')
@minLength(2)
@maxLength(20)
param baseName string

@description('Machine learning workspace name')
param machineLearningName string = 'aml-${baseName}'

// optional parameters
@description('Machine learning workspace display name')
param machineLearningFriendlyName string = '${baseName} (sandbox/open)'

@description('Machine learning workspace description')
param machineLearningDescription string = 'An open AzureML workspace with no specific security settings.'

@description('Specifies the location of the Azure Machine Learning workspace and dependent resources.')
param location string = resourceGroup().location

@description('Specifies whether to reduce telemetry collection and enable additional encryption.')
param hbiWorkspace bool = false

@description('Name of the application insights log analytics workspace')
param appInsightLogAnalyticsName string = 'logs-${baseName}'

@description('Name of the application insights resource')
param applicationInsightsName string = 'appi-${baseName}'

@description('Name of the container registry resource')
param containerRegistryName string = replace('cr-${baseName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the workspace secrets store (shared keyvault) resource')
param keyVaultName string = 'ws-shkv-${baseName}'

@description('Name of the storage account resource')
param storageAccountName string = replace('st-${baseName}','-','') // replace because only alphanumeric characters are supported

@description('Tags to curate the resources in Azure.')
param tags object = {}


// ************************************************************
// Dependent resources for the Azure Machine Learning workspace
// ************************************************************

resource storage 'Microsoft.Storage/storageAccounts@2022-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    // Allow or disallow public access to all blobs or containers in the storage account.
    allowBlobPublicAccess: false

    // Allow or disallow cross AAD tenant object replication.
    allowCrossTenantReplication: false

    // Allow or disallow public network access to Storage Account.
    publicNetworkAccess: 'Enabled'

    // Encryption settings to be used for server-side encryption for the storage account.
    encryption: {
      keySource: 'Microsoft.Storage'
      requireInfrastructureEncryption: false
      services: {
        blob: {
          enabled: true
          keyType: 'Account'
        }
        file: {
          enabled: true
          keyType: 'Account'
        }
        queue: {
          enabled: true
          keyType: 'Service'
        }
        table: {
          enabled: true
          keyType: 'Service'
        }
      }
    }

    // Allows https traffic only to storage service if sets to true. 
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

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2021-09-01' = {
  name: containerRegistryName
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2021-06-01' = {
  name: appInsightLogAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Disabled'
  }
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: (((location == 'eastus2') || (location == 'westcentralus')) ? 'southcentralus' : location)
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
    Flow_Type: 'Bluefield'
  }
}

// ********************************
// Azure Machine Learning workspace
// ********************************

resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2022-10-01' = {
  name: machineLearningName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    // workspace organization
    friendlyName: machineLearningFriendlyName
    description: machineLearningDescription

    // dependent resources
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: applicationInsights.id
    containerRegistry: containerRegistry.id
    hbiWorkspace: hbiWorkspace

    // configuration for workspaces with private link endpoint
    publicNetworkAccess: 'Enabled'    
  }
}


// *******
// Outputs
// *******

output storageName string = storage.name
output workspaceName string = machineLearning.name
output region string = location
output workspaceSecretStoreId string = keyVault.id
output workspaceSecretStoreName string = keyVault.name
