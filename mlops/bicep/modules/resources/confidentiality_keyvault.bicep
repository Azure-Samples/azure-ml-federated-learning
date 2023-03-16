// Create a keyvault dedicated to "confidentiality"
// i.e. store keys for data encryption at rest
// creates a development RSA key as consumed by our confidentiality samples

@description('Name of the keyvault.')
param keyVaultName string

@description('Region of the keyvault.')
param region string = resourceGroup().location

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('List of principal ids that will be allowed cryptographic operations (Key Vault Crypto User).')
param identitiesEnabledCryptoOperations array = []

@description('Just in case you need a second list of identities as argument.')
param secondaryIdentitiesEnabledCryptoOperations array = []

param createDevRSAKey bool = true
param devKeyName string = 'dev-rsa-key'

// Create a "confidentiality" keyvault external to the workspace
resource confidentialityKeyVault 'Microsoft.KeyVault/vaults@2022-07-01' = {
  name: keyVaultName
  location: region
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    createMode: 'default'

    // loss protection
    enablePurgeProtection: true
    enableSoftDelete: true

    // permissions
    enableRbacAuthorization: true // we'll use rbac to allow UAIs to perform crypto operations
    publicNetworkAccess: 'Enabled'
    accessPolicies: []

    // features
    enabledForDeployment: false
    enabledForDiskEncryption: true
    enabledForTemplateDeployment: false
  }
}

resource key 'Microsoft.KeyVault/vaults/keys@2022-07-01' = if (createDevRSAKey) {
  parent: confidentialityKeyVault
  name: devKeyName
  properties: {
    kty: 'RSA'
    keyOps: [
      'encrypt'
      'decrypt'
      'sign'
      'verify'
    ]
    keySize: 2048
  }
}

var keyVaultCryptoUserRoleId = '12338af0-0e69-4776-bea7-57ae8d297424' // Key Vault Crypto User

resource roleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for identityPrincipalId in identitiesEnabledCryptoOperations: {
  scope: confidentialityKeyVault
  name: guid(resourceGroup().id, keyVaultName, identityPrincipalId, keyVaultCryptoUserRoleId)
  properties: {
    roleDefinitionId: '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/${keyVaultCryptoUserRoleId}'
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}]

resource secondaryRoleAssignments 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for identityPrincipalId in secondaryIdentitiesEnabledCryptoOperations: {
  scope: confidentialityKeyVault
  name: guid(resourceGroup().id, keyVaultName, identityPrincipalId, keyVaultCryptoUserRoleId)
  properties: {
    roleDefinitionId: '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/${keyVaultCryptoUserRoleId}'
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}]
