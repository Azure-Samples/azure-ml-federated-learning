// Creates a KeyVault with Private Link Endpoint

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

@description('The Azure Region to deploy the resources into')
param location string = resourceGroup().location

@description('Tags to apply to the Key Vault Instance')
param tags object = {}

@description('The name of the Key Vault')
param keyvaultName string

@description('The Subnet ID where the Key Vault Private Link is to be created')
param subnetId string

@description('Name of the private DNS zone')
param privateDNSZoneName string = 'privatelink${environment().suffixes.keyvaultDns}'

resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' = {
  name: keyvaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    createMode: 'default'
    sku: {
      name: 'standard'
      family: 'A'
    }

    // usage
    enabledForDeployment: false
    enabledForDiskEncryption: true
    enabledForTemplateDeployment: false
    enableRbacAuthorization: true

    // loss protection
    enablePurgeProtection: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7

    // networking
    publicNetworkAccess: 'Disabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
    }
  }
}

module privateEndpoint '../networking/private_endpoint.bicep' = {
  name: '${keyVault.name}-endpoint-to-vnet'
  scope: resourceGroup()
  params: {
    tags: tags
    location: keyVault.location
    resourceServiceId: keyVault.id
    pleRootName: 'ple-${keyVault.name}'
    subnetId: subnetId
    privateDNSZoneName: privateDNSZoneName
    groupId: 'vault'
  }
}

output keyvaultId string = keyVault.id
