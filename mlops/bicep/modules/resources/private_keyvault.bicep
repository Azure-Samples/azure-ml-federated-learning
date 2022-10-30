// Creates a KeyVault with Private Link Endpoint
@description('The Azure Region to deploy the resources into')
param location string = resourceGroup().location

@description('Tags to apply to the Key Vault Instance')
param tags object = {}

@description('The name of the Key Vault')
param keyvaultName string

@description('The Subnet ID where the Key Vault Private Link is to be created')
param subnetId string

@description('The VNet ID where the Key Vault Private Link is to be created')
param virtualNetworkId string

@description('Name of the private DNS zone')
param privateDNSZoneName string = 'privatelink${environment().suffixes.keyvaultDns}'

@description('Location of the private DNS zone')
param privateDNSZoneLocation string = 'global'

resource keyVault 'Microsoft.KeyVault/vaults@2021-10-01' = {
  name: keyvaultName
  location: location
  tags: tags
  properties: {
    createMode: 'default'
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: false
    enableSoftDelete: true
    enableRbacAuthorization: true
    enablePurgeProtection: true
    publicNetworkAccess: 'Disabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
    }
    sku: {
      family: 'A'
      name: 'standard'
    }
    softDeleteRetentionInDays: 7
    tenantId: subscription().tenantId
  }
}

module privateEndpoint '../networking/private_endpoint.bicep' = {
  name: '${keyVault.name}-endpoint-to-vnet'
  scope: resourceGroup()
  params: {
    tags: tags
    location: keyVault.location
    resourceServiceId: keyVault.id
    resourceName: keyVault.name
    virtualNetworkId: virtualNetworkId
    subnetId: subnetId
    privateDNSZoneName: privateDNSZoneName
    privateDNSZoneLocation: privateDNSZoneLocation
    groupId: 'vault'
  }
}

output keyvaultId string = keyVault.id
