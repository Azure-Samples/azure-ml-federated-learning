// Creates an Azure Container Registry with Azure Private Link endpoint

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

@description('Azure region of the deployment')
param location string

@description('Tags to add to the resources')
param tags object = {}

@description('Container registry name')
param containerRegistryName string

@description('Resource ID of the subnet')
param subnetId string

@description('Name of the private DNS zone')
param privateDNSZoneName string = 'privatelink${environment().suffixes.acrLoginServer}'

var containerRegistryNameCleaned = replace(containerRegistryName, '-', '')

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2021-09-01' = {
  name: containerRegistryNameCleaned
  location: location
  tags: tags
  sku: {
    name: 'Premium'
  }
  properties: {
    adminUserEnabled: true
    dataEndpointEnabled: false
    networkRuleBypassOptions: 'AzureServices'
    publicNetworkAccess: 'Disabled'
    networkRuleSet: {
      defaultAction: 'Deny'
    }
    policies: {
      quarantinePolicy: {
        status: 'disabled'
      }
      retentionPolicy: {
        status: 'enabled'
        days: 7
      }
      trustPolicy: {
        status: 'disabled'
        type: 'Notary'
      }
    }
    zoneRedundancy: 'Disabled'
  }
}

module privateEndpoint '../networking/private_endpoint.bicep' = {
  name: '${containerRegistry.name}-endpoint-to-vnet'
  scope: resourceGroup()
  params: {
    tags: tags
    location: location
    resourceServiceId: containerRegistry.id
    resourceName: containerRegistry.name
    subnetId: subnetId
    privateDNSZoneName: privateDNSZoneName
    groupId: 'registry'
  }
}

output containerRegistryId string = containerRegistry.id
