// Creates a private link endpoint for a given resource
targetScope = 'resourceGroup'

@description('Azure region of the deployment')
param location string = resourceGroup().location

@description('Service ID of the resource to create private link endpoint to')
param resourceServiceId string

@description('Name of the storage blob private link endpoint')
param pleRootName string

@description('Resource ID of the subnet')
param subnetId string

@description('use privateIPAddress to assign a specific static IP address to PLE')
param useStaticIPAddress bool = false

@description('Specify the private IP address on the subnet.')
param privateIPAddress string = ''

@description('Name of the existing DNS zone to add the PLE to')
param privateDNSZoneName string

@description('Name of the DNS zone group to add to the PLE')
param groupId string

@description('Tags to add to the resources')
param tags object = {}

var ipConfigurationsDefinition = useStaticIPAddress ? [{
  name: '${pleRootName}-ipconfig'
  properties: {
    groupId: groupId
    memberName: groupId
    privateIPAddress: privateIPAddress
  }
}] : []

resource privateEndpoint 'Microsoft.Network/privateEndpoints@2022-01-01' = {
  name: pleRootName
  location: location
  tags: tags
  properties: {
    ipConfigurations: ipConfigurationsDefinition
    privateLinkServiceConnections: [ {
      name: pleRootName
      properties: {
        groupIds: [ groupId ]
        privateLinkServiceId: resourceServiceId
        privateLinkServiceConnectionState: {
          status: 'Approved'
          description: 'Auto-Approved'
          actionsRequired: 'None'
        }
      }
    }]
    subnet: {
      id: subnetId
    }
  }
}

resource privateDNSZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: privateDNSZoneName
}

resource privateEndpointDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2020-06-01' = {
  name: '${groupId}-PrivateDnsZoneGroup'
  parent: privateEndpoint
  properties:{
    privateDnsZoneConfigs: [
      {
        name: privateDNSZone.name
        properties:{
          privateDnsZoneId: privateDNSZone.id
        }
      }
    ]
  }
}

output name string = privateEndpoint.name
output id string = privateEndpoint.id
