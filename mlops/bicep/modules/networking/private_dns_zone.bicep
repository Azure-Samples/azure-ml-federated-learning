// Provision a private DNS Zone

@description('Name of the private DNS zone')
param name string

@description('Location of the private DNS zone (default: global)')
param location string = 'global'

@description('Tags for curation of resources')
param tags object = {}

@description('Optional: link the private DNS zone to a given virtual network')
param linkToVirtualNetworkId string = ''

// create the dns zone
resource privateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: name
  location: location
  tags: tags 
}

// create the link
resource privateDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = if (!empty(linkToVirtualNetworkId)) {
  name: uniqueString(linkToVirtualNetworkId, name, location)
  parent: privateDnsZone
  location: location
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: linkToVirtualNetworkId
    }
  }
}

output name string = privateDnsZone.name
output id string = privateDnsZone.id
