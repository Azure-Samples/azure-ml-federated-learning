param name string
param location string
param tags object = {}

param linkToVirtualNetworkId string = ''

resource privateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: name
  location: location
  tags: tags 
}

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

output id string = privateDnsZone.id
