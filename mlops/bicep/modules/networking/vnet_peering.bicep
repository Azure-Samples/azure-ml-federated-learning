// Peers two vnet (from different regions)
// see https://learn.microsoft.com/en-us/azure/virtual-network/virtual-network-peering-overview

targetScope = 'resourceGroup'

@description('Set the local VNet name')
param existingVirtualNetworkName1 string

@description('Set the remote VNet name')
param existingVirtualNetworkName2 string

@description('Sets the remote VNet Resource group')
param existingVirtualNetworkName2ResourceGroupName string = resourceGroup().name

resource _vnet_peering 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2021-02-01' = {
  name: '${existingVirtualNetworkName1}/peering-to-${existingVirtualNetworkName2}'
  properties: {
    allowVirtualNetworkAccess: true
    allowForwardedTraffic: false
    allowGatewayTransit: false
    useRemoteGateways: false
    remoteVirtualNetwork: {
      id: resourceId(existingVirtualNetworkName2ResourceGroupName, 'Microsoft.Network/virtualNetworks', existingVirtualNetworkName2)
    }
  }
}

resource _vnet_peering_back 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2021-02-01' = {
  name: '${existingVirtualNetworkName2}/peering-to-${existingVirtualNetworkName1}'
  properties: {
    allowVirtualNetworkAccess: true
    allowForwardedTraffic: false
    allowGatewayTransit: false
    useRemoteGateways: false
    remoteVirtualNetwork: {
      id: resourceId(resourceGroup().name, 'Microsoft.Network/virtualNetworks', existingVirtualNetworkName1)
    }
  }
}

output id string = _vnet_peering.id
