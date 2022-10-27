// Peers two vnet (from different regions)
// see https://learn.microsoft.com/en-us/azure/virtual-network/virtual-network-peering-overview

targetScope = 'resourceGroup'

@description('Set the local VNet name')
param existingVirtualNetworkNameSource string

@description('Set the remote VNet name')
param existingVirtualNetworkNameTarget string

@description('Sets the remote VNet Resource group')
param existingVirtualNetworkNameTargetResourceGroupName string = resourceGroup().name

param useGatewayFromSourceToTarget bool = false
param allowVirtualNetworkAccess bool = true

resource _vnet_peering 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2022-01-01' = {
  name: '${existingVirtualNetworkNameSource}/peering-to-${existingVirtualNetworkNameTarget}'
  properties: {
    allowVirtualNetworkAccess: allowVirtualNetworkAccess
    allowForwardedTraffic: false
    allowGatewayTransit: false
    useRemoteGateways: useGatewayFromSourceToTarget
    remoteVirtualNetwork: {
      id: resourceId(existingVirtualNetworkNameTargetResourceGroupName, 'Microsoft.Network/virtualNetworks', existingVirtualNetworkNameTarget)
    }
  }
}

resource _vnet_peering_back 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2022-01-01' = {
  name: '${existingVirtualNetworkNameTarget}/peering-to-${existingVirtualNetworkNameSource}'
  properties: {
    allowVirtualNetworkAccess: allowVirtualNetworkAccess
    allowForwardedTraffic: false
    allowGatewayTransit: useGatewayFromSourceToTarget
    useRemoteGateways: false
    remoteVirtualNetwork: {
      id: resourceId(resourceGroup().name, 'Microsoft.Network/virtualNetworks', existingVirtualNetworkNameSource)
    }
  }
}

output id string = _vnet_peering.id
