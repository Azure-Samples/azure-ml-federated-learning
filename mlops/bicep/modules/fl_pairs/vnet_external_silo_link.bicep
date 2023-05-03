// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

// @description('Set true to provision PLEs necessary for private workspace to interact with the pair')
// param machineLearningIsPrivate bool = true

@description('Specifies the location of the pair resources.')
param pairRegion string = resourceGroup().location

@description('Base name used for creating all pair resources.')
param pairBaseName string

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('Name of the UAI for the jobs running in the pair')
param jobsUaiName string = 'uai-jobs-${pairBaseName}'

// @description('Name of the default compute cluster for the pair')
// param connectedClusterName string = '${pairBaseName}-arc'

@allowed([
    'Disabled'
    'Enabled'
])
param publicNetworkAccess string = 'Disabled'

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${pairBaseName}'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${pairBaseName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Subnet address prefix')
param computeSubnetPrefix string = '10.0.0.0/24'

@description('Subnet address prefix')
param endpointsSubnetPrefix string = '10.0.1.0/24'

@description('Enable compute node public IP')
param enableNodePublicIp bool = true

// Virtual network and network security group
module nsg '../networking/azureml_compute_nsg.bicep' = {
  name: '${nsgResourceName}-deployment'
  params: {
    location: pairRegion
    nsgName: nsgResourceName
    tags: tags
    workspaceRegion: machineLearningRegion
    enableNodePublicIp: enableNodePublicIp
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: '${vnetResourceName}-deployment'
  params: {
    location: pairRegion
    virtualNetworkName: vnetResourceName
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: vnetAddressPrefix
    subnets: [
      {
        name: 'compute'
        addressPrefix: computeSubnetPrefix
      }
      {
        name: 'endpoints'
        addressPrefix: endpointsSubnetPrefix
      }
    ]
    tags: tags
  }
}

// provision a user assigned identify for this compute
resource uaiJobs 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: jobsUaiName
  location: pairRegion
  tags: tags
}

// https://learn.microsoft.com/en-us/azure/azure-arc/kubernetes/private-link
resource privateLinkScopeForArc 'Microsoft.HybridCompute/privateLinkScopes@2022-11-10' = {
  name: '${pairBaseName}-privatelinkscope'
  location: pairRegion
  tags: tags
  properties: {
    publicNetworkAccess: publicNetworkAccess
  }
}

module arcPrivateDnsZone '../networking/private_dns_zone.bicep' = {
  name: '${pairBaseName}-arc-dns-zone'
  scope: resourceGroup()
  params: {
    name: 'privatelink.his.arc.azure.com'
    location: 'global'
    linkToVirtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module guestconfigPrivateDnsZone '../networking/private_dns_zone.bicep' = {
  name: '${pairBaseName}-guestconfig-dns-zone'
  scope: resourceGroup()
  params: {
    name: 'privatelink.guestconfiguration.azure.com'
    location: 'global'
    linkToVirtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

module kubernetesconfigPrivateDnsZone '../networking/private_dns_zone.bicep' = {
  name: '${pairBaseName}-k8sconfig-dns-zone'
  scope: resourceGroup()
  params: {
    name: 'privatelink.db.kubernetesconfiguration.azure.com'
    location: 'global'
    linkToVirtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

// create a private endpoint
resource privateEndpoint 'Microsoft.Network/privateEndpoints@2022-01-01' = {
  name: 'ple-${pairBaseName}-arc'
  location: pairRegion
  tags: tags
  properties: {
    privateLinkServiceConnections: [ {
      name: 'ple-${pairBaseName}-arc'
      properties: {
        groupIds: [ 'hybridcompute' ]
        privateLinkServiceId: privateLinkScopeForArc.id
      }
    }]
    subnet: {
      id: '${vnet.outputs.id}/subnets/endpoints'
    }
  }
  dependsOn: [
    arcPrivateDnsZone
    guestconfigPrivateDnsZone
    kubernetesconfigPrivateDnsZone
  ]
}
