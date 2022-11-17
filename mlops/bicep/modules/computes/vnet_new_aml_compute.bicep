// This BICEP script will provision a new Azure ML compute
// behind a vnet and subnet, attached to a workspace
// plus managed identity for permissions management.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

@description('Name of the compute cluster to create')
param computeName string

@description('Specifies the location of the compute resources.')
param computeRegion string

@description('VM size for the default compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param computeIdentityType string = 'UserAssigned'

@description('Name of the UAI for the compute cluster (if computeIdentityType==UserAssigned)')
param computeUaiName string = 'uai-${computeName}'

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${computeName}'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${computeName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Subnet address prefix')
param subnetPrefix string = '10.0.0.0/24'

@description('Subnet name')
param subnetName string = 'snet-training'

@description('Enable compute node public IP')
param enableNodePublicIp bool = true

@description('Tags to curate the resources in Azure.')
param tags object = {}

// Virtual network and network security group
module nsg '../networking/nsg.bicep' = { 
  name: '${nsgResourceName}-deployment'
  params: {
    location: computeRegion
    nsgName: nsgResourceName
    tags: tags
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: '${vnetResourceName}-deployment'
  params: {
    location: computeRegion
    virtualNetworkName: vnetResourceName
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: vnetAddressPrefix
    subnets: [
      {
        name: subnetName
        addressPrefix: subnetPrefix
      }
    ]
    tags: tags
  }
}

// provision a user assigned identify for this compute
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = if (computeIdentityType == 'UserAssigned') {
  name: computeUaiName
  location: computeRegion
  tags: tags
}

var identityPrincipalId = computeIdentityType == 'UserAssigned' ? uai.properties.principalId : compute.identity.principalId
var userAssignedIdentities = computeIdentityType == 'SystemAssigned' ? null : {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2022-05-01' existing = {
  name: machineLearningName
}

// provision a compute cluster, and assign the user assigned identity to it
resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2021-07-01' = {
  name: computeName
  parent: workspace
  location: machineLearningRegion
  identity: {
    type: computeIdentityType
    userAssignedIdentities: userAssignedIdentities
  }
  properties: {
    computeType: 'AmlCompute'
    computeLocation: computeRegion
    disableLocalAuth: true

    properties: {
      vmPriority: 'Dedicated'
      vmSize: computeSKU
      osType: 'Linux'

      // how many nodes to provision
      scaleSettings: {
        maxNodeCount: computeNodes
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }

      // networking
      enableNodePublicIp: enableNodePublicIp
      isolatedNetwork: false
      remoteLoginPortPublicAccess: 'Disabled'

      // includes compute in the vnet/subnet
      subnet: {
        id: '${vnet.outputs.id}/subnets/${subnetName}'
      }

      // ???
      // propertyBag: any()
    }
  }
}

// output the compute config for next actions (permission model)
output identityPrincipalId string = identityPrincipalId
output compute string = compute.name
output region string = computeRegion
output vnetName string = vnet.outputs.name
output vnetId string = vnet.outputs.id
output subnetId string = '${vnet.outputs.id}/subnets/${subnetName}'
output subnetName string = subnetName
