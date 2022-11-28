// This BICEP script will provision a new Azure ML compute
// in a given AzureML workspace, without any specific security settings.

// IMPORTANT: This setup is intended only for demo purpose. The data is still accessible
// by the users when opening the storage accounts, and data exfiltration is easy.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

@description('Specifies the location of the compute resources.')
param computeRegion string

@description('Prefix of the compute clusters to create')
param computeName string

@description('VM size for the default cpu compute cluster')
param cpuComputeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default cpu compute cluster')
param cpuComputeNodes int = 4

@description('Name of the cpu compute cluster to create')
param cpuComputeName string = '${computeName}-cpu'

@description('Name of the gpu compute cluster to create')
param gpuComputeName string = '${computeName}-gpu'

@description('VM size for the default gpu compute cluster')
param gpuComputeSKU string = 'Standard_NC6'

@description('VM nodes for the default gpu compute cluster')
param gpuComputeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param computeIdentityType string = 'UserAssigned'

@description('Name of the UAI for the compute cluster (if computeIdentityType==UserAssigned)')
param computeUaiName string = 'uai-${computeName}'

@description('Tags to curate the resources in Azure.')
param tags object = {}


// provision a user assigned identify for this compute
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = if (computeIdentityType == 'UserAssigned') {
  name: computeUaiName
  location: computeRegion
  tags: tags
}

var identityPrincipalId = computeIdentityType == 'UserAssigned' ? uai.properties.principalId : cpuCompute.identity.principalId
var userAssignedIdentities = computeIdentityType == 'SystemAssigned' ? null : {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2022-05-01' existing = {
  name: machineLearningName
}

// provision a cpu compute cluster, and assign the user assigned identity to it
resource cpuCompute 'Microsoft.MachineLearningServices/workspaces/computes@2021-07-01' = {
  name: cpuComputeName
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
      vmSize: cpuComputeSKU
      osType: 'Linux'

      // how many nodes to provision
      scaleSettings: {
        maxNodeCount: cpuComputeNodes
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }

      // networking
      enableNodePublicIp: true
      isolatedNetwork: false
      remoteLoginPortPublicAccess: 'Disabled'

      subnet: json('null')
    }
  }
}

// provision a gpu compute cluster, and assign the user assigned identity to it
resource gpuCompute 'Microsoft.MachineLearningServices/workspaces/computes@2021-07-01' = {
  name: gpuComputeName
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
      vmSize: gpuComputeSKU
      osType: 'Linux'

      // how many nodes to provision
      scaleSettings: {
        maxNodeCount: gpuComputeNodes
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }

      // networking
      enableNodePublicIp: true
      isolatedNetwork: false
      remoteLoginPortPublicAccess: 'Disabled'

      subnet: json('null')
    }
  }
}

// output the compute config for next actions (permission model)
output identityPrincipalId string = identityPrincipalId
output cpuCompute string = cpuCompute.name
output gpuCompute string = gpuCompute.name
output region string = computeRegion
