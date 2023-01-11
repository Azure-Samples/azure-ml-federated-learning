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

@description('Name of the compute cluster to create')
param computeName string

@description('Specifies the location of the compute resources.')
param computeRegion string

@description('Name of the UAI for the compute cluster (if computeIdentityType==UserAssigned)')
param computeUaiName string

@description('VM size for the default compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param computeIdentityType string = 'UserAssigned'

@description('Tags to curate the resources in Azure.')
param tags object = {}


// get an existing user assigned identify for this compute
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' existing = {
  name: computeUaiName
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
      enableNodePublicIp: true
      isolatedNetwork: false
      remoteLoginPortPublicAccess: 'Disabled'

      subnet: json('null')
    }
  }
}

// output the compute config for next actions (permission model)
output identityPrincipalId string = identityPrincipalId
output compute string = compute.name
output region string = computeRegion
