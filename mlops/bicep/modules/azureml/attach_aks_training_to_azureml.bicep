// This BICEP script will attach an AKS cluster
// to a given AzureML workspace for training (NOT inferencing).

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

@description('Resource ID of the AKS cluster.')
param aksResourceId string

@description('Region of the AKS cluster.')
param aksRegion string

@description('How to name this compute in Azure ML')
param amlComputeName string

@description('Name of the existing UAI for the compute cluster.')
param computeUaiName string

// provision a user assigned identify for this silo
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' existing = {
  name: computeUaiName
  scope: resourceGroup()
}

var identityPrincipalId = uai.properties.principalId
var userAssignedIdentities = {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2022-05-01' existing = {
  name: machineLearningName
  scope: resourceGroup()
}

// attach the AKS cluster to the workspace
resource aksAzuremlCompute 'Microsoft.MachineLearningServices/workspaces/computes@2021-01-01' = {
  name: amlComputeName
  parent: workspace
  location: machineLearningRegion
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: userAssignedIdentities
  }
  properties: {
    computeType: 'Kubernetes'
    computeLocation: aksRegion
    resourceId: aksResourceId
    description: 'AKS cluster attached to AzureML workspace'
    properties: {
    }
  }
}

// output the compute config for next actions (permission model)
output identityPrincipalId string = identityPrincipalId
output compute string = aksAzuremlCompute.name
