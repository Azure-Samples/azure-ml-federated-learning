// This BICEP script will provision an AKS cluster with confidential computes
// behind a vnet and subnet, attached to a workspace
// plus managed identity for permissions management.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

@description('The name of the Managed Cluster resource.')
param computeName string

@description('Specifies the location of the compute resources.')
param computeRegion string

@description('Optional DNS prefix to use with hosted Kubernetes API server FQDN.')
@maxLength(54)
param dnsPrefix string = replace('dnxprefix-${computeName}', '-', '')

// see https://learn.microsoft.com/en-us/azure/virtual-machines/dcv3-series
@description('The size of the Virtual Machine.')
param agentVMSize string = 'Standard_DC2as_v5'

@description('The number of nodes for the cluster pool.')
@minValue(1)
@maxValue(50)
param agentCount int = 2

@description('Disk size (in GB) to provision for each of the agent pool nodes. This value ranges from 0 to 1023. Specifying 0 will apply the default disk size for that agentVMSize.')
@minValue(0)
@maxValue(1023)
param osDiskSizeGB int = 0

@description('Name of the UAI for the compute cluster')
param computeUaiName string

@description('Subnet ID')
param subnetId string

@description('Subnet name')
param subnetName string = 'snet-training'

@description('Tags to curate the resources in Azure.')
param tags object = {}

// get an existing user assigned identify for this compute
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' existing = {
  name: computeUaiName
}

var identityPrincipalId = uai.properties.principalId
var userAssignedIdentities = {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}


resource aks 'Microsoft.ContainerService/managedClusters@2023-01-02-preview' = {
  name: computeName
  location: computeRegion
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: userAssignedIdentities
  }
  sku: {
    name: 'Base'
    tier: 'Free'
  }
  properties: {
    dnsPrefix: dnsPrefix
    //fqdnSubdomain: 'foo'
    addonProfiles: {
      // enable the provisioning of confidential computes nodes
      ACCSGXDevicePlugin: {
        enabled: true
        config: {
          ACCSGXQuoteHelperEnabled: 'false'
        }
      }
    }
    agentPoolProfiles: [
      {
        name: 'confcompool'
        count: agentCount
        // enableAutoScaling: true
        // maxCount: 5
        // minCount: 2        

        vmSize: agentVMSize
        osType: 'Linux'
        mode: 'System'
        osDiskSizeGB: osDiskSizeGB
        vnetSubnetID: subnetId
      }
    ]
    apiServerAccessProfile: {
      // IMPORTANT: use this for demo only, it is not a private AKS cluster
      authorizedIPRanges: []
      enablePrivateCluster: false
      enablePrivateClusterPublicFQDN: false
      enableVnetIntegration: false
    }
    networkProfile:{
      networkPlugin: 'azure'
    }
    
  }
}

//module azuremlExtension '../azureml/deploy_aks_azureml_extension.bicep' = {
module azuremlExtension '../azureml/deploy_aks_azureml_extension_via_script.bicep' = {
  name: 'deploy-aml-extension-${computeName}'
  scope: resourceGroup()
  params: {
    clusterName: computeName
  }
  dependsOn: [
    aks
  ]
}

module deployAttachToWorkspace '../azureml/attach_aks_training_to_azureml.bicep' = {
  name: 'attach-${computeName}-to-aml-${machineLearningName}'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion
    aksResourceId: aks.id
    aksRegion: aks.location
    amlComputeName: computeName
    computeUaiName: computeUaiName
  }
  dependsOn: [
    azuremlExtension
  ]
}

// output the compute config for next actions (permission model)
output identityPrincipalId string = identityPrincipalId
output compute string = aks.name
output region string = computeRegion
output subnetName string = subnetName
