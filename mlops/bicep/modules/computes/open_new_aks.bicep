// This BICEP script will provision an AKS cluster
// attached to a given AzureML workspace, without any specific security settings.

// IMPORTANT: This setup is intended only for demo purpose. The data is still accessible
// by the users when opening the storage accounts, and data exfiltration is easy.

// NOTE: this can take up to 15 minutes to complete

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

@description('The name of the Managed Cluster resource.')
param aksClusterName string

@description('How to name this compute in Azure ML')
param amlComputeName string = aksClusterName

@description('Specifies the location of the compute resources.')
param computeRegion string

@description('Optional DNS prefix to use with hosted Kubernetes API server FQDN.')
@maxLength(54)
param dnsPrefix string = replace('dnxprefix-${aksClusterName}', '-', '')

@description('The number of nodes for the cluster pool.')
@minValue(1)
@maxValue(50)
param agentCount int = 4


@description('The size of the Virtual Machine.')
param agentVMSize string = 'Standard_DS3_v2' // 'Standard_DS3_v2' is for CPU; for GPU, 'Standard_NC6' would be a good default choice (don't forget to set computeIsGPU below to true if you want a GPU)

@description('Boolean to indicate if the compute cluster should be a GPU cluster')
param computeIsGPU bool = false // change to true if you want to use a GPU

@description('Disk size (in GB) to provision for each of the agent pool nodes. This value ranges from 0 to 1023. Specifying 0 will apply the default disk size for that agentVMSize.')
@minValue(0)
@maxValue(1023)
param osDiskSizeGB int = 0

@description('Name of the UAI for the compute cluster.')
param computeUaiName string = 'uai-${aksClusterName}'

@description('Tags to curate the resources in Azure.')
param tags object = {}


// provision a user assigned identify for this silo
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: computeUaiName
  location: computeRegion
  tags: tags
}

var identityPrincipalId = uai.properties.principalId
var userAssignedIdentities = {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}

resource aks 'Microsoft.ContainerService/managedClusters@2022-05-02-preview' = {
  name: aksClusterName
  location: computeRegion
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: userAssignedIdentities
  }
  properties: {
    dnsPrefix: dnsPrefix
    //fqdnSubdomain: 'foo'
    agentPoolProfiles: [
      {
        name: 'compool'
        count: agentCount
        // enableAutoScaling: true
        // maxCount: 5
        // minCount: 2        

        vmSize: agentVMSize
        osType: 'Linux'
        mode: 'System'
        osDiskSizeGB: osDiskSizeGB
      }
    ]
    apiServerAccessProfile: {
      // IMPORTANT: use this for demo only, it is not a private AKS cluster
      authorizedIPRanges: []
      enablePrivateCluster: false
      enablePrivateClusterPublicFQDN: false
      enableVnetIntegration: false
    }
  }
}

//module azuremlExtension '../azureml/deploy_aks_azureml_extension.bicep' = {
module azuremlExtension '../azureml/deploy_aks_azureml_extension_via_script.bicep' = {
  name: 'deploy-aml-extension-${aksClusterName}'
  scope: resourceGroup()
  params: {
    clusterName: aksClusterName
    installNvidiaDevicePlugin: computeIsGPU
    installDcgmExporter: computeIsGPU
  }
  dependsOn: [
    aks
  ]
}

module deployAttachToWorkspace '../azureml/attach_aks_training_to_azureml.bicep' = {
  name: 'attach-${aksClusterName}-to-aml-${machineLearningName}'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion
    aksResourceId: aks.id
    aksRegion: aks.location
    amlComputeName: amlComputeName
    computeUaiName: computeUaiName
  }
  dependsOn: [
    azuremlExtension
  ]
}

// output the compute config for next actions (permission model)
output identityPrincipalId string = identityPrincipalId
output compute string = amlComputeName
output region string = computeRegion
output aksControlPlaneFQDN string = aks.properties.fqdn
output aksId string = aks.id
