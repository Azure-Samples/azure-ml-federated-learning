// This BICEP script will provision a compute+storage pair
// in a given AzureML workspace, using a vnet and subnet to secure
// the communication between compute and storage, plus managed identity
// for permissions management.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('Specifies the location of the pair resources.')
param region string = resourceGroup().location

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('Base name used for creating all pair resources.')
param pairBaseName string

@description('Name of the storage account resource to create for the pair')
param storageAccountName string = replace('st${pairBaseName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the storage container resource to create for the pair')
param storageContainerName string = 'private'

@description('Name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = replace('datastore_${pairBaseName}','-','_')

@description('Name of the default compute cluster for the pair')
param computeName string = 'cpu-cluster-${pairBaseName}'

@description('VM size for the default compute cluster')
param computeSKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@allowed(['UserAssigned','SystemAssigned'])
param identityType string = 'UserAssigned'

@description('Name of the UAI for the pair compute cluster')
param uaiName string = 'uai-${pairBaseName}'

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${pairBaseName}'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${pairBaseName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Subnet address prefix')
param subnetPrefix string = '10.0.0.0/24'

@description('Subnet name')
param subnetName string = 'snet-training'

@description('Enable compute node public IP')
param enableNodePublicIp bool = true


// Virtual network and network security group
module nsg '../networking/nsg.bicep' = { 
  name: '${nsgResourceName}-deployment'
  params: {
    location: region
    nsgName: nsgResourceName
    tags: tags
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: '${vnetResourceName}-deployment'
  params: {
    location: region
    virtualNetworkName: vnetResourceName
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: vnetAddressPrefix
    subnetPrefix: subnetPrefix
    subnetName: subnetName
    tags: tags
  }
}

var storageAccountCleanName = substring(storageAccountName, 0, min(length(storageAccountName),24))

// deploy a storage account for the silo
module storageDeployment '../secure_resources/storage.bicep' = {
  name: '${storageAccountCleanName}-deployment'
  params: {
    location: region
    storageName: storageAccountCleanName
    storageSKU: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/${subnetName}'
    virtualNetworkId: vnet.outputs.id
    tags: tags
  }
}

// create a "private" container in the storage account
// this one will be readable only by silo compute
resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  name: '${storageAccountCleanName}/default/${storageContainerName}'
  properties: {
    metadata: {}
    publicAccess: 'None'
  }
  dependsOn: [
    storageDeployment
  ]
}

// attach as a datastore in AzureML
resource datastore 'Microsoft.MachineLearningServices/workspaces/datastores@2022-06-01-preview' = {
  name: '${machineLearningName}/${datastoreName}'
  properties: {
    credentials: {
      credentialsType: 'None'
      // For remaining properties, see DatastoreCredentials objects
    }
    description: 'Private storage in region ${region}'
    properties: {}
    datastoreType: 'AzureBlob'
    // For remaining properties, see DatastoreProperties objects
    accountName: storageAccountCleanName
    containerName: storageContainerName
    // endpoint: 'string'
    // protocol: 'string'
    resourceGroup: resourceGroup().name
    // serviceDataAccessAuthIdentity: 'string'
    subscriptionId: subscription().subscriptionId
  }
  dependsOn: [
    container
  ]
}

// provision a user assigned identify for this silo
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = if (identityType == 'UserAssigned') {
  name: uaiName
  location: region
  tags: tags
  dependsOn: [
    storageDeployment // ensure the storage exists BEFORE we do UAI role assignments
  ]
}

var identityPrincipalId = identityType == 'UserAssigned' ? uai.properties.principalId : compute.identity.principalId
var userAssignedIdentities = identityType == 'SystemAssigned' ? null : {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}

// provision a compute cluster, and assign the user assigned identity to it
resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2022-05-01' = {
  name: '${machineLearningName}/${computeName}'
  location: region
  identity: {
    type: identityType
    userAssignedIdentities: userAssignedIdentities
  }
  properties: {
    computeType: 'AmlCompute'
    description: 'Silo cluster with R/W access to the silo storage'
    properties: {
      vmPriority: 'Dedicated'      
      vmSize: computeSKU
      enableNodePublicIp: enableNodePublicIp
      isolatedNetwork: false
      osType: 'Linux'
      remoteLoginPortPublicAccess: 'Disabled'
      scaleSettings: {
        maxNodeCount: computeNodes
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT300S' // 5 minutes
      }
      subnet: {
        id: '${vnet.outputs.id}/subnets/${subnetName}'
      }
    }
  }
  dependsOn: [
    storageDeployment  // ensure the storage exists BEFORE we do UAI role assignments
  ]
}

// output the pair config for next actions (permission model)
output identityPrincipalId string = identityPrincipalId
output storage string = storageAccountCleanName
output storageServiceId string = storageDeployment.outputs.storageId
output container string = container.name
output datastore string = datastore.name
output compute string = compute.name
output region string = region
output vnetName string = vnet.outputs.name
output vnetId string = vnet.outputs.id
output subnetId string = '${vnet.outputs.id}/subnets/${subnetName}'
