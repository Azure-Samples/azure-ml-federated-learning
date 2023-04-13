// This BICEP script will provision a compute+storage pair
// in a given AzureML workspace, using a vnet and subnet to secure
// the communication between compute and storage, plus managed identity
// for permissions management.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

@description('Specifies the location of the pair resources.')
param pairRegion string = resourceGroup().location

@description('Base name used for creating all pair resources.')
param pairBaseName string

@description('Name of the storage account resource to create for the pair')
param storageAccountName string = replace('st${pairBaseName}','-','') // replace because only alphanumeric characters are supported

@description('Name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = replace('datastore_${pairBaseName}','-','_')

@description('Name of the default compute cluster for the pair')
param compute1Name string = '${pairBaseName}-01'

@description('VM size for the compute cluster')
param compute1SKU string = 'Standard_DS3_v2'

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@description('Flag whether to create a second compute or not')
param compute2 bool = false

@description('The second VM used for creating compute clusters in orchestrator and silos.')
param compute2SKU string = 'Standard_DS3_v2'

@description('Name of the second compute cluster for the pair')
param compute2Name string = '${pairBaseName}-02'

@description('Name of the UAI for the jobs running in the pair')
param jobsUaiName string = 'uai-jobs-${pairBaseName}'

@description('Name of the Network Security Group resource (if createNewVnet==true)')
param nsgResourceName string = 'nsg-${pairBaseName}'

@description('Name of the vNET resource (if createNewVnet==true)')
param vnetResourceName string = 'vnet-${pairBaseName}'

@description('Virtual network address prefix (if createNewVnet==true)')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Subnet name to use for the compute cluster (if createNewVnet==false)')
param subnetName string = 'fl-pair-snet'

@description('Subnet address prefix (if createNewVnet==true)')
param subnetPrefix string = '10.0.0.0/24'

@description('Use a static ip for storage PLE')
param useStoragePLEStaticIP bool = false

@description('Which static IP to use for storage PLE (if useStorageStaticIP is true)')
param storagePLEStaticIP string = '10.0.0.50'

@description('Create a workspace PLE inside the compute virtual network (required if using private link workspace)')
param createWorkspacePLE bool = false

@description('Use a static ip for storage PLE')
param useWorkspacePLEStaticIP bool = false

@description('Which static IP to use for workspace PLE (if useWorkspacePLEStaticIP is true)')
param workspacePLEStaticIP string = '10.0.0.51'

@description('Allow other subnets into the storage (need to be in the same region)')
param allowedSubnetIds array = []

@description('Enable compute node public IP')
param enableNodePublicIp bool = true

@allowed(['Enabled','vNetOnly','Disabled'])
@description('Allow or disallow public network access to Storage Account.')
param storagePublicNetworkAccess string = 'Disabled'

@description('Allow compute cluster to access storage account with R/W permissions (using UAI)')
param applyDefaultPermissions bool = true

@description('Name of the private DNS zone for blob')
param blobPrivateDNSZoneName string = 'privatelink.blob.${environment().suffixes.storage}'

@description('Tags to curate the resources in Azure.')
param tags object = {}


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
        name: subnetName
        addressPrefix: subnetPrefix
      }
    ]
    tags: tags
  }
}

var subnetId = '${vnet.outputs.id}/subnets/${subnetName}'

// provision a user assigned identify for this compute
resource uaiJobs 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: jobsUaiName
  location: pairRegion
  tags: tags
}

// resource uaiEncryption 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
//   name: encryptionUaiName
//   location: pairRegion
//   tags: tags
// }

// private link to workspace in the new vnet
resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2022-05-01' existing = {
  name: machineLearningName
}

// *****************************************
// Azure Machine Learning private networking
// *****************************************

var amlPrivateDnsZoneNames =  {
  azureusgovernment: 'privatelink.api.ml.azure.us'
  azurechinacloud: 'privatelink.api.ml.azure.cn'
  azurecloud: 'privatelink.api.azureml.ms'
}
var amlPrivateDnsZoneName = amlPrivateDnsZoneNames[toLower(environment().name)]

module amlPLE '../networking/private_endpoint.bicep' = if (createWorkspacePLE){
  name: '${machineLearningName}-${pairBaseName}-ple-deployment'
  params: {
    location: pairRegion
    pleRootName: 'ple-${machineLearningName}-${pairBaseName}'
    resourceServiceId: machineLearning.id
    subnetId: subnetId
    tags: tags
    useStaticIPAddress: useWorkspacePLEStaticIP
    privateIPAddress: workspacePLEStaticIP
    privateDNSZoneName: amlPrivateDnsZoneName
    groupId: 'amlworkspace'
  }
}

resource amlPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = if (createWorkspacePLE){
  name: amlPrivateDnsZoneName
}
resource privateAmlDnsZoneVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = if (createWorkspacePLE){
  name: uniqueString(subscription().id, resourceGroup().id, vnetResourceName, amlPrivateDnsZoneName, 'global')
  parent: amlPrivateDnsZone
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.outputs.id
    }
  }
}


// create new Azure ML compute
module computeDeployment1 '../computes/vnet_new_aml_compute.bicep' = {
  name: '${pairBaseName}-vnet-aml-compute-01'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion

    // compute
    computeName: compute1Name
    computeRegion: pairRegion
    computeSKU: compute1SKU
    computeNodes: computeNodes

    // identity
    computeIdentityType: 'UserAssigned'
    computeUaiName: uaiJobs.name

    // networking
    subnetName: subnetName
    vnetId: vnet.outputs.id
    enableNodePublicIp: enableNodePublicIp

    tags: tags
  }
}


// create new second Azure ML compute
module computeDeployment2 '../computes/vnet_new_aml_compute.bicep' = if(compute2) {
  name: '${pairBaseName}-vnet-aml-compute-02'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion

    // compute
    computeName: compute2Name
    computeRegion: pairRegion
    computeSKU: compute2SKU
    computeNodes: computeNodes

    // identity
    computeIdentityType: 'UserAssigned'
    computeUaiName: uaiJobs.name

    // networking
    subnetName: subnetName
    vnetId: vnet.outputs.id
    enableNodePublicIp: enableNodePublicIp

    tags: tags
  }
}

// create new blob storage and datastore
module storageDeployment '../storages/new_blob_storage_datastore.bicep' = {
  name: '${pairBaseName}-vnet-storage'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion
    storageName: storageAccountName
    storageRegion: pairRegion
    datastoreName: datastoreName
    publicNetworkAccess: storagePublicNetworkAccess
    subnetIds: concat(
      [subnetId],
      allowedSubnetIds
    )
    tags: tags
  }
}

// Create a private service endpoints internal to each pair for their respective storages
module pairStoragePrivateEndpoint '../networking/private_endpoint.bicep' = if (storagePublicNetworkAccess == 'Disabled') {
  name: '${pairBaseName}-endpoint-to-insilo-storage'
  scope: resourceGroup()
  params: {
    location: pairRegion
    tags: tags
    resourceServiceId: storageDeployment.outputs.storageId
    pleRootName: 'ple-${storageDeployment.outputs.storageName}-to-${pairBaseName}-st-blob'
    subnetId: subnetId
    useStaticIPAddress: useStoragePLEStaticIP
    privateIPAddress: storagePLEStaticIP
    privateDNSZoneName: blobPrivateDNSZoneName
    groupId: 'blob'
  }
  dependsOn: [
    storageDeployment
  ]
}

// Set R/W permissions for orchestrator UAI towards orchestrator storage
module pairInternalPermissions '../permissions/msi_storage_rw.bicep' = if(applyDefaultPermissions) {
  name: '${pairBaseName}-internal-rw-perms'
  scope: resourceGroup()
  params: {
    storageAccountName: storageDeployment.outputs.storageName
    identityPrincipalId: computeDeployment1.outputs.identityPrincipalId
  }
  dependsOn: [
    storageDeployment
    computeDeployment1
  ]
}

// output the pair config for next actions (permission model)
output identityPrincipalId string = computeDeployment1.outputs.identityPrincipalId
output storageName string = storageDeployment.outputs.storageName
output storageServiceId string = storageDeployment.outputs.storageId
output computeName string = computeDeployment1.outputs.compute
output region string = pairRegion
// output vnetName string = computeDeployment.outputs.vnetName
// output vNetId string = computeDeployment.outputs.vNetId
// output subnetId string = computeDeployment.outputs.subnetId
output vnetName string = vnet.outputs.name
output vNetId string = vnet.outputs.id
output subnetId string = subnetId
