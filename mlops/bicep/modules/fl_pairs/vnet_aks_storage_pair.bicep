// This BICEP script will provision an AKS cluster 
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
param computeName string = '${pairBaseName}-01'

@description('VM size for the compute cluster')
param computeSKU string = 'Standard_DS3_v2' // 'Standard_DS3_v2' is for CPU; for GPU, 'Standard_NC6' would be a good default choice (don't forget to set computeIsGPU below to true if you want a GPU)

@description('Boolean to indicate if the compute cluster should be a GPU cluster')
param computeIsGPU bool = false // change to true if you want to use a GPU

@description('VM nodes for the default compute cluster')
param computeNodes int = 4

@description('Name of the UAI for the pair compute cluster')
param uaiName string = 'uai-${pairBaseName}'

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${pairBaseName}'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${pairBaseName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string

@description('Subnet address prefix')
param subnetPrefix string

@description('Use a static ip for storage PLE')
param useStorageStaticIP bool = false

@description('Which static IP to use for storage PLE (if useStorageStaticIP is true)')
param storagePLEStaticIP string = '172.19.0.50'

@description('Subnet name')
param subnetName string = 'snet-training'

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

@description('Location of the private DNS zone for blob')
param blobPrivateDNSZoneLocation string = 'global'

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

// provision a user assigned identify for this compute
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: uaiName
  location: pairRegion
  tags: tags
}


// create new Azure ML compute
module computeDeployment '../computes/vnet_new_aks.bicep' = {
  name: '${pairBaseName}-vnet-aml-compute'
  scope: resourceGroup()
  params: {
    machineLearningName: machineLearningName
    machineLearningRegion: machineLearningRegion

    // compute
    computeName: computeName
    computeRegion: pairRegion
    agentVMSize: computeSKU
    computeIsGPU: computeIsGPU
    agentCount: computeNodes

    // identity
    computeUaiName: uai.name

    // networking
    subnetName: subnetName
    subnetId: '${vnet.outputs.id}/subnets/${subnetName}'

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
      ['${vnet.outputs.id}/subnets/${subnetName}'],
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
    resourceName: storageDeployment.outputs.storageName
    pleRootName: 'ple-${storageDeployment.outputs.storageName}-to-${pairBaseName}-st-blob'
    virtualNetworkId: vnet.outputs.id
    subnetId: '${vnet.outputs.id}/subnets/${subnetName}'
    useStaticIPAddress: useStorageStaticIP
    privateIPAddress: storagePLEStaticIP
    privateDNSZoneName: blobPrivateDNSZoneName
    privateDNSZoneLocation: blobPrivateDNSZoneLocation
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
    identityPrincipalId: computeDeployment.outputs.identityPrincipalId
  }
  dependsOn: [
    storageDeployment
    computeDeployment
  ]
}

// output the pair config for next actions (permission model)
output identityPrincipalId string = computeDeployment.outputs.identityPrincipalId
output storageName string = storageDeployment.outputs.storageName
output storageServiceId string = storageDeployment.outputs.storageId
output computeName string = computeDeployment.outputs.compute
output region string = pairRegion
output vnetName string = vnet.outputs.name
output vnetId string = vnet.outputs.id
output subnetId string = '${vnet.outputs.id}/subnets/${subnetName}'
