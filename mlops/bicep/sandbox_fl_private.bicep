targetScope = 'resourceGroup'

// please specify the base name for all resources
@description('Base name of the demo, used for creating all resources as prefix')
@minLength(2)
@maxLength(20)
param sandboxBaseName string = 'flsbox'

// below parameters are optionals and have default values
@description('Region of the orchestrator (workspace, central storage and compute).')
param orchestratorRegion string = resourceGroup().location

@description('Set network access for the AzureML workspace.')
@allowed([
  'public'
  'private'
])
param workspaceNetworkAccess string = 'public'

@description('Set the orchestrator storage network access as private, with private endpoints into each silo.')
@allowed([
  'public'
  'private'
])
param orchestratorStorageNetworkAccess string = 'private'

@description('Set the silo storage network access as private, with private endpoints into each silo.')
@allowed([
  'public'
  'private'
])
param siloStorageNetworkAccess string = 'private'

@description('List of each region in which to create an internal silo.')
param siloRegions array = [
  'australiaeast'
  'eastus'
  'westeurope'
]

@description('The VM used for creating compute clusters in orchestrator and silos.')
param compute1SKU string = 'Standard_DS4_v2'

@description('Flag whether to create a second compute or not')
param compute2 bool = false

@description('The VM used for creating a second compute cluster in orchestrator and silos.')
param compute2SKU string = 'Standard_NC6'

@description('WARNING: turn true to apply vNet peering from silos to orchestrator allowing compute to compute communication.')
param applyVNetPeering bool = true

@description('Tags to curate the resources in Azure.')
param tags object = {
  Owner: 'AzureML Samples'
  Project: 'azure-ml-federated-learning'
  Environment: 'dev'
  Toolkit: 'bicep'
  Docs: 'https://github.com/Azure-Samples/azure-ml-federated-learning'
}

// Virtual network and network security group of the workspace resources
module nsg './modules/networking/azureml_workspace_nsg.bicep' = { 
  name: 'nsg-${sandboxBaseName}'
  scope: resourceGroup()
  params: {
    location: orchestratorRegion
    nsgName: 'nsg-${sandboxBaseName}-orch'
    tags: tags
  }
}

module vnet './modules/networking/vnet.bicep' = { 
  name: 'vnet-${sandboxBaseName}-deployment'
  scope: resourceGroup()
  params: {
    location: orchestratorRegion
    virtualNetworkName: 'vnet-${sandboxBaseName}-ws'
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: '10.0.0.0/16'
    subnets: [
      {
        name: 'workspace'
        addressPrefix: '10.0.0.0/24'
      }
    ]
    tags: tags
  }
}

// create this one in this scope so we can use it in the silo modules
resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.blob.${environment().suffixes.storage}'
  location: 'global'
  tags: tags 
}

// Create Azure Machine Learning workspace, including private DNS zones
module workspace './modules/azureml/private_azureml_workspace.bicep' = {
  name: '${sandboxBaseName}-aml-${orchestratorRegion}'
  scope: resourceGroup()
  params: {
    machineLearningName: 'aml-${sandboxBaseName}'
    machineLearningDescription: 'Azure ML demo workspace for federated learning'
    baseName: sandboxBaseName
    location: orchestratorRegion
    workspacePublicNetworkAccess: workspaceNetworkAccess == 'public' ? 'Enabled' : 'Disabled'
    virtualNetworkId: vnet.outputs.id
    subnetName: 'workspace'
    createPrivateDNSZones: true
    tags: tags
  }
}

// In order to be able to record this storage in dns zone with static ip
// we need to set this storage account name ourselves here
var orchestratorStorageAccountName = replace('st${sandboxBaseName}orch','-','')
var orchestratorStorageAccountCleanName = substring(orchestratorStorageAccountName, 0, min(length(orchestratorStorageAccountName),24))

// Create an orchestrator compute+storage pair and attach to workspace
module orchestrator './modules/fl_pairs/vnet_compute_storage_pair.bicep' = {
  name: '${sandboxBaseName}-orchestrator-pair'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion
    machineLearningIsPrivate: true

    pairRegion: orchestratorRegion
    tags: tags

    pairBaseName: '${sandboxBaseName}-orch'

    compute1Name: 'orchestrator-01' // let's not use demo base name in cluster name
    compute1SKU: compute1SKU
    computeNodes: 2
    compute2: compute2
    compute2SKU: compute2SKU
    compute2Name: 'orchestrator-02'

    storageAccountName: orchestratorStorageAccountCleanName
    datastoreName: 'datastore_orchestrator' // let's not use demo base name

    // set R/W permissions for orchestrator UAI towards orchestrator storage
    applyDefaultPermissions: true

    // networking
    vnetAddressPrefix: '10.0.1.0/24'
    subnetPrefix: '10.0.1.0/24'
    amlPLEStaticIPs: '10.0.1.240,10.0.1.241,10.0.1.242' // default,notebook,inference
    storagePLEStaticIP: '10.0.1.243'

    // IMPORTANT: compute still has public ip to let workspace submit job
    // traffic regulated by NSG
    enableNodePublicIp: false

    // IMPORTANT: below means all traffic allowed (with permissions via UAI)
    // alternative is vNetOnly for specific vnets, or Disabled for service endpoints
    storagePublicNetworkAccess: orchestratorStorageNetworkAccess == 'public' ? 'Enabled' : 'Disabled'

    //allowedSubnetIds: [for i in range(0, siloCount): silos[i].outputs.subnetId]
    blobPrivateDNSZoneName: blobPrivateDnsZone.name
  }
  dependsOn: [
    workspace
  ]
}

var siloCount = length(siloRegions)

// Create all silos as a compute+storage pair and attach to workspace
// This pair will be considered eyes-off
module silos './modules/fl_pairs/vnet_compute_storage_pair.bicep' = [for i in range(0, siloCount): {
  name: '${sandboxBaseName}-silo-${i}-pair'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion
    machineLearningIsPrivate: true
    createMachineLearningPLE : !applyVNetPeering // if peering is applied, PLE goes through the peering
    pairRegion: siloRegions[i]
    tags: tags

    pairBaseName: '${sandboxBaseName}-silo${i}'

    compute1Name: 'silo${i}-01' // let's not use demo base name in cluster name
    compute1SKU: compute1SKU
    computeNodes: 2
    compute2: compute2
    compute2SKU: compute2SKU
    compute2Name: 'silo${i}-02'
    datastoreName: 'datastore_silo${i}' // let's not use demo base name

    // set R/W permissions for orchestrator UAI towards orchestrator storage
    applyDefaultPermissions: true

    // networking
    vnetAddressPrefix: (applyVNetPeering ? '10.0.${i+2}.0/24' : '10.0.1.0/24' )
    subnetPrefix: (applyVNetPeering ? '10.0.${i+2}.0/24' : '10.0.1.0/24' )
    amlPLEStaticIPs: (applyVNetPeering ? '10.0.${i+2}.240,10.0.${i+2}.241,10.0.${i+2}.242' : '10.0.1.240,10.0.1.241,10.0.1.242')
    // leave 243 for orchestrator
    storagePLEStaticIP: (applyVNetPeering ? '10.0.${i+2}.244' : '10.0.1.244')

    // IMPORTANT: compute still has public ip to let workspace submit job
    // traffic regulated by NSG
    enableNodePublicIp: false

    // IMPORTANT: below Disabled means data will be only accessible via private service endpoints
    storagePublicNetworkAccess: siloStorageNetworkAccess == 'public' ? 'Enabled' : 'Disabled'

    blobPrivateDNSZoneName: blobPrivateDnsZone.name
  }
  dependsOn: [
    workspace
  ]
}]

// Attach orchestrator and silos together with private endpoints and RBAC
// Create a private service endpoints internal to each pair for their respective storages
module orchToSiloPrivateEndpoints './modules/networking/private_endpoint.bicep' = [for i in range(0, siloCount): if (orchestratorStorageNetworkAccess == 'private' && !applyVNetPeering) {
  name: '${sandboxBaseName}-orch-to-silo${i}-storage-ple'
  scope: resourceGroup()
  params: {
    location: silos[i].outputs.region
    tags: tags
    resourceServiceId: orchestrator.outputs.storageServiceId
    pleRootName: 'ple-${orchestrator.outputs.storageName}-to-${sandboxBaseName}-silo${i}-st-blob'
    subnetId: silos[i].outputs.subnetId
    useStaticIPAddress: true
    privateIPAddress: (applyVNetPeering ? '10.0.${i+2}.243' : '10.0.1.243')
    privateDNSZoneName: 'privatelink.blob.${environment().suffixes.storage}'
    groupId: 'blob'
  }
  dependsOn: [
    silos[i]
  ]
}]

// Set R/W permissions for silo identity towards (eyes-on) orchestrator storage
module siloToOrchPermissions './modules/permissions/msi_storage_rw.bicep' = [for i in range(0, siloCount): {
  name: '${sandboxBaseName}-rw-perms-silo${i}-to-orch'
  scope: resourceGroup()
  params: {
    storageAccountName: orchestrator.outputs.storageName
    identityPrincipalId: silos[i].outputs.identityPrincipalId
  }
  dependsOn: [
    silos
  ]
}]


// WARNING: apply vNet peering on top of everything
// to allow for interconnections between computes
module vNetPeerings './modules/networking/vnet_peering.bicep' = [for i in range(0, siloCount): if(applyVNetPeering) {
  name: '${sandboxBaseName}-vnetpeering-orch-to-silo${i}'
  scope: resourceGroup()
  params: {
    existingVirtualNetworkNameSource: silos[i].outputs.vnetName
    existingVirtualNetworkNameTarget: orchestrator.outputs.vnetName
    existingVirtualNetworkNameTargetResourceGroupName: resourceGroup().name
    useGatewayFromSourceToTarget: false
  }
  dependsOn: [
    orchestrator
    silos[i]
  ]
}]



// returned outputs
output workspaceSecretStoreId string = workspace.outputs.workspaceSecretStoreId
output workspaceSecretStoreName string = workspace.outputs.workspaceSecretStoreName
