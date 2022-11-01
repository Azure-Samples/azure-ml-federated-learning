// This BICEP script will fully provision a federated learning sandbox
// based on internal silos kept eyes-off using a combination of vnets
// and private service endpoints, to support the communication
// between compute and storage.

// IMPORTANT:
// - the orchestrator is considered eyes-on and is
//   secured with UAIs (no private service endpoints).
// - the computes still have an open public IP to allow
//   communication with the AzureML workspace.

// The permission model is represented by the following matrix:
// |               | orch.compute | siloA.compute | siloB.compute |
// |---------------|--------------|---------------|---------------|
// | orch.storage  |     R/W      |      R/W      |      R/W      |
// | siloA.storage |      -       |      R/W      |       -       |
// | siloB.storage |      -       |       -       |      R/W      |

// Usage (sh):
// > az login
// > az account set --name <subscription name>
// > az group create --name <resource group name> --location <region>
// > az deployment group create --template-file .\mlops\bicep\vnet_publicip_sandbox_setup.bicep \
//                              --resource-group <resource group name \
//                              --parameters demoBaseName="fldemo"

targetScope = 'resourceGroup'

// please specify the base name for all resources
@description('Base name of the demo, used for creating all resources as prefix')
param demoBaseName string = 'fldemo'

// below parameters are optionals and have default values
@allowed(['UserAssigned','SystemAssigned'])
@description('Type of identity to use for permissions model')
param identityType string = 'UserAssigned'

@description('Region of the orchestrator (workspace, central storage and compute).')
param orchestratorRegion string = resourceGroup().location

@description('Set the orchestrator storage as private, with endpoints into each silo.')
@allowed([
  'public'
  'private'
])
param orchestratorAccess string = 'public'

@description('List of each region in which to create an internal silo.')
param siloRegions array = [
  'westus'
  'francecentral'
  'brazilsouth'
]

@description('The VM used for creating compute clusters in orchestrator and silos.')
param computeSKU string = 'Standard_DS3_v2'

@description('WARNING: turn true to apply vNet peering from silos to orchestrator allowing compute to compute communication.')
param applyVNetPeering bool = false

@description('Tags to curate the resources in Azure.')
param tags object = {
  Owner: 'AzureML Samples'
  Project: 'azure-ml-federated-learning'
  Environment: 'dev'
  Toolkit: 'bicep'
  Docs: 'https://github.com/Azure-Samples/azure-ml-federated-learning'
}

// Create the storage DNS zone before the rest
resource storagePrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.blob.${environment().suffixes.storage}'
  location: 'global'
  tags: tags
}

// Create Azure Machine Learning workspace
module workspace './modules/azureml/open_azureml_workspace.bicep' = {
  name: '${demoBaseName}-aml-${orchestratorRegion}'
  scope: resourceGroup()
  params: {
    machineLearningName: 'aml-${demoBaseName}'
    machineLearningDescription: 'Azure ML demo workspace for federated learning (orchestratorAccess=${orchestratorAccess}, applyVNetPeering=${applyVNetPeering})'
    baseName: demoBaseName
    location: orchestratorRegion
    tags: tags
  }
}

// In order to be able to record this storage in dns zone with static ip
// we need to set this storage account name ourselves here
var orchestratorStorageAccountName = replace('st${demoBaseName}orch','-','')
var orchestratorStorageAccountCleanName = substring(orchestratorStorageAccountName, 0, min(length(orchestratorStorageAccountName),24))

// Create an orchestrator compute+storage pair and attach to workspace
module orchestrator './modules/fl_pairs/vnet_compute_storage_pair.bicep' = {
  name: '${demoBaseName}-vnetpair-orchestrator'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion

    pairRegion: orchestratorRegion
    tags: tags

    pairBaseName: '${demoBaseName}-orch'

    computeName: 'cpu-orchestrator' // let's not use demo base name in cluster name
    computeSKU: computeSKU
    computeNodes: 4

    storageAccountName: orchestratorStorageAccountCleanName
    datastoreName: 'datastore_orchestrator' // let's not use demo base name

    // identity for permissions model
    identityType: identityType

    // set R/W permissions for orchestrator UAI towards orchestrator storage
    applyDefaultPermissions: true

    // networking
    vnetAddressPrefix: '10.0.0.0/24'
    subnetPrefix: '10.0.0.0/24'

    // NOTE: when using storagePublicNetworkAccess = 'Disabled' we will need to
    // have multiple endpoints from the orchestrator storage
    // (to orch vnet and to each silo vnet)
    // we need to set static IP to create a unique record in DNS zone
    // with all the IPs to the orchestrator storage
    useStorageStaticIP: orchestratorAccess == 'Disabled'
    storagePLEStaticIP: '10.0.0.50'

    // IMPORTANT: compute still has public ip to let workspace submit job
    // traffic regulated by NSG
    enableNodePublicIp: true

    // IMPORTANT: below means all traffic allowed (with permissions via UAI)
    // alternative is vNetOnly for specific vnets, or Disabled for service endpoints
    storagePublicNetworkAccess: orchestratorAccess == 'public' ? 'Enabled' : 'Disabled'

    //allowedSubnetIds: [for i in range(0, siloCount): silos[i].outputs.subnetId]

    blobPrivateDNSZoneName: storagePrivateDnsZone.name
    blobPrivateDNSZoneLocation: storagePrivateDnsZone.location
  }
  dependsOn: [
    workspace
  ]
}

var siloCount = length(siloRegions)

// Create all silos as a compute+storage pair and attach to workspace
// This pair will be considered eyes-off
module silos './modules/fl_pairs/vnet_compute_storage_pair.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-vnetpair-silo-${i}'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion
    pairRegion: siloRegions[i]
    tags: tags

    pairBaseName: '${demoBaseName}-silo${i}-${siloRegions[i]}'

    computeName: 'cpu-silo${i}-${siloRegions[i]}' // let's not use demo base name
    computeSKU: computeSKU
    computeNodes: 4
    datastoreName: 'datastore_silo${i}_${siloRegions[i]}' // let's not use demo base name

    // identity for permissions model
    identityType: identityType

    // set R/W permissions for orchestrator UAI towards orchestrator storage
    applyDefaultPermissions: true

    // networking
    vnetAddressPrefix: '10.0.${i+1}.0/24'
    subnetPrefix: '10.0.${i+1}.0/24'

    // IMPORTANT: compute still has public ip to let workspace submit job
    // traffic regulated by NSG
    enableNodePublicIp: true

    // IMPORTANT: below Disabled means data will be only accessible via private service endpoints
    storagePublicNetworkAccess: 'Disabled'

    blobPrivateDNSZoneName: storagePrivateDnsZone.name
    blobPrivateDNSZoneLocation: storagePrivateDnsZone.location
  }
  dependsOn: [
    workspace
  ]
}]

// Attach orchestrator and silos together with private endpoints and RBAC
// Create a private service endpoints internal to each pair for their respective storages
module orchToSiloPrivateEndpoints './modules/networking/private_endpoint.bicep' = [for i in range(0, siloCount): if (orchestratorAccess == 'private') {
  name: '${demoBaseName}-orch-to-silo${i}-endpoint'
  scope: resourceGroup()
  params: {
    location: silos[i].outputs.region
    tags: tags
    resourceServiceId: orchestrator.outputs.storageServiceId
    resourceName: orchestrator.outputs.storageName
    linkVirtualNetwork: false // the link already exists at this point
    pleRootName: 'ple-${orchestrator.outputs.storageName}-to-${demoBaseName}-silo${i}-st-blob'
    virtualNetworkId: silos[i].outputs.vnetId
    subnetId: silos[i].outputs.subnetId
    // we need to set static IP to create a unique record in DNS zone
    // with all the IPs to the orchestrator storage
    useStaticIPAddress: true
    privateIPAddress: '10.0.${i+1}.50'
    privateDNSZoneName: storagePrivateDnsZone.name
    privateDNSZoneLocation: storagePrivateDnsZone.location
    groupId: 'blob'
  }
  dependsOn: [
    orchestrator
    silos[i]
  ]
}]

// NOTE: when creating multiple endpoints in multiple vnets using the same private DNS zone
// the IP address of each endpoint will overwrite the previous one.
// we are using static IP adresses so that we can create a unique record in the DNS zone
// with all the IP adresses from each vnet (orch + silos)
resource privateDnsARecordOrchestratorStorage 'Microsoft.Network/privateDnsZones/A@2020-06-01' = if (orchestratorAccess == 'private') {
  name: orchestratorStorageAccountCleanName
  parent: storagePrivateDnsZone
  properties: {
    ttl: 3600
    aRecords: [ for i in range(0, siloCount+1): {
        ipv4Address: '10.0.${i}.50'
    }]
  }
  dependsOn: [
    orchToSiloPrivateEndpoints
  ]
}

// Set R/W permissions for silo identity towards (eyes-on) orchestrator storage
module siloToOrchPermissions './modules/permissions/msi_storage_rw.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-rw-perms-silo${i}-to-orch'
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
  name: '${demoBaseName}-vnetpeering-orch-to-silo${i}'
  scope: resourceGroup()
  params: {
    existingVirtualNetworkNameSource: silos[i].outputs.vnetName
    existingVirtualNetworkNameTarget: orchestrator.outputs.vnetName
    existingVirtualNetworkNameTargetResourceGroupName: resourceGroup().name
    useGatewayFromSourceToTarget: true
  }
  dependsOn: [
    orchestrator
    silos[i]
  ]
}]
