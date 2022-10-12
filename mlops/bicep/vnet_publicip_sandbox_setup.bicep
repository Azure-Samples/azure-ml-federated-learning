// EXPERIMENTAL - please do not take production dependency on this setup

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

@description('List of each region in which to create an internal silo.')
param siloRegions array = [
  'westus'
  'francecentral'
  'brazilsouth'
]

@description('The VM used for creating compute clusters in orchestrator and silos.')
param computeSKU string = 'Standard_DS13_v2'

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

// Create Azure Machine Learning workspace
module workspace './modules/resources/open_azureml_workspace.bicep' = {
  name: '${demoBaseName}-deploy-aml-${orchestratorRegion}'
  scope: resourceGroup()
  params: {
    machineLearningName: 'aml-${demoBaseName}'
    location: orchestratorRegion
    tags: tags
  }
}

// Create an orchestrator compute+storage pair and attach to workspace
// This pair will be considered eyes-on
module orchestrator './modules/resources/vnet_compute_storage_pair.bicep' = {
  name: '${demoBaseName}-deploy-orchestrator'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspace
    machineLearningRegion: orchestratorRegion

    region: orchestratorRegion
    tags: tags

    pairBaseName: '${demoBaseName}-orch'

    computeName: 'cpu-orchestrator' // let's not use demo base name in cluster name
    computeSKU: computeSKU
    computeNodes: 4
    datastoreName: 'datastore_orchestrator' // let's not use demo base name

    // identity for permissions model
    identityType: identityType

    // networking
    vnetAddressPrefix: '10.0.0.0/24'
    subnetPrefix: '10.0.0.0/24'

    // IMPORTANT: compute still has public ip to let workspace submit job
    // traffic regulated by NSG
    enableNodePublicIp: true

    // IMPORTANT: below means all traffic allowed (with permissions via UAI)
    // alternative is vNetOnly for specific vnets, or Disabled for service endpoints
    storagePublicNetworkAccess: 'Enabled'

    //allowedSubnetIds: [for i in range(0, siloCount): silos[i].outputs.subnetId]
  }
  dependsOn: [
    workspace
  ]
}

// Set R/W permissions for orchestrator UAI towards orchestrator storage
module orchestratorPermission './modules/permissions/msi_storage_rw.bicep' = {
  name: '${demoBaseName}-deploy-orchestrator-permission-${orchestratorRegion}'
  scope: resourceGroup()
  params: {
    storageAccountName: orchestrator.outputs.storageName
    identityPrincipalId: orchestrator.outputs.identityPrincipalId
  }
  dependsOn: [
    orchestrator
  ]
}

var siloCount = length(siloRegions)

// Create all silos as a compute+storage pair and attach to workspace
// This pair will be considered eyes-off
module silos './modules/resources/vnet_compute_storage_pair.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-deploy-silo-${i}-${siloRegions[i]}'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspace
    machineLearningRegion: orchestratorRegion
    region: siloRegions[i]
    tags: tags

    pairBaseName: '${demoBaseName}-silo${i}-${siloRegions[i]}'

    computeName: 'cpu-silo${i}-${siloRegions[i]}' // let's not use demo base name
    computeSKU: computeSKU
    computeNodes: 4
    datastoreName: 'datastore_silo${i}_${siloRegions[i]}' // let's not use demo base name

    // identity for permissions model
    identityType: identityType

    // networking
    vnetAddressPrefix: '10.0.${i+1}.0/24'
    subnetPrefix: '10.0.${i+1}.0/24'

    // IMPORTANT: compute still has public ip to let workspace submit job
    // traffic regulated by NSG
    enableNodePublicIp: true

    // IMPORTANT: below Disabled means data will be only accessible via private service endpoints
    storagePublicNetworkAccess: 'Disabled'
  }
  dependsOn: [
    workspace
  ]
}]

// Set R/W permissions for silo identity towards silo storage
module siloToSiloPermissions './modules/permissions/msi_storage_rw.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-deploy-silo${i}-permission'
  scope: resourceGroup()
  params: {
    storageAccountName: silos[i].outputs.storageName
    identityPrincipalId: silos[i].outputs.identityPrincipalId
  }
  dependsOn: [
    silos
  ]
}]

// Add a private DNS zone for all our private endpoints
resource siloStoragePrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.blob.${environment().suffixes.storage}'
  location: 'global'
}

// Create a private service endpoints internal to each silo for their respective storages
module silosStoragePrivateEndpoints './modules/networking/private_endpoint.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-deploy-internal-endpoint-${i}-${siloRegions[i]}-storage'
  scope: resourceGroup()
  params: {
    location: siloRegions[i]
    tags: tags
    privateLinkServiceId: silos[i].outputs.storageServiceId
    storagePleRootName: 'ple-${silos[i].outputs.storageName}-to-silo${i}${siloRegions[i]}-st-blob'
    subnetId: silos[i].outputs.subnetId
    virtualNetworkId: silos[i].outputs.vnetId
    privateDNSZoneName: siloStoragePrivateDnsZone.name
    privateDNSZoneId: siloStoragePrivateDnsZone.id
    groupIds: [
      'blob'
      //'file'
    ]
  }
  dependsOn: [
    silos[i]
  ]
}]


// Set R/W permissions for silo identity towards (eyes-on) orchestrator storage
module siloToOrchPermissions './modules/permissions/msi_storage_rw.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-deploy-silo${i}-to-orch-permission'
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
  name: '${demoBaseName}-deploy-vnet-peering-orch-to-${i}-${siloRegions[i]}'
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
