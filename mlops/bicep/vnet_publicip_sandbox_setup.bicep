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
@minLength(2)
@maxLength(20)
param demoBaseName string = 'fldemo'

// below parameters are optionals and have default values
@description('Region of the orchestrator (workspace, central storage and compute).')
param orchestratorRegion string = resourceGroup().location

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

@description('Name of the keyvault to use for storing actual secrets (ex: encryption at rest).')
param confidentialityKeyVaultName string = 'kv-${demoBaseName}'

@description('Tags to curate the resources in Azure.')
param tags object = {
  Owner: 'AzureML Samples'
  Project: 'azure-ml-federated-learning'
  Environment: 'dev'
  Toolkit: 'bicep'
  Docs: 'https://github.com/Azure-Samples/azure-ml-federated-learning'
}

// create this one in this scope so we can use it in the silo modules
resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
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
    machineLearningDescription: 'Azure ML demo workspace for federated learning (orchestratorStorageNetworkAccess=${orchestratorStorageNetworkAccess}, applyVNetPeering=${applyVNetPeering})'
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
  name: '${demoBaseName}-orchestrator-pair'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion
    machineLearningIsPrivate: false

    pairRegion: orchestratorRegion
    tags: tags

    pairBaseName: '${demoBaseName}-orch'

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
    vnetResourceName: 'vnet-${demoBaseName}-orch'
    vnetAddressPrefix: '10.0.0.0/16'
    computeSubnetPrefix: '10.0.1.0/24'
    endpointsSubnetPrefix: '10.0.0.0/24'

    // NOTE: when using storagePublicNetworkAccess = 'Disabled' we will need to
    // have multiple endpoints from the orchestrator storage
    // (to orch vnet and to each silo vnet)
    // we need to set static IP to create a unique record in DNS zone
    // with all the IPs to the orchestrator storage
    storagePLEStaticIP: '10.0.0.243'
  
    // IMPORTANT: compute still has public ip to let workspace submit job
    // traffic regulated by NSG
    enableNodePublicIp: true

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
  name: '${demoBaseName}-silo-${i}-pair'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspaceName
    machineLearningRegion: orchestratorRegion
    machineLearningIsPrivate: false
    createMachineLearningPLE: false
    pairRegion: siloRegions[i]
    tags: tags

    pairBaseName: '${demoBaseName}-silo${i}'

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
    vnetAddressPrefix: (applyVNetPeering ? '10.${i+2}.0.0/16' : '10.0.0.0/16' )
    computeSubnetPrefix: (applyVNetPeering ? '10.${i+2}.1.0/24' : '10.0.1.0/24' )
    endpointsSubnetPrefix: (applyVNetPeering ? '10.${i+2}.0.0/24' : '10.0.0.0/24' )
    // leave 243 for orchestrator
    storagePLEStaticIP: (applyVNetPeering ? '10.${i+2}.0.244' : '10.0.0.244')

    // IMPORTANT: compute still has public ip to let workspace submit job
    // traffic regulated by NSG
    enableNodePublicIp: true

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
  name: '${demoBaseName}-orch-to-silo${i}-storage-ple'
  scope: resourceGroup()
  params: {
    location: silos[i].outputs.region
    tags: tags
    resourceServiceId: orchestrator.outputs.storageServiceId
    pleRootName: 'ple-${orchestrator.outputs.storageName}-to-${demoBaseName}-silo${i}-st-blob'
    subnetId: '${silos[i].outputs.vNetId}/subnets/endpoints'
    useStaticIPAddress: true
    privateIPAddress: (applyVNetPeering ? '10.${i+2}.0.243' : '10.0.0.243')
    privateDNSZoneName: 'privatelink.blob.${environment().suffixes.storage}'
    groupId: 'blob'
  }
  dependsOn: [
    orchestrator
    silos[i]
  ]
}]

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
    useGatewayFromSourceToTarget: false
  }
  dependsOn: [
    orchestrator
    silos[i]
  ]
}]


// Create a "confidentiality" keyvault external to the workspace
// This keyvault will be used to store actual secrets (ex: encryption at rest)
var siloIdentities = [ for i in range(0, siloCount) : '${silos[i].outputs.identityPrincipalId}' ]

module confidentialityKeyVault './modules/resources/confidentiality_keyvault.bicep' = {
  name: '${demoBaseName}-kv-confidentiality'
  params: {
    keyVaultName: confidentialityKeyVaultName
    tags: tags
    region: orchestratorRegion
    identitiesEnabledCryptoOperations: siloIdentities
    // for some reason, concat doesn't work here, using secondary list
    secondaryIdentitiesEnabledCryptoOperations: [ '${orchestrator.outputs.identityPrincipalId}' ]
  }
  dependsOn: [
    silos
    orchestrator
  ]
}

// returned outputs
output workspaceSecretStoreId string = workspace.outputs.workspaceSecretStoreId
output workspaceSecretStoreName string = workspace.outputs.workspaceSecretStoreName
