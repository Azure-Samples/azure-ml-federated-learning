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
// > az deployment group create --template-file .\mlops\bicep\vnet_private_sandbox_setup.bicep \
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

@description('WARNING: make it possible to interact with the workspace through (public) azure portal: workspace and default storage on public network (RBAC controlled).')
@allowed([
  'public'
  'private'
])
param workspaceNetworkAccess string = 'private'

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

// Virtual network and network security group of the workspace resources
module nsg './modules/networking/azureml_workspace_nsg.bicep' = { 
  name: 'nsg-${demoBaseName}'
  scope: resourceGroup()
  params: {
    location: orchestratorRegion
    nsgName: 'nsg-${demoBaseName}'
    tags: tags
  }
}

module vnet './modules/networking/vnet.bicep' = { 
  name: 'vnet-${demoBaseName}-deployment'
  scope: resourceGroup()
  params: {
    location: orchestratorRegion
    virtualNetworkName: 'vnet-${demoBaseName}-ws'
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: '10.0.0.0/16'
    subnets: [
      // a subnet for all the compute resources (image-build-compute)
      {
        name: 'compute'
        addressPrefix: '10.0.1.0/24'
      }
      // a subnet for all the endpoints
      {
        name: 'endpoints'
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
  name: '${demoBaseName}-aml-${orchestratorRegion}'
  scope: resourceGroup()
  params: {
    machineLearningName: 'aml-${demoBaseName}'
    machineLearningDescription: 'Azure ML demo workspace for federated learning'
    baseName: demoBaseName
    location: orchestratorRegion
    tags: tags

    // networking settings
    workspacePublicNetworkAccess: workspaceNetworkAccess == 'public' ? 'Enabled' : 'Disabled'
    virtualNetworkId: vnet.outputs.id
    computeSubnetName: 'compute'
    endpointsSubnetName: 'endpoints'
    createPrivateDNSZones: true

    // we're forcing the IP to be the same as orchestrator vnet
    // to avoid private DNS zone conflicts
    acrPLEStaticIPs: '10.0.0.237,10.0.0.236'
    amlPLEStaticIPs: '10.0.0.240,10.0.0.241,10.0.0.242' // default,notebook,inference
    blobPLEStaticIP: '10.0.0.239'
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
    machineLearningIsPrivate: true

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

    // address has same range as workspace vnet
    // those two vnet will NOT be peered
    vnetAddressPrefix: '10.0.0.0/16'
    computeSubnetPrefix: '10.0.1.0/24'
    endpointsSubnetPrefix: '10.0.0.0/24'

    // we're forcing the IP to be the same as workspace vnet
    // to avoid private DNS zone conflicts
    amlPLEStaticIPs: '10.0.0.240,10.0.0.241,10.0.0.242' // default,notebook,inference
    storagePLEStaticIP: '10.0.0.243'

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

// Create an endpoint in the orchestrator vnet for the workspace storage (for local data upload)
module wsPLEsInOrchestratorVnet './modules/azureml/azureml_resources_ples.bicep' = {
  name: '${demoBaseName}-ws-ple-in-orch'
  scope: resourceGroup()
  params: {
    tags: tags
    machineLearningName: workspace.outputs.workspaceName
    pleRegion: orchestratorRegion
    virtualNetworkName: orchestrator.outputs.vnetName
    virtualNetworkId: orchestrator.outputs.vNetId
    subnetName: 'endpoints'

    linkAcrDnsToVirtualNetwork: true // link ACR DNS Zone (not done previously)
    createAcrPLE: true
    acrPLEStaticIPs: '10.0.0.237,10.0.0.236'

    linkKeyvaultDnsToVirtualNetwork: true // link KV DNS Zone (not done previously)
    createKeyVaultPLE: true
    keyVaultPLEStaticIP: '10.0.0.238'

    linkBlobDnsToVirtualNetwork: false // link was done during silo creation already
    createBlobPLE: true
    blobPLEStaticIP: '10.0.0.239'
  }
  dependsOn: [
    workspace
    orchestrator
  ]
}

// Set READ (only) so that orchestrator can read data from default workspace blob store (for local data upload)
module wsToOrchPermissions './modules/permissions/msi_storage_rw.bicep' = {
  name: '${demoBaseName}-rw-perms-ws-to-orch'
  scope: resourceGroup()
  params: {
    storageAccountName: workspace.outputs.workspaceStorageName
    identityPrincipalId: orchestrator.outputs.identityPrincipalId
    computeToStorageRoles : [
      '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1' // Storage Blob Data Reader
      '81a9662b-bebf-436f-a333-f67b29880f12' // Storage Account Key Operator Service Role
      'c12c1c16-33a1-487b-954d-41c89c60f349' // Reader and Data Access
    ]
  }
  dependsOn: [
    silos
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
    machineLearningIsPrivate: true
    createMachineLearningPLE : !applyVNetPeering // if peering is applied, PLE goes through the peering
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
    amlPLEStaticIPs: (applyVNetPeering ? '10.${i+2}.0.240,10.${i+2}.0.241,10.${i+2}.0.242' : '10.0.0.240,10.0.0.241,10.0.0.242')
    // leave 243 for orchestrator
    storagePLEStaticIP: (applyVNetPeering ? '10.${i+2}.0.244' : '10.0.0.244')

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

module wsPLEsInSilosVnet './modules/azureml/azureml_resources_ples.bicep' = [for i in range(0, siloCount): {
  name: '${demoBaseName}-ws-ple-in-silo${i}'
  scope: resourceGroup()
  params: {
    tags: tags
    machineLearningName: workspace.outputs.workspaceName
    pleRegion: siloRegions[i]
    virtualNetworkName: silos[i].outputs.vnetName
    virtualNetworkId: silos[i].outputs.vNetId
    subnetName: 'endpoints'

    linkAcrDnsToVirtualNetwork: true // link ACR DNS Zone (not done previously)
    createAcrPLE: !applyVNetPeering // if peering is applied, PLE goes through the peering
    acrPLEStaticIPs: '10.0.0.237,10.0.0.236' // unused arg is createAcrPLE=False

    linkKeyvaultDnsToVirtualNetwork: true // link KV DNS Zone (not done previously)
    createKeyVaultPLE: !applyVNetPeering // if peering is applied, PLE goes through the peering
    keyVaultPLEStaticIP: '10.0.0.238' // unused arg is createKeyVaultPLE=False

    linkBlobDnsToVirtualNetwork: false // link was done during silo creation already
    createBlobPLE: !applyVNetPeering // if peering is applied, PLE goes through the peering
    blobPLEStaticIP: '10.0.0.239' // unused arg is createBlobPLE=False
  }
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
