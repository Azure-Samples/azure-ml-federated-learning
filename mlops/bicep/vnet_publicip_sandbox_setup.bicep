// This BICEP script will fully provision a functional federated learning sandbox
// based on internal silos secured using a vnet and subnet to secure
// the communication between compute and storage, plus managed identity
// for permissions management.

// The demo permission model is represented by the following matrix:
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
param identityType string = 'UserAssigned'

@description('Location of the orchestrator (workspace, central storage and compute).')
param orchestratorRegion string = resourceGroup().location

@description('List of each region in which to create an internal silo.')
param siloRegions array = [
  'westus'
  'francecentral'
  'brazilsouth'
]

@description('The VM used for creating compute clusters in orchestrator and silos.')
param computeSKU string = 'Standard_DS13_v2'

@description('Apply vNet peering silos->orchestrator')
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
module orchestrator './modules/resources/vnet_compute_storage_pair.bicep' = {
  name: '${demoBaseName}-deploy-orchestrator'
  scope: resourceGroup()
  params: {
    machineLearningName: workspace.outputs.workspace
    machineLearningRegion: orchestratorRegion

    region: orchestratorRegion
    tags: tags

    pairBaseName: '${demoBaseName}-orch'

    computeName: 'cpu-orch' // let's not use demo base name in cluster name
    computeSKU: computeSKU
    computeNodes: 4
    datastoreName: 'datastore_orchestrator' // let's not use demo base name

    // identity for permissions model
    identityType: identityType

    // networking
    vnetAddressPrefix: '10.0.0.0/24'
    subnetPrefix: '10.0.0.0/24'
    enableNodePublicIp: true
  }
  dependsOn: [
    workspace
  ]
}

// set R/W permissions for orchestrator UAI towards orchestrator storage
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

// Create all silos using a provided bicep module
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

    identityType: identityType

    // networking
    vnetAddressPrefix: '10.0.${i+1}.0/24'
    subnetPrefix: '10.0.${i+1}.0/24'
    enableNodePublicIp: true
  }
  dependsOn: [
    workspace
  ]
}]

// set R/W permissions for silo identity towards silo storage
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

// set R/W permissions for silo identity towards orchestrator storage
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

// // Create a role assignment for the orchestrator compute to access the silo storage
// module crossgeoOrchToSiloPrivateEndpoints './modules/networking/private_endpoint.bicep' = [for i in range(0, siloCount): {
//   name: '${demoBaseName}-deploy-crossgeo-endpoint-orch-to-${i}-${siloRegions[i]}'
//   scope: resourceGroup()
//   params: {
//     location: siloRegions[i]
//     tags: tags
//     privateLinkServiceId: orchestrator.outputs.storageServiceId
//     storagePleRootName: 'ple-${orchestrator.outputs.storage}-to-silo${i}${siloRegions[i]}-st-blob'
//     subnetId: silos[i].outputs.subnetId
//     groupIds: [
//       'blob'
//       //'file'
//     ]
//   }
//   dependsOn: [
//     silos[i]
//   ]
// }]

// module vNetPeerings './modules/networking/vnet_peering.bicep' = [for i in range(0, siloCount): if(applyVNetPeering) {
//   name: '${demoBaseName}-deploy-vnet-peering-orch-to-${i}-${siloRegions[i]}'
//   scope: resourceGroup()
//   params: {
//     existingVirtualNetworkName1: silos[i].outputs.vnetName
//     existingVirtualNetworkName2: orchestrator.outputs.vnetName
//     existingVirtualNetworkName2ResourceGroupName: resourceGroup().name
//   }
//   dependsOn: [
//     orchestrator
//     silos[i]
//   ]
// }]
