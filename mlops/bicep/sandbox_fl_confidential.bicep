// This BICEP script will fully provision a federated learning sandbox
// with eyes-off orchestrator and silos storages
// and only confidential compute clusters in orchestrator and silos

targetScope = 'resourceGroup'

// please specify the base name for all resources
@description('Base name of the demo, used for creating all resources as prefix')
param demoBaseName string = 'fldemo'

@description('Region of the orchestrator (workspace, central storage and compute).')
param orchestratorRegion string = 'eastus'

@description('List of each region in which to create an internal silo.')
param siloRegions array = [
  'eastus'
  'eastus'
  'eastus'
]

@description('The VM used for creating compute clusters in orchestrator and silos.')
param computeSKU string = 'Standard_DC4as_v5'

@description('Uses public network access for the orchestrator storage, allowing it to be eyes-on.')
param orchestratorEyesOn bool = false

@description('Apply vnet peering to allow for vertical FL')
param applyVNetPeering bool = true

@description('Provide your Kaggle API user name to run our samples relying on Kaggle datasets.')
param kaggleUsername string = ''

@description('Provide your Kaggle API key to run our samples relying on Kaggle datasets.')
@secure()
param kaggleKey string = ''


// run the generic sandbox bicep script with proper arguments
module sandbox 'vnet_publicip_sandbox_aks_confcomp_setup.bicep' = {
  name: 'sandbox'
  params: {
    demoBaseName: demoBaseName
    orchestratorRegion: orchestratorRegion
    siloRegions: siloRegions

    // computes
    computeSKU: computeSKU
  
    // eyes-on/eyes-off settings
    orchestratorStorageNetworkAccess: orchestratorEyesOn ? 'public' : 'private'
    siloStorageNetworkAccess: 'private'

    // ready for vertical FL
    applyVNetPeering: applyVNetPeering
  }
}

// Add kaggle secrets if given
resource kaggleSecretUsername 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = if (!empty(kaggleUsername)) {
  name: 'ws-shkv-${demoBaseName}/kaggleusername'
  properties: {
    value: kaggleUsername
  }
  dependsOn: [
    sandbox
  ]
}

resource kaggleSecretKey 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = if (!empty(kaggleUsername)) {
  name: 'ws-shkv-${demoBaseName}/kagglekey'
  properties: {
    value: kaggleKey
  }
  dependsOn: [
    sandbox
  ]
}
