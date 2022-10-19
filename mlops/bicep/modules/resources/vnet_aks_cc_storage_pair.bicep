// https://github.com/ssarwa/Bicep/blob/master/modules/Identity/role.bicep

// This BICEP script will provision a compute+storage pair
// in a given AzureML workspace, using a vnet and subnet to secure
// the communication between compute and storage, plus managed identity
// for permissions management.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Base name used for creating all pair resources.')
param pairBaseName string

@description('Specifies the location of the pair resources.')
param pairRegion string = resourceGroup().location

@description('Tags to curate the resources in Azure.')
param tags object = {}

@description('Name of the storage account resource to create for the pair')
param storageAccountName string = replace('st${pairBaseName}','-','') // replace because only alphanumeric characters are supported
var storageAccountCleanName = substring(storageAccountName, 0, min(length(storageAccountName),24))

@description('Name of the storage container resource to create for the pair')
param storageContainerName string = 'private'

@description('Name of the default compute cluster for the pair')
param aksClusterName string = 'aks-${pairBaseName}'

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

@description('Allow other subnets into the storage (need to be in the same region)')
param allowedSubnetIds array = []

@description('Make compute notes reachable through public IP')
param enableNodePublicIp bool = true

@description('Make this AKS private')
param privateAKS bool = true

@description('Optional DNS prefix to use with hosted Kubernetes API server FQDN.')
param dnsPrefix string = 'dnsprefix-${pairBaseName}'

@allowed(['Enabled','vNetOnly','Disabled'])
@description('Allow or disallow public network access to Storage Account.')
param storagePublicNetworkAccess string = 'Disabled'

@description('Name of the private DNS zone to create for the AKS')
param privateAKSDnsZoneName string = '${replace(pairBaseName, '-','')}.privatelink.${pairRegion}.azmk8s.io'
// [a-zA-Z0-9-]{1,32}.privatelink.${pairRegion}.azmk8s.io'

@description('Name of the existing private DNS zone for blob storage endpoints')
param privateBlobDnsZoneName string = 'privatelink.blob.core.windows.net'

@description('Allow compute cluster to access storage account with R/W permissions (using UAI)')
param applyDefaultPermissions bool = true

@description('Disk size (in GB) to provision for each of the agent pool nodes. This value ranges from 0 to 1023. Specifying 0 will apply the default disk size for that agentVMSize.')
@minValue(0)
@maxValue(1023)
param osDiskSizeGB int = 0

@description('The number of nodes for the cluster pool.')
@minValue(1)
@maxValue(50)
param agentCount int = 4

// see https://learn.microsoft.com/en-us/azure/virtual-machines/dcv3-series
@description('The size of the Virtual Machine.')
param agentVMSize string = 'Standard_DC4ds_v3'

// Virtual network and network security group
module nsg '../networking/nsg.bicep' = { 
  name: '${nsgResourceName}-deployment'
  params: {
    location: pairRegion
    nsgName: nsgResourceName
    tags: tags
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: '${vnetResourceName}-deployment'
  params: {
    location: pairRegion
    virtualNetworkName: vnetResourceName
    networkSecurityGroupId: nsg.outputs.id
    vnetAddressPrefix: vnetAddressPrefix
    subnetPrefix: subnetPrefix
    subnetName: subnetName
    tags: tags
  }
}

// Look for existing private DNS zone for all our private endpoints
resource pairAKSPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: privateAKSDnsZoneName
  location: 'global'
}

// deploy a storage account for the pair
module storageDeployment './storage_private.bicep' = {
  name: '${storageAccountCleanName}-deployment'
  params: {
    location: pairRegion
    storageName: storageAccountCleanName
    storageSKU: 'Standard_LRS'
    subnetIds: concat(
      ['${vnet.outputs.id}/subnets/${subnetName}'],
      allowedSubnetIds
    )
    publicNetworkAccess: storagePublicNetworkAccess
    tags: tags
  }
}

resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: privateBlobDnsZoneName
}

// Create a private service endpoints internal to each silo for their respective storages
module pairStoragePrivateEndpoint '../networking/private_endpoint.bicep' = {
  name: '${pairBaseName}-deploy-private-endpoint-to-internalstorage'
  scope: resourceGroup()
  params: {
    location: pairRegion
    tags: tags
    privateLinkServiceId: storageDeployment.outputs.storageId
    storagePleRootName: 'ple-${storageAccountCleanName}-to-${pairBaseName}-st-blob'
    subnetId: '${vnet.outputs.id}/subnets/${subnetName}'
    virtualNetworkId: vnet.outputs.id
    privateDNSZoneName: blobPrivateDnsZone.name
    privateDNSZoneId: blobPrivateDnsZone.id
    groupIds: [
      'blob'
      //'file'
    ]
  }
  dependsOn: [
    storageDeployment
  ]
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

// provision a user assigned identify for this silo
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: uaiName
  location: pairRegion
  tags: tags
  dependsOn: [
    storageDeployment // ensure the storage exists BEFORE we do UAI role assignments
  ]
}

// provision a compute cluster, and assign the user assigned identity to it
resource aks 'Microsoft.ContainerService/managedClusters@2022-05-02-preview' = {
  name: aksClusterName
  location: pairRegion
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}
    }
  }
  properties: {
    dnsPrefix: dnsPrefix
    addonProfiles: {
      // omsagent: {
      //   config: {
      //     logAnalyticsWorkspaceResourceID: logworkspaceid
      //   }
      //   enabled: true
      // }
      azurepolicy: {
        enabled: true
      }
      // enable the provisioning of confidential computes nodes
      ACCSGXDevicePlugin: {
        enabled: true
        config: {
          ACCSGXQuoteHelperEnabled: 'false'
        }
      }
    }
    agentPoolProfiles: [
      {
        name: 'confcompool'

        count: agentCount
        // enableAutoScaling: true
        // maxCount: 5
        // minCount: 2        

        vmSize: agentVMSize
        mode: 'System'
        osDiskSizeGB: osDiskSizeGB
        osType: 'Linux'
        // maxPods: 50
        enableNodePublicIP: enableNodePublicIp
        type: 'VirtualMachineScaleSets'
        vnetSubnetID: '${vnet.outputs.id}/subnets/${subnetName}'
      }
    ]
    networkProfile: {
      loadBalancerSku: 'standard'
      networkPlugin: 'azure'
      //outboundType: 'loadBalancer'
      dockerBridgeCidr: '172.17.0.1/16'
      dnsServiceIP: '10.0.11.10'
      serviceCidr: '10.0.11.0/24'
      networkPolicy: 'azure'
    }
    apiServerAccessProfile: {
      enablePrivateCluster: privateAKS
      privateDNSZone: pairAKSPrivateDnsZone.id
    }
    // enableRBAC: true
    // aadProfile: {
    //   adminGroupObjectIDs: aadGroupdIds
    //   enableAzureRBAC: true
    //   managed: true
    //   tenantID: subscription().tenantId
    // }
    // linuxProfile: {
    //   adminUsername: linuxAdminUsername
    //   ssh: {
    //     publicKeys: [
    //       {
    //         keyData: sshRSAPublicKey
    //       }
    //     ]
    //   }
    // }
  }
  dependsOn: [
    storageDeployment // ensure the storage exists BEFORE we do UAI role assignments
    aksPrivateDNSZoneRoles
  ]
}

// need to add AKS identity as contributor to DNS zone???
var aksPrivateDNSRoleIds = [
  // Private DNS Zone Contributor
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/b12aa53e-6015-4669-85d0-8515ebb3ae7f'
  // Network Contributor
  '/subscriptions/${subscription().subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/4d97b98b-1d4f-4787-a291-c67834d212e7'
]
resource aksPrivateDNSZoneRoles 'Microsoft.Authorization/roleAssignments@2022-04-01' = [ for roleId in aksPrivateDNSRoleIds: {
  scope: pairAKSPrivateDnsZone
  name: guid(resourceGroup().id, uai.id, roleId)
  properties: {
    roleDefinitionId: roleId
    principalId: uai.properties.principalId
    principalType: 'ServicePrincipal'
  }
}]

// Set R/W permissions for orchestrator UAI towards orchestrator storage
module pairDefaultRWPermissions '../permissions/msi_storage_rw.bicep' = if(applyDefaultPermissions) {
  name: '${pairBaseName}-${pairRegion}-deploy-internal-default-permission'
  scope: resourceGroup()
  params: {
    storageAccountName: storageAccountCleanName
    identityPrincipalId: uai.properties.principalId
  }
  dependsOn: [
    storageDeployment
    aks
  ]
}

// output the pair config for next actions (permission model)
output identityPrincipalId string = uai.properties.principalId
output storageName string = storageAccountCleanName
output storageServiceId string = storageDeployment.outputs.storageId
output container string = container.name
output aksId string = aks.id
output region string = pairRegion
output vnetName string = vnet.outputs.name
output vnetId string = vnet.outputs.id
output subnetId string = '${vnet.outputs.id}/subnets/${subnetName}'
