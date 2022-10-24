// This BICEP script will provision an AKS cluster with confidential computes
// attached to a given AzureML workspace, using a vnet and subnet to secure
// the communication between compute and storage, plus managed identity
// for permissions management.

// NOTE: this can take up to 15 minutes to complete

// https://github.com/ssarwa/Bicep/blob/master/modules/Identity/role.bicep

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('The region of the machine learning workspace')
param machineLearningRegion string = resourceGroup().location

@description('The name of the Managed Cluster resource.')
param aksClusterName string

@description('How to name this compute in Azure ML')
param amlComputeName string = aksClusterName

@description('Specifies the location of the compute resources.')
param computeRegion string

@description('Name of the Network Security Group resource')
param nsgResourceName string = 'nsg-${aksClusterName}'

@description('Name of the vNET resource')
param vnetResourceName string = 'vnet-${aksClusterName}'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Subnet address prefix')
param subnetPrefix string = '10.0.0.0/24'

@description('Subnet name')
param subnetName string = 'snet-training'

@description('Make compute notes reachable through public IP')
param enableNodePublicIp bool = true

@description('Make this AKS private')
param privateAKS bool = true

@description('Optional DNS prefix to use with hosted Kubernetes API server FQDN.')
@maxLength(54)
param dnsPrefix string = replace('dnxprefix-${aksClusterName}', '-', '')

@description('Name of the private DNS zone to create for the AKS')
param privateAKSDnsZoneName string = '${replace(aksClusterName, '-','')}.privatelink.${computeRegion}.azmk8s.io'

@description('The number of nodes for the cluster pool.')
@minValue(1)
@maxValue(50)
param agentCount int = 4

// see https://learn.microsoft.com/en-us/azure/virtual-machines/dcv3-series
@description('The size of the Virtual Machine.')
param agentVMSize string = 'Standard_DC4ds_v3'

@description('Disk size (in GB) to provision for each of the agent pool nodes. This value ranges from 0 to 1023. Specifying 0 will apply the default disk size for that agentVMSize.')
@minValue(0)
@maxValue(1023)
param osDiskSizeGB int = 0

@description('Name of the UAI for the compute cluster.')
param computeUaiName string = 'uai-${aksClusterName}'

@description('Tags to curate the resources in Azure.')
param tags object = {}

// Virtual network and network security group
module nsg '../networking/nsg.bicep' = { 
  name: '${nsgResourceName}-deployment'
  params: {
    location: computeRegion
    nsgName: nsgResourceName
    tags: tags
  }
}

module vnet '../networking/vnet.bicep' = { 
  name: '${vnetResourceName}-deployment'
  params: {
    location: computeRegion
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

// provision a user assigned identify for this compute
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: computeUaiName
  location: computeRegion
  tags: tags
}

var identityPrincipalId = uai.properties.principalId
var userAssignedIdentities = {'/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}}

resource aks 'Microsoft.ContainerService/managedClusters@2022-05-02-preview' = {
  name: aksClusterName
  location: computeRegion
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: userAssignedIdentities
  }
  properties: {
    dnsPrefix: dnsPrefix
    addonProfiles: {
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
        osType: 'Linux'
        mode: 'System'
        osDiskSizeGB: osDiskSizeGB
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

module azuremlExtension '../azureml/deploy_aks_azureml_extension_via_script.bicep' = {
  name: 'deploy-aml-extension-${aksClusterName}'
  scope: resourceGroup()
  params: {
    clusterName: aksClusterName
  }
  dependsOn: [
    aks
  ]
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2022-05-01' existing = {
  name: machineLearningName
  scope: resourceGroup()
}

// attach the AKS cluster to the workspace
resource aksAzuremlCompute 'Microsoft.MachineLearningServices/workspaces/computes@2021-07-01' = {
  name: amlComputeName
  parent: workspace
  location: machineLearningRegion
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: userAssignedIdentities
  }
  properties: {
    computeType: 'AKS'
    properties: {
      agentCount: aks.properties.agentPoolProfiles[0].count
      agentVmSize: aks.properties.agentPoolProfiles[0].vmSize
      // aksNetworkingConfiguration: {
      //   dnsServiceIP: aks.properties.networkProfile.dnsServiceIP
      //   dockerBridgeCidr: aks.properties.networkProfile.dockerBridgeCidr
      //   serviceCidr: aks.properties.networkProfile.serviceCidr
      //   //subnetId: aks.properties.networkProfile.
      // }
      clusterFqdn: aks.properties.fqdn
      clusterPurpose: 'DevTest'
      // loadBalancerSubnet: 'string'
      // loadBalancerType: aks.properties.
      // sslConfiguration: {
      //   cert: 'string'
      //   cname: 'string'
      //   key: 'string'
      //   leafDomainLabel: 'string'
      //   overwriteExistingDomain: bool
      //   status: 'string'
      // }
    }
  }
  dependsOn: [
    aks
    azuremlExtension
  ]
}

// output the compute config for next actions (permission model)
output identityPrincipalId string = identityPrincipalId
output compute string = aksAzuremlCompute.name
output region string = computeRegion
output aksControlPlaneFQDN string = aks.properties.fqdn
output aksId string = aks.id
output vnetName string = vnet.outputs.name
output vnetId string = vnet.outputs.id
output subnetName string = subnetName
output subnetId string = '${vnet.outputs.id}/subnets/${subnetName}'
