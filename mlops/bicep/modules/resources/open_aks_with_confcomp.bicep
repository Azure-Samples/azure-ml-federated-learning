// This BICEP script will provision an AKS cluster with confidential computes
// IMPORTANT: This setup is intended only for demo purpose.

// resource group must be specified as scope in az cli or module call
targetScope = 'resourceGroup'

// required parameters
@description('The name of the Managed Cluster resource.')
param clusterName string

@description('The location of the Managed Cluster resource.')
param region string = resourceGroup().location

@description('Optional DNS prefix to use with hosted Kubernetes API server FQDN.')
param dnsPrefix string = '${clusterName}-${resourceGroup().name}-${uniqueString(clusterName, resourceGroup().id)}'

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

param uaiName string = 'uai-${clusterName}'

@description('Tags to curate the resources in Azure.')
param tags object = {}

// provision a user assigned identify for this aks
resource uai 'Microsoft.ManagedIdentity/userAssignedIdentities@2022-01-31-preview' = {
  name: uaiName
  location: region
  tags: tags
}

resource aks 'Microsoft.ContainerService/managedClusters@2022-05-02-preview' = {
  name: clusterName
  location: region
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/${uai.name}': {}
    }
  }
  properties: {
    dnsPrefix: dnsPrefix
    addonProfiles: {
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
        osDiskSizeGB: osDiskSizeGB
        count: agentCount
        vmSize: agentVMSize
        osType: 'Linux'
        mode: 'System'
      }
    ]
  }
}

output aksControlPlaneFQDN string = aks.properties.fqdn
output aksId string = aks.id
