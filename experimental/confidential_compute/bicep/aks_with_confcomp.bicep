// https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-bicep?tabs=azure-cli%2CCLI

@description('The name of the Managed Cluster resource.')
param clusterName string

@description('The location of the Managed Cluster resource.')
param region string = resourceGroup().location

@description('User name for the Linux Virtual Machines.')
param linuxAdminUsername string

@description('Configure all linux machines with the SSH RSA public key string. Your key should include three parts, for example \'ssh-rsa AAAAB...snip...UcyupgH azureuser@linuxvm\'')
param sshRSAPublicKey string

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


resource aks 'Microsoft.ContainerService/managedClusters@2022-05-02-preview' = {
  name: clusterName
  location: region
  identity: {
    type: 'SystemAssigned'
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
    linuxProfile: {
      adminUsername: linuxAdminUsername
      ssh: {
        publicKeys: [
          {
            keyData: sshRSAPublicKey
          }
        ]
      }
    }
  }
}

output aksControlPlaneFQDN string = aks.properties.fqdn
output aksId string = aks.id
