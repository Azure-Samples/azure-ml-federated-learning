// Creates a network security group preconfigured for use with Azure ML
// To learn more, see:
// https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-workspace-vnet
// https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-azureml-behind-firewall

targetScope = 'resourceGroup'

@description('Name of the network security group')
param nsgName string

@description('Azure region of the deployment')
param location string = resourceGroup().location

@description('Region of the AzureML workspace')
param workspaceRegion string = resourceGroup().location

@description('Set rules to allow for compute with public IP')
param enableNodePublicIp bool = false

@description('Tags to add to the resources')
param tags object = {}


resource nsg 'Microsoft.Network/networkSecurityGroups@2022-07-01' = {
  name: nsgName
  location: location
  tags: tags
}

// AzureML compute required inbound rules - service tags
// required only when using public IP
resource amlPublicIPInboundSecurityRule 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= if (enableNodePublicIp) {
  name: 'AzureMLPublicIPInbound'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRange: '44224'
    sourceAddressPrefix: 'AzureMachineLearning'
    destinationAddressPrefix: '*'
    access: 'Allow'
    priority: 130
    direction: 'Inbound'
  }
}

// AzureML compute required outbound rules - service tags
resource amlOutboundTcpSecurityRule 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'AzureMLOutboundTcp'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRanges: ['443','8787','18881']
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'AzureMachineLearning'
    access: 'Allow'
    priority: 140
    direction: 'Outbound'
  }
}

// AzureML compute required outbound rules - service tags
resource amlOutboundUdpSecurityRule 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'AzureMLOutboundUdp'
  parent: nsg
  properties: {
    protocol: 'Udp'
    sourcePortRange: '*'
    destinationPortRange: '5831'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'AzureMachineLearning'
    access: 'Allow'
    priority: 150
    direction: 'Outbound'
  }
}

// AzureML compute required outbound rules - service tags
resource BatchNodeManagementOutbound 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'BatchNodeManagementOutbound'
  parent: nsg
  properties: {
    protocol: '*'
    sourcePortRange: '*'
    destinationPortRange: '443'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'BatchNodeManagement.${workspaceRegion}'
    access: 'Allow'
    priority: 160
    direction: 'Outbound'
  }
}

// to interact with the orchestrator storage
resource OrchestratorStorageAccount 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'OrchestratorStorageAccount'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRange: '443'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'Storage.${workspaceRegion}'
    access: 'Allow'
    priority: 170
    direction: 'Outbound'
  }
}

// to interact with the silo storage
resource SiloStorageAccount 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'SiloStorageAccount'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRange: '443'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'Storage.${location}'
    access: 'Allow'
    priority: 171
    direction: 'Outbound'
  }
}

// unused?
// resource armRule 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
//   name: 'AzureResourceManager'
//   parent: nsg
//   properties: {
//     protocol: 'Tcp'
//     sourcePortRange: '*'
//     destinationPortRange: '443'
//     sourceAddressPrefix: '*'
//     destinationAddressPrefix: 'AzureResourceManager'
//     access: 'Allow'
//     priority: 160
//     direction: 'Outbound'
//   }
// }

// to interact with the Azure Key Vault
resource aadRule 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'AzureActiveDirectory'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRange: '*'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'AzureActiveDirectory'
    access: 'Allow'
    priority: 180
    direction: 'Outbound'
  }
}

// required for ACR interaction (also see acrRule)
resource afdRule 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'AzureFrontDoor'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRange: '443'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'AzureFrontDoor.FrontEnd'
    access: 'Allow'
    priority: 190
    direction: 'Outbound'
  }
}

// required for ACR interaction
resource acrOrchRule 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'AzureContainerRegistryOrch'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRange: '443'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'AzureContainerRegistry.${workspaceRegion}'
    access: 'Allow'
    priority: 200
    direction: 'Outbound'
  }
}

// required for ACR interaction
resource acrSiloRule 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'AzureContainerRegistrySilo'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRange: '443'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'AzureContainerRegistry.${location}'
    access: 'Allow'
    priority: 201
    direction: 'Outbound'
  }
}

// required for pulling images directly from MCR
resource mcrRule 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'MicrosoftContainerRegistry'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRange: '443'
    sourceAddressPrefix: 'VirtualNetwork'
    destinationAddressPrefix: 'MicrosoftContainerRegistry'
    access: 'Allow'
    priority: 210
    direction: 'Outbound'
  }
}

output id string = nsg.id
