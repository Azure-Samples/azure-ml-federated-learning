// Creates a network security group preconfigured for use with Azure ML computes
// To learn more, see https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-training-vnet?tabs=cli%2Crequired#compute-instancecluster-with-no-public-ip

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
    priority: 150
    direction: 'Outbound'
  }
}

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
    priority: 160
    direction: 'Outbound'
  }
}

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
    priority: 170
    direction: 'Outbound'
  }
}

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
    priority: 180
    direction: 'Outbound'
  }
}

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
    priority: 181
    direction: 'Outbound'
  }
}

output id string = nsg.id
