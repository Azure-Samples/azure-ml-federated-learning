// Creates a network security group preconfigured for use with Azure ML computes
// To learn more, see https://learn.microsoft.com/en-us/azure/machine-learning/how-to-secure-training-vnet?tabs=cli%2Crequired#compute-instancecluster-with-no-public-ip

@description('Azure region of the deployment')
param location string

@description('Region of the AzureML workspace')
param workspaceRegion string

@description('Tags to add to the resources')
param tags object

@description('Name of the network security group')
param nsgName string

@description('Set rules to allow for compute with public IP')
param enableNodePublicIp bool = false

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
    priority: 151
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
    priority: 152
    direction: 'Outbound'
  }
}

resource AzureStorageAccount 'Microsoft.Network/networkSecurityGroups/securityRules@2022-07-01'= {
  name: 'AzureStorageAccount'
  parent: nsg
  properties: {
    protocol: 'Tcp'
    sourcePortRange: '*'
    destinationPortRange: '443'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: 'Storage.${workspaceRegion}'
    access: 'Allow'
    priority: 143
    direction: 'Outbound'
  }
}

output id string = nsg.id
