// This script will provision a confidential compute virtual machine
// on a given virtual network and subnet.

// Optionally, it will also create a bastion host to access the VM.

// VNET SETTINGS
@description('Name of virtual network to attach jumpbox to')
param vnetName string
@description('Name of subnet to attach jumpbox to')
param subnetName string
@description('Name of network security group on the vnet')
param nsgName string

// JUMPBOX SETTINGS
@description('Name of the virtual machine resource')
param jumpboxVmName string = 'jumpbox-cc-vm'
@description('Azure region for Bastion and virtual network')
param location string = resourceGroup().location
@description('Name of the network interface')
param jumpboxNetworkInterfaceName string = 'nic-${jumpboxVmName}'
@description('User name for the Virtual Machine.')
param adminUsername string
@description('Password for the Virtual Machine.')
@secure()
param adminPassword string

@description('The size of the Virtual Machine.')
@allowed([
  // see https://learn.microsoft.com/en-us/azure/virtual-machines/dcv3-series
  'Standard_DC1ds_v3'
  'Standard_DC2ds_v3'
  'Standard_DC4ds_v3'
  'Standard_DC8ds_v3'
  'Standard_DC16ds_v3'
  'Standard_DC24ds_v3'
  'Standard_DC32ds_v3'
  'Standard_DC48ds_v3'
  // see https://learn.microsoft.com/en-us/azure/virtual-machines/dcasv5-dcadsv5-series
  'Standard_DC2as_v5'
  'Standard_DC4as_v5'
  'Standard_DC8as_v5'
  'Standard_DC16as_v5'
  'Standard_DC32as_v5'
  'Standard_DC48as_v5'
  'Standard_DC64as_v5'
  'Standard_DC96as_v5'
  'Standard_DC2ads_v5'
  'Standard_DC4ads_v5'
  'Standard_DC8ads_v5'
  'Standard_DC16ads_v5'
  'Standard_DC32ads_v5'
  'Standard_DC48ads_v5'
  'Standard_DC64ads_v5'
  'Standard_DC96ads_v5'
])
param jumpboxVmSize string = 'Standard_DC4ads_v5'

@allowed([
  'linux'
  'windows'
])
param jumpboxOs string = 'linux'

@description('Specifies the storage account type for the managed disk.')
@allowed([
  'PremiumV2_LRS'
  'Premium_LRS'
  'Premium_ZRS'
  'StandardSSD_LRS'
  'StandardSSD_ZRS'
  'Standard_LRS'
  'UltraSSD_LRS'
])
param osDiskType string = 'Premium_LRS'

@description('Specifies whether OS Disk should be deleted or detached upon VM deletion.')
@allowed([
  'Delete'
  'Detach'
])
param osDiskDeleteOption string = 'Delete'

// BASTION SETTINGS (OPTIONAL)
param provisionBastion bool = false
param bastionHostName string = 'jumpbox-bastion'
param bastionNsgName string = 'nsg-bastion-${bastionHostName}'
param bastionSubnetName string = 'AzureBastionSubnet'
param bastionSubnetPrefix string = '10.0.250.0/27'
param bastionPublicIpName string = 'publicip-bastion-${bastionHostName}'

@description('Tags to add to the resources')
param tags object = {}

// look for the existing vnet,subnet,nsg in which to create jumpbox
resource vnet 'Microsoft.Network/virtualNetworks@2020-07-01' existing = {
  name: vnetName
}
resource subnet 'Microsoft.Network/virtualNetworks/subnets@2020-07-01' existing = {
  parent: vnet
  name: subnetName
}
resource nsg 'Microsoft.Network/networkSecurityGroups@2020-07-01' = {
  name: nsgName
  location: location
}

// we're picking specific images depending on required OS
var imageReference = jumpboxOs == 'linux' ? {
  publisher: 'canonical'
  offer: '0001-com-ubuntu-confidential-vm-focal'
  sku: '20_04-lts-cvm'
  version: 'latest'
} : {
  publisher: 'MicrosoftWindowsServer'
  offer: 'WindowsServer'
  sku: '2019-datacenter-smalldisk-g2'
  version: 'latest'
}
// and settings specifics to the OS
var osProfile = jumpboxOs == 'linux' ? {
  computerName: jumpboxVmName
  adminUsername: adminUsername
  adminPassword: adminPassword
  linuxConfiguration: {
    patchSettings: {
      patchMode: 'ImageDefault'
    }
  }
} : {
  computerName: jumpboxVmName
  adminUsername: adminUsername
  adminPassword: adminPassword
  windowsConfiguration: {
    enableAutomaticUpdates: true
    provisionVmAgent: true
    patchSettings: {
        enableHotpatching: false
        patchMode: 'AutomaticByOS'
    }
  }
}

// network interface for the jumpbox VM
resource jumpboxNetworkInterface 'Microsoft.Network/networkInterfaces@2021-03-01' = {
  name: jumpboxNetworkInterfaceName
  location: location
  tags: tags 
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: {
            id: subnet.id //plug into the target vnet
          }
          privateIPAllocationMethod: 'Dynamic'
        }
      }
    ]
    networkSecurityGroup: {
      id: nsg.id
    }
  }
}

resource virtualMachine 'Microsoft.Compute/virtualMachines@2022-03-01' = {
  name: jumpboxVmName
  location: location
  tags: tags 
  zones: [
    '2'
  ]
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    hardwareProfile: {
      vmSize: jumpboxVmSize
    }
    storageProfile: {
      osDisk: {
        createOption: 'fromImage'
        managedDisk: {
          storageAccountType: osDiskType
          securityProfile: {
            securityEncryptionType: 'DiskWithVMGuestState'
          }
        }
        deleteOption: osDiskDeleteOption
      }
      imageReference: imageReference
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: jumpboxNetworkInterface.id
          properties: {
            deleteOption: 'Detach'
          }
        }
      ]
    }
    osProfile: osProfile
    securityProfile: {
      securityType: 'ConfidentialVM'
      uefiSettings: {
        secureBootEnabled: true
        vTpmEnabled: true
      }
    }
    diagnosticsProfile: {
      bootDiagnostics: {
        enabled: true
      }
    }
  }
}



// this public ip is for the bastion host
resource bastionPublicIpAddress 'Microsoft.Network/publicIpAddresses@2020-07-01' = if (provisionBastion) {
  name: bastionPublicIpName
  location: location
  tags: tags 
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

// these rules are designed to allow access to the bastion host
resource bastionNsg 'Microsoft.Network/networkSecurityGroups@2020-07-01' = if (provisionBastion) {
  name: bastionNsgName
  location: location
  tags: tags 
  properties: {
    securityRules: [
      {
        name: 'AllowHttpsInBound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: 'Internet'
          destinationPortRange: '443'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 100
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowGatewayManagerInBound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: 'GatewayManager'
          destinationPortRange: '443'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 110
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowLoadBalancerInBound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: 'AzureLoadBalancer'
          destinationPortRange: '443'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 120
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowBastionHostCommunicationInBound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationPortRanges: [
            '8080'
            '5701'
          ]
          destinationAddressPrefix: 'VirtualNetwork'
          access: 'Allow'
          priority: 130
          direction: 'Inbound'
        }
      }
      {
        name: 'DenyAllInBound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          sourceAddressPrefix: '*'
          destinationPortRange: '*'
          destinationAddressPrefix: '*'
          access: 'Deny'
          priority: 1000
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowSshRdpOutBound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: '*'
          destinationPortRanges: [
            '22'
            '3389'
          ]
          destinationAddressPrefix: 'VirtualNetwork'
          access: 'Allow'
          priority: 100
          direction: 'Outbound'
        }
      }
      {
        name: 'AllowAzureCloudCommunicationOutBound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: '*'
          destinationPortRange: '443'
          destinationAddressPrefix: 'AzureCloud'
          access: 'Allow'
          priority: 110
          direction: 'Outbound'
        }
      }
      {
        name: 'AllowBastionHostCommunicationOutBound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationPortRanges: [
            '8080'
            '5701'
          ]
          destinationAddressPrefix: 'VirtualNetwork'
          access: 'Allow'
          priority: 120
          direction: 'Outbound'
        }
      }
      {
        name: 'AllowGetSessionInformationOutBound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: 'Internet'
          destinationPortRanges: [
            '80'
            '443'
          ]
          access: 'Allow'
          priority: 130
          direction: 'Outbound'
        }
      }
      {
        name: 'DenyAllOutBound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
          access: 'Deny'
          priority: 1000
          direction: 'Outbound'
        }
      }
    ]
  }
}

// create a subnet only for the bastion host
resource bastionSubnet 'Microsoft.Network/virtualNetworks/subnets@2020-07-01' = if (provisionBastion) {
  parent: vnet
  name: bastionSubnetName
  properties: {
    addressPrefix: bastionSubnetPrefix
    networkSecurityGroup: {
      id: bastionNsg.id
    }
  }
}

// create the bastion host using public ip, on its own subnet
resource bastionHost 'Microsoft.Network/bastionHosts@2020-07-01' = if (provisionBastion) {
  name: bastionHostName
  location: location
  tags: tags 
  properties: {
    ipConfigurations: [
      {
        name: 'IpConf'
        properties: {
          subnet: {
            id: bastionSubnet.id
          }
          publicIPAddress: {
            id: bastionPublicIpAddress.id
          }
        }
      }
    ]
  }
  dependsOn: [
    subnet
  ]
}
