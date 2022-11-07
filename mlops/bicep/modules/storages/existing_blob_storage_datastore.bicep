// Creates a datastore for an existing storage account in the same tenant
@description('Name of AzureML workspace to attach compute+storage to.')
param machineLearningName string

@description('Existing storage account name to attach to the pair.')
param storageAccountName string

@description('Azure region of the storage to create')
param storageRegion string

@description('Resource group of the existing storage account to attach to the pair.')
param storageAccountResourceGroup string = resourceGroup().name

@description('SubscriptionId of the existing storage account to attach to the pair.')
param storageAccountSubscriptionId string = subscription().subscriptionId

@description('Name of the storage container resource to create for the pair')
param containerName string = 'private'

@description('Name of the datastore for attaching the storage to the AzureML workspace.')
param datastoreName string = replace('datastore_${storageAccountName}','-','_')

@description('Tags to add to the resources')
param tags object = {}

var storageId = '/subscriptions/${storageAccountSubscriptionId}/resourceGroups/${storageAccountResourceGroup}/providers/Microsoft.Storage/storageAccounts/${storageAccountName}'

// attach as a datastore in AzureML
resource datastore 'Microsoft.MachineLearningServices/workspaces/datastores@2022-06-01-preview' = {
  name: '${machineLearningName}/${datastoreName}'
  properties: {
    credentials: {
      credentialsType: 'None'
    }
    description: 'Private storage in region ${storageRegion}'
    properties: {}
    datastoreType: 'AzureBlob'

    accountName: storageAccountName
    containerName: containerName
    resourceGroup: storageAccountResourceGroup
    subscriptionId: storageAccountSubscriptionId
    tags: tags
  }
}

// output storage references
output storageId string = storageId
output storageName string = storageAccountName
output containerName string = containerName
output datastoreName string = datastore.name
