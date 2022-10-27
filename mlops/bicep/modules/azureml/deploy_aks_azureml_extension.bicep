// Deploy AzureML extension on Azure Kubernetes Service (AKS) cluster
// Note: this can take up to 10 minutes.

// resource group must be specified as scope in az cli or module call
targetScope='resourceGroup'

// required
@description('Name of the AKS cluster in the resource group.')
param clusterName string

// Azure ML extension options
// see https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-kubernetes-extension
param extensionDeploymentName string = 'azmlext'

@description('Must be set to True for AzureML extension deployment with Machine Learning model training and batch scoring support.')
param enableTraining bool = true

@description('Must be set to True for AzureML extension deployment with Machine Learning inference support.')
param enableInference bool = false

@description('Required if enableInference=True.')
@allowed(['loadBalancer', 'nodePort', 'clusterIP'])
param inferenceRouterServiceType string = 'loadBalancer'

@description('Can be set to True to use inference HTTP endpoints for development or test purposes.')
param allowInsecureConnections bool = true

@description('Set to azure to allow the inference router using internal load balancer. This config is only applicable for Azure Kubernetes Service(AKS) cluster now.')
param internalLoadBalancerProvider string = 'azure'

@description('To ensure high availability of azureml-fe routing service (for clusters with 3 nodes or more).')
param inferenceLoadBalancerHA bool = false

@description('To enable ML workloads on NVIDIA GPU hardware.')
param installNvidiaDevicePlugin bool = false

@description('AzureML extension needs prometheus operator to manage prometheus. Set to False to reuse the existing prometheus operator.')
param installPromOp bool = true

@description('AzureML extension needs volcano scheduler to schedule the job. Set to False to reuse existing volcano scheduler.')
param installVolcano bool = true

@description('Dcgm-exporter can expose GPU metrics for AzureML workloads, which can be monitored in Azure portal. Set installDcgmExporter to True to install dcgm-exporter.')
param installDcgmExporter bool = false

// // Generic extension options
// // see https://learn.microsoft.com/en-us/azure/aks/cluster-extensions#optional-parameters
// @description('Specifies if the extension minor version will be upgraded automatically or not.')
// param autoUpgradeMinorVersion bool = true

// @description('Extension authors can publish versions in different release trains such as Stable, Preview, etc.')
// param releaseTrain string = 'Stable'

// @description('indicates the namespace within which the release is to be created. This parameter is only relevant if scope parameter is set to cluster.')
// param releaseNamespace string = 'azureml'

resource aks 'Microsoft.ContainerService/managedClusters@2022-05-02-preview' existing = {
  name: clusterName
}

resource deployAksAzuremlExtension 'Microsoft.KubernetesConfiguration/extensions@2022-07-01' = {
  name: 'deploy-aks-azureml-extensions-to-${clusterName}'
  scope: aks
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    aksAssignedIdentity: {
      type: 'UserAssigned'
    }
    configurationSettings: {
      extensionDeploymentName: extensionDeploymentName
      enableTraining: '${enableTraining}'
      enableInference: '${enableInference}'
      inferenceRouterServiceType: inferenceRouterServiceType
      allowInsecureConnections: '${allowInsecureConnections}'
      internalLoadBalancerProvider: internalLoadBalancerProvider
      inferenceLoadBalancerHA: '${inferenceLoadBalancerHA}'
      installNvidiaDevicePlugin: '${installNvidiaDevicePlugin}'
      installPromOp: '${installPromOp}'
      installVolcano: '${installVolcano}'
      installDcgmExporter: '${installDcgmExporter}'
    }
    extensionType: 'Microsoft.AzureML.Kubernetes'
    scope: {
      cluster: {
        releaseNamespace: 'azureml'
      }
    }
  }
}
