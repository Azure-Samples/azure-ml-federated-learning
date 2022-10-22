// required
param clusterName string

// options
param extensionDeploymentName string = 'azmlext'
param enableTraining bool = true
param enableInference bool = false
param inferenceRouterServiceType string = 'LoadBalancer'
param allowInsecureConnections bool = true
param inferenceLoadBalancerHA bool = false

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
      inferenceLoadBalancerHA: '${inferenceLoadBalancerHA}'
    }
    extensionType: 'Microsoft.AzureML.Kubernetes'
    scope: {
      cluster: {
        releaseNamespace: 'azureml'
      }
    }
  }
}
