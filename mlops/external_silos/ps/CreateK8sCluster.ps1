Param(
    [Parameter(Mandatory=$true,
    HelpMessage="The guid of the subscription where we'll be preparing the silo.")]
    [string]
    $SubscriptionId,
    [Parameter(Mandatory=$false,
    HelpMessage="The name of the K8s cluster to create (this will NOT be the name of the attached compute in the Azure ML portal).")]
    [string]
    $K8sClusterName="cont-k8s-01",
    [Parameter(Mandatory=$false,
    HelpMessage="The region where to create the K8s cluster (westus2 by default).")]
    [string]
    $RGLocation="westus2",
    [Parameter(Mandatory=$false,
    HelpMessage="The number of agents in the K8s cluster (1 by default).")]
    [int]
    $AgentCount=1,
    [Parameter(Mandatory=$false,
    HelpMessage="The agent VM SKU ('Standard_A2_v2' by default).")]
    [string]
    $AgentVMSize="Standard_A2_v2"
)

# load useful functions
. "$PSScriptRoot/AzureUtilities.ps1"

# making sure we're in the right subscription
Write-Output "We'll be setting up a silo in this subscription: $SubscriptionId."
# log in
az login
az account set --subscription $SubscriptionId


# create RG, if it doesn't exist
$RGName = $K8sClusterName+"-rg" # name is derived from the K8s cluster name
Deploy-RGIfInexistent $RGName $RGLocation "K8s cluster"

# create cluster, if it doesn't exist
Write-Output "Name of the K8s cluster to create: $K8sClusterName, with $AgentCount agent(s) of SKU '$AgentVMSize'." 
$ManagedClusters =  az aks list --resource-group $RGName --query "[?name=='$K8sClusterName']" | ConvertFrom-Json
if ($ManagedClusters.Length -eq 0){
    Write-Output "Creating the K8s cluster..."
    $DeploymentName = $K8sClusterName + "-deployment"
    az deployment group create --resource-group $RGName --name $DeploymentName --template-file "./arm/managed_k8s.json" --parameters aksClusterName=$K8sClusterName dnsPrefix=$K8sClusterName agentCount=$AgentCount agentVMSize=$AgentVMSize
} else {
    Write-Output "The K8s cluster $K8sClusterName already exists."
}

# Get the kubeconfig file ready
Write-Output "Preparing the kube config file to connect to the K8s cluster..."
az aks get-credentials --resource-group $RGName --name $K8sClusterName