Param(
    [Parameter(Mandatory=$true,
    HelpMessage="The guid of the subscription where K8s cluster lives in.")]
    [string]
    $SubscriptionId,
    [Parameter(Mandatory=$true,
    HelpMessage="The name of the resource group.")]
    [string]
    $RGName,
    [Parameter(Mandatory=$true,
    HelpMessage="The name of the provisioned AKS compute.")]
    [string]
    $AKSName
 )

 $ScriptPath = split-path -parent $MyInvocation.MyCommand.Definition

az login
az account set --subscription $SubscriptionId
az aks get-credentials --resource-group $RGName --name $AKSName
Write-Output "Connecting to the k8s"

$kubenodes = kubectl get nodes
$kubenodes_name = $kubenodes[1].Split(" ")[0]
Write-Output "Find the node $kubenodes_name"
# kubectl label nodes $kubenodes_name training_node=my_beefy_node
kubectl apply -f $ScriptPath/instance-type-name.yaml
Write-Output "Done!"