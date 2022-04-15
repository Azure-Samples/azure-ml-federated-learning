Param(
    [Parameter(Mandatory=$true,
    HelpMessage="The guid of the subscription where K8s cluster lives in.")]
    [string]
    $SubscriptionId,
    [Parameter(Mandatory=$false,
    HelpMessage="The name of the created K8s cluster.")]
    [string]
    $K8sClusterName="cont-k8s-01"
 )

$K8sClusterRGName = $K8sClusterName + "-rg"

az login
az account set --subscription $SubscriptionId
az aks get-credentials --resource-group $K8sClusterRGName --name $K8sClusterName
Write-Output "Connecting to the k8s"

$kubenodes = kubectl get nodes
$kubenodes_name = $kubenodes[1].Split(" ")[0]
Write-Output "Find the node $kubenodes_name"
kubectl label nodes $kubenodes_name training_node=my_beefy_node
kubectl apply -f kube_instance/my_instance_type.yaml
Write-Output "Done!"