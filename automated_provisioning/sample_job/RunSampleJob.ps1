Param(
    [Parameter(Mandatory=$true,
    HelpMessage="The guid of the subscription to which the orchestrator belongs.")]
    [string]
    $SubscriptionId,
    [Parameter(Mandatory=$true,
    HelpMessage="The name of the orchestrator AML workspace.")]
    [string]
    $WorkspaceName,
    [Parameter(Mandatory=$true,
    HelpMessage="The name of the orchestrator AML workspace resource group.")]
    [string]
    $ResourceGroup  
)

# log in
az login
# making sure we're in the right subscription
az account set --subscription $SubscriptionId


# run the job
Write-Output "Submitting the job..."
az ml job create -f ./sample_job/job.yml --web -g $ResourceGroup -w $WorkspaceName
