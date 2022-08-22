function Wait-SuccessfulRegistration {
    param (
        $ProviderName
    )
    Write-Output "Waiting for successful registration of the $ProviderName provider."
    $Provider = az provider show -n $ProviderName | ConvertFrom-Json
    while ($Provider.RegistrationState -ne "Registered"){
        Start-Sleep -Seconds 60
        $Provider = az provider show -n $ProviderName | ConvertFrom-Json
    }
    Write-Output "The $ProviderName provider has been successfuly registered."
}

function Deploy-RGIfInexistent {
    Param (
        $RGName,
        $RGLocation,
        $Purpose
    )
    Write-Output "Name of the $Purpose resource group to create: $RGName, in $RGLocation location."
    if ( $(az group exists --name $RGName) -eq $true ){
        Write-Output "The resource group '$RGName' already exists."
    } else {
        Write-Output "Creating the resource group..."
        az deployment sub create --name $RGName --location $RGLocation --template-file $PSScriptRoot/../bicep/ResourceGroup.bicep --parameters rgname=$RGName rglocation=$RGLocation
    }
}

function Confirm-Name {
    Param(
        $Name,
        $ResourceType
    )
    Write-Output "Validating requested $ResourceType name..."
    if ($ResourceType -eq "Compute"){
        $RegEx = '^[a-z0-9-]{2,16}$'
    } elseif ($ResourceType -eq "AMLWorkspace"){
        $RegEx = '^[a-zA-Z0-9-]{3,21}$'
    } else {
        Write-Error "Invalid resource type: $ResourceType. It should be either 'Compute' or 'AMLWorkspace'."
        exit
    }
    
    if ($Name -match $RegEx){
        Write-Output "$ResourceType name $Name is valid."
    } else{
        Write-Error "$ResourceType name $Name is invalid. It can include letters, digits and dashes. It must start with a letter, end with a letter or digit, and be between {2 for Compute, 3 for AMLWorkspace} and {16 for Compute, 21 for AMLWorkspace} characters in length."
        if ($ResourceType -eq "AMLWorkspace"){
            Write-Error "Note that even though Azure ML does accept underscores in workspace names, this provisioning tool does not. This restriction could be lifted in a future iteration if needed."
        }
        exit
    }
}