# Provisioning a Federated Learning setup with internal silos

## Contents

This directory contains resources to help you provision a Federated Learning setup with _internal_ silos, _i.e._ silos that are in the same Azure tenant as the orchestrator. You will be able to use this setup to run the examples in the `../../examples/pipelines` directory.

For this first iteration, we are only provisioning a _vanilla_ setup, _i.e._ a setup with no security or governance features, where the silos are NOT locked down. We will add these features in future iterations. In the mean time, you should NOT be uploading sensitive data to this setup. However, you CAN use this setup for refining your training pipelines and algorithms. The only elements that will need to change when working on a _real_ secure setup will just be the orchestrator and silos names in a config file - you will be able to re-use all your code as-is.

> Due to a bug in the v2 Azure ML SDK/CLI, the location of the silos computes cannot currently be customized - all silos computes will be in the same region as the orchestrator. This makes the setup "even more vanilla", but does not really change anything else.

## Outcome

By the end, you will have an Azure ML workspace with a compute cluster (the orchestrator), 3 additional computes (the 3 silos), and train/test data for each silo. 

You will be able to use this setup as a sandbox to refine your pipelines and algorithms before you run them on production data in a secure setting. Porting your code to run on a production setup will not require any modification besides adjusting orchestrator and silo names in a config file.

## Prerequisites

To ensure that you can provision the setup, you will need the following:
- an Azure subscription in which you have the Contributor role;
- a shell application on your local machine (_e.g._ bash, PowerShell, etc...);
- Docker and Git on your local machine.

## Procedure

In this section, we outline the procedure to automatically provision the setup. 

1. First, clone the current repo onto your local machine.
2. Second, open the config file [`mlops/internal_silos/YAML/setup_info.yml`](./YAML/setup_info.yml) and enter your subscription Id. Feel free to change the name and resource group of the orchestrator workspace if you so wish, and the names of the silos too.
    - Note that for the time being, due to the bug mentioned in the **Contents** section, there is no point modifying the silos' locations. 
3. Now prepare your docker environment.
    - Go to the `mlops` folder.
    - Build the docker image by running the following command (don't forget the dot at the end).
    ```ps1
    docker build --file ./internal_silos/Dockerfile -t fl-rp-vanilla .
    ```
    - Start your docker environment by running the following.
    ```ps1
    docker run -i fl-rp-vanilla
    ```
4. Run the provisioning script from your docker environment.
    - In the docker environment, the working directory should now be `mlops/internal_silos`.
    - Run the provisioning script with this command.
    ```ps1
    ./ps/ProvisionVanillaSetup.ps1
    ```

> If you need to modify any file after having completed step 3, you will need to either modify the file _from within the docker environment_, or _modify on your local machine and rebuild the docker environment_ for the changes to be taken into account.